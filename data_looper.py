#
# Multiprocessing looper for .data
# cutting injection
# 
# (2026) NanaVan@github
# Inspired by loopr.py by (2025) xaratustrah@github
#

import os, sys, re, time, signal, multiprocessing, concurrent.futures


def handle_windows(window_length, window=None, beta=None):
    '''
    handling various windows

    window_length:      length of the tapering window
    window:         to be chosen from ["bartlett", "blackman", "hamming", "hanning", "kaiser"]
                        if None, a rectangular window is implied
                        if "kaiser" is given, an additional argument of beta is expected
    '''
    import numpy as np
    if window is None:
        window_sequence = np.ones(window_length)
    elif window == "kaiser":
        if beta is None:
            raise ValueError("additional argument beta is empty!")
        else:
            window_sequence = np.kaiser(window_length, beta)
    else:
        window_func = getattr(np, window)
        window_sequence = window_func(window_length)
    return window_sequence



def _async_processing_task(x_stack, info):
    """
    子线程任务：负责严格的 pyfftw 计算、归一化、绘图和存盘
    x_stack: 形状为 (n_frames, n_average, window_length) 的三维数组
             或者是单次注入拼接好的完整一维信号
    """
    # 启用 pyfftw 缓存以提升频繁创建 plan 的性能
    import numpy as np
    import pyfftw, os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = None
    try:
        todo = info['todo']
        output_dir = info['output_dir']
        base_name = info['base_name']
        W = info['window_length']
        D = info['D']
        n_average = info['n_average']
        n_hop = info['n_hop']
        sampling_rate = info['sampling_rate']
        window_sequence = info['window_sequence']
        win_sq_sum = info['win_sq_sum']
        
        # 1. 严格复现原有的 stride + window 逻辑
        # 假设传进来的是该 injection 的一维完整信号 x_injection
        x_inj = x_stack
        n_frames = (len(x_inj) - (W + D * (n_average - 1))) // n_hop + 1

        
        # 构造 3D 视图: (帧数, 平均段数, 窗口长度)
        # 这一步完全复现了你原有的交叠逻辑
        shape = (n_frames, n_average, W)
        strides = (x_inj.strides[0] * n_hop, x_inj.strides[0] * D, x_inj.strides[0])
        grid_signal = np.lib.stride_tricks.as_strided(x_inj, shape=shape, strides=strides)
        
        # 2. 严格使用 pyfftw 计算
        # 为当前任务创建对齐的内存
        fft_input = pyfftw.empty_aligned(shape, dtype='complex64')
        fft_input[:] = grid_signal * window_sequence
        
        # 创建 FFT 计划 (在子线程内独立)
        fft_obj = pyfftw.builders.fftn(fft_input, axes=(-1,), threads=1)
        fft_res = fft_obj()
        
        # 3. 能量归一化与平均 (严格复现原公式)
        # fftshift, 绝对值平方, 能量修正
        psd_all = np.absolute(np.fft.fftshift(fft_res, axes=-1))**2 / win_sq_sum / sampling_rate
        psd_array = np.mean(psd_all, axis=1) # 对 n_average 维度取平均，得到 (n_frames, W)

        if info['additional_psd'] is not None:
            try:
                psd_array = np.vstack((info['additional_psd'], psd_array))
                n_frames += info['additional_psd'].shape[0] 
            except Exception as ve:
                return f"FAILED: {base_name} during vstack. Error: {ve}"
        
        # 4. 频率裁剪与坐标生成
        frequencies = np.linspace(-sampling_rate/2, sampling_rate/2, W+1)
        if W % 2 == 1: frequencies += sampling_rate / (2*W)
        freq_idx_0, freq_idx_1 = np.searchsorted(frequencies, [-info['span']/2, info['span']/2])
        
        x_freq = frequencies[freq_idx_0:freq_idx_1+1] + info['center_frequency']
        y_time = np.arange(n_frames+1) / sampling_rate * n_hop
        z_psd = psd_array[:, freq_idx_0:freq_idx_1]

        # 5. I/O 保存与绘图
        if 'data_spectrum' in todo:
            np.savez(os.path.join(output_dir, f"{base_name}_spectrum.npz"), 
                     frequencies=x_freq[:-1], psd=np.mean(z_psd, axis=0))
        
        if 'data_spectrogram' in todo:
            np.savez(os.path.join(output_dir, f"{base_name}_spectrogram.npz"), 
                     frequencies=x_freq, times=y_time, psd_arrays=z_psd)

        if 'png_spectrum' in todo:
            fig = Figure(figsize=(10, 6))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            ax.plot(x_freq[:-1]*1e-3, np.mean(z_psd, axis=0))
            ax.set_yscale('log')
            ax.set_title(f'Average Spectrum\n{base_name}')
            ax.set_xlabel('Frequency [kHz]')
            ax.set_ylabel('Power Spectral Density [arb. unit]')
            ax.grid(True, which='both', ls='--', alpha=0.5)
            canvas.print_figure(os.path.join(output_dir, f"{base_name}_spectrum.png"), dpi=200)
            fig.clf()

        if 'png_spectrogram' in todo:
            fig = Figure(figsize=(12, 10))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            norm = colors.LogNorm(vmin=max(z_psd.min(), 1e-18), vmax=z_psd.max())
            pcm = ax.pcolormesh(x_freq*1e-3, y_time*1e3, z_psd, shading='flat', cmap='viridis', norm=norm)
            ax.set_xlabel('Frequency [kHz]')
            ax.set_ylabel('Time [ms]')
            ax.set_title(f'Waterfall Plot\n{base_name}')
            fig.colorbar(pcm, ax=ax).set_label('Power Spectral Density [arb. unit]')
            canvas.print_figure(os.path.join(output_dir, f"{base_name}_spectrogram.png"), dpi=200)
            fig.clf()
            
        return f"SUCCESS: {base_name}"
    except Exception as e:
        return f"FAILED: {base_name} with error {e}"
    finally:
        if fig is not None:
            fig.clf()
        plt.close('all') # 彻底清理 Matplotlib 后端句柄，防止现场长时间运行 OOM
        import gc
        gc.collect()


class FileProcessor:
    def __init__(self, SOURCE_DIR, OUTPUT_DIR, FILE_PREFIX, EXPECTED_SIZE, CHECK_INTERVAL):
        self.SOURCE_DIR = SOURCE_DIR
        self.OUTPUT_DIR = OUTPUT_DIR
        self.FILE_PREFIX = FILE_PREFIX
        self.EXPECTED_SIZE = EXPECTED_SIZE
        self.CHECK_INTERVAL = CHECK_INTERVAL
        self.running = True
        self._init_executor()
        self.futures = []
        self.current_index = self._get_start_index()
        # 注册信号捕获，处理 Ctrl+C
        signal.signal(signal.SIGINT, self._handle_exit)

    def _init_executor(self):
        '''初始化或重建进程池'''
        workers = max(1, multiprocessing.cpu_count()-1)
        print('[*] 正在启动/重建进程池 (workers: {:})'.format(workers))
        # 创建长周期的线程池，建议：进程数 = 物理核心数 - 1（给系统留一个核心）
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=self._worker_ini_fn)
        

    def _get_start_index(self):
        '''
        启动时检测输出文件夹，寻找最大的 index
        '''
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
            return 0
        
        existing_files = os.listdir(self.OUTPUT_DIR)
        indices = []
        for f in existing_files:
            indices.append(int(os.path.basename(f).split('_')[2].split('-')[-1]))
        if not indices:
            return 0

        last_index = max(indices)
        print(f"检测到已处理至索引：{last_index}，将从{last_index+1}开始。")
        return last_index + 1

    def _find_last_file(self, index):
        '''检索出最后生成的不完整文件'''
        index = index - 1
        target_files = []
        for filename in os.listdir(self.OUTPUT_DIR):
            if not filename.endswith('.npz'):
                continue
            if 'incomplete' in filename:
                try:
                    if int(filename.split('_')[2].split('-')[-1]) == index:
                        target_files.append(filename)
                except:
                    continue
        try:
            return os.path.basename(target_files[0])
        except:
            return None

    def _delete_assigned_incomplete_file(self, last_file):
        '''
        检查前序不完整文件是否是当前生成文件中已覆盖的：
        前序文件存在触发，当前生成文件使用了前序文件末尾数据合并成了频谱
        删除该不完整文件（特征是含有trigger_i_incomplete，其中i>0）
            8249_PY82ch1_0010_trigger_i_incomplete_2026-04-08T02-00-44_spectrogram.npz
        '''
        _match = re.search(r'trigger_(\d+)_incomplete', last_file)
        if _match:
            trigger_i = int(_match.group(1))
            if trigger_i > 0:
                file_path = os.path.join(self.OUTPUT_DIR, last_file)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print('[!] 删除掉前序不完整文件：{:}'.format(last_file))

    @staticmethod
    def _worker_ini_fn():
        # 子进程不处理 Ctrl+C，由主进程统一调度
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _handle_exit(self, signum, frame):
        '''捕获到 Ctrl+C 时的响应：只改变状态，不退出'''
        if self.running:
            print("\n[!] 接收到退出信号。正在处理当前文件，请稍后...")
            self.running = False # 停止新文件探索

    def is_file_ready(self, path):
        '''完整性校验：检查大小、写入停顿时间以及写锁状态'''
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < self.EXPECTED_SIZE:
            return False
        # 校验修改时间（防止采集软件 Flush 间隔误判，至少稳定 2 秒未修改）
        mtime = os.path.getmtime(path)
        if time.time() - mtime < 2.0:
            return False
        # 再次确认文件没有再写入
        s1 = os.path.getsize(path) # Bytes
        time.sleep(0.3)
        if s1 != os.path.getsize(path):
            return False
        # 尝试以追加/独占模式打开，确认采集程序已解除文件锁
        try:
            with open(path, 'a+b') as f:
                pass
        except IOError:
            return False

        return True

    def run(self, WIN_LEN, N_AVER, OVERLAPR, N_HOP, TODO):
        '''主处理循环'''
        print("程序启动，当前文件索引：{:}".format(self.current_index))

        try:
            while True:
                target_file = os.path.join(self.SOURCE_DIR, '{:}_{:}.data'.format(self.FILE_PREFIX, self.current_index))

                # 1. 先判断文件是否 ready
                if self.is_file_ready(target_file):

                    last_file = self._find_last_file(self.current_index)
                    lastfile_path = os.path.join(self.OUTPUT_DIR, self._find_last_file(self.current_index)) if last_file else None
                    time_0 = time.time()

                    try:
                        self.file_cutInjection(target_file, self.OUTPUT_DIR, WIN_LEN, N_AVER, OVERLAPR, self.executor, todo=TODO, n_hop=N_HOP, window='kaiser', beta=14, last_file=lastfile_path)
                        # 2. 阻塞等待当前文件的所有异步任务完成
                        if self.futures:
                            done, _ = concurrent.futures.wait(self.futures)
                            for f in done:
                                try:
                                    res = f.result()
                                    if "FAILED" in res: print("[!] 任务失败：{:}".format(res))
                                except Exception as e:
                                    print("[!] 任务执行崩溃：{:}".format(e))
                            self.futures = [] # 彻底清空任务列表

                    except concurrent.futures.process.BrokenProcessPool:
                        print("[!] 检测到进程池损坏，尝试重启进程池，并重启当前任务...")
                        self.executor.shutdown(wait=False)
                        self._init_executor()
                        continue

                    print('[*] 处理该文件耗时 {:.3f} sec'.format(time.time()-time_0))
                    if last_file is not None:
                        self._delete_assigned_incomplete_file(last_file)
                    self.current_index += 1

                    # 3. 手动触发垃圾回收
                    import gc
                    gc.collect()

                    # 处理完一个文件后，才去检查退出标志
                    if not self.running:
                        print("[!] 检测到退出信号，当前文件已处理结束。安全退出。")
                        break
                else: # 文件不 ready, 且用户按了 Ctrl+C 
                    if not self.running:
                        break
                    # 正常等待
                    time.sleep(self.CHECK_INTERVAL)
        finally:
            print("[*] 正在关闭进程池 ...")
            self.executor.shutdown(wait=True)
            print('[√] 程序已安全退出。')

    def file_cutInjection(self, input_file, output_dir, window_length, n_average, overlap_ratio, executor, todo, n_hop=None, window=None, beta=None, last_file=None):
        import numpy as np
        import pyfftw
        from preprocessing import Preprocessing
        
        # 参数初始化 ---
        window_sequence = handle_windows(window_length, window, beta)
        win_sq_sum = np.sum(window_sequence**2)
        D = int((1 - overlap_ratio) * window_length) 
        N = int(window_length + D * (n_average - 1))
        n_point = window_length
        if n_hop is None: n_hop = N
        
        _current_fileIndex = int(os.path.basename(input_file).split('_')[1].split('.')[0])
        _prefix = os.path.dirname(input_file).split('/')[-1].split('_')[0] + '_' + os.path.basename(input_file).split('_')[0]  # 8249_PY82ch1
    
        bud = Preprocessing(input_file, puyuan_new=True, abs_trigger=False)
        ThisFileTimestamp = bud.date_time + np.timedelta64(8, 'h') # convert to '+08' timezone
    
        # 处理前序文件的残余内容
        if last_file is None:
            additional_x, lastTriggerData_remain, trigger_frame, offset = np.array([]), 0, 0, 0
            trigger_inLastFile = False
            ThisDataTimestamp = ThisFileTimestamp
        else:
            trigger_inLastFile = False
            _macth = re.search(r'trigger_\d+', os.path.basename(last_file))
            if _macth: # 前序文件存在触发信号，trigger_inLastFile = True，反之亦然
                trigger_inLastFile = True if int(_macth.group().split('_')[-1]) > 0 else False
            with np.load(last_file) as _f:
                if trigger_inLastFile:
                    additional_x, lastTriggerData_remain, psd_array = _f['addition_data'], len(_f['addition_data']), _f['psd_arrays']
                    trigger_frame, offset = len(_f['times'])-1, 0
                    _match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', os.path.basename(last_file))
                    if _match:
                        ThisDataTimestamp = np.datetime64(_match.group(1)[:11] + _match.group(1)[11:].replace('-', ':'), 's')
                else:
                    additional_x, lastTriggerData_remain = _f['addition_data'], len(_f['addition_data'])
                    trigger_frame, offset = 0, 0
                    _match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})', os.path.basename(last_file))
                    if _match:
                        ThisDataTimestamp = np.datetime64(_match.group(1)[:11] + _match.group(1)[11:].replace('-', ':'), 's') + np.timedelta64(int(_f['times'][-1]), 's')
                    else:
                        ThisDataTimestamp = ThisFileTimestamp
    
            
        # 如果整个文件中都不存在触发信号，那么执行以下策略：1.前序文件含有触发信号，就将前序文件末尾的新注入数据与本文件可组成频谱的内容合并为新文件（incomplete），未组合成频谱部分放置于addition_data中；2.前序文件不含有触发信号，仅将前序文件addition_data中的数据与本文件可组成频谱的内容合并未新文件（incomplete），未组合成频谱部分放置于addition_data中。3.无前序文件（见于PY8*ch*_0.data），将本文件可组成频谱的内容合并为新文件（incomplete），未组合成频谱部分放置于addition_data中。
        if len(bud.trigger_timestamp) == 0:
            print('[!] 当前文件内无触发信号，将按普通频谱处理！')
            dummy = pyfftw.empty_aligned((n_average, window_length))
            fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=1)
            ThisTriggerData_remain = bud.n_sample
            while True:
                if ThisTriggerData_remain > N:
                    x = np.hstack((additional_x, bud.load(N-lastTriggerData_remain, offset)[1]))
                    _signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
                    if trigger_frame == 0:
                        psd_array = np.mean(np.absolute(np.fft.fftshift(fft(_signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)
                    else:
                        psd_array = np.vstack((psd_array, np.mean(np.absolute(np.fft.fftshift(fft(_signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)))
                    ThisTriggerData_remain -= n_hop
                    trigger_frame += 1
                    if lastTriggerData_remain > 0:
                        if lastTriggerData_remain - n_hop > 0:
                            offset = 0
                            additional_x = additional_x[n_hop:]
                            lastTriggerData_remain = len(additional_x)
                        else:
                            offset = n_hop - lastTriggerData_remain
                            additional_x = np.array([])
                            lastTriggerData_remain = 0
                    else:
                        offset += n_hop
                else:
                    additional_x = np.hstack((additional_x, bud.load(ThisTriggerData_remain, offset)[1]))
                    lastTriggerData_remain = len(additional_x)
                    offset = 0
                    frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1) # Hz
                    if n_point % 2 ==1: frequencies += bud.sampling_rate / (2*n_point)
                    x_frequency = frequencies + bud.center_frequency # Hz
                    y_time = np.arange(trigger_frame+1) / bud.sampling_rate * n_hop # s
                    z_psd_array = psd_array

                    time_str = ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S')
                    if trigger_inLastFile or (last_file is None):
                        out_name = f'{_prefix}_{_current_fileIndex:04d}_trigger_0_incomplete_{time_str}_spectrogram.npz' # 前序不含有触发信号或者前序文件不存在，生成格式为 8249_PY82ch1_0010_trigger_0_incomplete_2026-04-08T22-12-23_spectrogram.npz
                        print('[*] 当前无触发文件的前序文件也无触发或不存在，保留时频谱 ...')
                    else:
                        out_name = f'{_prefix}_{_current_fileIndex-1:04d}-{_current_fileIndex:04d}_trigger_incomplete_{time_str}_spectrogram.npz' # 前序文件含有触发信号，生成文件格式为 8249_PY82ch1_0010-0011_trigger_incomplete_2026-04-08T22-12-23_spectrogram.npz
                        print('[*] 当前无触发文件将与前序文件剩余部分合并，保留时频谱 ...')
                    np.savez(os.path.join(output_dir, out_name), frequencies=x_frequency, times=y_time, psd_arrays=z_psd_array, addition_data=additional_x)
                    print(f'[√] 文件生成：{out_name}')
                    break
            return
    
    
        # --- 核心修改：在触发信号循环中 ---
        for trigger_i, trigger_timestamp in enumerate(bud.trigger_timestamp):
            # --- 内存管理：定期清理已完成的任务 ---
            if len(self.futures) > 8:
                done, not_done = concurrent.futures.wait(
                    self.futures, timeout=0.1,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                self.futures = list(not_done)
        
            # 1. 仍然按照你的逻辑计算 ThisTriggerData_remain
            # 2. 将该 Injection 需要的所有原始数据一次性 load 出来（形成一个长向量 x_injection）
            #    这样可以避免在 while True 中频繁进行 vstack 导致的效率低下
            ThisTriggerData_remain = trigger_timestamp * bud.data_len + lastTriggerData_remain - offset
            if trigger_i == 0 and trigger_frame == 0:
                print('[!] 当前含新注入的文件为文件夹中起始文件，生成文件将从首次注入开始 ...')
                offset = trigger_timestamp * bud.data_len
                ThisDataTimestamp = ThisTriggerData_remain + np.timedelta64(int(offset/bud.sampling_rate), 's')
                continue
            
            if trigger_i == 0:
                time_str = ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S')
                base_name = f'{_prefix}_{_current_fileIndex-1:04d}-{_current_fileIndex:04d}_trigger_{time_str}'
                additional_psd = psd_array.copy()
                print('[!] 当前文件将和前序文件残余部分拼接，生成文件 {:}'.format(base_name))
            else:
                time_str = ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S')
                base_name = f'{_prefix}_{_current_fileIndex:04d}_trigger_{trigger_i}_{time_str}'
                additional_psd, additional_x  = None, np.array([])
            x_injection = np.hstack((additional_x, bud.load(trigger_timestamp * bud.data_len - offset, offset)[1]))
            # 3. 准备子线程所需的元数据
            info = {
                'window_length': window_length, 'D': D, 'n_average': n_average, 'n_hop': n_hop,
                'sampling_rate': bud.sampling_rate, 'window_sequence': window_sequence,
                'win_sq_sum': win_sq_sum, 'span': bud.span, 'center_frequency': bud.center_frequency,
                'todo': todo, 'output_dir': output_dir, 'base_name': base_name, 'additional_psd': additional_psd # 动态生成
            }
            # 4. 异步提交任务
            try:
                future = executor.submit(_async_processing_task, x_injection.copy(), info)
                self.futures.append(future)
            except concurrent.futures.process.BrokenProcessPool:
                raise

            lastTriggerData_remain, trigger_frame = 0, 0
            offset = trigger_timestamp * bud.data_len
            ThisDataTimestamp = ThisFileTimestamp + np.timedelta64(int(offset/bud.sampling_rate), 's')
    
        # 返回给主循环所需的断点信息
        trigger_i += 1
        ThisTriggerData_remain = bud.n_sample + lastTriggerData_remain - offset
        dummy = pyfftw.empty_aligned((n_average, window_length))
        fft = pyfftw.builders.fft(dummy, n=window_length, overwrite_input=True, threads=1)
        print('[*] 正在处理本文件剩余数据...')
        while True:
            if ThisTriggerData_remain > N:
                x = np.hstack((additional_x, bud.load(N-lastTriggerData_remain, offset)[1]))
                _signal = np.lib.stride_tricks.as_strided(x, (n_average, window_length), (x.strides[0] * D, x.strides[0])) * window_sequence
                if trigger_frame == 0:
                    psd_array = np.mean(np.absolute(np.fft.fftshift(fft(_signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)
                else:
                    psd_array = np.vstack((psd_array, np.mean(np.absolute(np.fft.fftshift(fft(_signal), axes=-1))**2 / np.sum(window_sequence**2) / bud.sampling_rate, axis=0)))
                ThisTriggerData_remain -= n_hop
                trigger_frame += 1
                offset += n_hop
            else:
                additional_x = np.hstack((additional_x, bud.load(ThisTriggerData_remain, offset)[1]))
                lastTriggerData_remain = len(additional_x)
                offset = 0
                frequencies = np.linspace(-bud.sampling_rate/2, bud.sampling_rate/2, n_point+1) # Hz
                if n_point % 2 ==1: frequencies += bud.sampling_rate / (2*n_point)
                x_frequency = frequencies + bud.center_frequency # Hz
                y_time = np.arange(trigger_frame+1) / bud.sampling_rate * n_hop # s
                z_psd_array = psd_array

                time_str = ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S')
                out_name = f'{_prefix}_{_current_fileIndex:04d}_trigger_{trigger_i}_incomplete_{time_str}_spectrogram.npz'
                np.savez(os.path.join(output_dir, out_name), frequencies=x_frequency, times=y_time, psd_arrays=z_psd_array, addition_data=additional_x)
                print(f'[√] 本文件剩余部分保存：{out_name}')
                break
    

if __name__ == "__main__":
    # 这一行在 Windows 下对于防止 spawn 死循环极其重要
    multiprocessing.freeze_support() 
    processor = FileProcessor(SOURCE_DIR, OUTPUT_DIR, FILE_PREFIX, EXPECTED_SIZE, CHECK_INTERVAL)
    processor.run(WIN_LEN, N_AVER, OVERLAPR, N_HOP, TODO)
