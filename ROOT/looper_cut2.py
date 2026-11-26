#
# Multiprocessing looper for .data
# cutting injection
# 
# (2026) NanaVan@github
# Inspired by loopr.py by (2025) xaratustrah@github
#

import os, sys, re, time, signal, multiprocessing, concurrent.futures
from pathlib import Path
from SpectrumAnalyzer import SpectrumAnalyzer

# 单次 _async_processing_task 中每批处理的帧数
FFT_CHUNK_SIZE = 64


def _writer_injection(spectrogram_queue, bg_precomputed, spectrogram_analysis_config, peaks_summary_path):
    """
    独立单次注入进程：从队列中取出 spectrogram 数据，执行全部 I/O 操作
    (.npz 保存、.png 渲染、Spectrogram 分析)，避免多 worker 并发写磁盘。
    """
    import signal, numpy as np
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(f'[*] 单次注入频谱分析进程已启动 (处理 .npz + .png + 分析)')
    try:
        while True:
            item = spectrogram_queue.get()
            if item is None:
                break
            (x_freq, y_time, z_psd, output_dir, base_name, todo) = item[:6]
            spectrogram_cfg = item[6] if len(item) > 6 else None
            freq_low = item[7] if len(item) > 7 else (x_freq[0] if len(x_freq) else 0)
            freq_high = item[8] if len(item) > 8 else (x_freq[-1] if len(x_freq) else 0)

            try:
                # ── 1. 保存 .npz 文件 ──
                if 'data_spectrum' in todo:
                    np.savez(os.path.join(output_dir, f"{base_name}_spectrum.npz"),
                             frequencies=x_freq[:-1], psd=np.mean(z_psd, axis=0))
                if 'data_spectrogram' in todo:
                    np.savez(os.path.join(output_dir, f"{base_name}_spectrogram.npz"),
                             frequencies=x_freq, times=y_time, psd_arrays=z_psd)

                # ── 2. 绘制 .png 图片 ──
                if 'png_spectrum' in todo:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.colors as colors
                    from matplotlib.figure import Figure
                    from matplotlib.backends.backend_agg import FigureCanvasAgg
                    fig = Figure(figsize=(10, 6))
                    canvas = FigureCanvasAgg(fig)
                    ax = fig.add_subplot(111)
                    ax.plot(x_freq[:-1]*1e-3, np.mean(z_psd, axis=0))
                    ax.set_yscale('log')
                    ax.set_title(f'Average Spectrum\n{base_name}')
                    ax.set_xlabel('Frequency [kHz]')
                    ax.set_ylabel('Power Spectral Density [arb. unit]')
                    ax.grid(True, which='both', ls='--', alpha=0.5)
                    fig.savefig(os.path.join(output_dir, f"{base_name}_spectrum.png"), transparent=False)
                    canvas.print_figure(os.path.join(output_dir, f"{base_name}_spectrum.png"), dpi=200)
                    fig.clf(); del fig

                if 'png_spectrogram' in todo:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.colors as colors
                    from matplotlib.figure import Figure
                    from matplotlib.backends.backend_agg import FigureCanvasAgg
                    fig = Figure(figsize=(12, 10))
                    canvas = FigureCanvasAgg(fig)
                    ax = fig.add_subplot(111)
                    norm = colors.LogNorm(vmin=max(z_psd.min(), 1e-18), vmax=z_psd.max())
                    pcm = ax.pcolormesh(x_freq*1e-3, y_time*1e3, z_psd, shading='flat', cmap='viridis', norm=norm)
                    ax.set_xlabel('Frequency [kHz]')
                    ax.set_ylabel('Time [ms]')
                    ax.set_title(f'Waterfall Plot\n{base_name}')
                    fig.colorbar(pcm, ax=ax).set_label('Power Spectral Density [arb. unit]')
                    fig.savefig(os.path.join(output_dir, f"{base_name}_spectrogram.png"), transparent=False)
                    canvas.print_figure(os.path.join(output_dir, f"{base_name}_spectrogram.png"), dpi=200)
                    fig.clf(); del fig

                # ── 3. Spectrogram 分析 ──
                if spectrogram_analysis_config is not None and spectrogram_cfg is not None:
                    peak_info_list = SpectrumAnalyzer.run_inline_analysis(
                        frequencies=x_freq, times=y_time, psd_array=z_psd,
                        output_dir=output_dir, base_name=base_name,
                        center_freq=0,
                        freq_low=freq_low, freq_high=freq_high,
                        bg_precomputed=bg_precomputed,
                        **spectrogram_cfg
                    )
                    if peaks_summary_path and peak_info_list:
                        _append_peaks_to_summary(peaks_summary_path, peak_info_list, base_name)
                        # 同步生成 CNN 格式（路径由 txt 同目录推导，随 txt 同生同灭）
                        cnn_path = os.path.join(os.path.dirname(peaks_summary_path),
                                                "all_peaks_cnn.csv")
                        cnn_peaks = _root_peak_to_cnn(peak_info_list, x_freq, y_time, z_psd, bg_precomputed)
                        if cnn_peaks:
                            _apply_pairing(cnn_peaks, y_time[-1], y_time[1] - y_time[0])
                            for _p in cnn_peaks:
                                _p['filename'] = base_name
                            _append_peaks_to_cnn_summary(cnn_path, cnn_peaks)
                        del cnn_peaks
            except Exception as e:
                print(f"[!] 单次注入分析失败: {base_name} - {e}")
            finally:
                del x_freq, y_time, z_psd
                import gc; gc.collect()
    except (KeyboardInterrupt, SystemExit):
        pass
    print('[*] 单次注入频谱分析进程已退出')


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



def _append_peaks_to_summary(summary_path, peak_info_list, base_name):
    """将单次注入的峰列表追加写入全局汇总文件（线程安全，每次打开/关闭）。"""
    try:
        with open(summary_path, 'a', encoding='utf-8') as f:
            for idx, peak in enumerate(peak_info_list, 1):
                f.write(f"{0:<6}\t"  # 总序号由用户后续处理
                        f"{idx:<6}\t"
                        f"{peak.get('mean_mhz', 0):<12.6f}\t"
                        f"{peak.get('peak_amp', 0):<12.6f}\t"
                        f"{peak.get('fwhm', 0):<12.6e}\t"
                        f"{peak.get('lifetime_start', -1):<12.6f}\t"
                        f"{peak.get('lifetime_end', -1):<12.6f}\t"
                        f"{peak.get('lifetime_time', -1):<12.6f}\t"
                        f"{peak.get('lifetime_time_bin', -1):<10}\t"
                        f"{peak.get('lifetime_area', 0):<12.4f}\t"
                        f"{peak.get('avg_per_bin', 0):<12.6f}\t"
                        f"{base_name:<30}\n")
    except Exception as e:
        print(f"[!] 写入峰汇总文件失败: {e}")


def _root_peak_to_cnn(peak_info_list, x_freq, y_time, z_psd, bg_precomputed):
    """把 ROOT 版峰 dict 换算成 CNN step4 的字段（不含 valid/pair_num/filename）。"""
    import numpy as np
    if not peak_info_list or not bg_precomputed:
        return []
    bg_mean = bg_precomputed.get('bg_mean')
    if bg_mean is None:
        return []

    # x_freq 是 bin 边界（比 z_psd 列多 1），裁剪到与 PSD 列对齐
    n_freq = z_psd.shape[1]
    if len(x_freq) > n_freq:
        x_freq = x_freq[:n_freq]
    psd_mean = np.mean(z_psd, axis=0)

    # 本底按索引截断对齐（沿用 run_inline_analysis 的假设：同 span、同点数）
    n = min(len(x_freq), len(bg_mean))
    x_freq = x_freq[:n]
    psd_mean = psd_mean[:n]
    bg_mean = np.asarray(bg_mean[:n], dtype=np.float64)
    z_psd = z_psd[:, :n]

    time_interval = y_time[1] - y_time[0] if len(y_time) > 1 else 1.0
    total_time = y_time[-1]

    out = []
    for p in peak_info_list:
        mean_hz = p.get('mean_mhz', 0.0) * 1e6
        sigma_hz = p.get('fwhm', 0.0) / 2.355
        exist_time = p.get('lifetime_time', -1.0)

        mask = (x_freq >= mean_hz - 6 * sigma_hz) & (x_freq <= mean_hz + 6 * sigma_hz)
        if not np.any(mask):
            continue

        # n_eff（用于 err_pos / err_sigma）
        w = np.maximum(psd_mean[mask] / bg_mean[mask] - 1.0, 0.0)
        max_w = w.max() if w.size else 0.0
        n_eff = max(w.sum() / max_w, 1.1) if max_w > 1e-30 else 1.1
        err_pos = sigma_hz / np.sqrt(n_eff)
        err_sigma = sigma_hz / np.sqrt(2.0 * n_eff)

        # height_ratio（峰频点处 psd/bg - 1）
        pk = int(np.argmin(np.abs(x_freq - mean_hz)))
        height_ratio = psd_mean[pk] / bg_mean[pk] - 1.0

        # height_ion（存活段内窗口峰值 psd/bg - 1）
        s_bin = int(round(p.get('lifetime_start', 0.0) / time_interval))
        e_bin = int(round(p.get('lifetime_end', 0.0) / time_interval))
        s_bin = max(0, min(s_bin, z_psd.shape[0] - 1))
        e_bin = max(0, min(e_bin, z_psd.shape[0] - 1))
        seg_mean = np.mean(z_psd[s_bin:e_bin + 1][:, mask], axis=0)
        height_ion = np.max(seg_mean / bg_mean[mask]) - 1.0

        # exist_state（由起止相对观测窗口判断）
        s, e = p.get('lifetime_start', 0.0), p.get('lifetime_end', 0.0)
        if s < 0.1 * total_time and e > 0.9 * total_time:
            state = 0
        elif s < 0.1 * total_time:
            state = 1
        else:
            state = 2

        out.append({
            'peak_pos': mean_hz, 'err_pos': err_pos,
            'sigma': sigma_hz, 'err_sigma': err_sigma,
            'height_ratio': height_ratio, 'height_ion': height_ion,
            'exist_state': state, 'exist_time': exist_time,
        })
    return out


def _apply_pairing(cnn_peaks, total_time, time_interval):
    """照抄 CNN step4 的母核-子核配对，回填 valid / pair_num。"""
    for p in cnn_peaks:
        p['pair_num'] = 0
        p['valid'] = 0 if p.get('exist_state') == 2 else 1

    pair_counter, used = 0, set()
    for i in range(len(cnn_peaks)):
        if cnn_peaks[i]['exist_state'] == 1 and i not in used:
            for j in range(len(cnn_peaks)):
                if cnn_peaks[j]['exist_state'] == 2 and j not in used:
                    time_ok = (abs(cnn_peaks[i]['exist_time']
                                   + cnn_peaks[j]['exist_time'] - total_time)
                               <= 2 * time_interval)
                    pos_ok = (cnn_peaks[i]['peak_pos'] < cnn_peaks[j]['peak_pos']
                              and abs(cnn_peaks[i]['peak_pos']
                                      - cnn_peaks[j]['peak_pos']) <= 80e3)
                    if time_ok and pos_ok:
                        pair_counter += 1
                        cnn_peaks[i]['pair_num'] = cnn_peaks[j]['pair_num'] = pair_counter
                        cnn_peaks[i]['valid'] = cnn_peaks[j]['valid'] = 1
                        used.add(i)
                        used.add(j)
                        break


def _append_peaks_to_cnn_summary(summary_path, cnn_peaks):
    """将换算后的 CNN 峰列表追加写入汇总文件（与 _append_peaks_to_summary 同构）。"""
    try:
        with open(summary_path, 'a', encoding='utf-8') as f:
            for p in cnn_peaks:
                f.write(f"{p.get('peak_pos', 0):<16.3f}\t"
                        f"{p.get('err_pos', 0):<14.6e}\t"
                        f"{p.get('sigma', 0):<14.6e}\t"
                        f"{p.get('err_sigma', 0):<14.6e}\t"
                        f"{p.get('height_ratio', 0):<14.6f}\t"
                        f"{p.get('height_ion', 0):<14.6f}\t"
                        f"{p.get('exist_state', 0):<12}\t"
                        f"{p.get('exist_time', 0):<14.6f}\t"
                        f"{p.get('valid', 0):<6}\t"
                        f"{p.get('pair_num', 0):<9}\t"
                        f"{p.get('filename', ''):<30}\n")
    except Exception as e:
        print(f"[!] 写入 CNN 峰汇总文件失败: {e}")


def _async_processing_task(x_stack, info):
    """
    子线程任务：负责严格的 pyfftw 计算、归一化、绘图和存盘
    x_stack: 形状为 (n_frames, n_average, window_length) 的三维数组
             或者是单次注入拼接好的完整一维信号
    """
    # 启用 pyfftw 缓存并限制大小（避免无限增长）
    import numpy as np
    import pyfftw, gc
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(30)  # 30秒后清除缓存
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
        
        # 1. 构造总帧数（不创建 3D 数组，避免内存爆炸）
        x_inj = x_stack
        n_frames = (len(x_inj) - (W + D * (n_average - 1))) // n_hop + 1

        # 2. 分批 FFT 计算（每批 CHUNK_SIZE 帧，峰值内存可控）
        #    每批: (CHUNK_SIZE, n_average, W) × complex64 × 2(输入+输出) ≲ 0.5 GB
        psd_array = np.empty((n_frames, W), dtype=np.float64)

        for start in range(0, n_frames, FFT_CHUNK_SIZE):
            end = min(start + FFT_CHUNK_SIZE, n_frames)
            bsize = end - start

            # 用 as_strided 提取本批数据的视图（不拷贝原数据）
            offset = start * n_hop
            bshape = (bsize, n_average, W)
            bstrides = (x_inj.strides[0] * n_hop,
                        x_inj.strides[0] * D,
                        x_inj.strides[0])
            bgrid = np.lib.stride_tricks.as_strided(x_inj[offset:],
                                                     shape=bshape,
                                                     strides=bstrides)

            # 加窗并 FFT（每批独立创建 plan，pyfftw 缓存会复用同形状的 plan）
            fft_in = pyfftw.empty_aligned(bshape, dtype='complex64')
            fft_in[:] = bgrid * window_sequence
            fft_out = pyfftw.empty_aligned(bshape, dtype='complex64')
            pyfftw.builders.fftn(fft_in, axes=(-1,), threads=2)(fft_in, fft_out)

            # 功率谱 + 帧内平均
            psd_array[start:end] = np.mean(
                np.abs(np.fft.fftshift(fft_out, axes=-1))**2
                / win_sq_sum / sampling_rate,
                axis=1)

            # 释放本批内存
            del fft_in, fft_out, bgrid
            gc.collect()

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

        # ── 将 spectrogram 数据放入单次注入队列（FFT worker 不做任何 I/O） ──
        sq = info.get('_spectrogram_queue')
        if sq is not None:
            config = info['spectrogram_analysis_config']
            spectrogram_cfg = {k: v for k, v in config.items()
                        if k not in ('freq_low', 'freq_high', 'amp_threshold')}
            freq_low = config.get('freq_low', x_freq[0])
            freq_high = config.get('freq_high', x_freq[-1])
            item = (x_freq.copy(), y_time.copy(), z_psd.copy(),
                    output_dir, base_name, todo,
                    spectrogram_cfg, freq_low, freq_high)
            try:
                sq.put_nowait(item)
            except:
                del item  # 队列满则丢弃（不阻塞 FFT worker）

        #print(f"SUCCESS: {base_name}")
            
        # ── 显式清理子进程大数组 ──
        for _v in ('x_freq','y_time','z_psd','psd_array','x_inj','x_stack'):
            try: del locals()[_v]
            except: pass
        gc.collect()

        return f"SUCCESS: {base_name}"
    except Exception as e:
        return f"FAILED: {base_name} with error {e}"
    finally:
        if 'fig' in locals(): fig.clf(); del fig
        import gc
        gc.collect()


class FileProcessor:
    def __init__(self, SOURCE_DIR, OUTPUT_DIR, FILE_PREFIX, EXPECTED_SIZE, CHECK_INTERVAL, SPECTROGRAM_ANALYSIS_CONFIG=None, WORKERS=None, START_INDEX=None, STOP_INDEX=None, SPECTROGRAM_WORKERS=1):
        self.SOURCE_DIR = SOURCE_DIR
        self.OUTPUT_DIR = OUTPUT_DIR
        self.FILE_PREFIX = FILE_PREFIX
        self.EXPECTED_SIZE = EXPECTED_SIZE
        self.CHECK_INTERVAL = CHECK_INTERVAL
        self.SPECTROGRAM_ANALYSIS_CONFIG = SPECTROGRAM_ANALYSIS_CONFIG
        self.WORKERS = WORKERS  # None 表示自动 = cpu_count() - 1
        self.START_INDEX = START_INDEX  # None = 自动检测，≥0 = 指定起始
        self.STOP_INDEX = STOP_INDEX  # None = 不限制，≥0 = 结束序号（不含）
        # 在主进程中预计算本底阈值（避免子进程重复加载1GB文件）
        self._bg_precomputed = self._precompute_background_if_needed()
        # 预初始化峰汇总文件（后续 run() 中重建，但单次注入进程需要先有路径）
        self.peaks_summary_path = self._init_peaks_summary()
        self.running = True

        # ── Spectrogram 分析单次注入进程（必须在 _init_executor 之前启动） ──
        self.spectrogram_queue = None
        self.injection_processes = []
        if SPECTROGRAM_ANALYSIS_CONFIG is not None:
            self.spectrogram_queue = multiprocessing.Manager().Queue(maxsize=4)
            for _ in range(SPECTROGRAM_WORKERS):
                p = multiprocessing.Process(
                    target=_writer_injection,
                    args=(self.spectrogram_queue, self._bg_precomputed,
                          SPECTROGRAM_ANALYSIS_CONFIG, self.peaks_summary_path),
                    daemon=True
                )
                p.start()
                self.injection_processes.append(p)
            print(f'[*] 启动 {SPECTROGRAM_WORKERS} 个单次注入频谱分析进程')

        self._init_executor()
        self.futures = []
        self.current_index = self._get_start_index()
        # 注册信号捕获，处理 Ctrl+C
        signal.signal(signal.SIGINT, self._handle_exit)

    def _precompute_background_if_needed(self):
        """如果配置了本底文件，在主进程中预计算阈值数据。"""
        if not self.SPECTROGRAM_ANALYSIS_CONFIG:
            return None
        bg_path = self.SPECTROGRAM_ANALYSIS_CONFIG.get('background_path')
        if bg_path is None:
            return None
        n_sigma = self.SPECTROGRAM_ANALYSIS_CONFIG.get('n_sigma', 5.0)
        return SpectrumAnalyzer.precompute_background(bg_path, n_sigma=n_sigma)

    def _init_executor(self):
        '''初始化或重建进程池'''
        if self.WORKERS is not None:
            n_workers = self.WORKERS
        else:
            n_workers = max(1, multiprocessing.cpu_count() - 1)
        print('[*] 正在启动/重建进程池 (workers: {:})'.format(n_workers))
        # 队列通过 info 字典传递给子进程（Manager.Queue proxy 可 pickle）
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, initializer=self._worker_ini_fn)
        

    def _get_start_index(self):
        '''
        启动时检测输出文件夹，寻找最大的 index。
        如果 START_INDEX 被指定（≥0）：
          - 自动检测的 index < START_INDEX → 从 START_INDEX 开始
          - 自动检测的 index ≥ START_INDEX → 从断点续跑（不倒退）
        如果 START_INDEX 为 None → 纯自动检测。
        '''
        # 扫描输出目录
        if not os.path.exists(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR)
            auto_index = 0
        else:
            indices = []
            for f in os.listdir(self.OUTPUT_DIR):
                if not f.endswith('.npz'):
                    continue
                if os.path.isdir(os.path.join(self.OUTPUT_DIR, f)):
                    continue
                try:
                    indices.append(int(os.path.basename(f).split('_')[2].split('-')[-1]))
                except (IndexError, ValueError):
                    continue
            auto_index = max(indices) if indices else 0

        # START_INDEX 作为下限：不倒退，但允许跳进
        if self.START_INDEX is not None and self.START_INDEX >= 0:
            if auto_index < self.START_INDEX:
                print(f"START_INDEX={self.START_INDEX} > 检测到的最大 index={auto_index}，从 {self.START_INDEX} 开始")
                return self.START_INDEX
            else:
                print(f"检测到已处理至索引：{auto_index}（≥START_INDEX={self.START_INDEX}），将从{auto_index+1}续跑")
                return auto_index + 1

        # 无 START_INDEX，纯自动
        if auto_index == 0 and not os.listdir(self.OUTPUT_DIR):
            return 0
        print(f"检测到已处理至索引：{auto_index}，将从{auto_index+1}开始。")
        return auto_index + 1

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
                os.remove(os.path.join(self.OUTPUT_DIR, last_file))
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
        '''完整性校验：检查大小且确认不再增长'''
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < self.EXPECTED_SIZE:
            return False
        # 再次确认文件没有再写入
        s1 = os.path.getsize(path) # Bytes
        time.sleep(0.2)
        if s1 != os.path.getsize(path):
            return False
        return True

    def _init_peaks_summary(self):
        """新建全局峰汇总文件，写入表头后合并已有的 *_peaks.txt。"""
        if not self.SPECTROGRAM_ANALYSIS_CONFIG:
            return None
        summary_dir = Path(self.OUTPUT_DIR) / "spectrogram_analysis"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / "all_peaks_summary.txt"

        header = (f"{'总序号':<6}\t"
                  f"{'峰序号':<6}\t"
                  f"{'频率[MHz]':<12}\t"
                  f"{'峰高':<12}\t"
                  f"{'FWHM[MHz]':<12}\t"
                  f"{'start[s]':<12}\t"
                  f"{'end[s]':<12}\t"
                  f"{'寿命[s]':<12}\t"
                  f"{'寿命[bin]':<10}\t"
                  f"{'面积':<12}\t"
                  f"{'每bin面积':<12}\t"
                  f"{'文件名':<30}\n")

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write("-" * 170 + "\n")

            # 合并已有 *_peaks.txt 中的数据行（格式不同，需补总序号列）
            existing_peaks = sorted(summary_dir.glob("*_peaks.txt"))
            total_peaks = 0
            for pk in existing_peaks:
                try:
                    with open(pk, 'r', encoding='utf-8') as pf:
                        for line in pf:
                            stripped = line.strip()
                            # 跳过表头、分隔线、空行
                            if not stripped or stripped.startswith('峰序号') or stripped.startswith('-'):
                                continue
                            # *_peaks.txt 无"总序号"列，补 0 占位
                            f.write(f"{0:<6}\t{line.rstrip()}\n")
                            total_peaks += 1
                except Exception as e:
                    print(f"[!] 读取已有峰文件失败: {pk} - {e}")

            if total_peaks > 0:
                print(f"[*] 从 {len(existing_peaks)} 个已有峰文件中合并 {total_peaks} 条峰数据")

        # 同步生成 CNN 格式汇总文件（与 txt 同目录、同生同灭）
        cnn_path = summary_dir / "all_peaks_cnn.csv"
        if not cnn_path.exists() or cnn_path.stat().st_size == 0:
            cnn_header = (f"{'peak_pos[Hz]':<16}\t"
                          f"{'err_pos[Hz]':<14}\t"
                          f"{'sigma[Hz]':<14}\t"
                          f"{'err_sigma[Hz]':<14}\t"
                          f"{'height_ratio':<14}\t"
                          f"{'height_ion':<14}\t"
                          f"{'exist_state':<12}\t"
                          f"{'exist_time[s]':<14}\t"
                          f"{'valid':<6}\t"
                          f"{'pair_num':<9}\t"
                          f"{'filename':<30}\n")
            with open(cnn_path, 'w', encoding='utf-8') as f:
                f.write(cnn_header)
                f.write("-" * 170 + "\n")
        print(f"[*] CNN 峰汇总文件: {cnn_path}")

        print(f"[*] 全局峰汇总文件: {summary_path}")
        return str(summary_path)

    def run(self, WIN_LEN, N_AVER, OVERLAPR, N_HOP, TODO):
        '''处理逻辑（单文件 + Spectrogram 分析单次注入进程分离）'''
        print("程序启动，当前文件索引：{:}".format(self.current_index))
        self._cached_incomplete_path = None  # 缓存上次 incomplete 文件路径
        # ── 计时日志 ──
        from pathlib import Path as _Path
        _timing_log = _Path(__file__).parent / "processing_timing.txt"
        with open(_timing_log, 'w', encoding='utf-8') as _f:
            _f.write(f"{'文件名':<45}\t{'处理耗时(s)':<16}\t{'等待间隔(s)':<16}\n")
            _f.write("-" * 77 + "\n")
        _last_finish = time.time()
        _pending = None  # (filename, proc_time)


        try:
            while True:
                target_file = os.path.join(self.SOURCE_DIR, '{:}_{:}.data'.format(self.FILE_PREFIX, self.current_index))

                if self.is_file_ready(target_file):
                    # 直接从缓存读 incomplete 文件，避免遍历目录
                    lastfile_path = self._cached_incomplete_path
                    last_file = os.path.basename(lastfile_path) if lastfile_path else None
                    _wait = time.time() - _last_finish
                    if _pending is not None:
                        with open(_timing_log, 'a', encoding='utf-8') as _f:
                            _f.write(f"{_pending[0]:<45}\t{_pending[1]:<16.3f}\t{_wait:<16.3f}\n")
                    time_0 = time.time()

                    try:
                        self.file_cutInjection(target_file, self.OUTPUT_DIR, WIN_LEN, N_AVER, OVERLAPR, self.executor, todo=TODO, n_hop=N_HOP, window='kaiser', beta=14, last_file=lastfile_path)

                        if self.futures:
                            done, _ = concurrent.futures.wait(self.futures)
                            for f in done:
                                try:
                                    res = f.result()
                                    if "FAILED" in res: print("[!] 任务失败：{:}".format(res))
                                    del res
                                except Exception as e:
                                    print("[!] 任务执行崩溃：{:}".format(e))
                            self.futures = []
                            del done
                    except concurrent.futures.process.BrokenProcessPool:
                        print("[!] 检测到进程池损坏，尝试重启进程池，并重启当前任务...")
                        self.executor.shutdown(wait=False)
                        self._init_executor()
                        continue

                    print('[*] 处理该文件耗时 {:.3f} sec'.format(time.time()-time_0))
                    _pending = (os.path.basename(target_file), time.time() - time_0)
                    _last_finish = time.time()
                    if last_file is not None:
                        self._delete_assigned_incomplete_file(last_file)
                    del last_file, lastfile_path, time_0
                    self.current_index += 1

                    if self.STOP_INDEX is not None and self.current_index > self.STOP_INDEX:
                        print(f"[!] 已达到 STOP_INDEX={self.STOP_INDEX}，停止处理。")
                        if _pending is not None:
                            with open(_timing_log, 'a', encoding='utf-8') as _f:
                                _f.write(f"{_pending[0]:<45}\t{_pending[1]:<16.3f}\t{'—':>16}\n")
                            _pending = None
                        break

                    import gc
                    gc.collect()

                    if not self.running:
                        print("[!] 检测到退出信号，当前文件已处理结束。安全退出。")
                        if _pending is not None:
                            with open(_timing_log, 'a', encoding='utf-8') as _f:
                                _f.write(f"{_pending[0]:<45}\t{_pending[1]:<16.3f}\t{'—':>16}\n")
                            _pending = None
                        break
                else:
                    if not self.running:
                        break
                    time.sleep(self.CHECK_INTERVAL)
        finally:

            if _pending is not None:
                with open(_timing_log, 'a', encoding='utf-8') as _f:
                    _f.write(f"{_pending[0]:<45}\t{_pending[1]:<16.3f}\t{'程序退出':>16}\n")

            # ── 关闭单次注入进程 ──
            if self.spectrogram_queue is not None:
                for _ in self.injection_processes:
                    try:
                        self.spectrogram_queue.put(None, timeout=5)
                    except:
                        pass
                for p in self.injection_processes:
                    p.join(timeout=5)
            print("[*] 正在关闭进程池 ...")
            self.executor.shutdown(wait=True)
            print('[√] 程序已安全退出。')

    def file_cutInjection(self, input_file, output_dir, window_length, n_average, overlap_ratio, executor, todo, n_hop=None, window=None, beta=None, last_file=None):
        import numpy as np
        import pyfftw
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        # 启用 pyfftw 缓存以提升频繁创建 plan 的性能
        pyfftw.interfaces.cache.enable()
        from preprocessing import Preprocessing
        
        # --- [参数初始化部分：保持你原有的逻辑] ---
        window_sequence = handle_windows(window_length, window, beta)
        win_sq_sum = np.sum(window_sequence**2)
        D = int((1 - overlap_ratio) * window_length) 
        N = int(window_length + D * (n_average - 1))
        n_point = window_length
        if n_hop is None: n_hop = N
        
        # ... (此处省略 bud 初始化、Timestamp 获取等原有代码) ...
        _current_fileIndex = int(os.path.basename(input_file).split('_')[1].split('.')[0])
        _prefix = os.path.dirname(input_file).split('/')[-1].split('_')[0] + '_' + os.path.basename(input_file).split('_')[0]  # 8249_PY82ch1
    
        bud = Preprocessing(input_file, puyuan_new=True, abs_trigger=False)
        ThisFileTimestamp = bud.date_time + np.timedelta64(8, 'h') # convert to '+08' timezone
    
        # 处理前序文件的内容
        if last_file is None:
            additional_x, lastTriggerData_remain, trigger_frame, offset = np.array([]), 0, 0, 0
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
    
    
            
        # 如果整个文件中都不存在触发信号，那么执行以下策略：1.前序文件含有触发信号，就将前序文件末尾的新注入数据与本文件可组成频谱的内容合并为新文件（incomplete），未组合成频谱部分放置于addition_data中；2.前序文件不含有触发信号，仅将前序文件addition_data中的数据与本文件可组成频谱的内容合并未新文件（incomplete），未组合成频谱部分放置于addition_data中。3.无前序文件（见于PY8*ch*_0.data），将本文件可组成频谱的内容合并为新文件（incomplete），未组合成频谱部分放置于addition_data中。
        if len(bud.trigger_timestamp) == 0:
            print('[!] 当前文件内无触发信号，将按普通频谱处理！')
            dummy = pyfftw.empty_aligned((n_average, window_length))
            fft = pyfftw.builders.fft(dummy, n=n_point, overwrite_input=True, threads=multiprocessing.cpu_count())
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
                    if trigger_inLastFile or (last_file is None):
                        print('[*] 当前无触发文件的前序文件也无触发或不存在，保留时频谱 ...')
                        _incomplete_name = '{:}_{:04}_trigger_0_incomplete_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))
                        np.savez(os.path.join(output_dir, _incomplete_name), frequencies=x_frequency, times=y_time, psd_arrays=z_psd_array, addition_data=additional_x)
                        print('[√] 文件生成：{:}'.format(_incomplete_name))
                        self._cached_incomplete_path = os.path.join(output_dir, _incomplete_name)
                    else:
                        print('[*] 当前无触发文件将与前序文件剩余部分合并，保留时频谱 ...')
                        _incomplete_name = '{:}_{:04d}-{:04}_trigger_incomplete_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex-1, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))
                        np.savez(os.path.join(output_dir, _incomplete_name), frequencies=x_frequency, times=y_time, psd_arrays=z_psd_array, addition_data=additional_x)
                        print('[√] 文件生成：{:}'.format(_incomplete_name))
                        self._cached_incomplete_path = os.path.join(output_dir, _incomplete_name)
                    break
            del bud, window_sequence, additional_x, dummy, fft
            import gc; gc.collect()
            return
    
    
        # --- 核心修改：在触发信号循环中 ---
        for trigger_i, trigger_timestamp in enumerate(bud.trigger_timestamp):
            # --- 内存管理：定期清理已完成的任务 ---
            if len(self.futures) > 10:
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
                base_name = '{:}_{:04d}-{:04d}_trigger_{:}'.format(_prefix, _current_fileIndex-1, _current_fileIndex, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%H-%S'))
                additional_psd = psd_array.copy()
                print('[!] 当前文件将和前序文件残余部分拼接，生成文件 {:}'.format(base_name))
            else:
                base_name = '{:}_{:04d}_trigger_{:}_{:}'.format(_prefix, _current_fileIndex, trigger_i, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))
                additional_psd, additional_x  = None, np.array([])
            x_injection = np.hstack((additional_x, bud.load(trigger_timestamp * bud.data_len - offset, offset)[1]))
            # 3. 准备子线程所需的元数据
            info = {
                'window_length': window_length, 'D': D, 'n_average': n_average, 'n_hop': n_hop,
                'sampling_rate': bud.sampling_rate, 'window_sequence': window_sequence,
                'win_sq_sum': win_sq_sum, 'span': bud.span, 'center_frequency': bud.center_frequency,
                'todo': todo, 'output_dir': output_dir, 'base_name': base_name, 'additional_psd': additional_psd, # 动态生成
                'spectrogram_analysis_config': self.SPECTROGRAM_ANALYSIS_CONFIG,   # 在线分析配置
                'bg_precomputed': self._bg_precomputed,              # 预计算本底阈值（小尺寸）
                'peaks_summary_path': getattr(self, 'peaks_summary_path', None),
                '_spectrogram_queue': self.spectrogram_queue,  # Manager.Queue proxy，可 pickle
            }
            # 4. 异步提交任务
            try:
                future = executor.submit(_async_processing_task, x_injection.copy(), info)
                self.futures.append(future)
                #print(f"DEBUG: 任务 {base_name} 已存入 futures 列表中，当前列表长度： {len(self.futures)}")
                # ── 父进程不再需要 x_injection 和 info，立即释放 ──
                del x_injection, info
            except concurrent.futures.process.BrokenProcessPool:
                raise

            lastTriggerData_remain, trigger_frame = 0, 0
            offset = trigger_timestamp * bud.data_len
            ThisDataTimestamp = ThisFileTimestamp + np.timedelta64(int(offset/bud.sampling_rate), 's')
    
        # 返回给主循环所需的断点信息
        trigger_i += 1
        ThisTriggerData_remain = bud.n_sample + lastTriggerData_remain - offset
        dummy = pyfftw.empty_aligned((n_average, window_length))
        fft = pyfftw.builders.fft(dummy, n=window_length, overwrite_input=True, threads=2)
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
                _incomplete_name = '{:}_{:04d}_trigger_{:}_incomplete_{:}_spectrogram.npz'.format(_prefix, _current_fileIndex, trigger_i, ThisDataTimestamp.astype('datetime64[s]').item().strftime('%Y-%m-%dT%H-%M-%S'))
                np.savez(os.path.join(output_dir, _incomplete_name), frequencies=x_frequency, times=y_time, psd_arrays=z_psd_array, addition_data=additional_x)
                print('[√] 本文件剩余部分保存：{:}'.format(_incomplete_name))
                # 缓存 incomplete 路径，供下一文件直接使用
                self._cached_incomplete_path = os.path.join(output_dir, _incomplete_name)
                break






        # ── 清理 file_cutInjection 中的大对象 ──
        del bud, window_sequence, additional_x
        for _n in ("dummy","fft","x","_signal","psd_array","x_frequency","y_time","z_psd_array"):
            try: del locals()[_n]
            except: pass
        import gc; gc.collect()

if __name__ == "__main__":
    # 这一行在 Windows 下对于防止 spawn 死循环极其重要
    multiprocessing.freeze_support() 
    processor = FileProcessor(SOURCE_DIR, OUTPUT_DIR, FILE_PREFIX, EXPECTED_SIZE, CHECK_INTERVAL)
    processor.run(WIN_LEN, N_AVER, OVERLAPR, N_HOP, TODO)
