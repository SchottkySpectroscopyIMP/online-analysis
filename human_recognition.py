#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DataLabeler:
    def __init__(self, data_dir, output_csv):
        self.data_dir = data_dir
        self.output_csv = output_csv
        
        # 1. 检查目录
        if not os.path.exists(data_dir):
            print(f"错误: 找不到文件夹 {data_dir}")
            return

        # 2. 扫描文件并按“主名”归组去重
        self.group_map = self._scan_and_group_files()
        all_base_keys = sorted(list(self.group_map.keys()))
        
        # 3. 加载已有进度 (根据主名断点续传)
        self.labeled_keys = self._get_already_labeled()
        
        # 4. 过滤掉已经标注过的 key
        self.remaining_keys = [k for k in all_base_keys if k not in self.labeled_keys]
        
        print("-" * 30)
        print(f"分组去重后总数: {len(all_base_keys)}")
        print(f"已标注主名数:   {len(self.labeled_keys)}")
        print(f"剩余待办主名:   {len(self.remaining_keys)}")
        print("-" * 30)

        if not self.remaining_keys:
            print("所有数据组已处理完毕！")
            return

        # 5. 初始化绘图窗口
        self.current_idx = 0
        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        self.fig.canvas.manager.set_window_title('数据标注工具 - 自动去重与断点续传')
        
        # 绑定键盘事件
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.show_next()
        plt.show()

    def _scan_and_group_files(self):
        """
        扫描文件夹，根据前缀提取主名，并配对 spectrum 与 spectrogram 文件
        """
        group = {}
        for fname in os.listdir(self.data_dir):
            if not fname.endswith('.npz'):
                continue
            
            # 匹配后缀并提取 base_key
            if fname.endswith('_spectrum.npz'):
                base_key = fname[:-13]  # 移除 '_spectrum.npz'
                suffix_type = 'spectrum'
            elif fname.endswith('_spectrogram.npz'):
                base_key = fname[:-16] # 移除 '_spectrogram.npz'
                suffix_type = 'spectrogram'
            else:
                continue

            if base_key not in group:
                group[base_key] = {}
            
            group[base_key][suffix_type] = fname
            
        return group

    def _get_already_labeled(self):
        """从CSV中读取已经处理过的 base_key 集合"""
        if os.path.exists(self.output_csv):
            try:
                df = pd.read_csv(self.output_csv)
                if 'base_key' in df.columns:
                    return set(df['base_key'].astype(str).tolist())
                elif 'filename' in df.columns:  # 兼容之前写入的文件名格式
                    return set(df['filename'].astype(str).tolist())
            except Exception as e:
                print(f"读取进度文件失败，将从头开始: {e}")
        return set()

    def save_result(self, base_key, label):
        """追加保存结果"""
        file_exists = os.path.exists(self.output_csv)
        # 记录 base_key，确保全局唯一
        df = pd.DataFrame([[base_key, label]], columns=['base_key', 'label'])
        df.to_csv(self.output_csv, mode='a', index=False, header=not file_exists)

    def load_data_for_key(self, base_key):
        """
        根据 base_key 优先加载 spectrum 文件；若无则加载 spectrogram 并现场求均值
        """
        files_dict = self.group_map[base_key]
        
        # 优先读取 _spectrum.npz
        if 'spectrum' in files_dict:
            file_path = os.path.join(self.data_dir, files_dict['spectrum'])
            with np.load(file_path) as data:
                freq = data['frequencies'][:-1]
                psd = data['psd']
                # 兼容维度可能不一致的情况
                if len(psd) > len(freq):
                    psd = psd[:len(freq)]
            return freq, psd, files_dict['spectrum']
            
        # 备选读取 _spectrogram.npz
        elif 'spectrogram' in files_dict:
            file_path = os.path.join(self.data_dir, files_dict['spectrogram'])
            with np.load(file_path) as data:
                freq = data['frequencies'][:-1]
                psd_arrays = data['psd_arrays']
                psd = np.mean(psd_arrays, axis=0)
                if len(psd) > len(freq):
                    psd = psd[:len(freq)]
            return freq, psd, files_dict['spectrogram']
            
        else:
            raise FileNotFoundError(f"未找到 {base_key} 对应的有效 npz 文件")

    def show_next(self):
        """显示剩余队列中的下一个文件"""
        if self.current_idx >= len(self.remaining_keys):
            print("\n恭喜！本批次所有文件处理完毕。")
            plt.close()
            return

        base_key = self.remaining_keys[self.current_idx]
        
        try:
            freq, psd, loaded_filename = self.load_data_for_key(base_key)
            
            self.ax.clear()
            self.ax.plot(freq, psd, linewidth=0.7, color='steelblue')
            self.ax.set_xlabel('Frequency [Hz]')
            self.ax.set_xlim(freq[0], freq[-1])
            self.ax.set_yscale('log')
            
            progress_str = f"Remaining process: {self.current_idx + 1}/{len(self.remaining_keys)}"
            self.ax.set_title(f"{progress_str}\nLoaded: {loaded_filename}\nBaseKey: {base_key}")
            self.ax.set_ylabel('PSD')
            self.ax.grid(True, which='both', alpha=0.3)
            
            self.fig.canvas.draw()
            
        except Exception as e:
            print(f"\n跳过损坏或加载失败的数据 [BaseKey: {base_key}]: {e}")
            self.current_idx += 1
            self.show_next()

    def on_key(self, event):
        if event.key is None: return
        key = event.key.lower()
        
        if key in ['0', '1']:
            base_key = self.remaining_keys[self.current_idx]
            label = int(key)
            
            # 立即保存到磁盘
            self.save_result(base_key, label)
            
            print(f"[{self.current_idx+1}] {base_key} -> {label} (已保存)", end='\r')
            
            self.current_idx += 1
            self.show_next()
            
        elif key == 'q':
            print("\n检测到退出指令，进度已安全保存。")
            plt.close()


