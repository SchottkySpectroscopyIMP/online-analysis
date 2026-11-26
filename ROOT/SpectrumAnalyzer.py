#!/usr/bin/env python3
import argparse
import glob
from pathlib import Path

import numpy as np
from ROOT import (
    TH2F, TH1F, TCanvas, TFile, TPad, TLine,
    TSpectrum,   # ← 新增
    gROOT, gStyle, TF1, TMarker, TBox
)


class SpectrumAnalyzer:
    """频谱数据分析与可视化主类"""
    
    def __init__(self, args):
        self.args = args
        self.center_freq = args.center_freq
        self.freq_low = args.freq_low
        self.freq_high = args.freq_high
        self.t_min_proj = args.t_min
        self.t_max_proj = args.t_max
        self.date_start = args.date_start
        self.date_end = args.date_end

        self.do_projection = (self.t_min_proj is not None and self.t_max_proj is not None)

        # 输出文件类型控制（默认全部开启）
        self.output_files = getattr(args, 'output_files',
                                    ['peaks_txt', 'spectrogram_png', 'root_file'])

        # 本底相关
        self.bg_psd = None
        self.bg_threshold = None

        # 输出相关
        self.result_dir = None
        self.strong_dir = None
        self.weak_dir = None
        self.signal_files = []

        self.h_sum = None
        self.h_proj_sum = None
        self.saved_count = 0

        self._setup_root_style()
        self.all_peaks = []          # ← 新增：用于汇总所有峰
        self.bg_mean = None      # 新增
        self.bg_std = None       # 新增
        self.bg_threshold = None
        self.freq_for_proj = None
        self.proj_bin_count = 0
        self._load_background()

    def _setup_root_style(self):
        """设置 ROOT 全局绘图样式"""
        gROOT.SetBatch(True)
        gStyle.SetPalette(51)
        gStyle.SetNumberContours(255)
        gStyle.SetOptStat(0)

    def _load_background(self):
        """加载并处理本底数据"""
        if not self.args.background:
            return
        bg_path = Path(self.args.background)
        if not bg_path.exists():
            raise FileNotFoundError(f"本底文件不存在: {bg_path}")
        
        bg_data = np.load(bg_path)
        bg_frequencies = bg_data['frequencies'] + self.center_freq
        self.bg_psd_array = bg_data['psd_arrays']
        
        # 保证形状为 [n_time, n_freq]
        if self.bg_psd_array.ndim == 2 and self.bg_psd_array.shape[0] > self.bg_psd_array.shape[1]:
            self.bg_psd_array = self.bg_psd_array.T
        
        # ====================== 关键修复：强制对齐长度 ======================
        n_freq_data = self.bg_psd_array.shape[1]
        n_freq_bg = len(bg_frequencies)
        
        if n_freq_data != n_freq_bg:
            print(f"本底频率维度不匹配: frequencies={n_freq_bg}, psd={n_freq_data}")
            # 取较小长度对齐
            min_len = min(n_freq_data, n_freq_bg)
            bg_frequencies = bg_frequencies[:min_len]
            self.bg_psd_array = self.bg_psd_array[:, :min_len]
            print(f"→ 已截断对齐至 {min_len} 个频率点")
        
        self.bg_mean, self.bg_std, self.bg_threshold = \
                    self._compute_background_threshold(
                        self.bg_psd_array, 
                        n_sigma=self.args.n_sigma,
                        smooth_window=51          # ← 可调整，推荐 11~51（奇数最佳）
                    )        
        self._save_background_threshold_to_root(bg_frequencies, self.bg_threshold)

    def _save_background_threshold_to_root(self, frequencies: np.ndarray, threshold: np.ndarray):
        """保存本底阈值"""
        if self.result_dir is None:
            bg_output_dir = Path(self.args.output_dir)
        else:
            bg_output_dir = self.result_dir
        bg_output_dir.mkdir(parents=True, exist_ok=True)

        n_bins = min(len(frequencies), len(threshold))
        freq_bin_width = float(np.median(np.diff(frequencies))) if len(frequencies) > 1 else 0

        h_bg_threshold = TH1F("h_bg_threshold",
                              f"Background Threshold ({self.args.n_sigma}#sigma);"
                              f"Frequency [MHz];Threshold",
                              n_bins,
                              frequencies[0] / 1e6,
                              (frequencies[n_bins-1] + freq_bin_width) / 1e6)

        for i in range(n_bins):
            bin_idx = i + 1
            h_bg_threshold.SetBinContent(bin_idx, float(threshold[i]))

        h_bg_threshold.SetLineColor(2)
        h_bg_threshold.SetLineWidth(2)
        
        root_filename = bg_output_dir / f"background_threshold_{int(self.args.n_sigma)}sigma.root"
        root_file = TFile(str(root_filename), "RECREATE")
        h_bg_threshold.Write()
        root_file.Close()
    
    @staticmethod
    def _compute_background_threshold(bg_psd: np.ndarray, n_sigma: float = 5.0, 
                                      smooth_window: int = 21):
        """计算背景均值、标准差和阈值，并对 bg_mean 和 bg_std 进行平滑"""
        if bg_psd.ndim == 2 and bg_psd.shape[0] > bg_psd.shape[1]:
            bg_psd = bg_psd.T
       
        bg_mean = np.mean(bg_psd, axis=0)
        bg_std = np.std(bg_psd, axis=0)
       
        # 避免 std=0 的通道
        bg_std = np.maximum(bg_std, 1e-10)

        # ====================== 新增：平滑处理 ======================
        if smooth_window > 1:
            # 使用 Savitzky-Golay 滤波器（保形效果好，适合谱线平滑）
            # 如果不想引入 scipy，可改用移动平均
            try:
                from scipy.signal import savgol_filter
                # window_length 必须为奇数
                window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
                bg_mean = savgol_filter(bg_mean, window_length=window, polyorder=3)
                bg_std = savgol_filter(bg_std, window_length=window, polyorder=3)
                print(f"   → 已对 bg_mean 和 bg_std 进行 Savitzky-Golay 平滑 (window={window})")
            except ImportError:
                # 回退到简单移动平均
                print(f"   → scipy未安装，使用移动平均平滑 (window={smooth_window})")
                kernel = np.ones(smooth_window) / smooth_window
                bg_mean = np.convolve(bg_mean, kernel, mode='same')
                bg_std = np.convolve(bg_std, kernel, mode='same')
        
        threshold = bg_mean + n_sigma * bg_std
       
        return bg_mean, bg_std, threshold

    def _get_timestamp_key(self, fpath: str) -> str:
        """从文件名提取时间戳用于排序和过滤"""
        print("chenrj ... ",fpath)
        print("chenrj ... Path(fpath).stem.split('_')[-1] =", Path(fpath).stem.split('_')[-1])
        
        return Path(fpath).stem.split('_')[-2]

    def _filter_files_by_date(self, npz_files: list) -> list:
        """按日期范围过滤文件"""
        if not (self.date_start or self.date_end):
            return npz_files

        filtered = []
        for f in npz_files:
            ts = self._get_timestamp_key(f).replace('-', '').replace('T', '')
            start_key = self.date_start.replace('-', '').replace('T', '') if self.date_start else None
            end_key = self.date_end.replace('-', '').replace('T', '') if self.date_end else None

            if start_key and ts < start_key:
                continue
            if end_key and ts > end_key:
                continue
            filtered.append(f)

        print(f"日期过滤后剩余 {len(filtered)} 个文件 (原 {len(npz_files)} 个)")
        return filtered

    def prepare_output_dirs(self):
        """准备输出目录结构"""
        bg_tag = f"_bgsub{int(self.args.n_sigma)}sigma" if self.args.background else ""

        if self.date_start and self.date_end:
            date_tag = f"{self.date_start.replace('T', '-')}_{self.date_end.replace('T', '-')}{bg_tag}"
        else:
            date_tag = f"all{bg_tag}"

        self.result_dir = Path(self.args.output_dir) / date_tag
        self.result_dir.mkdir(parents=True, exist_ok=True)

        print(f"输出目录已创建: {self.result_dir}")

    def process_all_files(self):
        """主处理流程：遍历所有 npz 文件并累加强信号"""
        data_dir = Path(self.args.data_dir)
        all_npz = sorted(glob.glob(str(data_dir / "*_spectrogram.npz")),
                         key=self._get_timestamp_key)

        npz_files = self._filter_files_by_date(all_npz)

        if not npz_files:
            raise RuntimeError("未找到任何 npz 文件！")

        print(f"\n=== 开始处理 {len(npz_files)} 个注入文件 ===")
        for idx, fpath in enumerate(npz_files, 1):
            self._process_single_file(idx, fpath)
            
        # 最后生成汇总文件
        self._save_all_peaks_summary()
        
    def _process_single_file(self, idx: int, fpath: str):
        """处理单个 npz 文件 - 极简维度修正版"""
        data = np.load(fpath)
        frequencies = data['frequencies'] + self.center_freq
        times = data['times'].astype(float)
        psd_array = data['psd_arrays']
        
        base_name = Path(fpath).stem
        print(f"[{idx:3d}] {base_name}")

        # ====================== 简单直接的维度对齐 ======================
        # 保证形状为 [n_time, n_freq]
        if psd_array.ndim == 2 and psd_array.shape[0] > psd_array.shape[1]:
            psd_array = psd_array.T

        # 2. 关键：按照 psd_array 的实际尺寸裁剪 frequencies 和 times
        n_time = psd_array.shape[0]
        n_freq = psd_array.shape[1]

        # 裁剪 times（如果 times 比 psd_array 时间维度多，就去掉后面的点）
        if len(times) > n_time:
            times = times[:n_time]

        # 裁剪 frequencies（如果 frequencies 比 psd_array 频率维度多，就去掉后面的点）
        if len(frequencies) > n_freq:
            frequencies = frequencies[:n_freq]

        # 安全检查
        if psd_array.shape[0] != len(times) or psd_array.shape[1] != len(frequencies):
            raise ValueError(f"维度对齐失败！ psd_array={psd_array.shape}, times={len(times)}, freq={len(frequencies)}")

        # 创建直方图并继续处理
        h, h_proj = self._create_histograms(psd_array, frequencies, times, base_name)
        self.current_peak_info = self.analyze_ion_lifetime(h_proj, psd_array, frequencies, times)
        self._draw_and_save_single(h, h_proj, base_name)

        # === 收集到全局列表（用于最终汇总）===
        for peak in self.current_peak_info:
            peak_copy = peak.copy()
            peak_copy['filename'] = Path(fpath).stem
            self.all_peaks.append(peak_copy)

    def analyze_ion_lifetime(self, h_proj_temp: TH1F, psd_array: np.ndarray,
                             frequencies: np.ndarray, times: np.ndarray):
        """使用 TSpectrum 分析峰，支持频率自适应 sigma 阈值（amp_threshold = n_sigma）
           直接使用 bg_mean / bg_std 对应索引（不插值）"""

        spectrum = TSpectrum()
        # ====================== 基本检查 ======================
        max_bin_content = h_proj_temp.GetMaximum()
        if max_bin_content <= 1e-8:
            print(f" [跳过分析] 投影谱最大值过小 ({max_bin_content:.2e})")
            return []

        # ====================== 长度对齐检查 ======================
        n_freq = len(frequencies)
        if len(self.bg_mean) != n_freq or len(self.bg_std) != n_freq:
            print(f"⚠️  背景与数据频率点数不匹配: bg={len(self.bg_mean)}, data={n_freq}")
            # 强制截断对齐（安全处理）
            min_len = min(len(self.bg_mean), n_freq)
            bg_mean_use = self.bg_mean[:min_len]
            bg_std_use = self.bg_std[:min_len]
            bg_threshold_use = self.bg_threshold[:min_len]
            freq_use = frequencies[:min_len]
        else:
            bg_mean_use = self.bg_mean
            bg_std_use = self.bg_std
            bg_threshold_use = self.bg_threshold
            freq_use = frequencies

        # ====================== 频率投影用于找峰 ======================
        h_proj_search = h_proj_temp.Clone("h_proj_search")
        
        # 直接使用对齐后的 bg 数据
        n_bins = h_proj_search.GetNbinsX()
        
        # 应用阈值（低于阈值的 bin 置 0）
        for binx in range(1, n_bins + 1):
            idx = binx - 1
            thresh = bg_threshold_use[idx]          # 使用本底阈值（bg_mean + n_sigma * bg_std）
            if h_proj_search.GetBinContent(binx) < thresh:
                h_proj_search.SetBinContent(binx, 0.0)

        # ====================== 手动寻找独立峰（不使用 TSpectrum） ======================
        peaks = []
        i = 1
        while i <= n_bins:
            if h_proj_search.GetBinContent(i) <= 0.0:
                i += 1
                continue
            
            # 找到一个连续高于阈值的区域（一个峰）
            start_bin = i
            max_val = 0.0
            max_bin = i
            
            # 向右扫描整个连续区域
            while i <= n_bins and h_proj_search.GetBinContent(i) > 0.0:
                current = h_proj_search.GetBinContent(i)
                if current > max_val:
                    max_val = current
                    max_bin = i
                i += 1
            
            # 只保留有效峰（至少跨 3 个 bin，避免噪声）
            if max_val > 0.0 and (i - start_bin) >= 3:
                mean_mhz = h_proj_search.GetXaxis().GetBinCenter(max_bin)
                peak_amp = max_val
                
                peaks.append((mean_mhz, peak_amp))

        peaks.sort(key=lambda x: x[0])   # 按频率排序

        peak_info_list = []

        for i, (mean_mhz, peak_amp) in enumerate(peaks):
            # ====================== FWHM 计算 ======================
            bin_center = h_proj_search.GetXaxis().FindBin(mean_mhz)
            bin_low = max(1, bin_center - 15)
            bin_high = min(h_proj_search.GetNbinsX(), bin_center + 15)

            h_proj_search.GetXaxis().SetRange(bin_low, bin_high)
            local_rms_mhz = h_proj_search.GetRMS()
            h_proj_search.GetXaxis().SetRange(0, 0)

            half_max = peak_amp * 0.5
            bin_half_low = bin_center
            bin_half_high = bin_center

            for b in range(bin_center, bin_low - 1, -1):
                if h_proj_search.GetBinContent(b) < half_max:
                    bin_half_low = b
                    break
            for b in range(bin_center, bin_high + 1):
                if h_proj_search.GetBinContent(b) < half_max:
                    bin_half_high = b
                    break

            fwhm_mhz = (h_proj_search.GetXaxis().GetBinCenter(bin_half_high) -
                        h_proj_search.GetXaxis().GetBinCenter(bin_half_low))

            print(f" 峰 {i+1}: mean = {mean_mhz:.6f} MHz, FWHM = {fwhm_mhz:.6f} MHz, "
                  f"A = {peak_amp:.6f}")

            # ====================== 峰窗口内精确时间投影 ======================
            freq_low_win = mean_mhz -  5 * fwhm_mhz / 2.35
            freq_high_win = mean_mhz + 5 * fwhm_mhz / 2.35
            
            time_proj_peak = np.zeros(len(times), dtype=np.float64)
            for ti in range(len(times)):
                for fi in range(len(freq_use)):
                    f_mhz = freq_use[fi] / 1e6
                    if freq_low_win <= f_mhz <= freq_high_win:
                        signal = psd_array[ti, fi]
                        if signal > bg_threshold_use[fi]:
                            time_proj_peak[ti] += signal
                            
            # ====================== 离子寿命计算（连续3点低于阈值） ======================
            if np.max(time_proj_peak) < 1e-8:
                print(f"  → 峰 {i+1} 时间投影过弱，跳过寿命计算")
                lifetime_start = lifetime_end = -1.0
                lifetime_time = lifetime_time_bin = 0
                lifetime_area = avg_per_bin = 0.0
            else:
                max_val = np.max(time_proj_peak)
                max_bin = np.argmax(time_proj_peak)
                threshold_value = max_val * 0.2

                # ====================== 离子寿命计算（连续7点低于阈值） ======================
                if np.max(time_proj_peak) < 1e-8:
                    print(f" → 峰 {i+1} 时间投影过弱，跳过寿命计算")
                    lifetime_start = lifetime_end = -1.0
                    lifetime_time = lifetime_time_bin = 0
                    lifetime_area = avg_per_bin = 0.0
                else:
                    max_val = np.max(time_proj_peak)
                    max_bin = np.argmax(time_proj_peak)
                    threshold_value = max_val * 0.2
    
                    n_time = len(time_proj_peak)
                    consecutive = 7   # 可轻松修改为其他数值
    
                    # ====================== 向左寻找起始点（连续7点低于阈值） ======================
                    lifetime_start_bin = max_bin
                    for b in range(max_bin, -1, -1):
                        # 检查是否还能取到连续7个点
                        if b < consecutive - 1:          # 不足7个点
                            lifetime_start_bin = 0
                            break
                        
                        # 检查从 b-6 到 b 是否连续7个点都低于阈值
                        if all(time_proj_peak[k] < threshold_value for k in range(b - 6, b + 1)):
                            lifetime_start_bin = b + 1   # 取第一个低于阈值的点作为起始
                            break
                    
                    # 如果左边界直接低于阈值
                    if lifetime_start_bin == max_bin and max_bin < consecutive:
                        lifetime_start_bin = 0
    
                    # ====================== 向右寻找结束点（连续7点低于阈值） ======================
                    lifetime_end_bin = max_bin
                    for b in range(max_bin, n_time):
                        # 检查是否还能取到连续7个点
                        if b > n_time - consecutive:
                            lifetime_end_bin = n_time - 1
                            break
                        
                        # 检查从 b 到 b+6 是否连续7个点都低于阈值
                        if all(time_proj_peak[k] < threshold_value for k in range(b, b + 7)):
                            lifetime_end_bin = b - 1     # 取最后一个高于阈值的点作为结束
                            break
    
                    # ====================== 计算寿命 ======================
                    lifetime_start = times[lifetime_start_bin]
                    lifetime_end = times[lifetime_end_bin]
    
                    if lifetime_end_bin > lifetime_start_bin:
                        lifetime_time = lifetime_end - lifetime_start
                        lifetime_time_bin = lifetime_end_bin - lifetime_start_bin + 1
                        lifetime_area = float(np.sum(time_proj_peak[lifetime_start_bin:lifetime_end_bin + 1]))
                        avg_per_bin = lifetime_area / lifetime_time_bin
                        
                        #print(f"  信号起始: bin {lifetime_start_bin:4d}, t = {lifetime_start:.6f} s")
                        #print(f"  信号结束: bin {lifetime_end_bin:4d}, t = {lifetime_end:.6f} s")
                        #print(f"  持续时间: {lifetime_time:.6f} s ({lifetime_time_bin} bins)")
                    else:
                        lifetime_time = lifetime_time_bin = 0
                        lifetime_area = avg_per_bin = 0.0
                        print("  → 未检测到有效信号区间（持续时间为0）")

            peak_info_list.append({
                'mean_mhz': mean_mhz,
                'peak_amp': peak_amp,
                'fwhm': fwhm_mhz,
                'lifetime_start': lifetime_start,
                'lifetime_end': lifetime_end,
                'lifetime_time': lifetime_time,
                'lifetime_time_bin': lifetime_time_bin,
                'lifetime_area': lifetime_area,
                'avg_per_bin': avg_per_bin,
                'max_time_proj': float(np.max(time_proj_peak)),
                'max_bin': int(np.argmax(time_proj_peak))
            })

        return peak_info_list

    def _draw_and_save_single(self, h: TH2F, h_proj: TH1F, base_name: str):
        """绘制单个注入的二维谱 + 投影图，并保存峰信息到 txt 文件"""
        c = TCanvas(f"c", base_name, 1600, 1200)

        # 上半部分：二维谱
        pad1 = TPad("pad1", "pad1", 0.0, 0.5, 1.0, 1.0)
        pad1.SetBottomMargin(0.13)
        pad1.SetLeftMargin(0.13)
        pad1.SetRightMargin(0.15)
        pad1.SetFillColor(18)
        pad1.Draw()
        pad1.cd()

        h.GetXaxis().SetRangeUser(
            max(self.freq_low / 1e6, h.GetXaxis().GetXmin()),
            min(self.freq_high / 1e6, h.GetXaxis().GetXmax()))
        h.Draw("COLZ")

        # ====================== 绘制所有峰的 mean_mhz（绿色虚线） ======================
        if hasattr(self, 'current_peak_info') and self.current_peak_info:
            for idx, peak in enumerate(self.current_peak_info, 1):
                mean_mhz = peak.get('mean_mhz', None)
                fwhm = peak.get('fwhm', None)
                lifetime_start = peak.get('lifetime_start', None)
                lifetime_end = peak.get('lifetime_end', None)
                x1 = mean_mhz - 5*fwhm/2.35
                x2 = mean_mhz + 5*fwhm/2.35
                y1 = lifetime_start
                y2 = lifetime_end
                if mean_mhz is None:
                    continue
                
                # 创建绿色矩形框 (TBox)
                box = TBox(x1, y1, x2, y2)
                box.SetLineColor(3)      # 绿色边框
                box.SetLineStyle(2)      # 虚线
                box.SetLineWidth(2)
                box.SetFillStyle(0)      # 关键：无填充（透明）
                box.DrawClone("")         # "same" 表示叠加在当前画布上

        pad1.SetLogz(True)
        self._set_axis_style(h, title_size=0.055, label_size=0.05)

        if self.args.z_min is not None:
            h.SetMinimum(self.args.z_min)
        if self.args.z_max is not None:
            h.SetMaximum(self.args.z_max)

        # 投影时间范围标记线
        if self.do_projection:
            for t_val in (self.t_min_proj, self.t_max_proj):
                line = TLine(self.freq_low/1e6, t_val, self.freq_high/1e6, t_val)
                line.SetLineColor(3)
                line.SetLineStyle(2)
                line.SetLineWidth(2)
                line.DrawClone()

        # ====================== 下半部分：投影图 ======================
        c.cd()
        pad2 = TPad("pad2", "pad2", 0.0, 0.0, 1.0, 0.5)
        pad2.SetTopMargin(0.08)
        pad2.SetBottomMargin(0.16)
        pad2.SetLeftMargin(0.13)
        pad2.SetRightMargin(0.15)
        pad2.SetFillColor(18)
        pad2.Draw()
        pad2.cd()
        pad2.SetLogy(True)

        h_proj.GetXaxis().SetRangeUser(
            max(self.freq_low / 1e6, h_proj.GetXaxis().GetXmin()),
            min(self.freq_high / 1e6, h_proj.GetXaxis().GetXmax()))
        h_proj.Draw("HIST")
        self._set_axis_style(h_proj, title_size=0.055, label_size=0.05)

        if self.args.proj_min is not None:
            h_proj.SetMinimum(self.args.proj_min)
        if self.args.proj_max is not None:
            h_proj.SetMaximum(self.args.proj_max)
        # ====================== 【修改】绘制频率自适应阈值曲线 ======================
        if (self.bg_mean is not None and self.bg_std is not None and 
            hasattr(self, 'freq_for_proj') and self.freq_for_proj is not None):
            
            # 使用 _create_histograms 中保存的对齐频率
            freq_mhz = self.freq_for_proj / 1e6
            n_points = len(freq_mhz)
            # 计算每个频率的自适应阈值
            thresh_values = np.zeros(n_points)
            for i in range(n_points):
                idx = i   # 因为已经对齐
                thresh_values[i] = self.bg_threshold[idx]
            # 创建 TGraph 绘制曲线
            from ROOT import TGraph
            g_thresh = TGraph(n_points)
            for i in range(n_points):
                g_thresh.SetPoint(i, freq_mhz[i], thresh_values[i])
            
            g_thresh.SetLineColor(2)      # 红色
            g_thresh.SetLineStyle(2)      # 虚线
            g_thresh.SetLineWidth(2)
            g_thresh.Draw("L same")       # L = 折线
            
            #print("   → 已绘制频率自适应阈值曲线")


        # 在下半部分投影图上也绘制峰位置（绿色虚线）
        if hasattr(self, 'current_peak_info') and self.current_peak_info:
            for peak in self.current_peak_info:
                mean_mhz = peak.get('mean_mhz')
                peak_amp = peak.get('peak_amp')
                if mean_mhz is not None:
                    line_lower = TLine(mean_mhz, 0, mean_mhz, h_proj.GetMaximum() * 1.05)
                    line_lower.SetLineColor(3)
                    line_lower.SetLineStyle(2)
                    line_lower.SetLineWidth(2)
                    line_lower.DrawClone()
                if mean_mhz is not None and peak_amp is not None:
                    # 创建一个红色的倒三角标记（MarkerStyle 23 = 倒三角）
                    marker = TMarker(mean_mhz, peak_amp, 23)   # 23 = 实心倒三角
                    
                    marker.SetMarkerColor(2)      # 2 = 红色
                    marker.SetMarkerSize(2.0)     # 大小，可调整（1.5~3.0 较合适）
                    marker.SetMarkerStyle(23)     # 23=倒三角, 22=正三角
                    marker.DrawClone()
                    
        # ====================== 保存峰信息到 TXT ======================
        if 'peaks_txt' in self.output_files and hasattr(self, 'current_peak_info') and self.current_peak_info:
            txt_file = self.result_dir / f"{base_name}_peaks.txt"
           
            with open(txt_file, "w", encoding="utf-8") as f:
                # 表头
                f.write(f"{'峰序号':<6}\t"
                        f"{'频率[MHz]':<12}\t"
                        f"{'峰高':<12}\t"
                        f"{'FWHM[MHz]':<12}\t"
                        f"{'start[s]':<12}\t"
                        f"{'end[s]':<12}\t"
                        f"{'寿命[s]':<12}\t"
                        f"{'寿命[bin]':<10}\t"
                        f"{'面积':<12}\t"
                        f"{'每bin面积':<12}\t"
                        f"{'文件名':<30}\n")          # ← 这里加上 \n
                
                # 分隔线（建议加长一点，更美观）
                f.write("-" * 160 + "\n")               # ← 建议改成 160
                
                # 数据行
                for idx, peak in enumerate(self.current_peak_info, 1):
                    f.write(f"{idx:<6}\t"
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
           
            #print(f" [保存] {txt_file.name} （共 {len(self.current_peak_info)} 个峰）")
        
        # ====================== 保存 PNG ======================
        if 'spectrogram_png' in self.output_files:
            save_dir = self.result_dir
            c.SaveAs(str(save_dir / f"{base_name}.png"))

        # ====================== 保存到 ROOT 文件 ======================
        if 'root_file' in self.output_files:
            root_file_path = self.result_dir / f"{base_name}.root"
            root_file = TFile(str(root_file_path), "RECREATE")
            root_file.cd()
            h.SetName("h2d")
            h_proj.SetName("h_proj")
            c.SetName("canvas")
            h.Write()
            h_proj.Write()
            c.Write()
            root_file.Close()
        
    def _save_all_peaks_summary(self):
        """处理完所有文件后生成总汇总文件"""
        if not self.all_peaks:
            print("未检测到任何峰")
            return

        summary_file = self.result_dir / "all_peaks_summary.txt"
        
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"{'总序号':<6}\t"
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
            f.write("-" * 170 + "\n")

            for i, peak in enumerate(self.all_peaks, 1):
                f.write(f"{i:<6}\t"
                        f"{peak.get('local_peak_idx', i):<6}\t"   # 如果需要可自行添加
                        f"{peak.get('mean_mhz', 0):<12.6f}\t"
                        f"{peak.get('peak_amp', 0):<12.6f}\t"
                        f"{peak.get('fwhm', 0):<12.6e}\t"
                        f"{peak.get('lifetime_start', -1):<12.6f}\t"
                        f"{peak.get('lifetime_end', -1):<12.6f}\t"
                        f"{peak.get('lifetime_time', -1):<12.6f}\t"
                        f"{peak.get('lifetime_time_bin', -1):<10}\t"
                        f"{peak.get('lifetime_area', 0):<12.4f}\t"
                        f"{peak.get('avg_per_bin', 0):<12.6f}\t"
                        f"{peak.get('filename', ''):<30}\n")

        print(f"\n🎉 全部处理完成！")
        print(f"   • 单个文件峰信息文件：已保存")
        print(f"   • 总汇总文件：{summary_file}")
        print(f"   • 共检测到 {len(self.all_peaks)} 个峰")
    
    def _get_max_amp_in_target(self, psd_array: np.ndarray, frequencies: np.ndarray) -> float:
        """获取目标频率范围内的最大幅度（用于打印）"""
        mask = (frequencies >= self.freq_low) & (frequencies <= self.freq_high)
        return float(np.max(psd_array[:, mask])) if np.any(mask) else 0.0

    def _create_histograms(self, psd_array, frequencies, times, base_name):
        """创建 TH2F 和 TH1F，并让 h_proj 的 binning 与 frequencies 严格对齐"""
        freq_bin_width = float(np.median(np.diff(frequencies))) if len(frequencies) > 1 else 0
        time_bin_width = float(np.median(np.diff(times))) if len(times) > 1 else 0

        # ====================== 二维图（保持不变） ======================
        n_bins_x = max(1, int(round((self.freq_high - self.freq_low) / freq_bin_width)) + 1)
        new_freq_high = self.freq_low + (n_bins_x - 1) * freq_bin_width

        h = TH2F("h",
                 f"{base_name};Frequency [MHz] (bin={freq_bin_width/1000:.3f} kHz);"
                 f"Time [s] (bin={self._format_bin_width(time_bin_width)})",
                 n_bins_x, self.freq_low / 1e6, new_freq_high / 1e6,
                 len(times), float(times.min()), float(times.min() + len(times) * time_bin_width))

        for ti in range(len(times)):
            for fi in range(len(frequencies)):
                f_val = frequencies[fi]
                if self.freq_low <= f_val <= self.freq_high:
                    binx = h.GetXaxis().FindBin(f_val / 1e6)
                    if 1 <= binx <= n_bins_x:
                        h.SetBinContent(binx, ti + 1, float(psd_array[ti, fi]))

        # ====================== 关键修改：h_proj 与 frequencies 严格对齐 ======================
        # 只使用目标频率范围内的频率点
        mask = (frequencies >= self.freq_low) & (frequencies <= self.freq_high)
        freq_in_range = frequencies[mask]
        
        # 检查 mask 是否为空
        if len(freq_in_range) == 0:
            print(f"  ❌ 错误：mask 为空！没有频率点在指定范围内")
            print(f"     建议检查：")
            print(f"       1. 数据频率单位是否正确（Hz vs MHz）？")
            print(f"       2. --freq_low 和 --freq_high 参数是否合理？")
            print(f"       3. --center_freq 参数是否正确？")
            raise ValueError(f"频率范围无重叠，无法创建直方图！")
            
        n_bins_proj = len(freq_in_range)
        h_proj = TH1F("h_proj",
                      f"X Projection [{self.freq_low/1e6:.3f}-{self.freq_high/1e6:.3f}] MHz;"
                      f"Frequency [MHz];Counts",
                      n_bins_proj,
                      (freq_in_range[0] - freq_bin_width/2) / 1e6,   # bin 左边缘
                      (freq_in_range[-1] + freq_bin_width/2) / 1e6)  # bin 右边缘

        # ====================== 填充 h_proj（带平均） ======================
        n_valid_time = 0
        for ti in range(len(times)):
            if not self.do_projection or (self.t_min_proj <= times[ti] <= self.t_max_proj):
                n_valid_time += 1
        if n_valid_time == 0:
            n_valid_time = 1

        weight = 1.0 / n_valid_time
        for ti in range(len(times)):
            if not self.do_projection or (self.t_min_proj <= times[ti] <= self.t_max_proj):
                for fi in range(len(frequencies)):
                    if mask[fi]:   # 只填充目标范围内的 bin
                        binx = h_proj.GetXaxis().FindBin(frequencies[fi] / 1e6)
                        if 1 <= binx <= n_bins_proj:
                            h_proj.SetBinContent(binx, 
                                h_proj.GetBinContent(binx) + float(psd_array[ti, fi]) * weight)

        # ====================== 新增：保存频率映射信息（供 analyze 使用） ======================
        self.freq_for_proj = freq_in_range          # 保存原始频率数组
        self.proj_bin_count = n_bins_proj

        return h, h_proj

    @staticmethod
    def _format_bin_width(width: float) -> str:
        """格式化时间 bin 宽度"""
        if width >= 1e-3:
            return f"{width*1e3:.3f} ms"
        elif width >= 1e-6:
            return f"{width*1e6:.3f} us"
        else:
            return f"{width*1e9:.3f} ns"

    @staticmethod
    def _set_axis_style(h, title_size=0.045, label_size=0.04, title_offset=1.1):
        """统一设置坐标轴样式"""
        font = 42
        for axis in (h.GetXaxis(), h.GetYaxis()):
            axis.SetTitleSize(title_size)
            axis.SetLabelSize(label_size)
            axis.SetTitleOffset(title_offset)
            axis.SetTitleFont(font)
            axis.SetLabelFont(font)
            axis.CenterTitle(True)

        if hasattr(h, 'GetZaxis') and h.GetZaxis():
            zaxis = h.GetZaxis()
            zaxis.SetTitleSize(title_size * 0.9)
            zaxis.SetLabelSize(label_size * 0.9)
            zaxis.SetTitleOffset(0.05)
            zaxis.SetTitleFont(font)
            zaxis.SetLabelFont(font)
            zaxis.CenterTitle(True)

    # ──────────────────────────────────────────────────────────
    # 类方法：预计算本底阈值（在主进程执行，避免子进程重复加载1GB大文件）
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def precompute_background(background_path, n_sigma=5.0, smooth_window=51):
        """加载本底 .npz 文件并计算阈值，返回小尺寸数组字典。

        返回值
        ------
        dict : ``{'bg_frequencies', 'bg_mean', 'bg_std', 'bg_threshold'}``
            所有数组均为 1D float64，大小 ≈ 频率点数 × 4 × 8 bytes，通常 < 1 MB。
        """
        import numpy as np
        import time

        t0 = time.time()
        print(f"[*] 预计算本底阈值: {background_path}")
        bg_data = np.load(background_path)
        bg_frequencies = bg_data['frequencies']
        bg_psd_array = bg_data['psd_arrays']

        if bg_psd_array.ndim == 2 and bg_psd_array.shape[0] > bg_psd_array.shape[1]:
            bg_psd_array = bg_psd_array.T

        n_freq_data = bg_psd_array.shape[1]
        n_freq_bg = len(bg_frequencies)
        if n_freq_data != n_freq_bg:
            min_len = min(n_freq_data, n_freq_bg)
            bg_frequencies = bg_frequencies[:min_len]
            bg_psd_array = bg_psd_array[:, :min_len]
            print(f"  → 频率维度对齐至 {min_len}")

        bg_mean, bg_std, bg_threshold = SpectrumAnalyzer._compute_background_threshold(
            bg_psd_array, n_sigma=n_sigma, smooth_window=smooth_window)

        print(f"  → 本底阈值计算完成 ({time.time()-t0:.1f}s, "
              f"数据量 ~{bg_psd_array.nbytes/1e6:.0f} MB → "
              f"阈值 ~{bg_mean.nbytes*3/1024:.0f} KB)")
        return {
            'bg_frequencies': bg_frequencies,
            'bg_mean': bg_mean,
            'bg_std': bg_std,
            'bg_threshold': bg_threshold,
        }

    # ──────────────────────────────────────────────────────────
    # 类方法：从内存数组直接运行 Spectrogram 分析（供 looper_cut2 子进程调用）
    # ──────────────────────────────────────────────────────────
    @classmethod
    def run_inline_analysis(cls, frequencies, times, psd_array,
                             output_dir, base_name,
                             center_freq=0.0,
                             freq_low=None, freq_high=None,
                             t_min=None, t_max=None,
                             background_path=None, n_sigma=5.0,
                             bg_precomputed=None,
                             output_files=None,
                             z_min=None, z_max=None,
                             proj_min=None, proj_max=None):
        """从内存中的频谱数据直接运行 Spectrogram 分析（绕过文件 I/O）。

        由 ``looper_cut2._async_processing_task`` 在子进程中调用，
        在每个 injection 的 FFT 计算完成后即时执行 Spectrogram 分析。

        Parameters
        ----------
        bg_precomputed : dict or None
            由 ``precompute_background()`` 返回的预计算本底阈值字典。
            如果提供，子进程**不再**加载原始 1GB 本底文件。
        output_files : list or None
            控制输出文件类型，可选值：``peaks_txt``, ``spectrogram_png``, ``root_file``。
            None 或省略则全部输出。

        注意
        ----
        ``frequencies`` 应为**物理频率**（Hz），即已包含 ``center_freq`` 偏移。
        因此本方法的 ``center_freq`` 默认值为 0，避免重复叠加。
        """
        import argparse
        from pathlib import Path

        # ── 维度对齐（frequencies 可能比 psd_array 的频点列多1个右边界） ──
        if psd_array.ndim == 2:
            n_freq = psd_array.shape[1]
            if len(frequencies) > n_freq:
                frequencies = frequencies[:n_freq]
            if len(times) > psd_array.shape[0]:
                times = times[:psd_array.shape[0]]

        # ── 默认全部输出 ──
        if output_files is None:
            output_files = ['peaks_txt', 'spectrogram_png', 'root_file']

        # ── 用传入的 kwargs 构造 mock args ──
        # 如果有预计算本底，就不传 background_path（避免 _load_background 读大文件）
        actual_background = background_path if bg_precomputed is None else None
        args = argparse.Namespace(
            center_freq=center_freq,
            freq_low=freq_low if freq_low is not None else frequencies[0],
            freq_high=freq_high if freq_high is not None else frequencies[-1],
            
            t_min=t_min,
            t_max=t_max,
            date_start=None,
            date_end=None,
            background=actual_background,
            n_sigma=n_sigma,
            output_dir=str(output_dir),
            data_dir="",
            output_files=output_files,
            z_min=z_min,
            z_max=z_max,
            proj_min=proj_min,
            proj_max=proj_max,

        )

        # ── 创建临时分析器实例 ──
        analyzer = cls(args)
        # 如果提供了预计算本底，直接注入（_load_background 已被跳过）
        if bg_precomputed is not None:
            analyzer.bg_mean = bg_precomputed['bg_mean']
            analyzer.bg_std = bg_precomputed['bg_std']
            analyzer.bg_threshold = bg_precomputed['bg_threshold']
            # 保存原始 bg frequencies 供 analyze_ion_lifetime 对齐使用
            analyzer.bg_frequencies = bg_precomputed.get('bg_frequencies', None)

        analyzer.result_dir = Path(output_dir) / "spectrogram_analysis"
        analyzer.result_dir.mkdir(parents=True, exist_ok=True)

        # ── 运行分析管线（复用同名的私有方法） ──
        h, h_proj = analyzer._create_histograms(psd_array, frequencies,
                                                 times, base_name)
        analyzer.current_peak_info = analyzer.analyze_ion_lifetime(
            h_proj, psd_array, frequencies, times)
        analyzer._draw_and_save_single(h, h_proj, base_name)

        n_peaks = len(analyzer.current_peak_info)
        #print(f"    → Spectrogram 分析完成: {n_peaks} 个峰 → {analyzer.result_dir}")
        return analyzer.current_peak_info


def parse_arguments():
    parser = argparse.ArgumentParser(description="频谱分析 - 累加特定频率范围内有强信号的注入（SpectrumAnalyzer版）")
    parser.add_argument("--data_dir", type=str, required=True, help="npz 文件目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--center_freq", type=float, default=310e6, help="中心频率 (Hz)")
    parser.add_argument("--t_min", type=float, default=None, help="投影时间下限 (s)")
    parser.add_argument("--t_max", type=float, default=None, help="投影时间上限 (s)")
    parser.add_argument("--freq_low", type=float, required=True, help="感兴趣频率下限 (Hz)")
    parser.add_argument("--freq_high", type=float, required=True, help="感兴趣频率上限 (Hz)")
    parser.add_argument("--z_min", type=float, default=None, help="Z轴最小值")
    parser.add_argument("--z_max", type=float, default=None, help="Z轴最大值")
    parser.add_argument("--proj_min", type=float, default=None, help="投影图 Y轴最小值")
    parser.add_argument("--proj_max", type=float, default=None, help="投影图 Y轴最大值")
    parser.add_argument("--background", type=str, default=None, help="本底数据文件路径（.npz）")
    parser.add_argument("--n_sigma", type=float, default=5.0, help="本底扣除的 sigma 倍数，默认 5.0")
    parser.add_argument("--date_start", type=str, default=None, help="起始时间")
    parser.add_argument("--date_end", type=str, default=None, help="结束时间")
    parser.add_argument("--output_files", type=str, nargs='+',
                        default=['peaks_txt', 'spectrogram_png', 'root_file'],
                        choices=['peaks_txt', 'spectrogram_png', 'root_file'],
                        help="输出文件类型（可多选，默认全部）")
    return parser.parse_args()

def main():
    args = parse_arguments()
    analyzer = SpectrumAnalyzer(args)
    analyzer.prepare_output_dirs()
    analyzer.process_all_files()
    print("\n✅ 所有处理完成！")

if __name__ == "__main__":
    main()
