#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
五步流处理流水线：CNN识别 → 基线估计 → 谱重建 → 寻峰导出 → 精确衰变。

架构
----
- **Processor 层**（CNNIonDetector / BaselineEstimator / SpectrumReconstructor /
  PeakExtractor / PreciseDecayAnalyzer）：纯计算，每个提供 ``process_array()`` 和
  ``process_mean()`` 两种内存入口。
- **PipelineContext**：在步骤间传递数据的共享上下文，统一管理大数组的
  延迟加载 / 缓存 / 释放。
- **StreamPipeline**：可自由组合步骤的流处理管线，支持单文件、批量、目录监控。

四种使用场景
------------
1) 五步全跑，只要峰 CSV::

    pipe = StreamPipeline()
    pipe.add("cnn",      CNNIonDetector("model.pth"))
    pipe.add("baseline", BaselineEstimator())
    pipe.add("recon",    SpectrumReconstructor())
    pipe.add("peaks",    PeakExtractor(), sink=Sink.csv("peaks.csv", PeakExtractor.FIELDNAMES))
    pipe.add("decay",    PreciseDecayAnalyzer("/data/raw/"))
    pipe.run("/data/")

2) 五步全跑，选择性输出::

    pipe = StreamPipeline()
    pipe.add("cnn",      CNNIonDetector("model.pth"), sink=Sink.csv("cnn.csv", [...]))
    pipe.add("baseline", BaselineEstimator())                         # 不存盘
    pipe.add("recon",    SpectrumReconstructor(),  sink=Sink.npz("/out/recon/"))
    pipe.add("peaks",    PeakExtractor(),          sink=Sink.csv("peaks.csv", [...]))
    pipe.add("decay",    PreciseDecayAnalyzer("/data/raw/"),
             sink=Sink.csv("decay.csv", PreciseDecayAnalyzer.FIELDNAMES))

3) 只跑某一步::

    pipe = StreamPipeline()
    pipe.add("baseline", BaselineEstimator(), sink=Sink.npy("/out/baselines/"))
    pipe.run("/data/")

4) 四步 + 精确衰变（流处理）::

    pipe = StreamPipeline()
    pipe.add("cnn",      CNNIonDetector("model.pth"))
    pipe.add("baseline", BaselineEstimator())
    pipe.add("recon",    SpectrumReconstructor())
    pipe.add("peaks",    PeakExtractor())
    pipe.add("decay",    PreciseDecayAnalyzer("/data/raw/"),
             sink=Sink.csv("decay.csv", PreciseDecayAnalyzer.FIELDNAMES))
    pipe.process_stream("/data/")
"""

import torch
import torch.nn as nn
import numpy as np
import os
import csv
import re
import time
import signal
from nonparams_est import NONPARAMS_EST
from reconstruct_spectrum import reconstruct_ion_spectrum, extract_peaks_log_detect

from typing import Optional, List, Dict, Union, Any, Callable
from collections import OrderedDict
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.signal import butter, filtfilt
from preprocessing import Preprocessing


# ============================================================
# Model definition (must match training-time architecture exactly)
# ============================================================
class SpectrumCNN(nn.Module):
    """CNN model for spectrum classification — signal (1) vs noise (0)."""

    def __init__(self, num_classes: int = 2):
        super(SpectrumCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1024),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(256),
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================
# Step 1: CNN 离子信号检测
# ============================================================
class CNNIonDetector:
    """
    Step 1: CNN-based ion signal detector.

    Loads a pre-trained CNN model and classifies spectrum data as containing
    ion signal (label=1) or noise (label=0).

    Supports three usage modes:
      - ``process_array()``  — in-memory (streaming), no file I/O
      - ``process_file()``   — single .npz file
      - ``process_batch()``  — entire directory, with optional CSV output

    Parameters
    ----------
    model_path : str
        Path to the trained model weights (``.pth`` file).
    target_length : int
        Expected number of frequency bins in the input PSD data.
    device : str
        Torch device string: ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_path: str,
        target_length: int = 209715,
        device: str = "cpu",
    ):
        self.target_length = target_length
        self.device = torch.device(device)
        self.model_path = model_path

        # -- build & load model --
        self.model = SpectrumCNN()
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except FileNotFoundError:
            print(f"错误：未找到模型权重文件 {model_path}")
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _preprocess(self, avg_psd: np.ndarray) -> torch.Tensor:
        """
        Convert a 1-D averaged PSD array into the 2-channel input tensor
        expected by the CNN.

        Channel 1 — log10 + z-score normalisation.
        Channel 2 — background-subtracted enhancement.
        """
        avg_psd = avg_psd.astype(np.float32)

        # Pad or truncate
        if len(avg_psd) > self.target_length:
            avg_psd = avg_psd[: self.target_length]
        elif len(avg_psd) < self.target_length:
            avg_psd = np.pad(avg_psd, (0, self.target_length - len(avg_psd)))

        # Channel 1
        avg_psd_log = 10.0 * np.log10(avg_psd + 1e-12)
        mean_val = np.mean(avg_psd_log)
        std_val = np.std(avg_psd_log)
        ch1 = torch.from_numpy(
            (avg_psd_log - mean_val) / (std_val + 1e-8)
        ).float()

        # Channel 2
        x_tensor = (
            torch.from_numpy(avg_psd_log).float().unsqueeze(0).unsqueeze(0)
        )
        padding = 1001 // 2
        bg_trend = nn.functional.avg_pool1d(
            x_tensor, kernel_size=1001, stride=1, padding=padding
        )
        ch2 = x_tensor - bg_trend[..., : self.target_length]
        ch2 = ch2.squeeze()
        ch2 = ch2 / (ch2.std() + 1e-8)

        # (1, 2, target_length)
        return torch.stack([ch1, ch2], dim=0).unsqueeze(0).to(self.device)

    def _infer(self, input_tensor: torch.Tensor) -> tuple:
        """Run the model and return *(class_id, confidence)*."""
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_array(
        self,
        psd_arrays: np.ndarray,
        filename: str = "unknown",
    ) -> Dict[str, Union[str, int, float]]:
        """
        Process a raw PSD array **in memory** — the core streaming entry point.

        This is the preferred method when you want to chain steps together
        without touching the filesystem.  It accepts a numpy array and
        returns a result dict.

        Parameters
        ----------
        psd_arrays : np.ndarray
            2-D array of shape ``(n_sweeps, n_freq_bins)``.
        filename : str
            Label / identifier carried through to the result dict.

        Returns
        -------
        dict
            ``{'filename': str, 'label': int, 'confidence': float}``
        """
        avg_psd = np.mean(psd_arrays, axis=0)
        input_tensor = self._preprocess(avg_psd)
        class_id, conf_val = self._infer(input_tensor)
        return {
            "filename": filename,
            "label": class_id,
            "confidence": conf_val,
        }

    def process_mean(
        self, avg_psd: np.ndarray, filename: str = "unknown"
    ) -> Dict[str, Union[str, int, float]]:
        """
        处理已平均的 PSD（无需加载完整 2D 数组，节省内存）。

        用于流处理管线中与 PipelineContext 配合使用。
        """
        input_tensor = self._preprocess(avg_psd)
        class_id, conf_val = self._infer(input_tensor)
        return {"filename": filename, "label": class_id, "confidence": conf_val}

    def process_file(self, file_path: str) -> Dict[str, Union[str, int, float]]:
        """
        Process a single ``.npz`` file.

        Parameters
        ----------
        file_path : str
            Path to the ``.npz`` file (must contain key ``'psd_arrays'``).

        Returns
        -------
        dict
            ``{'filename': str, 'label': int, 'confidence': float}``
        """
        with np.load(file_path) as loader:
            if "psd_arrays" not in loader:
                raise ValueError(
                    f"File {file_path} does not contain 'psd_arrays'"
                )
            psd_arrays = loader["psd_arrays"]

        filename = os.path.basename(file_path)
        return self.process_array(psd_arrays, filename=filename)

    def process_batch(
        self,
        input_dir: str,
        output_csv: Optional[str] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Process **all** ``.npz`` files in a directory.

        Parameters
        ----------
        input_dir : str
            Directory containing ``.npz`` files.
        output_csv : str or None
            When a path is given, results are written to this CSV file.
            When ``None`` (the default), results are **only** returned in
            memory and no CSV is created — use this when chaining.
        verbose : bool
            Print per-file progress.

        Returns
        -------
        list of dict
            Each dict: ``{'filename': str, 'label': int, 'confidence': float}``
        """
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        files = sorted(
            [f for f in os.listdir(input_dir) if f.endswith(".npz")]
        )
        if not files:
            if verbose:
                print(f"[CNNIonDetector] No .npz files found in {input_dir}")
            return []

        if verbose:
            print(
                f"[CNNIonDetector] Found {len(files)} .npz files "
                f"in {input_dir}"
            )

        # 增量写 CSV：先写表头
        if output_csv is not None:
            with open(output_csv, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(["filename", "label", "confidence"])

        results: List[Dict] = []

        for i, f in enumerate(files):
            file_path = os.path.join(input_dir, f)
            try:
                result = self.process_file(file_path)
                results.append(result)
                # 每处理完一个文件立即追加到 CSV
                if output_csv is not None:
                    with open(output_csv, "a", newline="", encoding="utf-8") as fh:
                        csv.writer(fh).writerow(
                            [result["filename"], result["label"],
                             f"{result['confidence']:.4f}"])
                if verbose:
                    print(
                        f"[{i + 1}/{len(files)}] {f:<40} "
                        f"| label: {result['label']} "
                        f"| conf: {result['confidence']:.4f}"
                    )
            except Exception as exc:
                if verbose:
                    print(
                        f"[{i + 1}/{len(files)}] ERROR {f}: {exc}"
                    )

        if verbose and output_csv:
            print(f"[CNNIonDetector] CSV saved -> {os.path.abspath(output_csv)}")

        return results

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _write_csv(
        results: List[Dict[str, Union[str, int, float]]],
        path: str,
    ) -> None:
        """Persist inference results as a CSV file."""
        with open(path, mode="w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["filename", "label", "confidence"])
            for r in results:
                writer.writerow(
                    [r["filename"], r["label"], f"{r['confidence']:.4f}"]
                )

    @staticmethod
    def read_csv(path: str) -> List[Dict[str, Union[str, int, float]]]:
        """
        Read a CNN-results CSV back into memory.

        This is useful when step 1 was run earlier and downstream steps
        need to consume its output.
        """
        results = []
        with open(path, mode="r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                results.append({
                    "filename": row["filename"],
                    "label": int(row["label"]),
                    "confidence": float(row["confidence"]),
                })
        return results

    @staticmethod
    def filter_signal(
        results: List[Dict],
        min_confidence: float = 0.9,
    ) -> List[str]:
        """
        Convenience: extract filenames classified as *signal* with
        confidence >= *min_confidence*.

        Returns
        -------
        list of str
            Filenames that pass the filter.
        """
        return [
            r["filename"]
            for r in results
            if r["label"] == 1 and r["confidence"] >= min_confidence
        ]


# ============================================================
# Step 2: 基线估计
# ============================================================
class BaselineEstimator:
    """
    Step 2: 用 BrPLS 估计基线。

    使用方式:
        est = BaselineEstimator(l=1e9, ratio=1e-7)

        # 批量 + 保存 .npy
        est.process_batch("/data/cutInjection/", output_dir="/data/baseline/")

        # 批量 + 不保存（串联）
        results = est.process_batch("/data/cutInjection/")

        # 流式：单文件
        bl = est.process_file("/data/file.npz", output_path="/data/baseline/file.npy")

        # 流式：纯数组
        bl = est.process_array(psd_2d_array)

    Parameters
    ----------
    l : float
        平滑参数 (default: 1e9)。
    ratio : float
        终止条件阈值 (default: 1e-7)。
    method : str
        PLS 方法名 (default: 'BrPLS')。
    """
    def __init__(self, l=1e9, ratio=1e-7, method="BrPLS",
                 cache_by_data: bool = False, max_cache: int = 4):
        self.l = l
        self.ratio = ratio
        self.method = method
        # 按 data 复用基线：同一 data 的多个 trigger 只估计一次（流处理加速）
        self.cache_by_data = cache_by_data
        self.max_cache = max_cache
        self._baseline_cache: OrderedDict = OrderedDict()

    # ---------- 核心 ----------
    def estimate(self, psd_mean):
        """
        从平均 PSD 估计基线。

        Parameters
        ----------
        psd_mean : np.ndarray, 1-D (线性空间)

        Returns
        -------
        baseline : np.ndarray, 1-D (线性空间)
        """
        base_log = NONPARAMS_EST(np.log(psd_mean)).pls(
            method=self.method, l=self.l, ratio=self.ratio)
        return np.exp(base_log)

    # ---------- 公开接口 ----------
    def process_array(self, psd_arrays):
        """
        流式核心：接收 (n_sweeps, n_freq_bins)，返回基线数组。
        无文件 I/O。
        """
        return self.estimate(np.mean(psd_arrays, axis=0))

    def process_mean(self, psd_mean: np.ndarray) -> np.ndarray:
        """
        从已平均的 PSD 估计基线（无需加载完整 2D 数组，节省内存）。

        用于流处理管线中与 PipelineContext 配合使用。
        """
        return self.estimate(psd_mean)

    def process_mean_cached(self, psd_mean: np.ndarray, data_key) -> np.ndarray:
        """
        按 data 缓存基线估计结果。

        同一 data 文件下的多个 trigger 共享同一噪声背景，基线形状几乎不变。
        首次遇到某 data 时做完整 BrPLS 拟合并缓存，后续 trigger 直接复用，
        避免流处理中反复拟合导致的速度瓶颈。

        Parameters
        ----------
        psd_mean : np.ndarray
            当前 trigger 的平均 PSD（仅缓存未命中时才实际使用）。
        data_key : tuple
            缓存键，通常为 ``(channel, data_idx)``。
        """
        if data_key in self._baseline_cache:
            self._baseline_cache.move_to_end(data_key)
            return self._baseline_cache[data_key]

        baseline = self.estimate(psd_mean)
        if len(self._baseline_cache) >= self.max_cache:
            self._baseline_cache.popitem(last=False)
        self._baseline_cache[data_key] = baseline
        return baseline

    def process_file(self, file_path, output_path=None):
        """
        处理单个 .npz 文件。

        Parameters
        ----------
        output_path : str or None
            若给定，保存为 .npy；否则只返回数组。
        """
        data = np.load(file_path)
        baseline = self.process_array(data["psd_arrays"])
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            np.save(output_path, baseline)
        return baseline

    def process_batch(self, input_dir, output_dir=None, verbose=True):
        """
        批量处理目录下的所有 .npz 文件。

        Parameters
        ----------
        output_dir : str or None
            None -> 只返回 list，不写文件（用于串联）。
            给定路径 -> 保存为 baseline_<原名>.npy。
        """
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"目录不存在: {input_dir}")

        files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npz"))
        if verbose:
            print(f"[Step2] {input_dir} 中共 {len(files)} 个 .npz 文件")

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        results = []
        for i, f in enumerate(files):
            try:
                fpath = os.path.join(input_dir, f)
                out = os.path.join(output_dir, f"baseline_{os.path.splitext(f)[0]}.npy") if output_dir else None
                baseline = self.process_file(fpath, output_path=out)
                results.append({"filename": f, "baseline": baseline}
                               if output_dir is None else
                               {"filename": f, "path": out})
                if verbose:
                    print(f"  [{i+1}/{len(files)}] {f}")
            except Exception as e:
                if verbose:
                    print(f"  [{i+1}/{len(files)}] {f} 出错: {e}")

        return results


# ============================================================
# Step 3: 谱重建
# ============================================================
class SpectrumReconstructor:
    """
    Step 3: 多尺度 CWT 特征融合重建离子谱。

    使用方式:
        rec = SpectrumReconstructor(k_high=8.0, k_low=1.8)

        # 批量 + 保存 .npz
        rec.process_batch("/data/cutInjection/", "/data/baseline/", output_dir="/data/reconstructed/")

        # 批量 + 不保存（串联）
        results = rec.process_batch("/data/cutInjection/", "/data/baseline/")

        # 流式：单对文件
        result = rec.process_file("/data/file.npz", "/data/baseline/baseline_file.npy")

        # 流式：纯数组
        result = rec.process_array(psd_2d_array, baseline_1d, frequencies_1d)

    Parameters
    ----------
    k_high : float
        信号判定严格阈值 (default: 8.0)。
    k_low : float
        信号边界宽松阈值 (default: 1.8)。
    """
    def __init__(self, k_high=8.0, k_low=1.8):
        self.k_high = k_high
        self.k_low = k_low

    # ---------- 核心 ----------
    def reconstruct(self, psd_raw, baseline):
        """
        从原始 PSD 和基线重建纯信号谱。

        Parameters
        ----------
        psd_raw : np.ndarray, 1-D (线性空间)
        baseline : np.ndarray, 1-D (线性空间)

        Returns
        -------
        pure_signal_log : np.ndarray, 1-D (对数空间)
        """
        log_psd = np.log(psd_raw)
        log_baseline = np.log(baseline)
        rebuilt = reconstruct_ion_spectrum(
            log_psd, log_baseline,
            k_high=self.k_high, k_low=self.k_low)
        return rebuilt - log_baseline

    # ---------- 公开接口 ----------
    def process_array(self, psd_arrays, baseline, frequencies):
        """
        流式核心：接收原始数组和基线，返回重构结果。
        无文件 I/O。

        Returns
        -------
        dict
            {'frequencies': np.ndarray, 'psd_log': np.ndarray}
        """
        psd_raw = np.mean(psd_arrays, axis=0)
        pure_signal_log = self.reconstruct(psd_raw, baseline)
        return {"frequencies": frequencies[:-1], "psd_log": pure_signal_log}

    def reconstruct_from_mean(
        self, psd_mean: np.ndarray, baseline: np.ndarray, frequencies: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        从已平均的 PSD 重建（无需加载完整 2D 数组，节省内存）。

        用于流处理管线中与 PipelineContext 配合使用。
        """
        pure_signal_log = self.reconstruct(psd_mean, baseline)
        return {"frequencies": frequencies[:-1], "psd_log": pure_signal_log}

    def process_file(self, raw_path, baseline_path, output_path=None):
        """
        处理单对 raw .npz + baseline .npy 文件。

        Parameters
        ----------
        output_path : str or None
            若给定，保存为 .npz；否则只返回 dict。
        """
        raw_data = np.load(raw_path)
        baseline = np.load(baseline_path)
        result = self.process_array(
            raw_data["psd_arrays"], baseline, raw_data["frequencies"])
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            np.savez(output_path, **result)
        return result

    def process_batch(self, input_dir, baseline_dir,
                      output_dir=None, file_list=None, verbose=True):
        """
        批量处理目录下的 .npz 文件。

        Parameters
        ----------
        input_dir : str
            原始 .npz 文件目录。
        baseline_dir : str
            基线 .npy 文件目录（文件名为 baseline_<原名>.npy）。
        output_dir : str or None
            None -> 只返回 list，不写文件（用于串联）。
            给定路径 -> 保存为 reconstruct_<原名>.npz。
        file_list : list of str or None
            指定要处理的文件名列表；None 则处理 input_dir 下所有 .npz。
        """
        if file_list is not None:
            files = sorted(file_list)
        else:
            files = sorted(f for f in os.listdir(input_dir) if f.endswith(".npz"))

        if verbose:
            print(f"[Step3] 共 {len(files)} 个文件待处理")

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        results = []
        for i, f in enumerate(files):
            try:
                raw_path = os.path.join(input_dir, f)
                base_name = f"baseline_{os.path.splitext(f)[0]}.npy"
                base_path = os.path.join(baseline_dir, base_name)

                if not os.path.exists(base_path):
                    if verbose:
                        print(f"  [{i+1}/{len(files)}] {f} 跳过: 未找到基线 {base_name}")
                    continue

                out = os.path.join(output_dir, f"reconstruct_{f}") if output_dir else None
                result = self.process_file(raw_path, base_path, output_path=out)

                if np.count_nonzero(result["psd_log"]) == 0:
                    if verbose:
                        print(f"  [{i+1}/{len(files)}] {f} 跳过: 纯信号为0")
                    continue

                results.append({"filename": f, **result}
                               if output_dir is None else
                               {"filename": f, "path": out})
                if verbose:
                    print(f"  [{i+1}/{len(files)}] {f}")

            except Exception as e:
                if verbose:
                    print(f"  [{i+1}/{len(files)}] {f} 出错: {e}")

        return results


# ============================================================
# Step 4: 寻峰导出
# ============================================================
class PeakExtractor:
    """
    Step 4: 从重建谱中提取离子峰信息并导出 CSV。

    使用方式:
        pk = PeakExtractor(snr_factor=6.0)

        # 批量 + 写 CSV
        pk.process_batch("/data/reconstructed/", "/data/cutInjection/",
                         "/data/baseline/", output_csv="peaks.csv")

        # 批量 + 不写 CSV（串联）
        all_peaks = pk.process_batch("/data/reconstructed/", "/data/cutInjection/",
                                     "/data/baseline/")

        # 流式：单组文件
        peaks = pk.process_file("/data/reconstructed/recon_xxx.npz",
                                "/data/cutInjection/xxx.npz",
                                "/data/baseline/baseline_xxx.npy")

        # 流式：纯数组
        peaks = pk.process_array(frequencies, psd_log, psd_arrays, times, baseline)

    Parameters
    ----------
    snr_factor : float
        信噪比阈值 (default: 6.0)。
    """
    # CSV 字段顺序
    FIELDNAMES = [
        'peak_pos', 'err_pos', 'sigma', 'err_sigma',
        'height_ratio', 'height_ion',
        'exist_state', 'exist_time', 'valid', 'pair_num',
        'filename']

    def __init__(self, snr_factor=6.0):
        self.snr_factor = snr_factor

    # ---------- 核心 ----------
    def extract(self, frequencies, psd_log, psd_arrays, times, baseline):
        """
        从重建谱提取峰列表，含衰变配对标记。

        Parameters
        ----------
        frequencies : np.ndarray, 1-D
        psd_log : np.ndarray, 1-D — 重建后的纯信号 (对数空间)
        psd_arrays : np.ndarray, 2-D — 原始 PSD (n_sweeps x n_freq)
        times : np.ndarray, 1-D — 时间轴
        baseline : np.ndarray, 1-D — 基线 (线性空间)

        Returns
        -------
        list of dict — 每个 dict 为一个峰的属性
        """
        time_interval = times[1] - times[0]
        b_log = np.log(baseline)
        peaks = extract_peaks_log_detect(
            frequencies, psd_log, psd_arrays, time_interval, b_log,
            snr_factor=self.snr_factor)
        if peaks:
            self._apply_pairing(peaks, times[-1], time_interval)
        return peaks or []

    def _apply_pairing(self, peaks, total_time, time_interval):
        """同种离子激发态->基态衰变配对"""
        for p in peaks:
            p['pair_num'] = 0
            p['valid'] = 0 if p['exist_state'] == 2 else 1

        pair_counter, used = 0, set()
        for i in range(len(peaks)):
            if peaks[i]['exist_state'] == 1 and i not in used:
                for j in range(len(peaks)):
                    if peaks[j]['exist_state'] == 2 and j not in used:
                        time_ok = (np.abs(peaks[i]['exist_time']
                                          + peaks[j]['exist_time']
                                          - total_time)
                                   <= 2 * time_interval)
                        pos_ok = (peaks[i]['peak_pos'] < peaks[j]['peak_pos']
                                  and np.abs(peaks[i]['peak_pos']
                                             - peaks[j]['peak_pos']) <= 80e3)
                        if time_ok and pos_ok:
                            pair_counter += 1
                            peaks[i]['pair_num'] = peaks[j]['pair_num'] = pair_counter
                            peaks[i]['valid'] = peaks[j]['valid'] = 1
                            used.add(i)
                            used.add(j)
                            break

    # ---------- 公开接口 ----------
    def process_array(self, frequencies, psd_log, psd_arrays, times, baseline):
        """流式核心：接收所有数组，返回峰列表。无文件 I/O。"""
        return self.extract(frequencies, psd_log, psd_arrays, times, baseline)

    def process_file(self, recon_path, raw_path, baseline_path):
        """
        处理单组三个文件。

        Parameters
        ----------
        recon_path : str — 重建 .npz (含 frequencies, psd_log)
        raw_path : str   — 原始 .npz (含 psd_arrays, times)
        baseline_path : str — 基线 .npy
        """
        recon = np.load(recon_path)
        raw_data = np.load(raw_path)
        baseline = np.load(baseline_path)
        return self.extract(
            recon['frequencies'], recon['psd_log'],
            raw_data['psd_arrays'], raw_data['times'], baseline)

    def process_batch(self, recon_dir, raw_dir, baseline_dir,
                      output_csv=None, file_list=None,
                      resume=False, verbose=True):
        """
        批量处理目录下的重构 .npz 文件。

        Parameters
        ----------
        recon_dir : str — 重建 .npz 目录 (reconstruct_<原名>.npz)
        raw_dir : str   — 原始 .npz 目录 (<原名>.npz)
        baseline_dir : str — 基线 .npy 目录 (baseline_<原名>.npy)
        output_csv : str or None
            None -> 只返回 list，不写文件。
            给定路径 -> 增量写入 CSV（每处理完一个文件立即追加）。
        file_list : list of str or None
            要处理的文件名列表（reconstruct_ 开头的 .npz 名）。
            None -> 扫描 recon_dir。
        resume : bool
            True 且 output_csv 已存在时，跳过 CSV 中已有的 filename。
        """
        if file_list is not None:
            files = sorted(file_list)
        else:
            files = sorted(f for f in os.listdir(recon_dir) if f.endswith(".npz"))

        if verbose:
            print(f"[Step4] 共 {len(files)} 个文件待处理")

        # 断点续传
        processed = set()
        if resume and output_csv and os.path.exists(output_csv):
            with open(output_csv, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                processed = {row['filename'] for row in reader
                             if row.get('filename')}
            if verbose:
                print(f"[Step4] 续传模式：跳过 {len(processed)} 个已处理文件")

        # 写 CSV 表头（新建时）
        header_written = bool(processed)  # 续传时不重写表头
        if output_csv and not header_written:
            with open(output_csv, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=self.FIELDNAMES).writeheader()

        all_peaks = []
        for i, fname in enumerate(files):
            if fname in processed:
                continue

            try:
                recon_path = os.path.join(recon_dir, fname)
                # 路径映射: reconstruct_<原名>.npz -> <原名>.npz / baseline_<原名>.npy
                raw_name = fname.replace("reconstruct_", "", 1)
                raw_path = os.path.join(raw_dir, raw_name)
                base_name = "baseline_" + raw_name.replace(".npz", ".npy")
                base_path = os.path.join(baseline_dir, base_name)

                if not os.path.exists(base_path):
                    if verbose:
                        print(f"  [{i+1}/{len(files)}] {fname} 跳过: 未找到基线 {base_name}")
                    continue

                peaks = self.process_file(recon_path, raw_path, base_path)

                # 标注文件名 & 实时追加 CSV
                for p in peaks:
                    p['filename'] = fname
                all_peaks.extend(peaks)

                if output_csv and peaks:
                    with open(output_csv, "a", newline="", encoding="utf-8") as fh:
                        csv.DictWriter(fh, fieldnames=self.FIELDNAMES).writerows(peaks)

                if verbose:
                    print(f"  [{i+1}/{len(files)}] {fname} — {len(peaks)} 个峰")

            except Exception as e:
                if verbose:
                    print(f"  [{i+1}/{len(files)}] {fname} 出错: {e}")

        return all_peaks


# ============================================================
# Step 5 helpers — 精确衰变分析（纯计算，无状态）
# ============================================================

def _parse_trigger_filename(filename: str) -> Optional[Dict]:
    """
    从 trigger 级 .npz 文件名中解析 channel、data 序号、trigger 序号。

    支持两种格式::

        ..._PY82ch1_0030_trigger_5_...    → 单 data 文件
        ..._PY82ch1_0030-0031_...         → 跨 data 文件

    Returns
    -------
    dict or None
        {'channel', 'data_idx', 'trigger_idx', 'is_cross'[, 'data_idx_next']}
    """
    m = re.search(r'_(PY\d{2,3}ch\d)_(\d{4})_trigger_(\d+)_', filename)
    if m:
        return {
            "channel": m.group(1),
            "data_idx": int(m.group(2)),
            "trigger_idx": int(m.group(3)) - 1,   # 转为 0-based
            "is_cross": False,
        }
    m = re.search(r'_(PY\d{2,3}ch\d)_(\d{4})-(\d{4})_', filename)
    if m:
        return {
            "channel": m.group(1),
            "data_idx": int(m.group(2)),
            "data_idx_next": int(m.group(3)),
            "trigger_idx": None,
            "is_cross": True,
        }
    return None


def _extract_envelope(data: np.ndarray, center_freq: float, bw: float,
                      fs_val: float, order: int = 4) -> np.ndarray:
    """
    对复 IQ 数据做中心频率搬移 + 低通滤波，提取幅度包络。
    """
    t = np.arange(len(data)) / fs_val
    shifted = data * np.exp(-1j * 2 * np.pi * center_freq * t)
    b, a = butter(order, (bw / 2.0) / (0.5 * fs_val), btype="low")
    return np.abs(filtfilt(b, a, shifted.real) + 1j * filtfilt(b, a, shifted.imag))


def _is_decayed_in_window(env: np.ndarray, fs: float, start_idx: int,
                          threshold_ratio: float = 0.3) -> bool:
    """
    判断粒子在 start_idx 生成后，在后续观测窗口内是否发生了衰变消失。
    """
    if start_idx >= len(env) - int(0.05 * fs):
        return False

    end_idx_stable = min(len(env), start_idx + int(0.1 * fs))
    start_idx_stable = start_idx + int(0.01 * fs)
    if start_idx_stable >= end_idx_stable:
        return False

    stable_signal = env[start_idx_stable:end_idx_stable]
    if len(stable_signal) == 0:
        return False
    baseline_high = np.mean(stable_signal)

    tail_signal = np.mean(env[-int(0.05 * fs):])
    return bool(tail_signal < (baseline_high * threshold_ratio))


def _compute_precise_chain_decay(raw_data: np.ndarray, raw_times: np.ndarray,
                                 fs: float, parent_freq: float,
                                 daughter_freq: float,
                                 approx_t: float) -> tuple:
    """
    基于母核消失与子核生成的差分包络过零点，计算精确衰变时刻。
    """
    bandwidth = 1500.0
    parent_env = _extract_envelope(raw_data, parent_freq, bandwidth, fs)
    daughter_env = _extract_envelope(raw_data, daughter_freq, bandwidth, fs)
    diff_env = daughter_env - parent_env

    t_start = max(0.01, approx_t - 0.1)
    t_end = min(raw_times[-1] - 0.01, approx_t + 0.1)
    search_mask = (raw_times >= t_start) & (raw_times <= t_end)
    search_indices = np.where(search_mask)[0]

    if len(search_indices) == 0:
        return approx_t, 1.0 / (2.0 * bandwidth)

    diff_in_search = diff_env[search_indices]
    decision_window = int(0.006 * fs)

    found_idx = next(
        (i for i in range(len(diff_in_search) - decision_window)
         if diff_in_search[i] > 0
         and np.mean(diff_in_search[i:i + decision_window]) > 0),
        np.argmin(np.abs(diff_in_search)),
    )
    zero_cross_idx = search_indices[found_idx]
    decay_time_raw = raw_times[zero_cross_idx]

    # 噪声与斜率估计
    pre_mask = ((raw_times >= max(0, decay_time_raw - 0.15))
                & (raw_times < decay_time_raw - 0.05))
    post_mask = ((raw_times > decay_time_raw + 0.05)
                 & (raw_times <= min(decay_time_raw + 0.15, raw_times[-1])))
    if np.any(pre_mask) and np.any(post_mask):
        sigma_noise = np.sqrt(
            (np.std(diff_env[pre_mask]) ** 2 + np.std(diff_env[post_mask]) ** 2) / 2.0
        )
    else:
        sigma_noise = np.std(diff_in_search[:int(0.01 * fs)])

    fit_hw = int(0.001 * fs)
    fit_range = slice(
        max(0, zero_cross_idx - fit_hw),
        min(len(raw_times), zero_cross_idx + fit_hw),
    )
    if fit_range.stop - fit_range.start > 2:
        slope_K, _ = np.polyfit(raw_times[fit_range], diff_env[fit_range], 1)
        slope_K = abs(slope_K) if abs(slope_K) > 1e-6 else 1.0
    else:
        slope_K = 1.0

    sigma_stat = sigma_noise / slope_K
    sigma_filter = 1.0 / (2.0 * bandwidth)
    sigma_total = np.sqrt(sigma_stat ** 2 + sigma_filter ** 2)

    return decay_time_raw, sigma_total


def _compute_precise_single_decay(raw_data: np.ndarray, raw_times: np.ndarray,
                                  fs: float, peak_freq: float,
                                  exist_state: int,
                                  approx_time: float) -> tuple:
    """
    单信号边缘拟合 (erf 阶跃)。

    exist_state == 1  → 下降沿 (falling)
    exist_state >= 2  → 上升沿 (rising)
    """
    bandwidth = 1500.0
    env = _extract_envelope(raw_data, peak_freq, bandwidth, fs)

    ds = max(1, int(fs / 10000))
    t_sub = raw_times[::ds]
    env_sub = env[::ds]

    if exist_state == 1:
        def step_func(t, A, B, t0, sigma_t):
            return 0.5 * A * (1 - erf((t - t0) / (np.sqrt(2) * sigma_t))) + B
    else:
        def step_func(t, A, B, t0, sigma_t):
            return 0.5 * A * (1 + erf((t - t0) / (np.sqrt(2) * sigma_t))) + B

    p0 = [np.ptp(env_sub), np.min(env_sub), approx_time, 0.005]
    t0_min = max(0.0, approx_time - 0.2)
    t0_max = min(raw_times[-1], approx_time + 0.2)
    bounds = (
        [0, 0, t0_min, 0.0001],
        [np.inf, np.inf, t0_max, 0.05],
    )

    try:
        popt, pcov = curve_fit(
            step_func, t_sub, env_sub, p0=p0, bounds=bounds, maxfev=2000,
        )
        t_event = popt[2]
        err_fit = np.sqrt(np.diag(pcov))[2]
        sigma_filter = 1.0 / (2.0 * bandwidth)
        sigma_total = np.sqrt(err_fit ** 2 + sigma_filter ** 2)
        return t_event, sigma_total
    except Exception:
        return approx_time, 1.0 / (2.0 * bandwidth)


# ============================================================
# Step 5: 精确衰变分析
# ============================================================
class PreciseDecayAnalyzer:
    """
    Step 5: 利用原始时域 IQ 数据精确求解离子衰变时刻。

    在 Step 4 寻峰结果的基础上：
    - 用更复杂的频差敏感配对逻辑替换简单配对
    - 通过包络提取 + 过零点检测（链式衰变）或 erf 阶跃拟合（单信号）
      将 ``exist_time`` 精度从频域帧级别提升至亚毫秒级

    新增输出字段：``err_exist_time``, ``is_stable``

    Parameters
    ----------
    raw_data_dir : str
        原始 .data 文件所在目录。
    channel_prefix : str or None
        通道名（如 ``"PY82ch1"``）。None 则从文件名自动解析。
    max_cache : int
        .data 文件 LRU 缓存上限（默认 2）。
    """

    # 经过精确衰变分析后的完整字段列表
    FIELDNAMES = [
        "peak_pos", "err_pos", "sigma", "err_sigma",
        "height_ratio", "height_ion",
        "exist_state", "exist_time", "err_exist_time",
        "valid", "pair_num", "is_stable",
        "filename",
    ]

    def __init__(self, raw_data_dir: str, channel_prefix: Optional[str] = None,
                 max_cache: int = 2):
        self.raw_data_dir = raw_data_dir
        self.channel_prefix = channel_prefix
        self.max_cache = max_cache
        self._cache: OrderedDict = OrderedDict()

    # ------------------------------------------------------------------
    # .data 文件缓存
    # ------------------------------------------------------------------
    def _data_path(self, channel: str, data_idx: int) -> str:
        """由 channel + data 序号拼出 .data 文件路径。"""
        return os.path.join(self.raw_data_dir, f"{channel}_{data_idx}.data")

    def _load_data(self, path: str) -> Optional[Preprocessing]:
        """带 LRU 的 .data 文件加载；文件不存在返回 None。"""
        if not os.path.exists(path):
            return None

        # 命中 → 移到队尾 (most-recent)
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        # 淘汰最旧的
        if len(self._cache) >= self.max_cache:
            self._cache.popitem(last=False)

        bud = Preprocessing(path, puyuan_new=True, abs_trigger=False, verbose=False)
        self._cache[path] = bud
        return bud

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------
    def process(self, ctx: "PipelineContext") -> None:
        """
        从 ctx 获取峰列表和文件名，加载原始 IQ 数据，精化衰变时刻。

        结果直接更新 ``ctx.peaks`` 中每个峰的 ``exist_time`` /
        ``err_exist_time`` / ``is_stable`` / ``pair_num`` 字段。
        """
        peaks = ctx.peaks
        if not peaks:
            return

        info = _parse_trigger_filename(ctx.filename)
        if info is None:
            return

        channel = self.channel_prefix or info["channel"]

        # -- 加载当前 data 文件 --
        bud = self._load_data(self._data_path(channel, info["data_idx"]))
        if bud is None:
            return
        fs = bud.sampling_rate
        total_triggers = len(bud.trigger_timestamp)

        # -- 确定 trigger 索引及 IQ 片段 --
        if info["is_cross"]:
            trigger_index = total_triggers - 1
        else:
            trigger_index = info["trigger_idx"]

        start_sample = max(
            0, int(bud.trigger_timestamp[trigger_index] * bud.data_len))
        if not info["is_cross"] and trigger_index < total_triggers - 1:
            end_sample = int(
                bud.trigger_timestamp[trigger_index + 1] * bud.data_len)
        else:
            end_sample = bud.n_sample

        slice_len = end_sample - start_sample
        _, iq_data = bud.load(size=slice_len, offset=start_sample, draw=False)
        iq_times = np.arange(slice_len) / fs

        # -- 跨文件拼接 --
        if info["is_cross"]:
            bud_next = self._load_data(
                self._data_path(channel, info["data_idx_next"]))
            if bud_next is not None:
                end_next = (
                    int(bud_next.trigger_timestamp[0] * bud_next.data_len)
                    if len(bud_next.trigger_timestamp) > 0
                    else bud_next.n_sample
                )
                _, iq_next = bud_next.load(size=end_next, offset=0, draw=False)
                iq_times_next = (np.arange(end_next) / fs) + (iq_times[-1] + 1.0 / fs)
                iq_data = np.concatenate([iq_data, iq_next])
                iq_times = np.concatenate([iq_times, iq_times_next])

        total_time = ctx.acquire_times()[-1]
        self._pair_and_refine(peaks, iq_data, iq_times, fs, total_time)
        ctx.peaks = peaks

    # ------------------------------------------------------------------
    # 核心：配对 + 精确计时
    # ------------------------------------------------------------------
    def _pair_and_refine(self, peaks: list, iq_data: np.ndarray,
                         iq_times: np.ndarray, fs: float,
                         total_time: float) -> None:
        """
        用频差敏感的两阶段配对替换 Step 4 的简单配对，
        然后对每个峰计算精确衰变 / 产生时刻。
        """
        # -- 重置 Step 4 的配对结果 --
        for p in peaks:
            p["pair_num"] = 0
            p["valid"] = 1
            p["is_stable"] = False

        # ======== 第一阶段：频差敏感配对 ========
        FREQ_DELTA_THRESH = 5000.0
        NORMAL_TIME_WINDOW = 0.30
        STRICT_TIME_WINDOW = 0.05

        pair_counter = 0
        node_t1_pairs: list = []   # (idx_A, idx_child, approx_t1, pair_num)
        matched_children: set = set()

        # State 1 → State 2
        for i in [i for i, p in enumerate(peaks) if p["exist_state"] == 1]:
            pA = peaks[i]
            tA_end = pA.get("exist_time", total_time / 2.0)
            freqA = pA["peak_pos"]

            best_j, best_diff = None, float("inf")
            for j, pC in enumerate(peaks):
                if pC["exist_state"] != 2 or j in matched_children or i == j:
                    continue
                tC_start = total_time - pC.get("exist_time", 0)
                dt = abs(tA_end - tC_start)
                df = abs(freqA - pC["peak_pos"])
                window = STRICT_TIME_WINDOW if df > FREQ_DELTA_THRESH else NORMAL_TIME_WINDOW
                if dt < window and dt < best_diff:
                    best_diff, best_j = dt, j

            if best_j is not None:
                pair_counter += 1
                node_t1_pairs.append((i, best_j, tA_end, pair_counter))
                matched_children.add(best_j)

        # State 2 → State 3
        for i in [i for i, p in enumerate(peaks) if p["exist_state"] == 2]:
            pA = peaks[i]
            tA_end = pA.get("exist_time", total_time / 2.0)
            freqA = pA["peak_pos"]

            best_j, best_diff = None, float("inf")
            for j, pC in enumerate(peaks):
                if pC["exist_state"] != 3 or j in matched_children or i == j:
                    continue
                tC_start = total_time - pC.get("exist_time", 0)
                dt = abs(tA_end - tC_start)
                df = abs(freqA - pC["peak_pos"])
                window = STRICT_TIME_WINDOW if df > FREQ_DELTA_THRESH else NORMAL_TIME_WINDOW
                if dt < window and dt < best_diff:
                    best_diff, best_j = dt, j

            if best_j is not None:
                pair_counter += 1
                node_t1_pairs.append((i, best_j, tA_end, pair_counter))
                matched_children.add(best_j)

        # ======== 第二阶段：精确节点时刻求解 ========
        t1_exact: dict = {}
        for idx_A, idx_child, approx_t1, p_num in node_t1_pairs:
            peaks[idx_A]["pair_num"] = p_num
            peaks[idx_child]["pair_num"] = p_num

            t1, err_t1 = _compute_precise_chain_decay(
                iq_data, iq_times, fs,
                peaks[idx_A]["peak_pos"], peaks[idx_child]["peak_pos"], approx_t1,
            )
            t1_exact[idx_A] = (t1, err_t1)
            peaks[idx_child]["_exact_birth"] = (t1, err_t1)

        # ======== 第三阶段：各峰存活时间赋值 ========
        for idx, p in enumerate(peaks):
            st = p["exist_state"]

            if st == 0:
                # 全程存在，未衰变
                p["exist_time"] = total_time
                p["err_exist_time"] = 0.0
                p["is_stable"] = True

            elif st == 1:
                # 母核 → 计算衰变时刻
                if idx in t1_exact:
                    t_decay, err_decay = t1_exact[idx]
                else:
                    app_t = p.get("exist_time", total_time / 2.0)
                    t_decay, err_decay = _compute_precise_single_decay(
                        iq_data, iq_times, fs, p["peak_pos"], 1, app_t,
                    )
                p["exist_time"] = max(0.0, t_decay)
                p["err_exist_time"] = err_decay
                p["is_stable"] = False

            elif st in (2, 3):
                # 子核 / 孙核 → 先生成，再判断是否衰变
                if "_exact_birth" in p:
                    t_birth, err_birth = p["_exact_birth"]
                else:
                    app_birth = total_time - p.get("exist_time", total_time / 2.0)
                    app_birth = max(0.01, min(total_time - 0.01, app_birth))
                    t_birth, err_birth = _compute_precise_single_decay(
                        iq_data, iq_times, fs, p["peak_pos"], 3, app_birth,
                    )

                env = _extract_envelope(iq_data, p["peak_pos"], 1500.0, fs)
                start_idx = int(t_birth * fs)
                if _is_decayed_in_window(env, fs, start_idx):
                    app_death = max(t_birth + 0.01,
                                    min(total_time - 0.01, t_birth + 0.5))
                    t_death, err_death = _compute_precise_single_decay(
                        iq_data, iq_times, fs, p["peak_pos"], 1, app_death,
                    )
                    p["exist_time"] = max(0.0, t_death - t_birth)
                    p["err_exist_time"] = np.sqrt(err_birth ** 2 + err_death ** 2)
                    p["is_stable"] = False
                else:
                    p["exist_time"] = max(0.0, total_time - t_birth)
                    p["err_exist_time"] = err_birth
                    p["is_stable"] = True

        # 清理临时字段
        for p in peaks:
            p.pop("_exact_birth", None)


# ============================================================
# PipelineContext — 内存感知的流处理上下文
# ============================================================
class PipelineContext:
    """
    在步骤间传递数据的共享上下文，统一管理大数组的加载 / 缓存 / 释放。

    **内存策略**

    - ``psd_arrays``（2D，可达数 GB）：首次访问时加载，之后缓存复用。
      峰值出现在 Step1，后续步骤不再增长。
    - ``psd_mean`` / ``frequencies`` / ``times``：一次计算/加载后缓存常驻
    - ``baseline`` / ``recon`` / ``peaks``：常驻

    **典型内存曲线**（2 GB 原始文件，四步全跑）::

        Step1(CNN) 加载 psd_arrays -> ~2 GB（峰值）
        Step2(基线)                 -> ~2 GB（不变）
        Step3(重建)                 -> ~2 GB（不变）
        Step4(寻峰) 复用缓存        -> ~2 GB（不重复读盘）
        完成 release_all()          -> ~0
    """

    def __init__(self, file_path: str = ""):
        # --- 文件 ---
        self.file_path = file_path
        self.filename = os.path.basename(file_path) if file_path else ""

        # --- 小型计算结果（常驻内存）---
        self.cnn:      Optional[dict] = None
        self.baseline: Optional[np.ndarray] = None
        self.recon:    Optional[dict] = None    # {"frequencies", "psd_log"}
        self.peaks:    Optional[list] = None    # list[dict]

        # --- 大型数据内部状态 ---
        self._npz:               Any = None   # 延迟打开的 NpzFile
        self._psd_arrays_cached: Optional[np.ndarray] = None  # 缓存，避免重复 I/O
        self._psd_mean:          Optional[np.ndarray] = None
        self._frequencies:       Optional[np.ndarray] = None
        self._times:             Optional[np.ndarray] = None

        # --- 流程控制 ---
        self.skipped:     bool = False
        self.skip_reason: str = ""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release_heavy()
        return False

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------
    @classmethod
    def from_file(cls, file_path: str) -> "PipelineContext":
        """从 .npz 文件路径创建上下文（不加载数据，只记录路径）。"""
        return cls(file_path=file_path)

    # ------------------------------------------------------------------
    # 外部数据注入（用于跳过前序步骤时手动填入已有数据）
    # ------------------------------------------------------------------
    def load_baseline(self, path_or_array):
        """
        从外部加载基线到上下文。用于已有基线文件、只想跑步骤 3/4 的场景。

        Parameters
        ----------
        path_or_array : str or np.ndarray
            .npy 文件路径，或已加载的基线数组。
        """
        if isinstance(path_or_array, str):
            self.baseline = np.load(path_or_array)
        else:
            self.baseline = path_or_array

    def load_cnn_result(self, label: int, confidence: float):
        """
        手动注入 CNN 结果。用于已有 CNN 分类结果、只想跑后续步骤的场景。
        """
        self.cnn = {"filename": self.filename, "label": label, "confidence": confidence}

    # ------------------------------------------------------------------
    # 数据获取（延迟加载 + 缓存）
    # ------------------------------------------------------------------
    def _open(self):
        """打开 .npz 文件（惰性——只打开 zip 索引，不立即加载数组内容）。"""
        if self._npz is None:
            if not self.file_path:
                raise ValueError("PipelineContext 没有绑定文件路径，无法打开")
            self._npz = np.load(self.file_path, allow_pickle=True)
        return self._npz

    def acquire_psd_arrays(self) -> np.ndarray:
        """
        获取完整的 2D PSD 数组（首次加载后缓存，峰值内存出现在此处）。

        缓存不会增加峰值——后续步骤复用已加载的数组，省去重复 I/O。
        """
        if self._psd_arrays_cached is not None:
            return self._psd_arrays_cached
        self._psd_arrays_cached = self._open()["psd_arrays"]
        return self._psd_arrays_cached

    def acquire_psd_mean(self) -> np.ndarray:
        """
        获取按 sweep 平均后的 1D PSD（~1.7 MB）。

        首次调用时从 psd_arrays 计算并缓存；后续调用直接返回缓存。
        """
        if self._psd_mean is None:
            psd = self.acquire_psd_arrays()
            self._psd_mean = np.mean(psd, axis=0)
        return self._psd_mean

    def acquire_frequencies(self) -> np.ndarray:
        """获取频率轴（~1.7 MB，缓存）。"""
        if self._frequencies is None:
            self._frequencies = self._open()["frequencies"]
        return self._frequencies

    def acquire_times(self) -> np.ndarray:
        """获取时间轴（~16 KB，缓存）。"""
        if self._times is None:
            self._times = self._open()["times"]
        return self._times

    # ------------------------------------------------------------------
    # 内存释放
    # ------------------------------------------------------------------
    def release_heavy(self):
        """
        释放 2D 原始数组及 .npz 文件句柄。
        缓存的小数组（psd_mean 等）不受影响。
        """
        self._psd_arrays_cached = None
        if self._npz is not None:
            self._npz.close()
            self._npz = None

    def release_all(self):
        """释放所有数据（含缓存）。用于处理完成后的彻底清理。"""
        self.release_heavy()
        self._psd_mean = None
        self._frequencies = None
        self._times = None
        self.baseline = None
        self.recon = None
        self.peaks = None


# ============================================================
# Sink — 步骤输出抽象
# ============================================================
class Sink:
    """
    将步骤计算结果写入磁盘。

    工厂方法
    --------
    - ``Sink.csv(path, fields)`` — 增量写 CSV
    - ``Sink.npy(dir_path)``    — 写 .npy 文件（基线）
    - ``Sink.npz(dir_path)``    — 写 .npz 文件（重建谱）
    - ``Sink.none()``           — 显式不输出（等价于传 None）
    """

    def __init__(self, kind: str, target: str,
                 fields: Optional[List[str]] = None,
                 path_builder: Optional[Callable[[
                     "PipelineContext", str], str]] = None):
        self.kind = kind
        self.target = target
        self.fields = fields
        self._path_builder = path_builder or (lambda ctx, t: t)

    # -- 工厂方法 -------------------------------------------------------
    @classmethod
    def csv(cls, path: str, fields: List[str]) -> "Sink":
        """创建一个 CSV 输出 Sink。每个文件处理完后增量追加一行/多行。"""
        return cls("csv", path, fields=fields)

    @classmethod
    def npy(cls, dir_path: str) -> "Sink":
        """创建一个 .npy 输出 Sink。文件名自动生成为 baseline_<原名>.npy。"""
        def _build(ctx, d):
            stem = os.path.splitext(ctx.filename)[0]
            return os.path.join(d, f"baseline_{stem}.npy")
        return cls("npy", dir_path, path_builder=_build)

    @classmethod
    def npz(cls, dir_path: str) -> "Sink":
        """创建一个 .npz 输出 Sink。文件名自动生成为 reconstruct_<原名>.npz。"""
        def _build(ctx, d):
            return os.path.join(d, f"reconstruct_{ctx.filename}")
        return cls("npz", dir_path, path_builder=_build)

    @classmethod
    def none(cls) -> "Sink":
        """创建一个空操作 Sink（显式表明不输出）。"""
        return cls("none", "")

    # -- 写入 -----------------------------------------------------------
    def write(self, ctx: "PipelineContext", step_name: str) -> Optional[str]:
        """
        从 ctx 提取对应步骤的数据并写入磁盘。

        Returns
        -------
        str or None
            写入的完整路径；如果 kind 为 "none" 则返回 None。
        """
        if self.kind == "none":
            return None
        if self.kind == "csv":
            return self._write_csv(ctx, step_name)
        if self.kind == "npy":
            return self._write_npy(ctx)
        if self.kind == "npz":
            return self._write_npz(ctx)
        raise ValueError(f"未知 Sink 类型: {self.kind}")

    # -- 内部实现 -------------------------------------------------------
    def _write_csv(self, ctx: "PipelineContext", step_name: str) -> str:
        path = self.target
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        if step_name in ("peaks", "decay"):
            rows = ctx.peaks or []
        else:
            rows = [ctx.cnn] if ctx.cnn else []

        if not rows:
            return path

        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fields)
            if write_header:
                w.writeheader()
            for row in rows:
                formatted = dict(row)
                # 对齐老代码：置信度保留 4 位小数
                if "confidence" in formatted and isinstance(formatted.get("confidence"), float):
                    formatted["confidence"] = f"{formatted['confidence']:.4f}"
                w.writerow(formatted)
        return os.path.abspath(path)

    def _write_npy(self, ctx: "PipelineContext") -> str:
        path = self._path_builder(ctx, self.target)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, ctx.baseline)
        return path

    def _write_npz(self, ctx: "PipelineContext") -> str:
        path = self._path_builder(ctx, self.target)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, **ctx.recon)
        return path


def _raise_keyboard_interrupt(sig, frame):
    """SIGINT handler — 确保 Ctrl+C 不被 numpy C 扩展吞掉。"""
    raise KeyboardInterrupt()


# ============================================================
# 步骤胶水函数
# ============================================================
# 每个函数签名: (PipelineContext, processor) -> None
# 职责: 从 ctx 按需获取数据 -> 调用 processor -> 写回 ctx -> 释放大数据
# ============================================================

def _step_cnn(ctx: PipelineContext, detector: CNNIonDetector) -> None:
    """Step 1: CNN 分类。首次加载 psd_arrays（峰值内存），之后缓存复用。"""
    psd_mean = ctx.acquire_psd_mean()
    ctx.cnn = detector.process_mean(psd_mean, filename=ctx.filename)


def _step_baseline(ctx: PipelineContext, estimator: BaselineEstimator) -> None:
    """Step 2: 基线估计。可选按 data 复用基线，避免同一 data 重复估计。"""
    if estimator.cache_by_data:
        info = _parse_trigger_filename(ctx.filename)
        # 跨文件 trigger 的基线跨越两个 data，不做缓存，单独估计
        if info is not None and not info["is_cross"]:
            data_key = (info["channel"], info["data_idx"])
            ctx.baseline = estimator.process_mean_cached(
                ctx.acquire_psd_mean(), data_key)
        else:
            ctx.baseline = estimator.process_mean(ctx.acquire_psd_mean())
    else:
        ctx.baseline = estimator.process_mean(ctx.acquire_psd_mean())


def _step_reconstruct(ctx: PipelineContext, rec: SpectrumReconstructor) -> None:
    """Step 3: 谱重建。需要 mean + baseline + frequencies（均为轻量缓存）。"""
    ctx.recon = rec.reconstruct_from_mean(
        ctx.acquire_psd_mean(), ctx.baseline, ctx.acquire_frequencies()
    )


def _step_peaks(ctx: PipelineContext, extractor: PeakExtractor) -> None:
    """Step 4: 峰提取。复用 Step1 已缓存的 psd_arrays 做时域衰变分析。"""
    ctx.peaks = extractor.extract(
        ctx.recon["frequencies"],   # 使用重建后的频率轴（与 psd_log 长度对齐）
        ctx.recon["psd_log"],
        ctx.acquire_psd_arrays(),   # 命中缓存，不重新读盘
        ctx.acquire_times(),
        ctx.baseline,
    )
    for p in (ctx.peaks or []):
        p["filename"] = f"reconstruct_{ctx.filename}"


def _step_decay(ctx: PipelineContext, analyzer: PreciseDecayAnalyzer) -> None:
    """Step 5: 精确衰变分析。需要原始 .data 目录（在 analyzer 中配置）。"""
    analyzer.process(ctx)


# ============================================================
# StreamPipeline — 可组合流处理管线
# ============================================================
class StreamPipeline:
    """
    可自由组合的流处理管线。

    三种核心用法
    ------------

    **需求 1 — 四步全跑，只要峰 CSV**::

        pipe = StreamPipeline(min_confidence=0.9)
        pipe.add("cnn",      CNNIonDetector("model.pth"))
        pipe.add("baseline", BaselineEstimator())
        pipe.add("recon",    SpectrumReconstructor())
        pipe.add("peaks",    PeakExtractor(), sink=Sink.csv("peaks.csv", PeakExtractor.FIELDNAMES))
        pipe.run("/data/")

    **需求 2 — 四步全跑，选择性输出**::

        pipe = StreamPipeline()
        pipe.add("cnn",      CNNIonDetector("model.pth"), sink=Sink.csv("cnn.csv", [...])  )
        pipe.add("baseline", BaselineEstimator())                                          # 不存
        pipe.add("recon",    SpectrumReconstructor(),  sink=Sink.npz("/out/recon/"))
        pipe.add("peaks",    PeakExtractor(),          sink=Sink.csv("peaks.csv", [...]))

    **需求 3 — 只跑某一步**::

        pipe = StreamPipeline()
        pipe.add("baseline", BaselineEstimator(), sink=Sink.npy("/out/baselines/"))
        pipe.run("/data/")
    """

    # 步骤注册表: name -> (step_fn, 是否需要 2D 大数组)
    _REGISTRY = {
        "cnn":      (_step_cnn,      True),
        "baseline": (_step_baseline, False),
        "recon":    (_step_reconstruct, False),
        "peaks":    (_step_peaks,    True),
        "decay":    (_step_decay,    False),
    }

    def __init__(self, min_confidence: float = 0.9, timing: bool = True):
        self._steps: List[dict] = []
        self.min_confidence = min_confidence
        self.timing = timing

    # ------------------------------------------------------------------
    # 步骤注册
    # ------------------------------------------------------------------
    def add(self, name: str, processor,
            sink: Optional[Sink] = None) -> "StreamPipeline":
        """
        向管线添加一个步骤。

        Parameters
        ----------
        name : str
            步骤名: ``"cnn"``, ``"baseline"``, ``"recon"``, ``"peaks"``。
        processor :
            CNNIonDetector / BaselineEstimator / SpectrumReconstructor /
            PeakExtractor 的实例。
        sink : Sink or None
            输出配置。None（默认）表示只计算不存盘。
        """
        if name not in self._REGISTRY:
            raise ValueError(f"未知步骤 '{name}'，可选: {list(self._REGISTRY)}")
        self._steps.append({
            "name": name,
            "fn": self._REGISTRY[name][0],
            "needs_heavy": self._REGISTRY[name][1],
            "proc": processor,
            "sink": sink if sink is not None else Sink.none(),
        })
        return self  # 支持链式调用

    # ------------------------------------------------------------------
    # 单文件处理（核心）
    # ------------------------------------------------------------------
    def process_file(self, file_path: str) -> PipelineContext:
        """
        流式处理单个文件。

        执行流程：
        1. 创建 PipelineContext（延迟加载，不立即读文件）
        2. 依次执行注册的步骤
        3. 每个步骤按需从 ctx 获取数据、写回结果
        4. 步骤间智能释放大数据
        5. 全部完成后释放所有数据

        Returns
        -------
        PipelineContext
            包含所有中间结果，可通过 ``ctx.peaks`` / ``ctx.baseline`` 等访问。
        """
        with PipelineContext.from_file(file_path) as ctx:
            for step in self._steps:
                name = step["name"]

                # -- 执行步骤 --
                _t0 = time.time()
                try:
                    step["fn"](ctx, step["proc"])
                except Exception as exc:
                    ctx.skipped = True
                    ctx.skip_reason = f"{name}: {exc}"
                    break
                _t1 = time.time()

                # -- 写入输出 --
                try:
                    step["sink"].write(ctx, name)
                except Exception as exc:
                    ctx.skipped = True
                    ctx.skip_reason = f"{name}.sink: {exc}"
                    break

                if self.timing:
                    print(f"  [{name}] {_t1 - _t0:.1f}s")

                # -- CNN 门控：若 CNN 判定为噪声，停止后续步骤 --
                if name == "cnn" and ctx.cnn is not None:
                    if (ctx.cnn["label"] != 1
                            or ctx.cnn["confidence"] < self.min_confidence):
                        ctx.skipped = True
                        ctx.skip_reason = (
                            f"cnn: label={ctx.cnn['label']} "
                            f"conf={ctx.cnn['confidence']:.3f}"
                        )
                        break

        return ctx

    def process_context(self, ctx: PipelineContext) -> PipelineContext:
        """
        处理一个预先构建好的 PipelineContext。

        用于已有部分中间数据的场景。例如已有基线文件，只想跑步骤 3::

            ctx = PipelineContext.from_file("/data/raw/xxx.npz")
            ctx.load_baseline("/data/baselines/baseline_xxx.npy")

            pipe = StreamPipeline()
            pipe.add("recon", SpectrumReconstructor(), sink=Sink.npz("/out/recon/"))
            pipe.process_context(ctx)
        """
        with ctx:
            for step in self._steps:
                name = step["name"]

                _t0 = time.time()
                try:
                    step["fn"](ctx, step["proc"])
                except Exception as exc:
                    ctx.skipped = True
                    ctx.skip_reason = f"{name}: {exc}"
                    break
                _t1 = time.time()

                try:
                    step["sink"].write(ctx, name)
                except Exception as exc:
                    ctx.skipped = True
                    ctx.skip_reason = f"{name}.sink: {exc}"
                    break

                if self.timing:
                    print(f"  [{name}] {_t1 - _t0:.1f}s")

                # -- CNN 门控：若 CNN 判定为噪声，停止后续步骤 --
                if name == "cnn" and ctx.cnn is not None:
                    if (ctx.cnn["label"] != 1
                            or ctx.cnn["confidence"] < self.min_confidence):
                        ctx.skipped = True
                        ctx.skip_reason = (
                            f"cnn: label={ctx.cnn['label']} "
                            f"conf={ctx.cnn['confidence']:.3f}"
                        )
                        break

        return ctx

    # ------------------------------------------------------------------
    # 批量 & 流式监控
    # ------------------------------------------------------------------
    def run(self, input_dir: str, verbose: bool = True, resume: bool = False,
            data_range=None):
        """
        批量处理目录下所有 spectrogram .npz 文件（单次扫描）。
        自动忽略 *_spectrum.npz。

        Parameters
        ----------
        input_dir : str
            包含 .npz 文件的目录。
        verbose : bool
            打印每个文件的处理进度。
        resume : bool
            True 时跳过 manifest 中已记录的文件（断点续跑）。
        data_range : tuple (a, b) or None
            可选，只处理 data 序号在 [a, b] 内的文件。
        """
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"目录不存在: {input_dir}")

        files = self._list_files(input_dir, data_range=data_range, skip_incomplete=True)

        processed = self._processed_files() if resume else set()
        if processed:
            files = [f for f in files if f not in processed]
            if verbose:
                print(f"[StreamPipeline] 续跑模式：跳过 {len(processed)} 个已处理文件，剩余 {len(files)}")

        if verbose:
            steps = [s["name"] for s in self._steps]
            print(f"[StreamPipeline] 步骤: {' -> '.join(steps)}")
            print(f"[StreamPipeline] {input_dir} 中共 {len(files)} 个文件待处理")

        prev = signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
        try:
            for i, f in enumerate(files):
                ctx = self.process_file(os.path.join(input_dir, f))
                if verbose:
                    if ctx.skipped:
                        print(f"  [{i+1}/{len(files)}] {f} -> 跳过({ctx.skip_reason})")
                    else:
                        n_peaks = len(ctx.peaks) if ctx.peaks else 0
                        print(f"  [{i+1}/{len(files)}] {f} -> {n_peaks} peaks")
                if not ctx.skipped:
                    self._mark_done(f)
        except KeyboardInterrupt:
            print(f"\n[StreamPipeline] 用户中断，已完成 {i+1}/{len(files)}")
        finally:
            signal.signal(signal.SIGINT, prev)

    def process_stream(self, input_dir: str, poll_interval: float = 5.0):
        """
        监控目录，新文件到达即处理（Ctrl-C 停止）。
        忽略 *_spectrum.npz 和 *_incomplete_* 文件，后者需等待完整文件到达。
        重启时自动跳过 manifest 中已记录的文件。

        Parameters
        ----------
        input_dir : str
            监控的目录。
        poll_interval : float
            轮询间隔（秒）。
        """
        print(f"[StreamPipeline] 监控目录: {input_dir}")
        step_names = [s["name"] for s in self._steps]
        print(f"[StreamPipeline] 步骤: {' -> '.join(step_names)}")
        for s in self._steps:
            if s["sink"].kind != "none":
                print(f"[StreamPipeline]   {s['name']} -> {s['sink'].target}")

        prev = signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
        try:
            while True:
                current = set(self._list_files(input_dir, skip_incomplete=True))
                processed = self._processed_files()
                new_files = sorted(current - processed)

                if new_files:
                    print(
                        f"[StreamPipeline] 发现 {len(new_files)} 个新文件")
                    for f in new_files:
                        ctx = self.process_file(
                            os.path.join(input_dir, f))
                        if ctx.skipped:
                            print(f"  {f} -> 跳过({ctx.skip_reason})")
                        else:
                            n = len(ctx.peaks) if ctx.peaks else 0
                            print(f"  {f} -> {n} peaks")
                        # 无论成功失败都记录，防止损坏文件被死循环重试
                        self._mark_done(f)

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n[StreamPipeline] 已停止。")
        finally:
            signal.signal(signal.SIGINT, prev)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 文件过滤 & 断点续跑
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_data_index(filename: str) -> int:
        """
        提取文件名中 data 序号。
        _0000_trigger → 0, _0030-0031_ → 30。
        文件名格式:
          ..._<data序号>_trigger_...  (单 trigger)
          ..._<data序号>-<data序号>_trigger_...  (跨 trigger)
        """
        m = re.search(r'_(?:(\d{4})_trigger|(\d{4})-\d{4}_)', filename)
        return int(m.group(1) or m.group(2)) if m else -1

    def _list_files(self, input_dir: str, data_range=None,
                    skip_incomplete: bool = False):
        """列出待处理文件。自动忽略 *_spectrum.npz。"""
        files = sorted(
            f for f in os.listdir(input_dir)
            if f.endswith(".npz") and not f.endswith("_spectrum.npz")
        )
        if skip_incomplete:
            files = [f for f in files if "_incomplete_" not in f]
        if data_range is not None:
            lo, hi = data_range
            files = [f for f in files
                     if lo <= self._parse_data_index(f) <= hi]
        return files

    @property
    def _done_path(self) -> str:
        """断点续跑 manifest 文件路径，放在第一个输出目录下。"""
        for s in self._steps:
            if s["sink"].kind == "csv":
                return os.path.join(os.path.dirname(s["sink"].target) or ".",
                                    ".pipeline_done")
        for s in self._steps:
            if s["sink"].kind != "none":
                return os.path.join(s["sink"].target, ".pipeline_done")
        return ".pipeline_done"

    def _processed_files(self) -> set:
        """从 manifest 文件读取已完整处理的文件列表。"""
        path = self._done_path
        if not os.path.exists(path):
            return set()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return {line.strip() for line in fh if line.strip()}
        except Exception:
            return set()

    def _mark_done(self, filename: str):
        """将一个文件记录为已完整处理。"""
        with open(self._done_path, "a", encoding="utf-8") as fh:
            fh.write(filename + "\n")


# ============================================================
# 便捷工厂函数（向后兼容旧版 IonAnalysisPipeline）
# ============================================================
def create_full_pipeline(
    model_path: str,
    l: float = 1e9,
    ratio: float = 1e-7,
    k_high: float = 8.0,
    k_low: float = 1.8,
    snr_factor: float = 6.0,
    min_confidence: float = 0.9,
    cache_by_data: bool = False,
    save_peaks_csv: Optional[str] = None,
    save_baseline_dir: Optional[str] = None,
    save_recon_dir: Optional[str] = None,
    save_cnn_csv: Optional[str] = None,
) -> StreamPipeline:
    """
    快速创建四步全跑管线（替代旧版 IonAnalysisPipeline 的常见用法）。

    用法::

        pipe = create_full_pipeline(
            "model.pth",
            save_peaks_csv="peaks.csv",
            save_baseline_dir="/data/baselines/",
        )
        pipe.run("/data/raw/")
    """
    pipe = StreamPipeline(min_confidence=min_confidence)

    pipe.add("cnn", CNNIonDetector(model_path),
             sink=Sink.csv(save_cnn_csv, ["filename", "label", "confidence"])
             if save_cnn_csv else None)
    pipe.add("baseline", BaselineEstimator(l=l, ratio=ratio,
                                           cache_by_data=cache_by_data),
             sink=Sink.npy(save_baseline_dir) if save_baseline_dir else None)
    pipe.add("recon", SpectrumReconstructor(k_high=k_high, k_low=k_low),
             sink=Sink.npz(save_recon_dir) if save_recon_dir else None)
    pipe.add("peaks", PeakExtractor(snr_factor=snr_factor),
             sink=Sink.csv(save_peaks_csv, PeakExtractor.FIELDNAMES)
             if save_peaks_csv else None)

    return pipe
