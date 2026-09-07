#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
run_pipeline.py 交互式生成器（纯终端，无需 GUI）。

用法::

    python generate_run_pipeline.py              # 交互式问答
    python generate_run_pipeline.py -o run.py    # 指定输出文件名

生成的文件可直接 ``python run.py`` 执行。
"""

import os
import sys
import argparse
from textwrap import dedent


def ask(prompt: str, default: str = "", required: bool = True) -> str:
    """带默认值的交互式提问。"""
    if default:
        hint = f" [{default}]"
    else:
        hint = " (必填)" if required else ""
    while True:
        raw = input(f"{prompt}{hint}: ").strip()
        if raw:
            return raw
        if default:
            return default
        if not required:
            return ""
        print("  ⚠ 此项为必填，请重新输入。")


def ask_yn(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def ask_choice(prompt: str, options: list, default: int = 0) -> str:
    print(f"\n{prompt}")
    for i, (label, _) in enumerate(options):
        mark = " ←" if i == default else ""
        print(f"  [{i}] {label}{mark}")
    while True:
        raw = input(f"请选择 [0-{len(options)-1}] (默认 {default}): ").strip()
        if not raw:
            return options[default][1]
        try:
            idx = int(raw)
            if 0 <= idx < len(options):
                return options[idx][1]
        except ValueError:
            pass
        print(f"  ⚠ 请输入 0-{len(options)-1} 之间的数字。")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="生成 run_pipeline.py 配置文件")
    parser.add_argument("-o", "--output", default=None,
                        help="输出文件名 (默认: run_pipeline.py，未指定则交互询问)")
    args = parser.parse_args()

    print("=" * 60)
    print("  run_pipeline.py 配置生成器")
    print("  按提示回答，回车使用默认值")
    print("=" * 60)

    # ---- 输出文件名（命令行未指定时交互询问） ----
    if args.output is None:
        output_name = ask("输出运行代码文件名", default="run_pipeline.py")
    else:
        output_name = args.output

    # ---- 路径配置 ----
    print("\n── 路径配置 ──")
    model_path = ask("CNN 模型权重路径 (.pth)", default="model_cpu_v1.pth")
    raw_dir = ask("原始 .npz 文件目录", default="/data/cutInjection/")
    out_dir = ask("输出根目录", default="/data/output/")
    raw_data_dir = ask("原始 .data 文件目录（Step 5 精确衰变需要）",
                       default="/data/raw_data/", required=False)

    # ---- 步骤选择 ----
    print("\n── 步骤选择 ──")
    steps = []
    steps.append(("CNN 离子信号检测", "cnn"))
    steps.append(("基线估计 (BrPLS)", "baseline"))
    steps.append(("谱重建 (CWT)", "recon"))
    steps.append(("寻峰导出", "peaks"))

    enabled = set()
    selected_steps = []
    print()
    for label, key in steps:
        if ask_yn(f"  启用 Step: {label}?", default=True):
            enabled.add(key)
            selected_steps.append(key)

    has_decay = False
    if raw_data_dir:
        if ask_yn(f"  启用 Step: 精确衰变分析?", default=True):
            enabled.add("decay")
            selected_steps.append("decay")
            has_decay = True
    else:
        if ask_yn(f"  启用 Step: 精确衰变分析? (需要配置 raw_data_dir)", default=False):
            raw_data_dir = ask("原始 .data 文件目录", required=True)
            enabled.add("decay")
            selected_steps.append("decay")
            has_decay = True

    if not enabled:
        print("  ⚠ 未选择任何步骤，退出。")
        sys.exit(0)

    # ---- 算法参数 ----
    print("\n── 算法参数 ──")
    min_conf = ask("CNN 置信度阈值", default="0.9") if "cnn" in enabled else "0.9"
    k_high = ask("重建 k_high (信号判定严格阈值)", default="8.0") if "recon" in enabled else "8.0"
    k_low = ask("重建 k_low (信号边界宽松阈值)", default="1.8") if "recon" in enabled else "1.8"
    snr = ask("寻峰 snr_factor", default="6.0") if "peaks" in enabled else "6.0"
    l_param = ask("基线平滑参数 l", default="1e9") if "baseline" in enabled else "1e9"
    ratio = ask("基线终止条件 ratio", default="1e-7") if "baseline" in enabled else "1e-7"

    # 基线缓存（流处理加速）
    cache_by_data = False
    if "baseline" in enabled:
        cache_by_data = ask_yn(
            "  按 data 复用基线（同一 data 的多个 trigger 只估计一次）?",
            default=False)

    # ---- 输出选择 ----
    print("\n── 输出选择 ──")
    outputs = {}
    if "cnn" in enabled:
        if ask_yn("  CNN 结果输出为 CSV?", default=True):
            outputs["cnn"] = f'{out_dir}/step1_cnn.csv'
    if "baseline" in enabled:
        if ask_yn("  基线输出为 .npy 文件?", default=True):
            outputs["baseline"] = f'{out_dir}/baselines/'
    if "recon" in enabled:
        if ask_yn("  重建谱输出为 .npz 文件?", default=True):
            outputs["recon"] = f'{out_dir}/reconstructed/'
    if "peaks" in enabled:
        if ask_yn("  峰信息输出为 CSV?", default=True):
            outputs["peaks"] = f'{out_dir}/step4_peaks.csv'
    if has_decay:
        if ask_yn("  精确衰变结果输出为 CSV?", default=True):
            outputs["decay"] = f'{out_dir}/step5_decay.csv'

    # ---- 运行模式 ----
    print("\n── 运行模式 ──")
    mode = ask_choice("选择运行模式:", [
        ("批量处理（单次扫描）", "run"),
        ("实时监控（持续轮询新文件）", "stream"),
    ], default=0)

    run_opts = ""
    if mode == "run":
        if ask_yn("  启用断点续跑?", default=False):
            run_opts += "    pipe.run(RAW_DIR, resume=True)\n"
        else:
            if ask_yn("  限制 data 序号范围?", default=False):
                lo = ask("  起始 data 序号", default="0")
                hi = ask("  结束 data 序号", default="999")
                run_opts += f"    pipe.run(RAW_DIR, data_range=({lo}, {hi}))\n"
            else:
                run_opts += "    pipe.run(RAW_DIR)\n"
    else:
        interval = ask("  轮询间隔 (秒)", default="5")
        run_opts += f"    pipe.process_stream(RAW_DIR, poll_interval={interval})\n"

    timing = "True" if ask_yn("  打印每步耗时?", default=True) else "False"

    # ============================================================
    # 生成脚本
    # ============================================================
    imports = [
        "StreamPipeline",
        "CNNIonDetector",
        "BaselineEstimator",
        "SpectrumReconstructor",
        "PeakExtractor",
    ]
    if has_decay:
        imports.append("PreciseDecayAnalyzer")
    imports.append("Sink")

    imports_str = ",\n    ".join(imports)

    script = f'''#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
流处理管线 — 由 generate_run_pipeline.py 自动生成。
步骤: {" → ".join(selected_steps)}
"""
from extracting_ion_information import (
    {imports_str},
)

# ============================================================
# 路径
# ============================================================
MODEL_PATH = {repr(model_path)}
RAW_DIR    = {repr(raw_dir)}
OUT_DIR    = {repr(out_dir)}
'''

    if has_decay:
        script += f'RAW_DATA_DIR = {repr(raw_data_dir)}\n'

    script += f'''
# ============================================================
# 参数
# ============================================================
MIN_CONF = {min_conf}
K_HIGH   = {k_high}
K_LOW    = {k_low}
SNR      = {snr}
L_PARAM  = {l_param}
RATIO    = {ratio}

# ============================================================
# 管线
# ============================================================
pipe = StreamPipeline(min_confidence=MIN_CONF, timing={timing})
'''

    if "cnn" in enabled:
        if "cnn" in outputs:
            script += f'''
pipe.add("cnn", CNNIonDetector(MODEL_PATH),
         sink=Sink.csv({repr(outputs["cnn"])},
                       ["filename", "label", "confidence"]))
'''
        else:
            script += '''
pipe.add("cnn", CNNIonDetector(MODEL_PATH))
'''

    if "baseline" in enabled:
        cache_arg = ", cache_by_data=True" if cache_by_data else ""
        if "baseline" in outputs:
            script += f'''
pipe.add("baseline", BaselineEstimator(l={l_param}, ratio={ratio}{cache_arg}),
         sink=Sink.npy({repr(outputs["baseline"])}))
'''
        else:
            script += f'''
pipe.add("baseline", BaselineEstimator(l={l_param}, ratio={ratio}{cache_arg}))
'''

    if "recon" in enabled:
        if "recon" in outputs:
            script += f'''
pipe.add("recon", SpectrumReconstructor(k_high=K_HIGH, k_low=K_LOW),
         sink=Sink.npz({repr(outputs["recon"])}))
'''
        else:
            script += f'''
pipe.add("recon", SpectrumReconstructor(k_high=K_HIGH, k_low=K_LOW))
'''

    if "peaks" in enabled:
        if "peaks" in outputs:
            script += f'''
pipe.add("peaks", PeakExtractor(snr_factor=SNR),
         sink=Sink.csv({repr(outputs["peaks"])},
                       PeakExtractor.FIELDNAMES))
'''
        else:
            script += '''
pipe.add("peaks", PeakExtractor(snr_factor=SNR))
'''

    if has_decay:
        if "decay" in outputs:
            script += f'''
pipe.add("decay", PreciseDecayAnalyzer(RAW_DATA_DIR),
         sink=Sink.csv({repr(outputs["decay"])},
                       PreciseDecayAnalyzer.FIELDNAMES))
'''
        else:
            script += '''
pipe.add("decay", PreciseDecayAnalyzer(RAW_DATA_DIR))
'''

    script += f'''
# ============================================================
# 执行
# ============================================================
if __name__ == "__main__":
{run_opts}
    # 其它常用选项:
    # pipe.run(RAW_DIR, resume=True)              # 断点续跑
    # pipe.run(RAW_DIR, data_range=(0, 99))       # 限定 data 序号范围
    # pipe.process_stream(RAW_DIR, poll_interval=5) # 实时监控
'''

    # 写文件
    out_path = os.path.abspath(output_name)
    if os.path.exists(out_path):
        if not ask_yn(f"\n文件 {out_path} 已存在，是否覆盖?", default=False):
            print("  已取消。")
            sys.exit(0)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(script)

    print(f"\n{'=' * 60}")
    print(f"  已生成: {out_path}")
    print(f"  运行方式: python {output_name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
