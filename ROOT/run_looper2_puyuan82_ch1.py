import multiprocessing
from looper_cut2 import FileProcessor

# --- 配置参数 ---
SOURCE_DIR = '/mnt/data1/raw_data/puyuan82_test/Data/ch1/8243_TestModePY82_26-04-07_22-12-25'      # 原始文件存放路径
OUTPUT_DIR = '/mnt/data2/analyzed_data/puyuan82_test/Data/ch1/8243_TestModePY82_26-04-07_22-12-25'   # 处理后生成文件的路径
FILE_PREFIX = 'PY82ch1'
EXPECTED_SIZE = 1024 * 1024 * 1024      # 原始文件固定大小, Bytes
CHECK_INTERVAL = 0.5              # 检查频率, sec

WIN_LEN = 262144                  # 频谱窗口长度
N_AVER = 4                        # 单帧平均次数
OVERLAPR = 0.60881                # 数据重叠率
N_HOP = 250108                    # 单帧数据间隔
#TODO = ['data_spectrogram', 'data_spectrum', 'png_spectrogram', 'png_spectrum']
TODO = ['data_spectrogram']#, 'data_spectrum']#, 'png_spectrogram', 'png_spectrum']

# --- 在线分析配置（可选，None 则跳过 Spectrogram 分析） ---
SPECTROGRAM_ANALYSIS_CONFIG = {
        'background_path': '/home/imsexp/xh_tang/0713TestProgress/8243_PY82ch1_0008_trigger_17_2026-04-07T22-12-26.npz',
        'n_sigma': 1


        }#None   # 暂不启用 Spectrogram 分析

# --- 并行进程数（None = 自动 = CPU核心数-1） ---
WORKERS = None   # 自动

# --- 起始文件序号（None = 自动检测，≥0 = 手动指定） ---
START_INDEX = None   # 自动检测（从上次中断处继续）

# --- 结束文件序号（None = 不限，≥0 = 处理到此序号为止） ---
STOP_INDEX = None    # 不限制

# --- 分析单次注入进程数 ---
SPECTROGRAM_WORKERS = 8  # 不启用单次注入进程（因 SPECTROGRAM_ANALYSIS_CONFIG=None）

# --- 运行程序: looper_cut2 ---
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    multiprocessing.freeze_support()
    processor = FileProcessor(SOURCE_DIR, OUTPUT_DIR, FILE_PREFIX, EXPECTED_SIZE, CHECK_INTERVAL, SPECTROGRAM_ANALYSIS_CONFIG, WORKERS, START_INDEX, STOP_INDEX, SPECTROGRAM_WORKERS)
    processor.run(WIN_LEN, N_AVER, OVERLAPR, N_HOP, TODO)
