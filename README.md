# online-analysis
Archive: online SIMS/SMS experiement processing pipeline

## Requisites

* Language environment: `Python 3`
* Multi-process, multi-thread: `multiprocessing`, `concurrent.futures`
* Sci-calculation: `scipy`, `numpy`, `pyfftw`
* Plots: `matplotlib`
* GUI: `pyQt5`, `pyqtgraph`

## Usage

1. Start the DAQ device (RIGOL 4-channel RF DAQ system). Run `data_looper.py` to cut the .data files based on injection and save them into PSD data files or/and pictures.

* make sure `preprocessing.py` and `data_looper.py` are in the same folder.
* `Ctrl + C` is the hotkey for quitting the program.

```Python
# --- configures ---
SOURCE_DIR = '/'  				# path for raw IQ data files
OUTPUT_DIR = '/'  				# path for produced files
FILE_PREFIX = 'PY82ch1'  		# based on the real IQ data prefix, example sourced from 'PY82ch1_0.data'
EXPECTED_SIZE = 1024*1024*1024  # size of the raw IQ data file, Bytes
CHECK_INTERVAL = 0.5  			# time interval for checking the SOURCE_DIR, seconds

WIN_LEN = 262144		# window length of the PSD
N_AVER = 4				# average number of one frame
OVERLAPR = 0.60881		# overlap of raw data for average
N_HOP = 250108			# data interval between individual frames

TODO = ['data_spectrogram', 'data_spectrum', 'png_spectrogram', 'png_spectrum']  # processing options

# --- run: data_looper.py ---
import multiprocessing
from data_looper import FileProcessor
if __name__ == '__main__':
	try:
		multiprocessing.set_start_method('spawn', force=True)
	except RuntimeError:
		pass
	multiprocessing.freeze_support()
	processor = FileProcessor(SOURCE_DIR, OUTPUT_DIR, FILE_PREFIX, EXPECTED_SIZE, CHECK_INTERVAL)
    processor.run(WIN_LEN, N_AVER, OVERLAPR, N_HOP, TODO)
``` 


