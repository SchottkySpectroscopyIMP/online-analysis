# online-analysis
Archive: online SIMS/SMS experiment processing pipeline

## Requisites

* Language environment: `Python 3`
* Multi-process, multi-thread: `multiprocessing`, `concurrent.futures`
* Sci-calculation: `scipy`, `numpy`, `pyfftw`, `sklearn`
* Plots: `matplotlib`
* Tables/DataFrame: `pandas`
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

2. Run `human_recognition.py`, visually inspect each injection to determine whether any ion signals are present. (option)

* manual steps, in preparation for the subsequent use of CNN to automatically identifiy ion peaks.
* hit `0` for no signal, `1` otherwise.
* hit `q` for quit.

```Python
# --- configures ---
DATA_DIR = '/'  # path for injection .npz 
OUTPUT_CSV = 'labels.csv'    # output file's path with filename

# --- run: human_recognition.py ---
from human_recognition import DataLabeler
if __name__ == "__main__":
    labeler = DataLabeler(DATA_DIR, OUTPUT_CSV)
```

3. Extracting ion information from data

> Method 1 (via `CERN-ROOT`): employ $\mu + 3 \sigma$ to separate background and ion signal, ...

* Requirements for input ...
* Output ...

```C++
// how to do
```

> Method 2 (via `Python 3`): employ CNN to distinguish ion signal, use CWT to reconstruct spectrum with pure ion signal from estimated baseline

* Needed results from `human_recognition.py` for CNN. But not necessary to do this every time. Only if the current data differs significantly from the previous training set (for example, if the resonance peaks are significantly offset) is it necessary to repeat the process.
* If there is at least one ion in the .data file, a corresponding .csv result will be generated.

```Python
# how to do
```

4. Tool for showing result

* `.csv` from step 3 is needed for `Data file`. `.csv` from PID result is needed for `Ref. file`.
* GMM can be used to distinguish different ion species from ion clusters.

```Shell
> python ion_monitor.py
```
