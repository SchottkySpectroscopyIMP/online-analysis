# online-analysis
Archive: online SIMS/SMS experiment processing pipeline

## Requisites

* Language environment: `Python 3`
* Multi-process, multi-thread: `multiprocessing`, `concurrent.futures`
* Sci-calculation: `scipy`, `numpy`, `pyfftw`, `sklearn`
* Plots: `matplotlib`
* Tables/DataFrame: `pandas`
* GUI: `pyQt5`, `pyqtgraph`
* Data analysis: `ROOT` (CERN PyROOT),  `torch`, `nptdms`

## Usage

1. Start the DAQ device (RIGOL 4-channel RF DAQ system). Run `data_looper.py` to cut the .data files based on injection and save them into PSD data files or/and pictures.

* make sure `preprocessing.py` and `data_looper.py` are in the same folder.
* `Ctrl + C` is the hotkey for quitting the program.

> **Note for CERN ROOT users:** 
> You only need to start the DAQ device. Please **skip** the `data_looper.py` execution below and proceed directly to Step 3 (this data cutting process is included in Step 3, Method 1).

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

> **Method 1 (via `CERN-ROOT`)**: employ $\mu + n \sigma$ to separate background and ion signal. This method essentially integrates ROOT-based analysis into the data cutting workflow from Step 1.

* Make sure all script files located in the `ROOT/` directory are placed in the same folder.
* **Background Spectrum Selection:** Since a background spectrum is required for the $\mu + n\sigma$ separation, the standard practice is:
  1. Run the script *without* setting the `background_path` (in this mode, it behaves exactly like Step 1).
  2. Pick one of the generated spectra that has **no injection signal** to serve as your background.
  3. Set this selected file as the background parameter and run the script again to extract the ion information.

**Input & Output:**
* **Input:** Raw IQ data files, background spectrum file (obtained as described above).
* **Output:** PSD data files and/or pictures (identical to Step 1), and extracted ion information results.

```Python
import multiprocessing
from looper_cut2 import FileProcessor

# --- configures ---
SOURCE_DIR = '/'                    # path for raw IQ data files
OUTPUT_DIR = '/'                    # path for produced files
FILE_PREFIX = 'PY82ch1'             # based on the real IQ data prefix, example sourced from 'PY82ch1_0.data'
EXPECTED_SIZE = 1024 * 1024 * 1024  # size of the raw IQ data file, Bytes
CHECK_INTERVAL = 0.5                # time interval for checking the SOURCE_DIR, seconds

WIN_LEN = 262144           # window length of the PSD
N_AVER = 4                 # average number of one frame
OVERLAPR = 0.60881         # overlap of raw data for average
N_HOP = 250108             # data interval between individual frames
TODO = ['data_spectrogram', 'data_spectrum', 'png_spectrogram', 'png_spectrum']     # processing options


# --- Online analysis configuration (optional; skip spectrogram analysis if set to None) ---
SPECTROGRAM_ANALYSIS_CONFIG = {
        'background_path': '',  # path of background spectrogram. LEAVE EMPTY to generate one!
        'n_sigma': 1            # sigma factor for peak detection (mu + n*sigma)
        }

# --- Number of parallel processes (None = Automatic = CPU cores - 1) ---
WORKERS = None   

# --- Starting file index (None = auto-detect, >=0 = manually specify) ---
START_INDEX = None   

# --- End file sequence number (None = no limit, >=0 = process up to this sequence number) ---
STOP_INDEX = None    

# --- Number of processes for analyzing a single injection ---
SPECTROGRAM_WORKERS = 8  

# --- run: looper_cut2.py ---
if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    multiprocessing.freeze_support()
    
    processor = FileProcessor(SOURCE_DIR, OUTPUT_DIR, FILE_PREFIX, EXPECTED_SIZE, CHECK_INTERVAL, 
                              SPECTROGRAM_ANALYSIS_CONFIG, WORKERS, START_INDEX, STOP_INDEX, SPECTROGRAM_WORKERS)
    processor.run(WIN_LEN, N_AVER, OVERLAPR, N_HOP, TODO)
```

> **Method 2 (via `Python 3`)**: employ CNN to distinguish ion signals, and use CWT to reconstruct spectra with pure ion signals from the estimated baseline.

* **Prerequisites:** Needed results from `human_recognition.py` for CNN. But not necessary to do this every time. Only if the current data differs significantly from the previous training set (for example, if the resonance peaks are significantly offset) is it necessary to repeat the process.
* **Script Preparation:** All required scripts are located in the `CNN/` directory. Ensure the following core scripts are kept in the same folder:
  * `extracting_ion_information.py`
  * `generate_run_pipeline.py`
  * `nonparams_est.py`
  * `reconstruct_spectrum.py`
  * `preprocessing.py` *(Note: Note: A copy of this script is already included in the `CNN/` folder for your convenience).*
* **Execution Workflow:** 
  1. Run `generate_run_pipeline.py`.
  2. Follow the on-screen prompts to fill in your configurations.
  3. A customized run script will be automatically generated in the same directory.
  4. Execute the newly generated script to start the analysis.

```bash
# 1. Navigate to the CNN directory
cd CNN/

# 2. Run the pipeline generator and follow the interactive prompts
python generate_run_pipeline.py

# 3. After the generator finishes, run the newly created execution file
# (Replace 'generated_run_file.py' with the actual filename produced)
python generated_run_file.py
```

4. Tool for showing result

* `.csv` from step 3 is needed for `Data file`. `.csv` from PID result is needed for `Ref. file`.
* GMM can be used to distinguish different ion species from ion clusters.

```Shell
> python ion_monitor.py
```
