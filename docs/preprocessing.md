# Data Preprocessing

The data preprocessing for this project involves several steps, including downloading the data, converting the audio files to a suitable format, and storing the data in an HDF5 file for efficient access during training.

## Data Download

The data is downloaded using a custom Python script and a Bash script. The Python script downloads the data files from a specified server and saves them to a local directory. The Bash script then unzips the downloaded file and removes the zip file if the unzipping was successful.

### Python Script

The Python script uses the `http.client` and `tqdm` libraries to download the data file. The `http.client` library is used to establish a connection to the server and send a GET request. The `tqdm` library is used to display a progress bar during the download.

Here is the Python script:

```python
# data_downloader
from pathlib import Path
import os
import http.client
from tqdm import tqdm

# ... (Python code here)

# Usage
url = 'https://datashare.ed.ac.uk/bitstream/handle/10283/2353/SiwisFrenchSpeechSynthesisDatabase.zip?sequence=3&isAllowed=y'
filename = Path('dataset/SiwisFrenchSpeechSynthesisDatabase.zip')
try:
    download_file(url, filename)
except Exception as e:
    print(e)
```

### Bash Script

The Bash script runs the Python script, unzips the downloaded file, and removes the zip file if the unzipping was successful. Here is the Bash script:

```bash
#!/bin/bash

# scripts/data_downloader/data_downloader.sh

# Run the Python script
python3 data_downloader/data_downloader.py

# Check if the Python script completed successfully
if [ $? -eq 0 ]; then
    echo "Python script completed successfully, unzipping file..."
    unzip -d dataset dataset/SiwisFrenchSpeechSynthesisDatabase.zip
    if [ $? -eq 0 ]; then
        echo "Unzipping completed successfully, removing the zip file..."
        rm dataset/SiwisFrenchSpeechSynthesisDatabase.zip
    else
        echo "Unzipping failed, not removing the zip file."
    fi
else
    echo "Python script failed, not unzipping file."
fi
```


Please note that the Python and Bash scripts are simplified for the purpose of this document. The actual scripts may contain additional error checking and logging.


## Audio Conversion

The audio files are converted to a suitable format using the `librosa` and `soundfile` libraries. The `librosa` library is used to load the audio files, convert them to a consistent sample rate, and perform various audio processing tasks. The `soundfile` library is used to save the converted audio data to .wav files.

Here is the Python script that performs the audio conversion:

```python
# scripts/data_preprocessing/audio_utils.py
import librosa
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler

# ... (Python code here)
```

### Functions

The script contains several functions that perform different audio processing tasks:

- `convert_to_mono`: This function converts a stereo audio file to mono by averaging the two channels.

- `remove_silence`: This function removes silence from the audio data. It uses the `librosa.effects.split` function to identify non-silent intervals, and then concatenates these intervals while keeping short silences that are shorter than a specified threshold.

- `sample_rate_conversion`: This function converts the sample rate of an audio file to a target sample rate.

- `extract_frames`: This function splits the audio data into frames of a specified size and stride.

- `extract_features`: This function extracts various features from the audio data, including the spectrogram, Mel-frequency cepstral coefficients (MFCC), chroma features, and spectral contrast.

- `normalize_features`: This function normalizes the extracted features using the `StandardScaler` from the `sklearn.preprocessing` module.

- `augment_audio`: This function augments the audio data by applying time stretching, pitch shifting, and adding noise.

These functions are used in the data preprocessing pipeline to convert the raw audio data into a format that can be used for machine learning.



## Data Storage

The audio data and corresponding transcriptions are stored in an HDF5 file using the `h5py` library. Each entry in the HDF5 file corresponds to a specific audio file, and the name of that file is used as the key for the entry. This allows for efficient access to the data during training.

## Data Verification

The integrity of the data in the HDF5 file is verified by comparing the .wav and .txt files generated from the HDF5 file to the original .wav and .txt files. This ensures that the data has been correctly converted and stored.

## Requirements

The following Python packages are required for the data preprocessing:

- numpy
- torch
- librosa
- soundfile
- pydub
- h5py
- pandas
- pytorch-lightning
- sklearn
- tqdm