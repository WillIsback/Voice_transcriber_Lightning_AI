#scripts/data_preprocessing/HDF5_dataset.py
import h5py
from torch.utils.data import Dataset
import librosa
from . import audio_utils
import numpy as np
import shutil
import soundfile as sf
import os
from tqdm import tqdm

class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'a' if os.path.exists(file_path) else 'w')
        if 'audio_data' not in self.file:
            self.audio_data = self.file.create_group('audio_data')
        else:
            self.audio_data = self.file['audio_data']
        if 'transcriptions' not in self.file:
            self.transcriptions = self.file.create_group('transcriptions')
        else:
            self.transcriptions = self.file['transcriptions']

    def flush(self):
        self.file.flush()

    def __getitem__(self, file_name):
        audio = self.audio_data[file_name][:]
        transcription = self.transcriptions[file_name][:]
        return audio, transcription

    def __len__(self):
        return len(self.audio_data)

    def __del__(self):
        self.file.close()

    def needs_preprocessing(self, audio_file_path):
        audio, sample_rate = librosa.load(audio_file_path, sr=None)

        # Check if audio is mono
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            #print(f"Audio at {audio_file_path} is not mono. Converting to mono.")
            audio = audio_utils.convert_to_mono(audio)

        # Check if audio has been normalized
        if np.max(audio) > 1.0 or np.min(audio) < -1.0:
            #print(f"Audio at {audio_file_path} has not been normalized. Normalizing.")
            audio = audio_utils.normalize_features(audio)

        # Check if silence has been removed
        non_silent_intervals = librosa.effects.split(audio, top_db=20)
        if len(non_silent_intervals) < len(audio):
            #print(f"Audio at {audio_file_path} contains silence. Removing silence.")
            audio = audio_utils.remove_silence(audio)

        # Convert sample rate
        target_sample_rate = 22050  # Replace with your target sample rate
        if sample_rate != target_sample_rate:
            #print(f"Audio at {audio_file_path} has different sample rate. Converting to {target_sample_rate}Hz.")
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)

        return audio

    def Preprocessing(self, base_directory_path, dry_run=False):
        
        with open(os.path.join(base_directory_path, 'lists/all_wavs.list'), 'r') as f:
            audio_files = [os.path.join(base_directory_path, 'wavs', line.strip()) for line in f]
        with open(os.path.join(base_directory_path, 'lists/all_text.list'), 'r') as f:
            transcription_files = [os.path.join(base_directory_path, 'text', line.strip()) for line in f]

        # If dry_run is True, only process the first 10 files
        if dry_run:
            audio_files = audio_files[:10]
            transcription_files = transcription_files[:10]

        for audio_file, transcription_file in tqdm(zip(audio_files, transcription_files), total=len(audio_files), ncols=100, desc=f'Preprocessing'):
            # Load the audio data
            audio = self.needs_preprocessing(audio_file)

            # Load the transcription
            with open(transcription_file, 'r') as f:
                transcription = f.read()

            # Get the file name without the extension
            file_name = os.path.splitext(os.path.basename(audio_file))[0]

            if dry_run:
                # Write the preprocessed audio data to a file
                dry_run_directory_path = os.path.join(base_directory_path, 'dry_run')
                os.makedirs(dry_run_directory_path, exist_ok=True)
                sf.write(os.path.join(dry_run_directory_path, file_name + '.wav'), audio, 22050)  # Replace 22050 with your sample rate

                # Write the transcription to a file
                with open(os.path.join(dry_run_directory_path, file_name + '.txt'), 'w') as f:
                    f.write(transcription)
                print(f"Writing audio data to HDF5 file: {audio}")

                self.audio_data.create_dataset(file_name, data=audio)

                print(f"Writing transcription to HDF5 file: {transcription}")
                self.transcriptions.create_dataset(file_name, data=transcription)
            else:
                # Add the preprocessed audio data and transcription to the HDF5 file
                #print(f"Writing audio data to HDF5 file: {audio}")
                self.audio_data.create_dataset(file_name, data=audio)

                #print(f"Writing transcription to HDF5 file: {transcription}")
                self.transcriptions.create_dataset(file_name, data=transcription)


    def HDF5_Reader(self, base_directory_path):
        print("Reading the HDF5 file...")
        # Get all entries
        audio_data_keys = list(self.audio_data.keys())
        transcriptions_keys = list(self.transcriptions.keys())

        # Base directory for dry run results
        dry_run_directory_path = os.path.join(base_directory_path, 'dry_run')

        # Directory for HDF5 results
        hdf5_directory_path = os.path.join(base_directory_path, 'hdf5_results')
        if not os.path.exists(hdf5_directory_path):
            os.makedirs(hdf5_directory_path)

        # Check each entry
        for audio_data_key, transcriptions_key in zip(audio_data_keys, transcriptions_keys):
            # Load the audio data and transcription from the HDF5 file
            hdf5_audio_data = self.audio_data[audio_data_key][()]
            hdf5_transcription = self.transcriptions[transcriptions_key][()]

            # Write the audio data and transcription to .wav and .txt files
            sf.write(os.path.join(hdf5_directory_path, audio_data_key + '.wav'), hdf5_audio_data, 22050)
            with open(os.path.join(hdf5_directory_path, audio_data_key + '.txt'), 'w') as f:
                f.write(hdf5_transcription.decode('utf-8'))

            # Load the dry run results
            dry_run_audio_data, _ = sf.read(os.path.join(dry_run_directory_path, audio_data_key + '.wav'))
            with open(os.path.join(dry_run_directory_path, audio_data_key + '.txt'), 'r') as f:
                dry_run_transcription = f.read()

