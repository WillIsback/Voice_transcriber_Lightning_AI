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
from unidecode import unidecode
import string


class HDF5Dataset(Dataset):
    def __init__(self, file_path, mode='r'):
        self.file = h5py.File(file_path, mode)
        self.base_directory_path = 'dataset/SiwisFrenchSpeechSynthesisDatabase'
        if mode == 'w':
            self.file_names = []
            self.audio_data = self.file.create_group('audio_data')
            self.transcriptions = self.file.create_group('transcriptions')
        else:
            self.file_names = list(self.file['audio_data'].keys())
            self.audio_data = self.file['audio_data']
            self.transcriptions = self.file['transcriptions']
        with open(os.path.join(self.base_directory_path, 'lists/all_text.list'), 'r') as f:
            self.transcription_files = [os.path.join(self.base_directory_path, 'text', line.strip()) for line in f]
        self.char_to_int, self.int_to_char = self._create_mapping()

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        audio = self.audio_data[file_name][:]
        audio_len = len(audio)
        transcription = self.transcriptions[file_name][:]
        transcription_len = len(transcription)
        return audio, audio_len, transcription, transcription_len

    def transcription_to_tokens(self, transcription):
        # Convert the transcription to lowercase
        transcription = transcription.lower()
        # Convert the transcription to a sequence of character tokens
        tokens = list(transcription)
        return tokens

    def _create_mapping(self):
        # Create a set of all characters that appear in the transcriptions
        characters = set()
        for transcription_file in self.transcription_files:
            with open(transcription_file, 'r') as f:
                transcription = f.read()
                transcription = self.preprocess_text(transcription)
                characters.update(transcription)
        # Create a character-to-integer mapping
        char_to_int = {char: i for i, char in enumerate(sorted(characters))}

        # Create an integer-to-character mapping
        int_to_char = {i: char for char, i in char_to_int.items()}

        return char_to_int, int_to_char

    def preprocess_text(self, transcription):
        transcription = unidecode(transcription)
        # Convert to lowercase
        transcription = transcription.lower()

        # Remove newline characters and replace '2' and '0' with appropriate words
        transcription = transcription.replace('\n', ' ').replace('2', 'two').replace('0', 'zero')

        # Remove punctuation
        transcription = transcription.translate(str.maketrans('', '', string.punctuation))
        
        return transcription

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

        audio_data = []
        transcriptions = []
        max_length_audio = 0
        max_length_transcription = 0
        

        # First pass: determine the maximum lengths
        for audio_file, transcription_file in tqdm(zip(audio_files, transcription_files), total=len(audio_files), ncols=100, desc='Determining max lengths'):
            # Load the audio data
            audio, sample_rate = librosa.load(audio_file, sr=None)

            # Load the transcription
            with open(transcription_file, 'r') as f:
                transcription = f.read()

            # Convert the transcription to a sequence of tokens
            transcription_tokens = self.transcription_to_tokens(transcription)

            # Update max_length_audio and max_length_transcription
            max_length_audio = max(max_length_audio, len(audio))
            max_length_transcription = max(max_length_transcription, len(transcription_tokens))
        # Second pass: process and write the data
        for audio_file, transcription_file in tqdm(zip(audio_files, transcription_files), total=len(audio_files), ncols=100, desc='Preprocessing'):
            # Load the audio data
            audio = self.needs_preprocessing(audio_file)

            # Load the transcription
            with open(transcription_file, 'r') as f:
                transcription = f.read()

            # Convert to unicode format convert to lowercase and remove punctuation
            transcription = self.preprocess_text(transcription)
            # Convert the transcription to a sequence of tokens
            transcription_tokens = self.transcription_to_tokens(transcription)

            # Convert the tokens to integers using the char_to_int mapping
            transcription_tokens = [self.char_to_int[token] for token in transcription_tokens]

            # Pad the audio data and transcription
            audio = np.pad(audio, (0, max_length_audio - len(audio)))
            transcription = np.pad(transcription_tokens, (0, max_length_transcription - len(transcription_tokens)))

            # Extract the file name from the audio file
            file_name = os.path.basename(audio_file)

            # Add the preprocessed audio data and transcription to the HDF5 file
            self.audio_data.create_dataset(file_name, data=audio)
            self.transcriptions.create_dataset(file_name, data=transcription, dtype=np.int32)