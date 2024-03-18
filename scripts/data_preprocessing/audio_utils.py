# scripts/data_preprocessing/audio_utils.py
import librosa
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler

def convert_to_mono(input_file_path, output_file_path):
    # Load the audio file
    y, sr = librosa.load(input_file_path, sr=None, mono=False)

    # Convert to mono
    y_mono = librosa.to_mono(y)

    # Write the mono audio file to the output path
    sf.write(output_file_path, y_mono, sr)

def remove_silence(audio_data, sample_rate=22050, top_db=20, silence_threshold=3):
    # Remove silence from the audio
    non_silent_intervals = librosa.effects.split(audio_data, top_db=top_db)

    # Initialize the start of the first non-silent interval
    start = 0

    # List to hold the audio data with short silences
    audio_with_short_silences = []

    for non_silent_start, non_silent_end in non_silent_intervals:
        # Calculate the duration of the silence
        silence_duration = (non_silent_start - start) / sample_rate

        # If the silence is shorter than the threshold, keep it
        if silence_duration < silence_threshold:
            audio_with_short_silences.append(audio_data[start:non_silent_end])
        else:
            audio_with_short_silences.append(audio_data[non_silent_start:non_silent_end])

        # Update the start of the next silence
        start = non_silent_end

    # Concatenate the audio data with short silences
    audio_with_short_silences = np.concatenate(audio_with_short_silences, axis=0)

    return audio_with_short_silences
    
def sample_rate_conversion(input_file_path, output_file_path, target_sample_rate=22050):
    # Load the audio file with the target sample rate
    y, sr = librosa.load(input_file_path, sr=target_sample_rate)

    # Write the audio file to the output path
    sf.write(output_file_path, y, sr)

def extract_frames(audio_data, sample_rate, frame_size=0.025, frame_stride=0.01):
    # Calculate frame length and stride (in samples)
    frame_length, frame_stride = frame_size * sample_rate, frame_stride * sample_rate

    # Convert to integers
    frame_length, frame_stride = int(round(frame_length)), int(round(frame_stride))

    # Calculate total frames
    total_frames = int(np.ceil(float(np.abs(len(audio_data) - frame_length)) / frame_stride))

    # Pad signal
    pad_audio_length = total_frames * frame_stride + frame_length
    pad_signal = np.pad(audio_data, (0, pad_audio_length - len(audio_data)), 'constant')

    # Apply frames
    indices = np.tile(np.arange(0, frame_length), (total_frames, 1)) + np.tile(np.arange(0, total_frames * frame_stride, frame_stride), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    return frames

def extract_features(audio_data, sample_rate):
    # Spectrogram
    spectrogram = np.abs(librosa.stft(audio_data))

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate)

    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)

    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)

    return spectrogram, mfcc, chroma, spectral_contrast

def normalize_features(features):
    scaler = StandardScaler()
    return scaler.fit_transform(features)

def augment_audio(audio_data, sample_rate):
    # Time stretching
    audio_data_stretch = librosa.effects.time_stretch(audio_data, rate=1.2)

    # Pitch shifting
    audio_data_shift = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=4)

    # Adding noise
    noise = np.random.normal(0, 0.005, len(audio_data))
    audio_data_noisy = audio_data + noise

    return audio_data_stretch, audio_data_shift, audio_data_noisy

