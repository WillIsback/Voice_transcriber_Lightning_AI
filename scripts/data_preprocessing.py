# scripts/data_preprocessing.py
from data_preprocessing.HDF5_dataset import HDF5Dataset
from data_preprocessing import audio_utils
import os
import librosa

def main(dry_run=False):
    # Define the base directory path
    base_directory_path = 'dataset/SiwisFrenchSpeechSynthesisDatabase'

    # Create an instance of the HDF5Dataset class in write mode
    dataset = HDF5Dataset("/teamspace/studios/this_studio/dataset/compressed_dataset.h5", mode='w')

    try:
        # Call the Preprocessing method on the instance
        dataset.Preprocessing(base_directory_path=base_directory_path, dry_run=dry_run)
    finally:
        # Close the file after writing to it
        dataset.file.close()
 
if __name__ == "__main__":
    main(dry_run=False)  # Set dry_run to True to only print the changes without saving themj  