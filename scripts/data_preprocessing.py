# scripts/data_preprocessing.py
from data_preprocessing.HDF5_dataset import HDF5Dataset
from data_preprocessing import audio_utils
import os
import librosa

def main(dry_run=False):
    # Create an instance of your dataset
    dataset = HDF5Dataset('dataset/compressed_dataset.h5')

    # Define the base directory path
    base_directory_path = 'dataset/SiwisFrenchSpeechSynthesisDatabase'


    try:
        dataset.Preprocessing(base_directory_path, dry_run=dry_run)
        if(dry_run):
            
            print("Dry run completed. No changes were saved.")
            dataset.HDF5_Reader(base_directory_path)
    except KeyboardInterrupt:
        print("Program interrupted by user. Saving changes...")
    except Exception as e:
        print(f"An error occurred: {e}. Saving changes...")
    finally:
        # Save and close the HDF5 file
        dataset.file.flush()
        dataset.file.close()
        print("Changes saved and HDF5 file closed.")

if __name__ == "__main__":
    main(dry_run=False)  # Set dry_run to True to only print the changes without saving them