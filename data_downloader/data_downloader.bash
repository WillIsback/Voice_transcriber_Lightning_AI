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