import h5py
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import nemo.collections.asr as nemo_asr
import numpy as np
from data_preprocessing.HDF5_dataset import HDF5Dataset


# Load the data
dataset = HDF5Dataset("/teamspace/studios/this_studio/dataset/compressed_dataset.h5")

# Print the size of the dataset
print(f"Total size of the dataset: {len(dataset)}")

# Determine the lengths of the splits
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Print the sizes of the splits
print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

# Split the data
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=32, num_workers=4)

# Load the pre-trained model
model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

# Create a character-to-integer mapping from the model's vocabulary
pretrained_char_to_int = {char: i for i, char in enumerate(model.decoder.vocabulary)}

# Update the labels_map to include all IDs up to max_id

# Get the labels used in the pre-trained model
pretrained_labels = set(pretrained_char_to_int.keys())

# Get the labels used in your training data
training_labels = set(dataset.char_to_int.keys())

# Print out the labels that are in the training data but not in the pre-trained model
print("Labels in training data not in pretrained model:", training_labels - pretrained_labels)

# Print out the labels that are in the pre-trained model but not in the training data
print("Labels in pre-trained model that are not in training data:", pretrained_labels - training_labels)
# Create a logger
logger = TensorBoardLogger("logs", name="training")

# Define learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Define model checkpoint callback
checkpoint = ModelCheckpoint(dirpath='checkpoints', monitor='val_loss', mode='min', save_top_k=1)

trainer = pl.Trainer(
    logger=logger,
    callbacks=[lr_monitor, early_stopping, checkpoint],
    max_epochs=100,
    precision=16 if torch.cuda.is_available() else 32,
    deterministic=True,
)

# Train the model
trainer.fit(model, train_loader, val_loader)