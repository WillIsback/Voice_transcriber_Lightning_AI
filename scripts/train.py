# train.py
import h5py
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import nemo
import nemo.collections.asr as nemo_asr
import numpy as np

class ASRDataset(Dataset):
    def __init__(self, file_path):
        with h5py.File(file_path, 'r') as f:
            self.audio_data = np.array(f['audio_data'])
            self.transcriptions = np.array(f['transcriptions'])
            self.length = len(self.audio_data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        audio_str = self.audio_data[index]
        audio = np.fromstring(audio_str, sep=' ')
        audio = torch.from_numpy(audio).float()
        audio = audio.squeeze(0)  # Remove the extra dimension
        transcription = self.transcriptions[index]
        return audio, len(audio), transcription, len(transcription)
with h5py.File("/teamspace/studios/this_studio/dataset/compressed_dataset.h5", 'r') as f:
    print(f.keys())
# Load the data
dataset = ASRDataset("/teamspace/studios/this_studio/dataset/compressed_dataset.h5")

# Determine the lengths of the splits
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Split the data
print("Splitting the data...")
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

dry_run = True

# Load the pre-trained model
print("Loading the pre-trained model...")
model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

# Create a logger
logger = TensorBoardLogger("logs", name="hyperparams-finetune")

trainer = Trainer(
    logger=logger,
    limit_val_batches=0.1,
    check_val_every_n_epoch=10 if not dry_run else 1,
    max_epochs=100 if not dry_run else 1,
)

print("Starting training...")
trainer.fit(model, train_loader, val_loader)
print("Finished training.")

# Define the PyTorch Lightning trainer
logger = TensorBoardLogger("tb_logs", name="Training-Logs")

# Define learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')

print("Starting final training...")

trainer = Trainer(
    max_epochs=100 if not dry_run else 1,
    logger=logger,
    progress_bar_refresh_rate=20,
    default_root_dir='models/',
    auto_lr_find=True,
    gradient_clip_val=0.5,
    precision=16 if torch.cuda.is_available() and not dry_run else 32,
    deterministic=True,
    resume_from_checkpoint=None,
)

# Train the model
trainer.fit(model, train_loader, val_loader)

print("Finished final training.")