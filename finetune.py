from pathlib import Path
import torch
from config import Config
try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

from dataset import LibriSpeechTraining, VivosTraining, WhisperDataCollatorWhithPadding

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import WhisperModelModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

if config.lang == "en":
    train_dataset = LibriSpeechTraining("train-clean-100")
    valid_dataset = LibriSpeechTraining("dev-clean")
elif config.lang == "vi":
    train_dataset = VivosTraining("train")
    valid_dataset = VivosTraining("test")

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.batch_size,num_workers=config.num_worker, collate_fn=WhisperDataCollatorWhithPadding()
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=config.batch_size,num_workers=config.num_worker, collate_fn=WhisperDataCollatorWhithPadding()
)


Path(config.log_output_dir).mkdir(exist_ok=True)
Path(config.check_output_dir).mkdir(exist_ok=True)

tflogger = TensorBoardLogger(save_dir=config.log_output_dir, name=config.train_name, version=config.train_id)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{config.check_output_dir}/checkpoint",
    filename="checkpoint-{epoch:04d}",
    save_top_k=-1,  # all model save
)

callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
model = WhisperModelModule(config,train_loader,valid_loader)

trainer = Trainer(
    # precision=2,
    accelerator=DEVICE,
    max_epochs=config.num_train_epochs,
    accumulate_grad_batches=config.gradient_accumulation_steps,
    logger=tflogger,
    callbacks=callback_list,
)

trainer.fit(model)
