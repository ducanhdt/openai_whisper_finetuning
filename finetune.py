from pathlib import Path
import numpy as np
import torch
import whisper
from config import Config

from dataset import LibriSpeechTraining, WhisperDataCollatorWhithPadding

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import WhisperModelModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

woptions = whisper.DecodingOptions(language="en", without_timestamps=True)
wtokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=woptions.task)

dataset = LibriSpeechTraining("test-clean", wtokenizer)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=cfg.batch_size, collate_fn=WhisperDataCollatorWhithPadding()
)


Path(cfg.log_output_dir).mkdir(exist_ok=True)
Path(cfg.check_output_dir).mkdir(exist_ok=True)

tflogger = TensorBoardLogger(save_dir=cfg.log_output_dir, name=cfg.train_name, version=cfg.train_id)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{cfg.check_output_dir}/checkpoint",
    filename="checkpoint-{epoch:04d}",
    save_top_k=-1,  # all model save
)

callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
model = WhisperModelModule(cfg, loader, loader)

trainer = Trainer(
    # precision=2,
    accelerator=DEVICE,
    max_epochs=cfg.num_train_epochs,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    logger=tflogger,
    callbacks=callback_list,
)

trainer.fit(model)
