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

log_output_dir = "content/logs"
check_output_dir = "content/artifacts"

train_name = "whisper"
train_id = "00001"

model_name = "base"
lang = "en"

woptions = whisper.DecodingOptions(language="en", without_timestamps=True)
model = whisper.load_model("tiny")
wtokenizer = whisper.tokenizer.get_tokenizer(True, language="en", task=woptions.task)

print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

dataset = LibriSpeechTraining("test-clean", wtokenizer)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, collate_fn=WhisperDataCollatorWhithPadding()
)

cfg = Config()

Path(log_output_dir).mkdir(exist_ok=True)
Path(check_output_dir).mkdir(exist_ok=True)

tflogger = TensorBoardLogger(save_dir=log_output_dir, name=train_name, version=train_id)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"{check_output_dir}/checkpoint",
    filename="checkpoint-{epoch:04d}",
    save_top_k=-1,  # all model save
)

callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
model = WhisperModelModule(cfg, model_name, lang, loader, loader)

trainer = Trainer(
    precision=16,
    accelerator=DEVICE,
    max_epochs=cfg.num_train_epochs,
    accumulate_grad_batches=cfg.gradient_accumulation_steps,
    logger=tflogger,
    callbacks=callback_list,
)

trainer.fit(model)
