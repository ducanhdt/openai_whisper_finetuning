import os
from pathlib import Path
import torch
from ultis import load_config_file

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

from dataset import LibriSpeechTraining, VivosTraining, VivosTrainingBothTask, WhisperDataCollatorWhithPadding,ZaloAiWithTimestampTraining  

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model import WhisperModelModule

from dotenv import load_dotenv

load_dotenv()
config = load_config_file(os.environ["CONFIG_PATH"])
print(config)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


if config["lang"] == "en":
    train_dataset = LibriSpeechTraining("train-clean-100")
    valid_dataset = LibriSpeechTraining("dev-clean")
elif config["lang"] == "vi":
    train_dataset = ZaloAiWithTimestampTraining("train")
    valid_dataset = ZaloAiWithTimestampTraining("test")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    num_workers=config["num_worker"],
    collate_fn=WhisperDataCollatorWhithPadding(),
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=config["batch_size"],
    num_workers=config["num_worker"],
    collate_fn=WhisperDataCollatorWhithPadding(),
)


Path(config["log_output_dir"]).mkdir(exist_ok=True)
Path(config["check_output_dir"]).mkdir(exist_ok=True)

tflogger = TensorBoardLogger(
    save_dir=config["log_output_dir"],
    name=config["train_name"],
    version=config["train_id"],
)

checkpoint_callback = ModelCheckpoint(
    dirpath=f'{config["check_output_dir"]}/checkpoint',
    filename="checkpoint-{epoch:04d}",
    save_top_k=-1,  # all model save
)

callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
model = WhisperModelModule(config, train_loader, valid_loader)

trainer = Trainer(
    # precision=2,
    accelerator=DEVICE,
    max_epochs=config["num_train_epochs"],
    accumulate_grad_batches=config["gradient_accumulation_steps"],
    logger=tflogger,
    callbacks=callback_list,
)

trainer.fit(model)
