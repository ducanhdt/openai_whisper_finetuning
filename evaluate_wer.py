import os
import jiwer
import numpy as np
import pandas as pd
import torch
import whisper

from ultis import load_config_file

try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer
from config import Config
from model import WhisperModelModule

from dataset import VivosTraining, LibriSpeechTraining, WhisperDataCollatorWhithPadding

from dotenv import load_dotenv
load_dotenv()
config = load_config_file(os.environ["CONFIG_PATH"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = config["checkpoint_path"]

# try:
module = WhisperModelModule(config)
try:
    state_dict = torch.load(checkpoint_path)
    state_dict = state_dict["state_dict"]
    module.load_state_dict(state_dict)
    print(f"load checkpoint successfully from {checkpoint_path}")
except Exception as e:
    print(e)
    print(f"load checkpoint failt using origin weigth of {config['model_name']} model")
model = module.model
model.to(device)
normalizer = EnglishTextNormalizer()

print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

if config["lang"] == "vi":
    dataset = VivosTraining("test")
    options = whisper.DecodingOptions(
        language="vi", without_timestamps=True, fp16=torch.cuda.is_available()
    )
elif config["lang"] == "en":
    dataset = LibriSpeechTraining("test-clean")
    options = whisper.DecodingOptions(
        language="en", without_timestamps=True, fp16=torch.cuda.is_available()
    )

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    num_workers=config["num_worker"],
    collate_fn=WhisperDataCollatorWhithPadding(),
)

hypotheses = []
references = []
print(model.device)
for sample in tqdm(loader):
    mels = sample["input_ids"].to(model.device)
    texts = sample["texts"]
    results = model.decode(mels, options)
    hypotheses.extend([result.text for result in results])
    references.extend(texts)

data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

data["hypothesis_clean"] = [
    normalizer(text) if config["lang"] == "en" else text.lower()
    for text in data["hypothesis"]
]
data["reference_clean"] = [
    normalizer(text) if config["lang"] == "en" else text.lower()
    for text in data["reference"]
]
print(data["hypothesis_clean"][:10])
print("___________")
print(data["reference_clean"][:10])
wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

print(f"WER: {wer * 100:.2f} %")
