import jiwer
import numpy as np
import pandas as pd
import torch
import whisper
try:
    import tensorflow  # required in Colab to avoid protobuf compatibility issues
except ImportError:
    pass

from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer
from config import Config
from model import WhisperModelModule

from dataset import VivosTraining, LibriSpeechTraining, WhisperDataCollatorWhithPadding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

checkpoint_path = "/home/ducanh/Desktop/WHISPER/content/artifacts/checkpoint/checkpoint-epoch=0002.ckpt"

# try:
module = WhisperModelModule(config)
try:
    state_dict = torch.load(checkpoint_path)
    state_dict = state_dict['state_dict']
    module.load_state_dict(state_dict)
    print(f"load checkpoint successfully from {checkpoint_path}")
except Exception as e:
    print(e)
    print(f"load checkpoint failt using origin weigth of {config.model_name} model")
model = module.model
normalizer = EnglishTextNormalizer()

print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

if config.lang == "vi":
    dataset = VivosTraining("test")
    options = whisper.DecodingOptions(language="vi", without_timestamps=True, fp16=torch.cuda.is_available())
elif config.lang == "en":
    dataset = LibriSpeechTraining('test-clean')
    options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=torch.cuda.is_available())

loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, collate_fn=WhisperDataCollatorWhithPadding()
)

hypotheses = []
references = []

for sample in tqdm(loader):
    mels = sample['input_ids'].to(model.device)
    texts = sample['labels']
    print("xxx",mels.device,model.device)
    results = model.decode(mels, options)
    hypotheses.extend([result.text for result in results])
    references.extend(texts)

data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
data["reference_clean"] = [normalizer(text) for text in data["reference"]]

wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

print(f"WER: {wer * 100:.2f} %")
