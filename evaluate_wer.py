import jiwer
import numpy as np
import pandas as pd
import torch
import whisper
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer

from dataset import LibriSpeech

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


dataset = LibriSpeech("test-clean")
loader = torch.utils.data.DataLoader(dataset, batch_size=1)

model = whisper.load_model("tiny")
normalizer = EnglishTextNormalizer()

print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)


options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=False)
# options = whisper.DecodingOptions(language="en", without_timestamps=True)
# options = whisper.DecodingOptions(language="vi", without_timestamps=True)

hypotheses = []
references = []

for mels, texts in tqdm(loader):
    results = model.decode(mels, options)
    hypotheses.extend([result.text for result in results])
    references.extend(texts)

data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
data["reference_clean"] = [normalizer(text) for text in data["reference"]]

wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

print(f"WER: {wer * 100:.2f} %")
