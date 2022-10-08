import numpy as np
import whisper
from whisper.normalizers import EnglishTextNormalizer
import torch

model = whisper.load_model("tiny")
normalizer = EnglishTextNormalizer()

print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)

# result = model.transcribe("audio.wav")

audio = whisper.load_audio("audio.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16=torch.cuda.is_available())
result = whisper.decode(model, mel, options)
# print(options)
# print the recognized text
print(result.audio_features.shape)
print(result.text)
