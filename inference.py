import numpy as np
import whisper
import torch

from model import WhisperModelModule
from ultis import load_config_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dotenv import load_dotenv
load_dotenv()
config = load_config_file(os.environ["CONFIG_PATH"])


checkpoint_path = config.checkpoint_path

# try:
module = WhisperModelModule(config)
try:
    state_dict = torch.load(checkpoint_path)
    state_dict = state_dict["state_dict"]
    module.load_state_dict(state_dict)
    print(f"load checkpoint successfully from {checkpoint_path}")
except Exception as e:
    print(e)
    print(f"load checkpoint failt using origin weigth of {config.model_name} model")
model = module.model
model.to(device)

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
options = whisper.DecodingOptions(
    language="vi", without_timestamps=True, fp16=torch.cuda.is_available()
)
result = whisper.decode(model, mel, options)
# print(options)
# print the recognized text
print(result.audio_features.shape)
print(result.text)
