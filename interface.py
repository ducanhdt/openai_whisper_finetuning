import io

import numpy as np
from model import WhisperModelModule
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import os
import torch
import whisper
from scipy.io.wavfile import read, write
import torchaudio.transforms as at

from ultis import load_config_file

st.title("Whisper Demo")
st.markdown("*Created by Dang Trung Duc Anh")

model_status = st.empty()
config_path = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_list = os.listdir("config")

@st.experimental_singleton
def load_model(config_path):
    # model_status.write("Loading model")
    config = load_config_file(f"config/{config_path}")
    module = WhisperModelModule(config)
    try:
        state_dict = torch.load(config["checkpoint_path"])
        state_dict = state_dict["state_dict"]
        module.load_state_dict(state_dict)
        # model_status.write(
        #     f"load checkpoint successfully from {config['checkpoint_path']}"
        # )
    except Exception as e:
        print(e)
        # model_status.write(
        #     f"load checkpoint failt using origin weigth of {config['model_name']} model"
        # )
    model = module.model
    model.to(device)
    return model

config_path = st.selectbox("Select the model", config_list)
model = load_model(config_path)
if st.button('Reload model'):
    model = load_model(config_path)

audio_bytes = audio_recorder()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

if st.button("Convert"):
    rate, audio = read(io.BytesIO(audio_bytes))
    write('tmp.wav', rate, audio.astype(np.int16))
    # print(rate, audio)
    # if rate != 16000:
    #     waveform = at.Resample(rate, 16000)(audio)
    audio = whisper.load_audio("tmp.wav",16000)
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
    st.text(result.text)
