# This is an unofficial code for finetune Whisper model with your own dataset
[[Original Repo]](https://github.com/openai/whisper) 
[[Example Colab]](https://colab.research.google.com/drive/1aXj6ssi_y3qow-h6M8M1Q5tFgHE0n3UP?usp=sharing)

In this setup we use a small part of the LibriSpeech Dataset for finetuning the English model, the other option is using the Vivos dataset for finetuning the Vietnamese model. In case you want to finetune in either another dataset or another language, check the "dataset.py". You are also able to change the hyperparameters by using other setup file base on the file "config/vn_base_example.yaml". The path to config file must be define in .env

Experiment on Vietnamese with Vivos Dataset, WER of the base Whisper model dropped from 45.56% to 24.27% after finetuning 5 epochs.

Python version: 3.7

Setup:
```bash
pip install -r requirements.txt
cp .env.copy .env
```

In case you want to finetune model in Vietnamese, run this command to download the dataset:
```bash
python data/download_data_vivos.py
tar -xvf vivos.tar.gz vivos
mv vivos data
```

To Finetune:
```bash
python finetune.py
```
To evaluate the model:
```bash
python evaluate_wer.py
```

To inference:

You are able to record your own audio file and convert it from speech to text using "record.py" and "inference.py"

### Todo list
- [x] Add python argument parser and refactor code
- [ ] Add dockerfile for deploy
- [ ] Add Vietnamese Text normalization / Postprocessing
- [ ] Add streamlit interface to record and inference

