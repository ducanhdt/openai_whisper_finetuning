# This is an unofficial code for finetune Whisper model with your own dataset
Original Repo: https://github.com/openai/whisper

python version: 3.7
setup:
```bash
pip install -r requirements.txt
```
In this setup we using a small part of LibriDataset for fineturning the smallest model. You can't change this setting and other hyperparameters by modify "config.py"

To Fineturning:
```bash
python finetune.py
```
To evaluate:
```bash
python evaluate_wer.py
```



