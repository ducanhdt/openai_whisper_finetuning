import json
import os
import numpy as np

import torch
import torchaudio

# import pandas as pd
import whisper
import torchaudio.transforms as at


class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and
    trim/pad the audio to 30 seconds.
    It will drop the last few seconds
    of a very small portion of the utterances.
    """

    def __init__(self, split="test-clean"):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("./data/"),
            url=split,
            download=True,
        )
        # self.dataset = [self.dataset[i] for i in range(100)] #sellect only first 100 sample

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)

        return (mel, text)


class LibriSpeechTraining(torch.utils.data.Dataset):
    def __init__(self, split="test-clean", tokenizer=None, sample_rate=16000) -> None:
        super().__init__()

        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("./data/"),
            url=split,
            download=True,
        )
        # self.dataset = [self.dataset[i] for i in range(100)]
        self.sample_rate = sample_rate
        self.options = whisper.DecodingOptions(language="en", without_timestamps=True)
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language="en", task=self.options.task
        )

    def load_wave(wave_path, sample_rate: int = 16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = at.Resample(sr, sample_rate)(waveform)
        return waveform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        audio, sample_rate, text, _, _, _ = self.dataset[id]

        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        text_token = [
            *self.tokenizer.sot_sequence_including_notimestamps
        ] + self.tokenizer.encode(text)
        labels = text_token[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text_token,
            "text": text,
        }


class WhisperDataCollatorWhithPadding:
    def __call__(sefl, features):
        input_ids, labels, dec_input_ids, texts = [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            texts.append(f["text"])
        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        labels = [
            np.pad(lab, (0, max_label_len - lab_len), "constant", constant_values=-100)
            for lab, lab_len in zip(labels, label_lengths)
        ]
        dec_input_ids = [
            np.pad(e, (0, max_label_len - e_len), "constant", constant_values=50257)
            for e, e_len in zip(dec_input_ids, dec_input_ids_length)
        ]  # 50257 is eot token id

        batch = {"labels": labels, "dec_input_ids": dec_input_ids}

        batch = {
            k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()
        }
        batch["input_ids"] = input_ids
        batch["texts"] = texts
        return batch


class VivosTraining(torch.utils.data.Dataset):
    def __init__(self, split="test", tokenizer=None, sample_rate=16000) -> None:
        super().__init__()

        root = f"data/vivos/{split}/waves"
        dataset = []
        with open(f"data/vivos/{split}/prompts.txt", "r") as f:
            a = f.read().split("\n")
        for i in a:
            x = i.find(" ")
            audio_id, text = i[:x], i[x + 1 :]
            speaker = audio_id.split("_")[0]
            audio_path = f"{root}/{speaker}/{audio_id}.wav"
            # print(audio_path)
            if os.path.isfile(audio_path):
                dataset.append((audio_id, audio_path, text))

        self.dataset = dataset
        # self.dataset = [self.dataset[i] for i in range(100)]
        self.sample_rate = sample_rate
        self.options = whisper.DecodingOptions(language="vi", without_timestamps=True)
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language="vi", task=self.options.task
        )

    def load_wave(self, wave_path, sample_rate: int = 16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = at.Resample(sr, sample_rate)(waveform)
        return waveform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        audio_id, audio_path, text = self.dataset[id]

        audio = self.load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)

        text_token = [
            *self.tokenizer.sot_sequence_including_notimestamps
        ] + self.tokenizer.encode(text)
        labels = text_token[1:] + [self.tokenizer.eot]

        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text_token,
            "text": text,
        }

class VivosTrainingBothTask(torch.utils.data.Dataset):
    def __init__(self, split="test", tokenizer=None, sample_rate=16000) -> None:
        super().__init__()

        root = f"data/vivos/{split}/waves"
        dataset = []
        with open(f"data/vivos/{split}/prompts.txt", "r") as f:
            transcribe_text = f.read().split("\n")
        with open(f"data/vivos/{split}/prompts_english.txt", "r") as f:
            translate_texts = f.read().split("\n")
        for text in transcribe_text:
            x = text.find(" ")
            audio_id, text = text[:x], text[x + 1 :]
            speaker = audio_id.split("_")[0]
            audio_path = f"{root}/{speaker}/{audio_id}.wav"
            if os.path.isfile(audio_path):
                dataset.append((audio_id, audio_path, text, "transcribe"))
            
        for text in translate_texts:
            x = text.find(" ")
            audio_id, text = text[:x], text[x + 1 :]
            
            speaker = audio_id.split("_")[0]
            audio_path = f"{root}/{speaker}/{audio_id}.wav"
            if os.path.isfile(audio_path):
                dataset.append((audio_id, audio_path, text, "translate"))
                    

        self.dataset = dataset
        # self.dataset = [self.dataset[i] for i in range(100)]
        self.sample_rate = sample_rate
        self.options = whisper.DecodingOptions(language="vi", without_timestamps=True)
        self.translate_tokenizer = whisper.tokenizer.get_tokenizer(
            True, language="vi", task="translate"
        )
        self.transcribe_tokenizer = whisper.tokenizer.get_tokenizer(
            True, language="vi", task=self.options.task
        )


    def load_wave(self, wave_path, sample_rate: int = 16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = at.Resample(sr, sample_rate)(waveform)
        return waveform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        audio_id, audio_path, text, task = self.dataset[id]

        audio = self.load_wave(audio_path, sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)
        tokenizer = getattr(self,f"{task}_tokenizer")
        text_token = [
            *tokenizer.sot_sequence_including_notimestamps
        ] + tokenizer.encode(text)
        labels = text_token[1:] + [tokenizer.eot]
        
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text_token,
            "text": text,
        }

class ZaloAiWithTimestampTraining(torch.utils.data.Dataset):
    def __init__(self, split = "train",tokenizer=None, sample_rate=16000) -> None:
        super().__init__()

        root = f"../train"
        self.song_path = f"{root}/songs"
        self.label_path = f"{root}/labels"
        
        song_paths_list = {i.replace(".wav","") for i in os.listdir(self.song_path) if ".wav" in i}
        id_list = [i.replace(".json","") for i in os.listdir(self.label_path) if ".json" in i and i.replace(".json","") in song_paths_list]
        self.dataset = []
        for i in id_list:
            self.dataset.append((i,"word"))
            if split == "train":
                self.dataset.append((i,"segment"))
        # self.dataset = [self.dataset[i] for i in range(100)]
        self.sample_rate = sample_rate
        self.options = whisper.DecodingOptions(language="vi", without_timestamps=True)
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            True, language="vi", task=self.options.task
        )

    def load_wave(self, wave_path, sample_rate: int = 16000) -> torch.Tensor:
        waveform, sr = torchaudio.load(wave_path, normalize=True)
        if sample_rate != sr:
            waveform = at.Resample(sr, sample_rate)(waveform)
        return waveform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, id):
        audio_id, token_type  = self.dataset[id]

        audio = self.load_wave(f"{self.song_path}/{audio_id}.wav", sample_rate=self.sample_rate)
        audio = whisper.pad_or_trim(audio.flatten())
        mel = whisper.log_mel_spectrogram(audio)
        
        with open(f"{self.label_path}/{audio_id}.json",'r') as f:
            target = json.load(f)
        
        text_token = [
            *self.tokenizer.sot_sequence
        ] 
        if token_type == "segment":
            for seg in target:
               text_token.append(int(seg['s']/1000.0/0.02)+self.tokenizer.timestamp_begin)
               text_token += self.tokenizer.encode(" ".join([tok['d'] for tok in seg['l']]))
               text_token.append(int(seg['e']/1000.0/0.02)+self.tokenizer.timestamp_begin)
        else:
            for seg in target:
                for tok in seg['l']:
                    text_token.append(int(tok['s']/1000.0/0.02)+self.tokenizer.timestamp_begin)
                    text_token += self.tokenizer.encode(tok['d'])
                    text_token.append(int(tok['e']/1000.0/0.02)+self.tokenizer.timestamp_begin)
            if len(text_token)>448:
                print("text word too long")
                text_token = [
                    *self.tokenizer.sot_sequence
                ] 
                if token_type == "segment":
                    for seg in target:
                        text_token.append(int(seg['s']/1000.0/0.02)+self.tokenizer.timestamp_begin)
                        text_token += self.tokenizer.encode(" ".join([tok['d'] for tok in seg['l']]))
                        text_token.append(int(seg['e']/1000.0/0.02)+self.tokenizer.timestamp_begin)
                            
        labels = text_token[1:] + [self.tokenizer.eot]
        text = ' '.join([tok['d'] for seg in target for tok in seg['l']])
        return {
            "input_ids": mel,
            "labels": labels,
            "dec_input_ids": text_token,
            "text": text,
        }



if __name__ == "__main__":
    # dataset = LibriSpeech("test-clean")
    # loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    # sample_mel, sample_text = dataset[0]
    # print(sample_mel.shape)
    # print(sample_text)
    model = whisper.load_model("tiny")
    wtokenizer = whisper.tokenizer.get_tokenizer(True, language="en")
    dataset = ZaloAiWithTimestampTraining(wtokenizer)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=WhisperDataCollatorWhithPadding()
    )
    for b in loader:
        print(b["labels"].shape)
        print(b["input_ids"].shape)
        print(b["dec_input_ids"].shape)

        for token, dec in zip(b["labels"], b["dec_input_ids"]):
            token[token == -100] = wtokenizer.eot
            text = wtokenizer.decode_with_timestamps(token)
            print(text)

            dec[dec == -100] = wtokenizer.eot
            text = wtokenizer.decode_with_timestamps(dec)
            print(text)
        break
    # with torch.no_grad():
    #     audio_features = model.encoder(b["input_ids"])
    #     input_ids = b["input_ids"]
    #     labels = b["labels"].long()
    #     dec_input_ids = b["dec_input_ids"].long()

    #     audio_features = model.encoder(input_ids)
    #     print(dec_input_ids)
    #     print(input_ids.shape, dec_input_ids.shape, audio_features.shape)
    #     print(audio_features.shape)
    #     print()
    # # out = model.decoder(dec_input_ids, input_ids)
    # out = model.decoder(dec_input_ids, audio_features)
    # print(out.shape)
    # print(out.view(-1, out.size(-1)).shape)
    # print(b["labels"].view(-1).shape)
    # tokens = torch.argmax(out, dim=2)
    # for token in tokens:
    #     token[token == -100] = wtokenizer.eot
    #     text = wtokenizer.decode(token, skip_special_tokens=True)
    #     print(text)
