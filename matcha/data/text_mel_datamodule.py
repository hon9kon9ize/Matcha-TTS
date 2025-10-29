import random
from pathlib import Path
from typing import Any, Dict, Optional
import librosa
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.utils import intersperse
from datasets import load_dataset, load_from_disk
from matcha.feature_extractions.spkemb_speechbrain import SpeechBrainSpkEmbExtractor


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        dataset_path,
        dataset_valid_ratio,
        batch_size,
        num_workers,
        pin_memory,
        add_blank,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
        skip_pos=False,
        skip_spk_emb=False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        import os

        if os.path.isdir(self.hparams.dataset_path):
            # Load local dataset saved with save_to_disk
            ds = load_from_disk(self.hparams.dataset_path)
            # For local datasets, split manually
            ds = ds.train_test_split(test_size=self.hparams.dataset_valid_ratio)
        else:
            # Load from HuggingFace Hub
            ds = load_dataset(self.hparams.dataset_path, split="train")
            ds = ds.train_test_split(test_size=self.hparams.dataset_valid_ratio)

        self.trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            ds["train"],
            self.hparams.n_spks,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            "tmp",
            self.hparams.skip_pos,
            self.hparams.skip_spk_emb,
        )
        self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            ds["test"],
            self.hparams.n_spks,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            "tmp",
            self.hparams.skip_pos,
            self.hparams.skip_spk_emb,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=1,  # Fixed to 1 due to pydips compatibility
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
            prefetch_factor=2,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=1,  # Fixed to 1 due to pydips compatibility
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
            prefetch_factor=2,
            persistent_workers=True,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        n_spks,
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        load_durations=False,
        tmp_dir="tmp",
        skip_pos=False,
        skip_spk_emb=False,
    ):
        self.dataset = dataset
        self.n_spks = n_spks
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations
        self.speaker_embedding_extractor = SpeechBrainSpkEmbExtractor(device="cpu") if not skip_spk_emb else None
        self.tmp_dir = Path(tmp_dir)
        self.skip_pos = skip_pos
        self.skip_spk_emb = skip_spk_emb

        # Create temporary directory if it does not exist
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
        random.seed(seed)

    def get_datapoint(self, row):
        text = row["text"]
        lang = row["lang"]  # Assume lang is always present in pre-processed data

        # Use pre-processed fields directly (assume prepare.py output)
        phone = row["phone_token_ids"]
        tones = row["tones"]
        word_positions = row["word_pos"]
        syllable_positions = row["syllable_pos"]
        audio = np.array(row["audio"]["array"])
        audio_path = row["audio"]["path"]
        sr = row["audio"]["sampling_rate"]

        # Shift phoneme tones by 1 to avoid conflict with blank tone 0
        if tones:
            pad_start = tones[0]
            pad_end = tones[-1]
            phoneme_tones = tones[1:-1]
            phoneme_tones = [t + 1 for t in phoneme_tones]
            tones = [pad_start] + phoneme_tones + [pad_end]

        text, phone, tone, word_pos, syllable_pos = self.get_text(
            text, phone, tones, word_positions, syllable_positions, add_blank=self.add_blank, skip_pos=self.skip_pos
        )
        audio16k = audio
        audio22k = audio
        # sr resampling audio from 16k -> 22050
        if sr == 16_000:
            audio22k = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        elif sr == 22_050:
            audio16k = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        mel = self.get_mel(audio22k, self.sample_rate)

        # Use pre-computed speaker embedding if available, otherwise compute it
        spk_emb = row.get("spk_emb")
        if spk_emb is None and not self.skip_spk_emb and self.speaker_embedding_extractor is not None:
            spk_emb = self.speaker_embedding_extractor.forward(audio16k, 16000)

        durations = self.get_durations(audio, text) if self.load_durations else None

        return {
            "x": phone,
            "y": mel,
            "filepath": audio_path,
            "x_text": text,
            "durations": durations,
            "tone": tone,
            "word_pos": word_pos,
            "syllable_pos": syllable_pos,
            "spk_emb": spk_emb,
            "lang": lang,
        }

    def get_durations(self, filepath, text):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem

        try:
            dur_loc = data_dir / "durations" / f"{name}.npy"
            durs = torch.from_numpy(np.load(dur_loc).astype(int))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first using: python matcha/utils/get_durations_from_trained_model.py \n"
            ) from e

        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs

    def get_mel(self, audio, sr: int):
        audio = torch.from_numpy(audio).unsqueeze(0).float()  # [1, T]
        assert sr == self.sample_rate
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel

    def get_text(self, text, phone, tones, word_pos, syllable_pos, add_blank=False, skip_pos=False):
        # Shift phoneme tones by 1 to avoid conflict with blank tone 0
        if tones:
            pad_start = tones[0]
            pad_end = tones[-1]
            phoneme_tones = tones[1:-1]
            phoneme_tones = [t + 1 for t in phoneme_tones]
            tones = [pad_start] + phoneme_tones + [pad_end]

        if add_blank:
            phone = intersperse(phone, 0)
            tones = intersperse(tones, 0)
            if not skip_pos:
                word_pos = intersperse(word_pos, 0)
                syllable_pos = intersperse(syllable_pos, 0)
            else:
                word_pos = [0] * len(phone)
                syllable_pos = [0] * len(phone)
        phone = torch.LongTensor(phone)
        tone = torch.LongTensor(tones)
        word_pos = torch.LongTensor(word_pos)
        syllable_pos = torch.LongTensor(syllable_pos)

        return text, phone, tone, word_pos, syllable_pos

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.dataset[index])
        return datapoint

    def __len__(self):
        return len(self.dataset)


class TextMelBatchCollate:
    def __init__(self, n_spks):
        self.n_spks = n_spks

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        tone = torch.zeros((B, x_max_length), dtype=torch.long)
        word_pos = torch.zeros((B, x_max_length), dtype=torch.long)
        syllable_pos = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)
        spk_embed = torch.zeros(B, 192, dtype=torch.float32)
        lang = torch.zeros(B, dtype=torch.long)
        y_lengths, x_lengths = [], []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            y_, x_, tone_, word_pos_, syllable_pos_, spk_embed_, lang_str = (
                item["y"],
                item["x"],
                item["tone"],
                item["word_pos"],
                item["syllable_pos"],
                item["spk_emb"],
                item["lang"],
            )
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            tone[i, : tone_.shape[-1]] = tone_
            word_pos[i, : word_pos_.shape[-1]] = word_pos_
            syllable_pos[i, : syllable_pos_.shape[-1]] = syllable_pos_
            if spk_embed_ is not None:
                spk_embed[i] = torch.tensor(spk_embed_).float()
            lang_id = {"en": 0, "yue": 1, "zh": 2}[lang_str]
            lang[i] = lang_id
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "tone": tone,
            "word_pos": word_pos,
            "syllable_pos": syllable_pos,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "spk_emb": spk_embed,
            "lang": lang,
            "durations": durations if not torch.eq(durations, 0).all() else None,
        }
