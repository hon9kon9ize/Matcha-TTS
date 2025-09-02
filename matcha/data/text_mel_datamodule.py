import random
from pathlib import Path
from typing import Any, Dict, Optional
import torchaudio.compliance.kaldi as kaldi
import librosa
import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
import onnxruntime
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.utils import intersperse
from datasets import load_dataset


def load_spk_embedding(onnx_path: str):
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=["CPUExecutionProvider"])
    return ort_session


def get_spk_embedding(audio, onnx_session):
    audio_tensor = None

    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(dim=0)
    elif isinstance(audio, torch.Tensor):
        if audio.dim() == 1:
            audio_tensor = audio.float().unsqueeze(dim=0)
        elif audio.dim() == 2:
            audio_tensor = audio.float()
        else:
            raise ValueError("Audio tensor must be 1D or 2D.")
    if audio_tensor is None:
        raise ValueError("Audio must be a numpy array or a torch tensor.")
    feat = kaldi.fbank(audio_tensor, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = (
        onnx_session.run(
            None,
            {onnx_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()},
        )[0]
        .flatten()
        .tolist()
    )

    return embedding


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        dataset_path,
        dataset_valid_ratio,
        speaker_embedding_model_path,
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
        ds = load_dataset(self.hparams.dataset_path, split="train")
        ds = ds.train_test_split(test_size=self.hparams.dataset_valid_ratio)

        speaker_embedding_onnx_session = load_spk_embedding(self.hparams.speaker_embedding_model_path)

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
            speaker_embedding_onnx_session,
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
            speaker_embedding_onnx_session,
            self.hparams.skip_pos,
            self.hparams.skip_spk_emb,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
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
        speaker_embedding_onnx_session=None,
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
        self.speaker_embedding_onnx_session = speaker_embedding_onnx_session
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
        phone = row["phone"]
        tone = row["tones"]
        word_pos = row["word_pos"]
        syllable_pos = row["syllable_pos"]
        audio = row["audio"]["array"]
        audio_path = row["audio"]["path"]
        sr = row["audio"]["sampling_rate"]
        text, phone, tone, word_pos, syllable_pos = self.get_text(
            text, phone, tone, word_pos, syllable_pos, add_blank=self.add_blank, skip_pos=self.skip_pos
        )
        audio16k = audio
        audio22k = audio
        # sr resampling audio from 16k -> 22050
        if sr == 16_000:
            audio22k = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        elif sr == 22_050:
            audio16k = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        mel = self.get_mel(audio22k, self.sample_rate)
        spk_emb = None

        if not self.skip_spk_emb and self.speaker_embedding_onnx_session is not None:
            spk_emb_path = self.tmp_dir / "spk_emb" / (audio_path + ".pt")

            if spk_emb_path.exists():
                spk_emb = torch.load(spk_emb_path)
            else:
                spk_emb = get_spk_embedding(audio16k, self.speaker_embedding_onnx_session)
                torch.save(spk_emb, spk_emb_path)

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
        y_lengths, x_lengths = [], []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            y_, x_, tone_, word_pos_, syllable_pos_, spk_embed_ = (
                item["y"],
                item["x"],
                item["tone"],
                item["word_pos"],
                item["syllable_pos"],
                item["spk_emb"],
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
            "durations": durations if not torch.eq(durations, 0).all() else None,
        }
