#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2025 Wen-Chin Huang (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Speaker embedding extractor based on SpeechBrain."""

import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier


class SpeechBrainSpkEmbExtractor:
    def __init__(self, device="cpu"):
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device}
        )

    def forward(
        self,
        audio_input,
        sr: int = 16000,
    ) -> np.array:
        if isinstance(audio_input, str):
            signal, fs = torchaudio.load(audio_input)
        elif isinstance(audio_input, np.ndarray):
            signal = torch.from_numpy(audio_input).float().unsqueeze(0)
            fs = sr
        else:
            raise ValueError("audio_input must be a file path (str) or numpy array")

        embeddings = self.classifier.encode_batch(signal)

        return embeddings.cpu().numpy().reshape(-1)
