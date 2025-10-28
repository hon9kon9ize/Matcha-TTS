import datetime as dt
import math
import random

import torch
import torch.nn.functional as F
import torch.nn as nn

import matcha.utils.monotonic_align as monotonic_align  # pylint: disable=consider-using-from-import
from matcha import utils
from matcha.models.baselightningmodule import BaseLightningClass
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)

log = utils.get_pylogger(__name__)


class MatchaTTS(BaseLightningClass):  # 🍵
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        n_feats,
        encoder,
        decoder,
        cfm,
        data_statistics,
        out_size,
        optimizer=None,
        scheduler=None,
        prior_loss=True,
        posterior_loss=False,
        diff_loss=True,
        dur_loss=True,
        use_precomputed_durations=False,
        freeze_decoder=False,
        freeze_encoder=False,
        pretrained_decoder_weight=None,
        pretrained_encoder_weight=None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.n_vocab = n_vocab
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.diff_loss = diff_loss
        self.dur_loss = dur_loss
        self.posterior_loss = posterior_loss
        self.use_precomputed_durations = use_precomputed_durations

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim=self.spk_emb_dim,
        )
        self.spk_embed_affine_layer = nn.Linear(self.spk_emb_dim, self.spk_emb_dim)
        self.lang_embed = nn.Embedding(3, self.spk_emb_dim)  # en, yue, zh

        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=self.spk_emb_dim,
        )

        if pretrained_decoder_weight is not None:
            log.info(f"Loading pretrained decoder weights from {pretrained_decoder_weight}")
            state_dict = torch.load(pretrained_decoder_weight, map_location="cpu")
            self.decoder.load_state_dict(state_dict, strict=False)

        if pretrained_encoder_weight is not None:
            log.info(f"Loading pretrained encoder weights from {pretrained_encoder_weight}")
            state_dict = torch.load(pretrained_encoder_weight, map_location="cpu")
            self.encoder.load_state_dict(state_dict, strict=False)

        if freeze_decoder:
            log.info("Freezing decoder and speaker embedding affine layer.")
            for param in self.decoder.parameters():
                param.requires_grad = False
            # Freeze speaker embedding affine layer if decoder is frozen
            for param in self.spk_embed_affine_layer.parameters():
                param.requires_grad = False

        if freeze_encoder:
            log.info("Freezing encoder.")
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.update_data_statistics(data_statistics)

    @torch.inference_mode()
    def synthesise(
        self,
        x,
        x_lengths,
        tone,
        word_pos,
        syllable_pos,
        n_timesteps,
        cond=None,
        spk_emb=None,
        lang=None,
        temperature=1.0,
        length_scale=1.0,
    ):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            tone (torch.Tensor): batch of tones.
                shape: (batch_size, max_text_length)
            word_pos (torch.Tensor): batch of word positions.
                shape: (batch_size, max_text_length)
            syllable_pos (torch.Tensor): batch of syllable positions.
                shape: (batch_size, max_text_length)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            spk_emb (bool, optional): speaker ids.
                shape: (batch_size, spk_emb_dim)
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
            prompt_feat (torch.Tensor, optional): prompt mel-spectrogram to condition the generation on.
                shape: (n_feats, prompt_length)

        Returns:
            dict: {
                "encoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Average mel spectrogram generated by the encoder
                "decoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Refined mel spectrogram improved by the CFM
                "attn": torch.Tensor, shape: (batch_size, max_text_length, max_mel_length),
                # Alignment map between text and mel spectrogram
                "mel": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Denormalized mel spectrogram
                "mel_lengths": torch.Tensor, shape: (batch_size,),
                # Lengths of mel spectrograms
                "rtf": float,
                # Real-time factor
            }
        """
        # For RTF computation
        t = dt.datetime.now()

        if spk_emb is not None:
            spk_emb = F.normalize(spk_emb, dim=1)
            spk_emb = self.spk_embed_affine_layer(spk_emb)

        if lang is not None:
            lang_emb = self.lang_embed(lang)
            if spk_emb is not None:
                spk_emb = spk_emb + lang_emb
            else:
                spk_emb = lang_emb

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, tone, word_pos, syllable_pos, spk_emb)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2).contiguous()  # [B, n_feats, max_mel_length]
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, spk_emb, cond)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }

    def forward(
        self,
        x,
        x_lengths,
        y,
        y_lengths,
        tone,
        word_pos,
        syllable_pos,
        spk_emb=None,
        out_size=None,
        cond=None,
        durations=None,
        lang=None,
    ):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotonic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            tone (torch.Tensor): batch of tones.
                shape: (batch_size, max_text_length)
            word_pos (torch.Tensor): batch of word positions.
                shape: (batch_size, max_text_length)
            syllable_pos (torch.Tensor): batch of syllable positions.
                shape: (batch_size, max_text_length)
            spk_emb (torch.Tensor, optional): speaker embeddings.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        if spk_emb is not None:
            spk_emb = F.normalize(spk_emb, dim=1)
            spk_emb = self.spk_embed_affine_layer(spk_emb)

        if lang is not None:
            lang_emb = self.lang_embed(lang)
            if spk_emb is not None:
                spk_emb = spk_emb + lang_emb
            else:
                spk_emb = lang_emb

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, tone, word_pos, syllable_pos, spk_emb)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        if self.use_precomputed_durations:
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y**2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()  # b, t_text, T_mel

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        # refered to as prior loss in the paper

        if self.dur_loss:
            logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
            dur_loss = duration_loss(logw, logw_, x_lengths)
        else:
            dur_loss = 0

        # Cut a small segment of mel-spectrogram in order to increase batch size
        #   - "Hack" taken from Grad-TTS, in case of Grad-TTS, we cannot train batch size 32 on a 24GB GPU without it
        #   - Do not need this hack for Matcha-TTS, but it works with it as well
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor(
                [torch.tensor(random.choice(range(start, end)) if end > start else 0) for start, end in offset_ranges]
            ).to(y_lengths)
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))  # [B, n_feats, T_mel]
        mu_y = mu_y.transpose(1, 2).contiguous()  # [B, T_mel, n_feats]

        # Compute loss of the decoder
        if self.diff_loss:
            diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spk_emb, cond=cond)
        else:
            diff_loss = 0

        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        return dur_loss, prior_loss, diff_loss, attn
