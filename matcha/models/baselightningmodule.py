"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""

import inspect
from abc import ABC
from typing import Any, Dict, Optional
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from matcha import utils
from matcha.utils.utils import plot_tensor

log = utils.get_pylogger(__name__)

# Vocos imports
from vocos import Vocos


class BaseLightningClass(LightningModule, ABC):
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            scheduler.last_epoch = current_epoch
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler.lightning_args.interval,
                    "frequency": self.hparams.scheduler.lightning_args.frequency,
                    "name": "learning_rate",
                },
            }

        return {"optimizer": optimizer}

    @property
    def logger_type(self):
        logger = self.logger
        if logger is None:
            return "none"
        elif isinstance(logger, WandbLogger):
            return "wandb"
        elif isinstance(self.logger, TensorBoardLogger):
            return "tb"
        else:
            return "other"

    def log_image(self, name: str, img, step: Optional[int] = None):
        if self.trainer.logger is not None:
            if self.logger_type == "tb":
                if img.shape[2] <= 4:
                    # WHC -> CWH
                    img = np.transpose(img, (2, 0, 1))
                self.trainer.logger.experiment.add_image(name, img, self.trainer.global_step)
            elif self.logger_type == "wandb":
                # if img is a list
                if isinstance(img, list):
                    img = img
                else:
                    img = [img]

                self.trainer.logger.log_image(key=name, images=img)
                # self.trainer.logger.experiment.log({name: wandb.Image(img)})

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        tone = batch["tone"]
        word_pos = batch["word_pos"]
        syllable_pos = batch["syllable_pos"]
        spk_emb = batch["spk_emb"]
        lang = batch["lang"]

        dur_loss, prior_loss, diff_loss, *_ = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            tone=tone,
            word_pos=word_pos,
            syllable_pos=syllable_pos,
            spk_emb=spk_emb,
            out_size=self.out_size,
            durations=batch["durations"],
            lang=lang,
        )

        return {
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "sub_loss/train_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        self.log(
            "sub_loss/val_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/val",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    y = y[:, : one_batch["y_lengths"][i]]
                    self.log_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                    )

            log.debug("Synthesising...")
            vocoder = Vocos.from_pretrained("BSC-LT/vocos-mel-22khz")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                tone = one_batch["tone"][i].unsqueeze(0).to(self.device)
                word_pos = one_batch["word_pos"][i].unsqueeze(0).to(self.device)
                syllable_pos = one_batch["syllable_pos"][i].unsqueeze(0).to(self.device)
                spk_emb = (
                    one_batch["spk_emb"][i].unsqueeze(0).to(self.device) if one_batch["spk_emb"] is not None else None
                )
                lang = one_batch["lang"][i].unsqueeze(0).to(self.device)
                output = self.synthesise(
                    x[:, :x_lengths],
                    x_lengths,
                    tone=tone[:, :x_lengths],
                    word_pos=word_pos[:, :x_lengths],
                    syllable_pos=syllable_pos[:, :x_lengths],
                    n_timesteps=10,
                    spk_emb=spk_emb,
                    lang=lang,
                )
                y_enc, y_dec, mel = output["encoder_outputs"], output["decoder_outputs"], output["mel"]
                attn = output["attn"]
                self.log_image(
                    f"generated_enc/{i}",
                    plot_tensor(y_enc.squeeze().cpu()),
                )
                self.log_image(
                    f"generated_dec/{i}",
                    plot_tensor(y_dec.squeeze().cpu()),
                )
                self.log_image(
                    f"alignment/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                )

                vocoder = vocoder.to(self.device)
                y_hat_audio = vocoder.decode(mel)

                self.logger.experiment.add_audio(
                    f"audio_{i}_lang_{lang.item()}",
                    y_hat_audio.squeeze().cpu(),
                    self.global_step,
                    22050,  # sample rate
                )

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})
