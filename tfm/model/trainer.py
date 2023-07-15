import io
from typing import Any, Dict

import pytorch_lightning as pl
import numpy as np
import pandas as pd
import seaborn

import PIL
import wandb
from matplotlib import pyplot as plt
from ray import tune
from sklearn import metrics
import torch
from torch import nn, optim
from torchmetrics import Accuracy
from torchvision.transforms.functional import to_pil_image

from tfm.constants import LABEL_TO_MOVEMENT, LABEL_TO_STRING
from tfm.model.base import Unet

torch.set_float32_matmul_precision("medium")


class Trainer(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model = Unet(
            layers=config["layers"],
            movements_layers=config["movements_layers"],
            out_channels=config["out_channels"],
            movements=config["movements"],
            input_size=config["input_size"],
            input_channels=config["input_channels"],
            block=config["block"],
        )
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.accuracy = Accuracy(task="multiclass", num_classes=config["num_movement"])
        self.dataset_name = config["dataset"]

        self.loss_fn_images = nn.MSELoss(reduction="none")
        self.loss_fn_movements = nn.CrossEntropyLoss(reduction="none")
        self.num_movement = config["num_movement"]
        self.confusion_matrix = np.zeros((self.num_movement, self.num_movement))
        self.has_logger = config.get("log", True)
        self.is_hyp = config.get("hyp", False)

        self.val_loss = []
        self.val_acc = []

    def plot_samples(self, x, movements_selected, movements_predicted, step):
        iterable = zip(x[:4].cpu(), movements_selected, movements_predicted)
        self.logger.experiment.log(
            {  # type: ignore
                f"Samples/{step}-images": [
                    wandb.Image(
                        to_pil_image(i),
                        caption=(
                            f"Movement: {LABEL_TO_MOVEMENT[self.dataset_name][ms.item()]}, "
                            f"Predicted: {LABEL_TO_MOVEMENT[self.dataset_name][mp.item()]}"
                        ),
                    )
                    for i, ms, mp in iterable
                ],
            },
            commit=False,
        )

    def plot_confusion_matrix(self, step: str, commit: bool):
        class_names = LABEL_TO_STRING[self.dataset_name].values()

        dataframe = pd.DataFrame(
            self.confusion_matrix,
            index=class_names,  # type: ignore
            columns=class_names,  # type: ignore
        )

        seaborn.heatmap(dataframe, annot=True, cbar=None, cmap="YlGnBu")

        plt.title("Confusion Matrix"), plt.tight_layout()

        plt.ylabel("True Class"),
        plt.xlabel("Predicted Class")
        with io.BytesIO() as output:
            plt.savefig(output, format="png")
            im = wandb.Image(PIL.Image.open(output))  # type: ignore

            self.logger.experiment.log(  # type: ignore
                {f"Confusion/{step}-confusion-matrix": im}, commit=commit
            )
        plt.clf()

    def on_train_start(self):
        if not self.has_logger:
            return

        for key, value in self.config.items():
            self.logger.experiment.config[key] = value  # type: ignore

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.is_hyp:
            return optimizer

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=7, verbose=True
        )

        return [optimizer], {
            "scheduler": scheduler,
            "monitor": "ptl/val_loss",
        }

    def forward(self, batch, batch_idx, step):
        x, y, m = batch
        batch_indices = np.arange(x.size()[0])

        m = m.to(torch.float32)
        y_pred, m_pred = self.model(x)

        images_losses = self.loss_fn_images(
            y, y_pred.repeat([1, self.num_movement, 1, 1])
        ).mean(dim=(-1, -2))
        movements_losses = self.loss_fn_movements(
            m[..., : self.num_movement],
            m_pred.unsqueeze(dim=-1).repeat([1, 1, self.num_movement]),
        )

        # Set not allowed to max value
        total_losses = images_losses + movements_losses
        allowed = m[..., self.num_movement] == 1
        total_losses[allowed] = 1e6
        movements_selected = total_losses.argmin(dim=-1)
        y_selected = y[batch_indices, movements_selected]

        images_loss = images_losses[batch_indices, movements_selected].mean()
        movements_loss = movements_losses[batch_indices, movements_selected].mean()
        movements_predicted = m_pred.argmax(dim=-1)

        loss = images_loss + movements_loss

        accuracy = self.accuracy(movements_predicted, movements_selected)

        self.log_step(
            movements_predicted,
            movements_selected,
            accuracy,
            images_loss,
            movements_loss,
            x,
            y_pred,
            y_selected,
            step,
            batch_idx,
        )

        if step == "train":
            return loss
        return loss, accuracy

    def log_step(
        self,
        movements_predicted,
        movements_selected,
        accuracy,
        images_loss,
        movements_loss,
        x,
        y_pred,
        y_selected,
        step,
        batch_idx,
    ):
        if not self.has_logger:
            return

        confusion_matrix = metrics.confusion_matrix(
            movements_predicted.cpu(),
            movements_selected.cpu(),
            labels=list(range(self.num_movement)),
        )

        self.confusion_matrix += confusion_matrix

        self.log(f"Acc/{step}_accuracy", accuracy)
        self.log(f"Loss/{step}_image_loss", images_loss)
        self.log(f"Loss/{step}_movement_loss", movements_loss)

        if batch_idx == 0:
            self.plot_samples(
                x, movements_selected, movements_predicted, f"{step}-input"
            )
            self.plot_samples(y_pred, movements_selected, movements_predicted, step)
            self.plot_samples(
                y_selected, movements_selected, movements_predicted, f"{step}-correct"
            )

    def training_step(self, batch, batch_idx):
        return self.forward(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.forward(batch, batch_idx, "val")

    def on_validation_batch_end(self, out, *_):
        loss, accuracy = out
        self.val_loss.append(loss)
        self.val_acc.append(accuracy)

    def on_validation_epoch_start(self) -> None:
        if not self.has_logger:
            return

        if self.global_step != 0:
            self.plot_confusion_matrix("train", False)
        self.confusion_matrix = np.zeros((self.num_movement, self.num_movement))

    def on_validation_epoch_end(self) -> None:
        if self.has_logger:
            if self.global_step != 0:
                self.plot_confusion_matrix("val", True)
            self.confusion_matrix = np.zeros((self.num_movement, self.num_movement))

        # TODO: Refactor this way of calculating the average
        avg_loss = torch.tensor(self.val_loss).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", torch.tensor(self.val_acc).mean())

        if self.is_hyp:
            tune.report(val_loss=avg_loss.item())

        self.val_loss = []
        self.val_acc = []
