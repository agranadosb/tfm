import io
from typing import Any, Dict

import pytorch_lightning as pl
import numpy as np
import pandas as pd
import seaborn

import PIL
import wandb
from matplotlib import pyplot as plt
from sklearn import metrics
import torch
from torch import nn, optim
from torchmetrics import Accuracy
from torchvision.transforms.functional import to_pil_image

from tfm.constants import LABEL_TO_MOVEMENT
from tfm.model.base import Unet


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
            block=config["block"]
        )
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.accuracy = Accuracy(task="multiclass", num_classes=4)

        self.loss_fn_images = nn.MSELoss(reduction="none")
        self.loss_fn_movements = nn.MSELoss(reduction="none")
        self.num_movement = config["num_movement"]
        self.confusion_matrix = np.zeros((self.num_movement, self.num_movement))
        self.confusion_matrix_changes = 0

        self.val_loss = []
        self.val_acc = []

    def plot_samples(self, x, movements_selected, movements_predicted, step, commit=False):
        self.logger.experiment.log({
            f"{step}-images": [
                wandb.Image(
                    to_pil_image(i),
                    caption=(
                        f"Movement: {LABEL_TO_MOVEMENT[ms.item()]}, "
                        f"Predicted: {LABEL_TO_MOVEMENT[mp.item()]}"
                    )
                )
                for i, ms, mp in zip(x[:4].cpu(), movements_selected, movements_predicted)
            ]
        }, commit=commit)

    def forward(self, batch, batch_idx, step):
        x, y, m = batch

        m = m.to(torch.float32)
        y_pred, m_pred = self.model(x)

        images_losses = self.loss_fn_images(
            y, y_pred.repeat([1, self.num_movement, 1, 1])
        ).mean(dim=(-1, -2))
        movements_losses = self.loss_fn_movements(
            m[..., :self.num_movement], m_pred.unsqueeze(dim=-1).repeat([1, 1, self.num_movement])
        ).mean(dim=-1)
        total_losses = images_losses + movements_losses

        # Set not allowed to max value
        allowed = m[..., self.num_movement] == 1
        total_losses[allowed] = 1e6
        images_losses[allowed] = 1e6
        movements_losses[allowed] = 1e6

        images_loss = images_losses.min(dim=-1).values.mean()
        movements_loss = movements_losses.min(dim=-1).values.mean()
        movements_selected = total_losses.argmin(dim=-1)
        movements_predicted = m_pred.argmax(dim=-1)

        accuracy = self.accuracy(movements_predicted, movements_selected)
        confusion_matrix = metrics.confusion_matrix(
            movements_predicted.cpu(), movements_selected.cpu(), labels=list(range(4))
        )

        self.confusion_matrix += confusion_matrix

        self.log(f"{step}_accuracy", accuracy)
        self.log(f"{step}_image_loss", images_loss)
        self.log(f"{step}_movement_loss", movements_loss)

        if batch_idx == 0:
            self.plot_samples(
                x, movements_selected, movements_predicted, f"{step}-input", commit=False
            )
            self.plot_samples(
                y_pred, movements_selected, movements_predicted, f"{step}", commit=False
            )

        loss = images_loss + movements_loss
        if step == "train":
            return loss
        return loss, accuracy

    def on_validation_batch_end(self, out, *_):
        loss, accuracy = out
        self.val_loss.append(loss)
        self.val_acc.append(accuracy)

    def training_step(self, batch, batch_idx):
        return self.forward(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self.forward(batch, batch_idx, 'val')

    def plot_confusion_matrix(self, step: str, commit: bool):
        class_names = ("right", "left", "top", "bottom")

        dataframe = pd.DataFrame(self.confusion_matrix, index=class_names, columns=class_names)

        seaborn.heatmap(dataframe, annot=True, cbar=None, cmap="YlGnBu")

        plt.title("Confusion Matrix"), plt.tight_layout()

        plt.ylabel("True Class"),
        plt.xlabel("Predicted Class")
        with io.BytesIO() as output:
            plt.savefig(output, format="png")
            im = wandb.Image(PIL.Image.open(output))

            self.logger.experiment.log(
                {f"{step}-confusion-matrix": im}, commit=commit
            )
        plt.clf()

    def on_validation_epoch_start(self) -> None:
        if self.global_step != 0:
            self.plot_confusion_matrix("train", False)
        self.confusion_matrix = np.zeros((self.num_movement, self.num_movement))

    def on_validation_epoch_end(self) -> None:
        if self.global_step != 0:
            self.plot_confusion_matrix("val", True)
        self.confusion_matrix = np.zeros((self.num_movement, self.num_movement))
        self.confusion_matrix_changes += 100

        # TODO: Refactor this way of calculating the average
        avg_loss = torch.tensor(self.val_loss).mean()
        avg_acc = torch.tensor(self.val_acc).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

        self.val_loss = []
        self.val_acc = []

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    def on_train_start(self):
        for key, value in self.config.items():
            self.logger.experiment.config[key] = value
