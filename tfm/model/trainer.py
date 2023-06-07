import lightning.pytorch as pl
import numpy as np
import pandas as pd
import seaborn
import torchvision
from matplotlib import pyplot as plt
from sklearn import metrics
import torch
from torch import nn, optim
from torchmetrics import Accuracy

from tfm.model.base import Unet


class Trainer(pl.LightningModule):
    def __init__(
        self,
        model: Unet,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        num_movement: int = 4,
    ):
        # TODO: Integrate mlflow as logger instead of tensorboard
        # https://mlflow.org/docs/latest/index.html
        # TODO: Use ray-tune for hyperparameter tuning
        # https://docs.ray.io/en/latest/tune/examples/tune-pytorch-lightning.html
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.accuracy = Accuracy(task="multiclass", num_classes=4)

        self.loss_fn_images = nn.MSELoss(reduction="none")
        self.loss_fn_movements = nn.MSELoss(reduction="none")
        self.num_movement = num_movement
        self.confusion_matrix = np.zeros((num_movement, num_movement))

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

        img_grid = torchvision.utils.make_grid(
            x, 16, normalize=True, scale_each=True, padding=2, pad_value=1.0,
        )
        self.logger.experiment.add_image(f"{step}-input-images", img_grid, self.global_step)
        img_grid = torchvision.utils.make_grid(
            y_pred, 16, normalize=True, scale_each=True, padding=2, pad_value=1.0,
        )
        self.logger.experiment.add_image(f"{step}-images", img_grid, self.global_step)

        return images_loss + movements_loss

    def training_step(self, batch, batch_idx):
        return self.forward(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        self.forward(batch, batch_idx, 'val')

    def plot_confusion_matrix(self, step: str):
        class_names = ("right", "left", "top", "bottom")

        dataframe = pd.DataFrame(self.confusion_matrix, index=class_names, columns=class_names)

        a = plt.figure(figsize=(8, 6))

        # Create heatmap
        seaborn.heatmap(dataframe, annot=True, cbar=None, cmap="YlGnBu")

        plt.title("Confusion Matrix"), plt.tight_layout()

        plt.ylabel("True Class"),
        plt.xlabel("Predicted Class")
        self.logger.experiment.add_figure(
            f"{step}-confusion-matrix", a, self.global_step
        )

    def on_train_epoch_end(self) -> None:
        # TODO: Check why the confusion matrix is not plotted
        self.plot_confusion_matrix("train")
        self.confusion_matrix = np.zeros((self.num_movement, self.num_movement))

    def on_validation_epoch_end(self) -> None:
        self.plot_confusion_matrix("val")
        self.confusion_matrix = np.zeros((self.num_movement, self.num_movement))

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
