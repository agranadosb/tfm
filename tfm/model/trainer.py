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
        weight_decay: float = 1e-3
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.accuracy = Accuracy(task="multiclass", num_classes=4)

        self.loss_fn_images = nn.MSELoss()
        self.loss_fn_movements = nn.MSELoss()
        self.confusion_matrix = np.zeros((4, 4))

    def forward(self, batch, batch_idx, step):
        x, y, m = batch

        m = m.to(torch.float32)
        y_pred, m_pred = self.model(x)

        images_loss = torch.zeros((1,)).to(self.device)
        movements_loss = torch.zeros((1,)).to(self.device)
        movement_selected = torch.zeros(m_pred.shape).to(self.device)
        for batch in range(m_pred.shape[0]):
            previous_loss = torch.full((1,), 1e6).to(self.device)
            previous_image_loss = torch.zeros(1).to(self.device)
            previous_movement_loss = torch.zeros(1).to(self.device)
            movement_selected_id = 0
            for i in range(4):
                if m[batch, i, -1] != 1:
                    image_loss = self.loss_fn_images(y_pred[batch], y[batch, :, i])
                    movement_loss = self.loss_fn_movements(m_pred[batch], m[batch, i, :-1])

                    total_loss = image_loss + movement_loss

                    if total_loss < previous_loss:
                        previous_loss = total_loss
                        previous_image_loss = image_loss
                        previous_movement_loss = movement_loss
                        movement_selected_id = i

            movement_selected[batch, movement_selected_id] = 1
            images_loss += previous_image_loss
            movements_loss += previous_movement_loss

        accuracy = self.accuracy(m_pred.argmax(1), movement_selected.argmax(1))
        confusion_matrix = metrics.confusion_matrix(
            m_pred.argmax(1).cpu(), movement_selected.argmax(1).cpu(), labels=list(range(4))
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
        self.plot_confusion_matrix("train")
        self.confusion_matrix = np.zeros((4, 4))

    def on_validation_epoch_end(self) -> None:
        self.plot_confusion_matrix("val")
        self.confusion_matrix = np.zeros((4, 4))

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer
