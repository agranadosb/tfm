from typing import Tuple, Optional, List

import torch
import torchvision
import tqdm as tqdm
from torch import nn, optim, Tensor
from torchmetrics.functional import accuracy, precision, recall
from torchvision import transforms

from tfm.constants import LABEL_TO_MOVEMENT
from tfm.data.puzzle import Puzzle8MnistGenerator
from tfm.model.net import MultiModelUnet
from tfm.plots.images import plot_images


class UnetMultiModalTrainer:
    def __init__(
        self, batch_size: int, batches: int, show_eval_samples: Optional[int] = 3
    ):
        self.batch_size = batch_size
        self.batches = batches
        self.show_eval_samples = show_eval_samples
        self.current_epoch = 0
        self.epochs = 0
        self.step_name = "none"

        self.puzzle = Puzzle8MnistGenerator(order=[1, 2, 3, 8, 0, 4, 7, 6, 5])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MultiModelUnet(1)
        self.image_criterion = nn.MSELoss()
        self.movements_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.losses = torch.Tensor()
        self.image_losses = torch.Tensor()
        self.movements_losses = torch.Tensor()

        self.val_losses = torch.Tensor()
        self.val_image_losses = torch.Tensor()
        self.val_movements_losses = torch.Tensor()

        self.total_loss = torch.Tensor()
        self.total_images_loss = torch.Tensor()
        self.total_movements_loss = torch.Tensor()
        self.total_accuracy = torch.Tensor()
        self.total_precision = torch.Tensor()
        self.total_recall = torch.Tensor()

    def samples(self) -> Tuple[Tensor, Tensor, Tensor]:
        images, transitions, movements = self.puzzle.get_batch(self.batch_size)
        images = torchvision.transforms.Resize((64, 64))(images)
        transitions = torchvision.transforms.Resize((64, 64))(transitions)

        images = images.to(self.device)
        transitions = transitions.to(self.device)
        movements = movements.to(self.device).to(torch.float32)

        return images, transitions, movements

    def loss(
        self,
        transitions: Tensor,
        predictions: Tensor,
        movements: Tensor,
        predictions_movements: Tensor,
        weights: Tuple[float, float, float] = (0.75, 0.25, 10.0),
    ) -> Tuple[Tensor, Tensor, Tensor]:
        images_loss = self.image_criterion(transitions, predictions)
        movements_loss = self.movements_criterion(movements, predictions_movements)

        total_loss = images_loss + movements_loss

        normalized_images_loss = (images_loss / total_loss) * weights[0] * weights[2]
        normalized_movements_loss = (
            (movements_loss / total_loss) * weights[1] * weights[2]
        )

        return (
            normalized_images_loss + normalized_movements_loss,
            normalized_images_loss,
            normalized_movements_loss,
        )

    def update_progress(self, progress_bar: tqdm.tqdm, message: str):
        progress_bar.set_description(
            f"{self.model.__class__.__name__} "
            f"{self.device} -> "
            f"Epoch {self.current_epoch:3d}/{self.epochs:3d} "
            f"{message}"
        )
        progress_bar.update(1)

    def movements_metrics(
        self, predictions_movements: Tensor, movements: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        one_hot_prediction_movements = torch.zeros(predictions_movements.size()).to(
            self.device
        )
        one_hot_prediction_movements[:, torch.argmax(predictions_movements, dim=1)] = 1

        batch_accuracy = accuracy(one_hot_prediction_movements, movements, "binary")
        batch_precision = precision(one_hot_prediction_movements, movements, "binary")
        batch_recall = recall(one_hot_prediction_movements, movements, "binary")

        return batch_accuracy, batch_precision, batch_recall

    def create_metrics(self):
        self.total_loss = torch.zeros(1).to(self.device)
        self.total_images_loss = torch.zeros(1).to(self.device)
        self.total_movements_loss = torch.zeros(1).to(self.device)

        self.total_accuracy = torch.zeros(1).to(self.device)
        self.total_precision = torch.zeros(1).to(self.device)
        self.total_recall = torch.zeros(1).to(self.device)

    def step(
        self,
        batch: int,
        progress_bar: tqdm.tqdm,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        images, transitions, movements = self.samples()

        predictions, predictions_movements = self.model(images)

        batch_accuracy, batch_precision, batch_recall = self.movements_metrics(
            predictions_movements, movements
        )

        loss, images_loss, movements_loss = self.loss(
            transitions, predictions, movements, predictions_movements
        )

        self.total_loss += loss
        self.total_images_loss += images_loss
        self.total_movements_loss += movements_loss

        self.total_accuracy += batch_accuracy
        self.total_precision += batch_precision
        self.total_recall += batch_recall

        batch = torch.Tensor([batch]).to(self.device)
        message = (
            f"{self.step_name} loss {(self.total_loss / batch).item():3.5f} "
            f"images loss {(self.total_images_loss / batch).item():3.5f} "
            f"movements metrics: loss {(self.total_movements_loss / batch).item():3.5f} "
            f"accuracy {(self.total_accuracy / batch).item():2.4%} "
            f"precision {(self.total_precision / batch).item():.4f} "
            f"recall {(self.total_recall / batch).item():.4f}"
        )
        self.update_progress(progress_bar, message)

        return loss, predictions, predictions_movements

    def epoch(self):
        self.create_metrics()
        self.step_name = "train"
        with tqdm.tqdm(total=self.batches) as progress_bar:
            for batch in range(1, self.batches + 1):
                self.optimizer.zero_grad()

                loss, _, _ = self.step(batch, progress_bar)

                loss.backward()

                self.optimizer.step()

        self.losses[self.current_epoch] = self.total_loss
        self.image_losses[self.current_epoch] = self.total_images_loss
        self.movements_losses[self.current_epoch] = self.total_movements_loss

    def eval(self):
        self.create_metrics()
        self.step_name = "eval"

        batches = self.batches // 2
        with tqdm.tqdm(total=batches) as progress_bar:
            for batch in range(1, batches + 1):
                images, transitions, movements = self.samples()

                with torch.no_grad():
                    loss, predictions, predictions_movements = self.step(
                        batch, progress_bar
                    )

                    if self.show_eval_samples:
                        size = self.show_eval_samples, 3
                        results: List[Tensor] = [  # noqa
                            None for _ in range(self.show_eval_samples * 3)
                        ]
                        titles: List[str] = [
                            "" for _ in range(self.show_eval_samples * 3)
                        ]

                        for i in range(self.show_eval_samples):
                            results[3 * i] = images[i]
                            results[3 * i + 1] = predictions[i]
                            results[3 * i + 2] = transitions[i]

                            titles[3 * i] = ""
                            titles[3 * i + 1] = LABEL_TO_MOVEMENT[
                                torch.argmax(predictions_movements[i]).item()
                            ]
                            titles[3 * i + 2] = LABEL_TO_MOVEMENT[
                                torch.argmax(movements[i]).item()
                            ]

                        plot_images(
                            [transforms.ToPILImage()(image) for image in results[:9]],
                            size=size,
                            titles=titles,
                        )

        self.val_losses[self.current_epoch] = self.total_loss
        self.val_image_losses[self.current_epoch] = self.total_images_loss
        self.val_movements_losses[self.current_epoch] = self.total_movements_loss

    def train(self, epochs: int):
        self.model.train()
        self.model.to(self.device)
        self.epochs = epochs

        self.losses = torch.zeros(epochs)
        self.image_losses = torch.zeros(epochs)
        self.movements_losses = torch.zeros(epochs)

        self.val_losses = torch.zeros(epochs)
        self.val_image_losses = torch.zeros(epochs)
        self.val_movements_losses = torch.zeros(epochs)

        for epoch in range(epochs):
            self.current_epoch = epoch
            self.epoch()
            self.eval()
