from typing import Tuple, Optional, List

import torch
import torchvision
import tqdm as tqdm
from torch import nn, optim, Tensor
from torchvision import transforms

from tfm.constants import LABEL_TO_MOVEMENT
from tfm.data.puzzle import Puzzle8MnistGenerator
from tfm.model.net import MultiModelUnet
from tfm.plots.images import plot_images


class UnetMultiModalTrainer:
    def __init__(self, batch_size: int, batches: int, show_eval_samples: Optional[int] = 3):
        self.batch_size = batch_size
        self.batches = batches
        self.show_eval_samples = show_eval_samples
        self.current_epoch = 0
        self.epochs = 0

        self.puzzle = Puzzle8MnistGenerator(order=[1, 2, 3, 8, 0, 4, 7, 6, 5])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MultiModelUnet(1)
        self.image_criterion = nn.MSELoss()
        self.movements_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

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
        weights: Tuple[float, float, float] = (0.75, 0.25, 10.0)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        images_loss = self.image_criterion(transitions, predictions)
        movements_loss = self.movements_criterion(movements, predictions_movements)

        total_loss = images_loss + movements_loss

        normalized_images_loss = (images_loss / total_loss) * weights[0] * weights[2]
        normalized_movements_loss = (movements_loss / total_loss) * weights[1] * weights[2]

        return normalized_images_loss + normalized_movements_loss, normalized_images_loss, normalized_movements_loss

    def update_progress(self, progress_bar: tqdm.tqdm, message: str):
        progress_bar.set_description(f"{self.model.__class__.__name__} -> {message}")
        progress_bar.update(1)

    def epoch(self):
        total_loss = torch.zeros(1).to(self.device)
        total_images_loss = torch.zeros(1).to(self.device)
        total_movements_loss = torch.zeros(1).to(self.device)
        with tqdm.tqdm(total=self.batches) as progress_bar:
            for batch in range(self.batches):
                self.optimizer.zero_grad()

                images, transitions, movements = self.samples()

                predictions, predictions_movements = self.model(images)

                loss, images_loss, movements_loss = self.loss(
                    transitions, predictions, movements, predictions_movements
                )

                total_loss += loss
                total_images_loss += images_loss
                total_movements_loss += movements_loss

                loss.backward()

                message = (
                    f"Epoch {self.current_epoch:3d}/{self.epochs:3d} "
                    f"train loss {(total_loss / self.batches).item():3.5f} "
                    f"images loss {(total_images_loss / self.batches).item():3.5f} "
                    f"movements loss {(total_movements_loss / self.batches).item():3.5f}"
                )
                self.update_progress(progress_bar, message)

                self.optimizer.step()

    def eval(self):
        total_loss = torch.zeros(1).to(self.device)
        total_images_loss = torch.zeros(1).to(self.device)
        total_movements_loss = torch.zeros(1).to(self.device)
        batches = self.batches // 2
        with tqdm.tqdm(total=batches) as progress_bar:
            for batch in range(batches):
                images, transitions, movements = self.samples()

                with torch.no_grad():
                    predictions, predictions_movements = self.model(images)
                    loss, images_loss, movements_loss = self.loss(
                        transitions, predictions, movements, predictions_movements
                    )

                    total_loss += loss
                    total_images_loss += images_loss
                    total_movements_loss += movements_loss

                    message = (
                        f"Epoch {self.current_epoch:3d}/{self.epochs:3d} "
                        f"eval loss {(total_loss / batches).item():3.5f} "
                        f"images loss {(total_images_loss / batches).item():3.5f} "
                        f"movements loss {(total_movements_loss / batches).item():3.5f}"
                    )
                    self.update_progress(progress_bar, message)

                    if self.show_eval_samples:
                        size = self.show_eval_samples, 3
                        results: List[Tensor] = [None for i in range(self.show_eval_samples * 3)]  # noqa
                        titles: List[str] = ["" for i in range(self.show_eval_samples * 3)]

                        for i in range(self.show_eval_samples):
                            results[3 * i] = images[i]
                            results[3 * i + 1] = predictions[i]
                            results[3 * i + 2] = transitions[i]

                            titles[3 * i] = ""
                            titles[3 * i + 1] = LABEL_TO_MOVEMENT[torch.argmax(predictions_movements[i]).item()]
                            titles[3 * i + 2] = LABEL_TO_MOVEMENT[torch.argmax(movements[i]).item()]

                        plot_images(
                            [transforms.ToPILImage()(image) for image in results[:9]],
                            size=size,
                            titles=titles
                        )

    def train(self, epochs: int):
        self.model.train()
        self.model.to(self.device)
        self.epochs = epochs

        for epoch in range(epochs):
            self.current_epoch = epoch
            self.epoch()
            self.eval()
