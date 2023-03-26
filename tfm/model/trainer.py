from dataclasses import dataclass
from typing import Tuple, Optional, List

import torch
import torchvision
import tqdm as tqdm
from torch import nn, optim, Tensor
from torch.cuda.amp import GradScaler
from torchmetrics.functional import accuracy, precision, recall
from torchvision import transforms

from tfm.constants import LABEL_TO_MOVEMENT
from tfm.data.puzzle import Puzzle8MnistGenerator
from tfm.model.net import (
    GeneralResnetUnet,
    ResNetBlock,
    ConvBlock,
    ResNetEncoder,
    ResNetDecoder,
    MovementsNet,
)
from tfm.plots.images import plot_images


@dataclass
class Configuration:
    batch_size: int
    batches: int
    show_eval_samples: Optional[int] = 3
    shuffle: bool = True


@dataclass
class LossItem:
    transitions: Tensor
    predictions: Tensor
    movements: Tensor
    predictions_movements: Tensor


class UnetMultiModalTrainer:
    def __init__(self, conf: Configuration):
        self.conf = conf
        self.current_epoch = 0
        self.epochs = 0
        self.step_name = "none"

        self.puzzle = Puzzle8MnistGenerator(order=[1, 2, 3, 8, 0, 4, 7, 6, 5])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder_blocks = [(1, 64), (64, 128), (128, 256)]
        self.decoder_blocks = [(512, 256), (256, 128), (128, 64)]
        self.conv_movements_blocks = [
            (512, 512),
            (512, 256),
            (256, 256),
            (256, 128),
            (128, 128),
            (128, 64),
            (64, 32),
        ]
        # 32 * 8 * 8
        self.dense_movements_blocks = [(2048, 512), (512, 64), (64, 8)]

        self.encoder_layer = lambda in_channels, out_channels: ConvBlock(
            in_channels, out_channels
        )
        self.decoder_layer = lambda in_channels, out_channels: ConvBlock(
            in_channels, out_channels
        )

        self.encoder_layer = lambda in_channels, out_channels: ResNetBlock(
            in_channels, out_channels, out_channels
        )
        self.decoder_layer = lambda in_channels, out_channels: ResNetBlock(
            in_channels, in_channels, out_channels
        )

        self.conv_movements_layer = lambda in_channels, out_channels: ResNetBlock(
            in_channels, out_channels, out_channels
        )

        self.encoder_net = ResNetEncoder(self.encoder_blocks, self.encoder_layer)
        self.decoder_net = ResNetDecoder(self.decoder_blocks, self.decoder_layer, True)
        self.movements_net = MovementsNet(
            self.conv_movements_blocks,
            self.dense_movements_blocks,
            self.conv_movements_layer,
        )

        self.model = GeneralResnetUnet(
            self.encoder_net, self.decoder_net, self.movements_net
        )
        self.image_criterion = nn.MSELoss()
        self.movements_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.losses = []
        self.image_losses = []
        self.movements_losses = []

        self.val_losses = []
        self.val_image_losses = []
        self.val_movements_losses = []

        self.total_loss = torch.Tensor()
        self.total_images_loss = torch.Tensor()
        self.total_movements_loss = torch.Tensor()
        self.total_accuracy = torch.Tensor()
        self.total_precision = torch.Tensor()
        self.total_recall = torch.Tensor()

    def samples(self) -> Tuple[Tensor, Tensor, Tensor]:
        images, transitions, movements = self.puzzle.get_batch(self.conf.batch_size)
        images = torchvision.transforms.Resize((64, 64))(images)
        transitions = torchvision.transforms.Resize((64, 64))(transitions)

        images = images.to(self.device)
        transitions = transitions.to(self.device)
        movements = movements.to(self.device).to(torch.float32)

        return images, transitions, movements

    def loss(
        self,
        loss_item: LossItem,
        weights: Tuple[float, float, float] = (0.75, 0.25, 10.0),
    ) -> Tuple[Tensor, Tensor, Tensor]:
        images_loss = self.image_criterion(loss_item.transitions, loss_item.predictions)
        movements_loss = self.movements_criterion(
            loss_item.movements, loss_item.predictions_movements
        )

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

        loss_item = LossItem(transitions, predictions, movements, predictions_movements)
        loss, images_loss, movements_loss = self.loss(loss_item)

        # TODO: Memory leak
        self.total_loss += loss.detach()
        self.total_images_loss += images_loss.detach()
        self.total_movements_loss += movements_loss.detach()

        self.total_accuracy += batch_accuracy.detach()
        self.total_precision += batch_precision.detach()
        self.total_recall += batch_recall.detach()

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
        scaler = GradScaler()
        with tqdm.tqdm(total=self.conf.batches) as progress_bar:
            for batch in range(1, self.conf.batches + 1):
                with torch.cuda.amp.autocast():
                    loss, _, _ = self.step(batch, progress_bar)

                scaler.scale(loss).backward()
                if (batch + 1) % 2 == 0:
                    scaler.step(self.optimizer)

                    scaler.update()

                    self.optimizer.zero_grad(set_to_none=True)

        self.losses[self.current_epoch] = self.total_loss.detach().item()
        self.image_losses[self.current_epoch] = self.total_images_loss.detach().item()
        self.movements_losses[self.current_epoch] = self.total_movements_loss.detach().item()

    def eval(self):
        self.create_metrics()
        self.step_name = "eval"

        batches = self.conf.batches // 2
        with tqdm.tqdm(total=batches) as progress_bar:
            for batch in range(1, batches + 1):
                images, transitions, movements = self.samples()

                with torch.no_grad():
                    loss, predictions, predictions_movements = self.step(
                        batch, progress_bar
                    )

                    if self.conf.show_eval_samples and False:
                        size = self.conf.show_eval_samples, 3
                        results: List[Tensor] = [  # noqa
                            None for _ in range(self.conf.show_eval_samples * 3)
                        ]
                        titles: List[str] = [
                            "" for _ in range(self.conf.show_eval_samples * 3)
                        ]

                        for i in range(self.conf.show_eval_samples):
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

        self.val_losses[self.current_epoch] = self.total_loss.detach().item()
        self.val_image_losses[self.current_epoch] = self.total_images_loss.detach().item()
        self.val_movements_losses[self.current_epoch] = self.total_movements_loss.detach().item()

    def save_results(self, filename: str, losses: List[float], images_loss: List[float], movements_loss: List[int]):
        with open(filename, "w") as file:
            file.write("epoch,loss,image_loss,movement_loss\n")
            for index, (loss, image_loss, movements_loss) in enumerate(zip(losses, images_loss, movements_loss)):
                file.write(f"{index},{loss},{image_loss},{movements_loss}")

    def train(self, epochs: int):
        self.model.train()
        self.model.to(self.device)
        self.epochs = epochs

        self.losses = [-1 for _ in range(epochs)]
        self.image_losses = self.losses.copy()
        self.movements_losses = self.losses.copy()

        self.val_losses = self.losses.copy()
        self.val_image_losses = self.losses.copy()
        self.val_movements_losses = self.losses.copy()

        for epoch in range(epochs):
            self.current_epoch = epoch
            self.epoch()
            self.eval()

        torch.save(self.model.state_dict(), "model")
        self.save_results("train-results.txt", self.losses, self.image_losses, self.movements_losses)
        self.save_results("val-results.txt", self.val_losses, self.val_image_losses, self.val_movements_losses)
