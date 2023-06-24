from typing import List

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel: int = 3,
        padding: int = 1,
        activation: nn.Module = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.GELU()
        self.conv = nn.Conv2d(in_features, out_features, kernel, padding=padding)
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = activation
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, kernel: int = 3, padding: int = 1
    ):
        super().__init__()
        self.conv1 = ConvBlock(in_features, out_features, kernel, padding)
        self.conv2 = ConvBlock(out_features, out_features, kernel, padding)

        self.skip = nn.Identity()
        if in_features != out_features:
            self.skip = ConvBlock(in_features, out_features, kernel, padding)

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + skip
        return x


class UnetEncBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module):
        super().__init__()
        self.input = block(in_features, out_features)
        self.mid = block(out_features, out_features)
        self.out = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.input(x)
        y = self.mid(x)
        x = self.out(y)
        return x, y


class UnetDecBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, block: nn.Module):
        super().__init__()
        self.input = block(in_features * 2, in_features)
        self.mid = block(in_features, in_features)
        self.out = nn.ConvTranspose2d(in_features, out_features, 2, 2)

    @staticmethod
    def crop(x, shape: torch.Size):
        """
        Function for cropping an image tensor: Given an image tensor and the new shape,
        crops to the center pixels (assumes that the input's size and the new size are
        even numbers).

        Parameters
        ----------
        x: Tensor
            image tensor of shape (batch size, channels, height, width).
        shape: Size
            a torch.Size object with the shape you want x to have.
        """
        _, _, h, w = x.shape
        _, _, h_new, w_new = shape

        ch, cw = h // 2, w // 2
        ch_new, cw_new = h_new // 2, w_new // 2
        x1 = int(cw - cw_new)
        y1 = int(ch - ch_new)
        x2 = int(x1 + w_new)
        y2 = int(y1 + h_new)
        return x[:, :, y1:y2, x1:x2]

    def forward(self, x, skip):
        if x.shape != skip.shape:
            skip = self.crop(skip, x.shape)
        x = torch.cat([x, skip], dim=1)
        x = self.input(x)
        x = self.mid(x)
        x = self.out(x)
        return x


class UnetEmbeddingBlock(nn.Module):
    def __init__(
        self, in_features: int, mid_features: int, out_features: int, block: nn.Module
    ):
        super().__init__()
        self.input = block(in_features, out_features)
        self.mid = block(out_features, mid_features)
        self.out = nn.ConvTranspose2d(mid_features, out_features, 2, 2)

    def forward(self, x):
        x = self.input(x)
        y = self.mid(x)
        x = self.out(y)
        return x, y


class UnetOutBlock(UnetDecBlock):
    def __init__(self, in_features: int, out_features: int, block: nn.Module):
        super().__init__(in_features, out_features, block)
        self.out = ConvBlock(
            in_features, out_features, 1, padding=0, activation=nn.ReLU()
        )


class UnetEncoder(nn.Module):
    def __init__(self, layers: List[int], block: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UnetEncBlock(in_features, out_features, block)
                for in_features, out_features in zip(layers, layers[1:])
            ]
        )

    def forward(self, x):
        features = []
        for block in self.blocks:
            x, y = block(x)
            features.append(y)
        return x, features


class UnetDecoder(nn.Module):
    def __init__(self, layers: List[int], block: nn.Module):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UnetDecBlock(in_features, out_features, block)
                for in_features, out_features in zip(layers, layers[1:])
            ]
        )

    def forward(self, x, features):
        for block, y in zip(self.blocks, features[::-1]):
            x = block(x, y)
        return x


class MovementsNet(nn.Module):
    def __init__(self, layers: List[int], movements: int, mid_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            *[
                ConvBlock(in_features, out_features, padding=1)
                for in_features, out_features in zip(layers, layers[1:])
            ]
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(mid_features, movements)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x


class Unet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        movements_layers: List[int],
        out_channels: int,
        movements: int,
        input_size: int,
        input_channels: int,
        block: nn.Module,
    ):
        super().__init__()
        layers = list(layers)
        movements_layers = list(movements_layers)
        # TODO: Document this way of compute the layers
        movements_mid_layer = (input_size // 2 ** len(layers)) ** 2 * movements_layers[
            -1
        ]

        layers = [input_channels] + layers
        movements_layers = [layers[-1] * 2] + movements_layers
        embedding_dim = layers[-1] * 2
        decoder_layers = layers[: -len(layers) : -1]

        self.encoder = UnetEncoder(layers, block)
        self.embedding = UnetEmbeddingBlock(
            layers[-1], embedding_dim, layers[-1], block
        )
        self.decoder = UnetDecoder(decoder_layers, block)

        self.out = UnetOutBlock(decoder_layers[-1], out_channels, block)

        self.movements = MovementsNet(movements_layers, movements, movements_mid_layer)

    def forward(self, x):
        x, features = self.encoder(x)
        x, embeddings = self.embedding(x)
        x = self.decoder(x, features[1:])

        x = self.out(x, features[0])

        return x, self.movements(embeddings)
