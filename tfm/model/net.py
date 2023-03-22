from typing import List, Tuple, Callable

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.relu(self.conv(x)))


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, mid_channels)
        self.conv2 = ConvBlock(mid_channels, out_channels)
        self.aux_conv = ConvBlock(in_channels, out_channels)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.apply_aux = in_channels != out_channels

    def forward(self, x):
        out = self.relu2(self.conv2(self.relu1(self.conv1(x))))
        if self.apply_aux:
            x = self.aux_conv(x)
        return x + out


class Base(nn.Module):
    def __init__(self, blocks: List[Tuple[int, int]], layer_function: Callable):
        super().__init__()
        self.layer_function = layer_function
        self.layers = [
            self.get_layer(in_channels, out_channels)
            for in_channels, out_channels in blocks
        ]

    def to(self, device):
        super().to(device)
        for layer in self.layers:
            for component in layer:
                component.to(device)


class ResNetEncoder(Base):
    def get_layer(
        self, in_channels: int, out_channels
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        return (
            self.layer_function(in_channels, out_channels),
            self.layer_function(out_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        partial_outs = [None] * len(self.layers)
        for index, (conv_1, conv_2, pool) in enumerate(self.layers):
            x = conv_2(conv_1(x))
            partial_outs[index] = x
            x = pool(x)
        return x, partial_outs  # noqa


class ResNetDecoder(Base):
    def __init__(self, blocks: List[Tuple[int, int]], layer_function: Callable, batched: bool = True):
        super().__init__(blocks, layer_function)

        self.cat_dim = 0
        if batched:
            self.cat_dim = 1

    def get_layer(
        self, in_channels: int, out_channels
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        return (
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            self.layer_function(in_channels, out_channels),
            self.layer_function(out_channels, out_channels),
        )

    def forward(
        self, x: torch.Tensor, encoder_outs: List[torch.Tensor]
    ) -> torch.Tensor:
        for encoder_out, (transpose, conv_1, conv_2) in zip(encoder_outs, self.layers):
            cat_result = torch.cat([encoder_out, transpose(x)], dim=self.cat_dim)
            x = conv_2(conv_1(cat_result))
        return x


class GeneralResnetUnet(nn.Module):
    def __init__(
        self,
        encoder_blocks: List[Tuple[int, int]],
        decoder_blocks: List[Tuple[int, int]],
        encoder_layer_function: Callable,
        decoder_layer_function: Callable,
        batched: bool = True,
    ):
        super().__init__()

        self.encoder = ResNetEncoder(encoder_blocks, encoder_layer_function)
        self.decoder = ResNetDecoder(decoder_blocks, decoder_layer_function, batched)

        in_embedding = encoder_blocks[-1][1]
        out_embedding = decoder_blocks[0][0]

        self.embedding_conv1 = ResNetBlock(in_embedding, in_embedding, out_embedding)
        self.embedding_conv2 = ResNetBlock(out_embedding, out_embedding, out_embedding)

        self.max_pool = nn.MaxPool2d(4)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(2048, 64)
        self.relu = nn.ReLU()

        self.prediction = nn.Linear(64, 4)
        self.softmax = nn.Softmax(dim=1)

        self.final_conv = ResNetBlock(64, 64, 1)

    def to(self, device):
        super().to(device)
        self.encoder.to(device)
        self.decoder.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x, encoder_outs = self.encoder(x)
        encoder_outs.reverse()

        x = self.embedding_conv2(self.embedding_conv1(x))

        movements_out = self.relu(self.linear(self.flatten(self.max_pool(x))))
        movements_out = self.softmax(self.prediction(movements_out))

        x = self.decoder(x, encoder_outs)

        return self.final_conv(x), movements_out
