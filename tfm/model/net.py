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


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, batched: bool = True):
        super().__init__()

        self.cat_dim = 0
        if batched:
            self.cat_dim = 1

        self.encoder_block1_conv1 = ResNetBlock(in_channels, 64, 64)
        self.encoder_block1_conv2 = ResNetBlock(64, 64, 64)
        self.encoder_block1_pool = nn.MaxPool2d(2)

        self.encoder_block2_conv1 = ResNetBlock(64, 128, 128)
        self.encoder_block2_conv2 = ResNetBlock(128, 128, 128)
        self.encoder_block2_pool = nn.MaxPool2d(2)

        self.encoder_block3_conv1 = ResNetBlock(128, 256, 256)
        self.encoder_block3_conv2 = ResNetBlock(256, 256, 256)
        self.encoder_block3_pool = nn.MaxPool2d(2)

        self.embedding_conv1 = ResNetBlock(256, 256, 512)
        self.embedding_conv2 = ResNetBlock(512, 512, 512)

        self.decoder_block1_upconv = nn.ConvTranspose2d(512, 256, 2, 2)
        self.decoder_block1_conv1 = ResNetBlock(512, 512, 256)
        self.decoder_block1_conv2 = ResNetBlock(256, 256, 256)

        self.decoder_block2_upconv = nn.ConvTranspose2d(256, 128, 2, 2)
        self.decoder_block2_conv1 = ResNetBlock(256, 256, 128)
        self.decoder_block2_conv2 = ResNetBlock(128, 128, 128)

        self.decoder_block3_upconv = nn.ConvTranspose2d(128, 64, 2, 2)
        self.decoder_block3_conv1 = ResNetBlock(128, 128, 64)
        self.decoder_block3_conv2 = ResNetBlock(64, 64, 64)

        self.final_conv = ResNetBlock(64, 64, 1)

    def forward(self, x: torch.Tensor):
        encoder_out_1 = self.encoder_block1_conv2(self.encoder_block1_conv1(x))
        encoder_out_2 = self.encoder_block2_conv2(
            self.encoder_block2_conv1(self.encoder_block1_pool(encoder_out_1))
        )
        encoder_out_3 = self.encoder_block3_conv2(
            self.encoder_block3_conv1(self.encoder_block2_pool(encoder_out_2))
        )

        embedding_out = self.embedding_conv2(
            self.embedding_conv1(self.encoder_block2_pool(encoder_out_3))
        )

        decoder_out_1 = self.decoder_block1_conv2(
            self.decoder_block1_conv1(
                torch.cat(
                    [encoder_out_3, self.decoder_block1_upconv(embedding_out)],
                    dim=self.cat_dim,
                )
            )
        )
        decoder_out_2 = self.decoder_block2_conv2(
            self.decoder_block2_conv1(
                torch.cat(
                    [encoder_out_2, self.decoder_block2_upconv(decoder_out_1)],
                    dim=self.cat_dim,
                )
            )
        )
        decoder_out_3 = self.decoder_block3_conv2(
            self.decoder_block3_conv1(
                torch.cat(
                    [encoder_out_1, self.decoder_block3_upconv(decoder_out_2)],
                    dim=self.cat_dim,
                )
            )
        )

        return self.final_conv(decoder_out_3)


class MultiModelUnet(UNet):
    def __init__(self, in_channels: int = 3, batched: bool = True):
        super().__init__(in_channels, batched)

        self.max_pool = nn.MaxPool2d(4)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(2048, 64)
        self.relu = nn.ReLU()

        self.prediction = nn.Linear(64, 4)
        self.softmax = nn.Softmax()

    def forward(self, x: torch.Tensor):
        encoder_out_1 = self.encoder_block1_conv2(self.encoder_block1_conv1(x))
        encoder_out_2 = self.encoder_block2_conv2(
            self.encoder_block2_conv1(self.encoder_block1_pool(encoder_out_1))
        )
        encoder_out_3 = self.encoder_block3_conv2(
            self.encoder_block3_conv1(self.encoder_block2_pool(encoder_out_2))
        )

        embedding_out = self.embedding_conv2(
            self.embedding_conv1(self.encoder_block2_pool(encoder_out_3))
        )

        movements_out = self.relu(self.linear(self.flatten(self.max_pool(embedding_out))))

        decoder_out_1 = self.decoder_block1_conv2(
            self.decoder_block1_conv1(
                torch.cat(
                    [encoder_out_3, self.decoder_block1_upconv(embedding_out)],
                    dim=self.cat_dim,
                )
            )
        )
        decoder_out_2 = self.decoder_block2_conv2(
            self.decoder_block2_conv1(
                torch.cat(
                    [encoder_out_2, self.decoder_block2_upconv(decoder_out_1)],
                    dim=self.cat_dim,
                )
            )
        )
        decoder_out_3 = self.decoder_block3_conv2(
            self.decoder_block3_conv1(
                torch.cat(
                    [encoder_out_1, self.decoder_block3_upconv(decoder_out_2)],
                    dim=self.cat_dim,
                )
            )
        )

        return self.final_conv(decoder_out_3), self.softmax(self.prediction(movements_out))
