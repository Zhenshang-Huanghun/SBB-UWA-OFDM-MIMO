import math

import torch
from torch import nn


def _groups(channels):
    return next(n for n in range(min(8, channels), 0, -1) if channels % n == 0)


class TimeEmbedding(nn.Module):
    def __init__(self, total_steps, width, embedding_dim):
        super().__init__()
        frequency = torch.exp(-torch.arange(0, width, 2).float() / width * math.log(10000))
        angles = torch.arange(total_steps).float()[:, None] * frequency[None]
        table = torch.zeros(total_steps, width)
        table[:, 0::2], table[:, 1::2] = angles.sin(), angles.cos()
        self.layers = nn.Sequential(
            nn.Embedding.from_pretrained(table, freeze=False),
            nn.Linear(width, embedding_dim), nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim))

    def forward(self, t):
        return self.layers(t)


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout=0.0):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(_groups(in_channels), in_channels), nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, 3, padding=1))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.block2 = nn.Sequential(
            nn.GroupNorm(_groups(out_channels), out_channels), nn.SiLU(),
            nn.Dropout(dropout), nn.Conv1d(out_channels, out_channels, 3, padding=1))
        self.shortcut = (nn.Identity() if in_channels == out_channels else
                         nn.Conv1d(in_channels, out_channels, 1))

    def forward(self, x, embedding):
        h = self.block1(x) + self.time_projection(embedding)[:, :, None]
        return self.block2(h) + self.shortcut(x)


class DDPMUNet(nn.Module):
    def __init__(self, total_steps=400, in_channels=8, out_channels=8,
                 base_channels=32, dropout=0.0):
        super().__init__()
        if base_channels < 2 or base_channels % 2:
            raise ValueError("base_channels must be a positive even integer.")
        self.total_steps = total_steps
        self.in_channels = in_channels
        self.out_channels = out_channels
        time_dim = 4 * base_channels
        self.time_embedding = TimeEmbedding(total_steps, base_channels, time_dim)
        widths = [base_channels, 2 * base_channels, 4 * base_channels]
        self.input_conv = nn.Conv1d(in_channels, widths[0], 3, padding=1)
        self.encoder_blocks, self.downsamples = nn.ModuleList(), nn.ModuleList()
        for level, width in enumerate(widths):
            self.encoder_blocks.append(ResBlock1D(width, width, time_dim, dropout))
            if level < len(widths) - 1:
                self.downsamples.append(nn.Conv1d(width, widths[level + 1], 4, stride=2, padding=1))
        self.middle = ResBlock1D(widths[-1], widths[-1], time_dim, dropout)
        self.upsamples, self.decoder_blocks = nn.ModuleList(), nn.ModuleList()
        for level in range(len(widths) - 2, -1, -1):
            self.upsamples.append(nn.ConvTranspose1d(widths[level + 1], widths[level],
                                                     4, stride=2, padding=1))
            self.decoder_blocks.append(ResBlock1D(2 * widths[level], widths[level], time_dim, dropout))
        self.output = nn.Sequential(
            nn.GroupNorm(_groups(widths[0]), widths[0]), nn.SiLU(),
            nn.Conv1d(widths[0], out_channels, 3, padding=1))

    def forward(self, x, t):
        if x.ndim != 3 or x.shape[1] != self.in_channels or x.shape[-1] % 4:
            raise ValueError("Expected (B, in_channels, K), with K divisible by four.")
        embedding = self.time_embedding(t)
        h, skips = self.input_conv(x), []
        for level, block in enumerate(self.encoder_blocks):
            h = block(h, embedding)
            if level < len(self.downsamples):
                skips.append(h)
                h = self.downsamples[level](h)
        h = self.middle(h, embedding)
        for up, block in zip(self.upsamples, self.decoder_blocks):
            h = block(torch.cat((up(h), skips.pop()), dim=1), embedding)
        return self.output(h)


class BridgeDenoiser(nn.Module):
    def __init__(self, total_steps=100, channels=8, base_channels=32, dropout=0.0):
        super().__init__()
        self.total_steps = total_steps
        self.net = DDPMUNet(total_steps, 2 * channels, channels, base_channels, dropout)

    def forward(self, x_t, h_ls, t):
        return self.net(torch.cat((x_t, h_ls), dim=1), t)
