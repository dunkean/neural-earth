"""Local convolutional students: no spatial normalization or attention.

The base student predicts the final blended T=2 field. Coarse and decoder
students initially replace complete production windows, preserving their fusion.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class StudentConfig:
    stage: str = 'base'
    width: int = 96
    depth: int = 4
    dilations: tuple[int, ...] = (1, 2, 4)

    @property
    def in_channels(self):
        return dict(base=32, coarse=16, decoder=16)[self.stage]

    @property
    def out_channels(self):
        return dict(base=5, coarse=6, decoder=1)[self.stage]


class PixelNorm(nn.Module):
    """Only normalize channels of the same pixel; arbitrary tiles stay local."""
    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(1, keepdim=True) + 1e-6).to(x.dtype)


class Block(nn.Module):
    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.layers = nn.Sequential(
            PixelNorm(), nn.Conv2d(cin, cout, 3, padding=dilation, dilation=dilation), nn.SiLU(),
            PixelNorm(), nn.Conv2d(cout, cout, 3, padding=dilation, dilation=dilation), nn.SiLU())

    def forward(self, x):
        return (self.skip(x) + self.layers(x)) * (2 ** -.5)


class Student(nn.Module):
    def __init__(self, config=StudentConfig()):
        super().__init__()
        if config.stage not in ('base', 'coarse', 'decoder'):
            raise ValueError(config.stage)
        if config.width < 16 or config.width % 16 or not 2 <= config.depth <= 5:
            raise ValueError('Width must be a multiple of 16; depth must be 2–5.')
        self.config = config
        widths = [config.width * min(2 ** i, 2) for i in range(config.depth)]
        self.stem = nn.Conv2d(config.in_channels, widths[0], 3, padding=1)
        self.encoders = nn.ModuleList(Block(w, w) for w in widths)
        self.down = nn.ModuleList(nn.Conv2d(a, b, 3, stride=2, padding=1)
                                  for a, b in zip(widths, widths[1:]))
        self.middle = nn.Sequential(*(Block(widths[-1], widths[-1], d) for d in config.dilations))
        self.decoders = nn.ModuleList(Block(a + b, b) for a, b in zip(widths[:0:-1], widths[-2::-1]))
        self.head = nn.Conv2d(widths[0], config.out_channels, 1)
        # Predict a residual on deterministic noise. This aids fitting high frequencies.
        self.direct = nn.Conv2d(config.in_channels, config.out_channels, 1, bias=False)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.direct.weight)

    @property
    def alignment(self):
        return 2 ** (self.config.depth - 1)

    @property
    def halo(self):
        # Conservative support radius: all conv paths, including nearest upsampling
        # alignment uncertainty. Rounded to the conditioning grid for base.
        jump, radius = 1, 1  # stem
        for level in range(self.config.depth):
            radius += 2 * jump  # encoder's two 3x3 convolutions
            if level < self.config.depth - 1:
                radius += jump
                jump *= 2
        radius += sum(2 * d * jump for d in self.config.dilations)
        for _ in range(self.config.depth - 1):
            radius += jump  # conservative nearest-neighbour phase allowance
            jump //= 2
            radius += 2 * jump
        alignment = math.lcm(32 if self.config.stage == 'base' else 8, self.alignment)
        return math.ceil(radius / alignment) * alignment

    def forward(self, x):
        if x.shape[-1] % self.alignment or x.shape[-2] % self.alignment:
            raise ValueError(f'Input must align to {self.alignment}.')
        original = x
        x = self.stem(x)
        skips = []
        for level, block in enumerate(self.encoders):
            x = block(x)
            if level < len(self.down):
                skips.append(x)
                x = self.down[level](x)
        x = self.middle(x)
        for skip, block in zip(reversed(skips), self.decoders):
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x) + self.direct(original)


def defaults(stage, width=None):
    if stage == 'base':
        return StudentConfig(stage, width or 96, 4, (1, 2, 4))
    return StudentConfig(stage, width or 64, 3 if stage == 'coarse' else 4,
                         (1, 2, 4))


def load_student(path, device='cpu', ema=True):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config'].copy()
    config['dilations'] = tuple(config['dilations'])
    model = Student(StudentConfig(**config))
    model.load_state_dict(checkpoint['ema'] if ema and 'ema' in checkpoint else checkpoint['model'])
    return model.to(device).eval(), checkpoint
