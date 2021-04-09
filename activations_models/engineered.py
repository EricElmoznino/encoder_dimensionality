import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math
from typing import List
from .base import Model


class CurvatureFiltersModel(Model):

    def __init__(self, n_filts=180, **kwargs):
        super(CurvatureFiltersModel, self).__init__(pool_map={'logits': 32},
                                                    **kwargs)

        assert n_filts % 15 == 0
        self.n_filts = n_filts

        self.n_ories = self.n_filts // 15
        self.curves = np.logspace(-2, -0.1, 15)
        self.gau_sizes = (5,)
        self.filt_size = 9
        self.fre = 1.2
        self.gamma = 1
        self.sigx = 1
        self.sigy = 1

        # Construct filters
        i = 0
        ories = np.arange(0, 2 * np.pi, 2 * np.pi / self.n_ories)
        w = torch.zeros(size=(len(ories) * len(self.curves) * len(self.gau_sizes),
                              1, self.filt_size, self.filt_size))
        for curve in self.curves:
            for gau_size in self.gau_sizes:
                for orie in ories:
                    w[i, 0, :, :] = banana_filter(gau_size, self.fre, orie, curve,
                                                  self.gamma, self.sigx, self.sigy,
                                                  self.filt_size)
                    i += 1
        self.weight = nn.Parameter(w)

    def forward(self, x):
        feats = F.conv2d(x, weight=self.weight,
                         padding=math.floor(self.filt_size / 2))
        feats = feats.abs()
        return feats

    def base_name(self) -> str:
        return f'curvature filters (n={self.n_filts})'

    def model_type(self) -> str:
        return 'engineered'

    def layers(self) -> List[str]:
        return ['logits']


class EdgeFiltersModel(Model):

    def __init__(self, n_filts=180, **kwargs):
        super(EdgeFiltersModel, self).__init__(pool_map={'logits': 32},
                                               **kwargs)

        self.n_filts = n_filts

        self.n_ories = self.n_filts
        self.curves = np.logspace(-2, -0.1, 15)
        self.gau_sizes = (5,)
        self.filt_size = 9
        self.fre = 1.2
        self.gamma = 1
        self.sigx = 1
        self.sigy = 1

        # Construct filters
        i = 0
        ories = np.arange(0, np.pi, np.pi / self.n_ories)
        w = torch.zeros(size=(len(ories) * len(self.gau_sizes),
                              1, self.filt_size, self.filt_size))
        for gau_size in self.gau_sizes:
            for orie in ories:
                w[i, 0, :, :] = banana_filter(gau_size, self.fre, orie, 0, self.gamma,
                                              self.sigx, self.sigy, self.filt_size)
                i += 1
        self.weight = nn.Parameter(w)

    def forward(self, x):
        feats = F.conv2d(x, weight=self.weight,
                         padding=math.floor(self.filt_size / 2))
        feats = feats.abs()
        return feats

    def base_name(self) -> str:
        return f'edge filters (n={self.n_filts})'

    def model_type(self) -> str:
        return 'engineered'

    def layers(self) -> List[str]:
        return ['logits']


class RandomFiltersModel(Model):

    def __init__(self, n_filts=180, **kwargs):
        super(RandomFiltersModel, self).__init__(pool_map={'logits': 32},
                                                 **kwargs)

        self.n_filts = n_filts

        self.filt_size = 9

        # Construct filters
        torch.manual_seed(27)
        w = torch.rand(n_filts, 1, self.filt_size, self.filt_size)
        w -= w.mean(dim=[2, 3], keepdim=True)
        self.weight = nn.Parameter(w)

    def forward(self, x):
        feats = F.conv2d(x, weight=self.weight,
                         padding=math.floor(self.filt_size / 2))
        feats = feats.abs()
        return feats

    def base_name(self) -> str:
        return f'random filters (n={self.n_filts})'

    def model_type(self) -> str:
        return 'engineered'

    def layers(self) -> List[str]:
        return ['logits']


class RawPixelsModel(Model):

    def forward(self, x):
        return x

    def base_name(self) -> str:
        return f'raw pixels'

    def model_type(self) -> str:
        return 'engineered'

    def layers(self) -> List[str]:
        return ['logits']


def banana_filter(s, fre, theta, cur, gamma, sigx, sigy, sz):
    # Define a matrix that used as a filter
    xv, yv = np.meshgrid(np.arange(np.fix(-sz / 2).item(), np.fix(sz / 2).item() + sz % 2),
                         np.arange(np.fix(sz / 2).item(), np.fix(-sz / 2).item() - sz % 2, -1))
    xv = xv.T
    yv = yv.T

    # Define orientation of the filter
    xc = xv * np.cos(theta) + yv * np.sin(theta)
    xs = -xv * np.sin(theta) + yv * np.cos(theta)

    # Define the bias term
    bias = np.exp(-sigx / 2)
    k = xc + cur * (xs ** 2)

    # Define the rotated Guassian rotated and curved function
    k2 = (k / sigx) ** 2 + (xs / (sigy * s)) ** 2
    G = np.exp(-k2 * fre ** 2 / 2)

    # Define the rotated and curved complex wave function
    F = np.exp(fre * k * 1j)

    # Multiply the complex wave function with the Gaussian function with a constant and bias
    filt = gamma * G * (F - bias)
    filt = np.real(filt)
    filt -= filt.mean()

    filt = torch.from_numpy(filt).float()
    return filt
