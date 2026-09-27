"""Small callable layers. No torch.nn layer implementations are used."""

from __future__ import annotations

import math

import torch


class Linear:
    def __init__(self, fan_in: int, fan_out: int, *, bias: bool = True, generator=None):
        self.weight = (torch.randn(fan_in, fan_out, generator=generator) / math.sqrt(fan_in)).requires_grad_()
        self.bias = torch.zeros(fan_out, requires_grad=True) if bias else None
        self.out = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
    """Channels are the final axis: (B,C) or (B,T,C)."""

    def __init__(self, dim: int, *, eps: float = 1e-5, momentum: float = 0.001,
                 incorrect_3d: bool = False):
        self.gamma = torch.ones(dim, requires_grad=True)
        self.beta = torch.zeros(dim, requires_grad=True)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        self.eps = eps
        self.momentum = momentum
        self.incorrect_3d = incorrect_3d
        self.training = True
        self.out = None
        self.last_mean = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim not in (2, 3):
            raise ValueError(f"BatchNorm1d expects 2D or 3D input, got {tuple(x.shape)}")
        if self.training:
            # The intentional bug reduces only B for 3D input, leaving T in
            # the statistics. The corrected version reduces both B and T.
            axes = 0 if x.ndim == 2 or self.incorrect_3d else (0, 1)
            mean = x.mean(dim=axes, keepdim=True)
            var = x.var(dim=axes, keepdim=True, unbiased=False)
            self.last_mean = mean.detach()
            with torch.no_grad():
                # Squeeze B only for the bad case: one statistic per position.
                saved_mean = mean.squeeze(0) if x.ndim == 3 and self.incorrect_3d else mean.reshape(-1)
                saved_var = var.squeeze(0) if x.ndim == 3 and self.incorrect_3d else var.reshape(-1)
                if self.running_mean.shape != saved_mean.shape:
                    self.running_mean = torch.zeros_like(saved_mean)
                    self.running_var = torch.ones_like(saved_var)
                self.running_mean.lerp_(saved_mean, self.momentum)
                self.running_var.lerp_(saved_var, self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        self.out = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]


class Tanh:
    def __init__(self):
        self.out = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.out = torch.tanh(x)
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return []


class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int, *, generator=None):
        self.weight = torch.randn(num_embeddings, embedding_dim, generator=generator).requires_grad_()
        self.out = None

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        self.out = self.weight[indices]
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight]


class Flatten:
    """Merge consecutive positions; n=8 is the ordinary flat MLP."""

    def __init__(self, n: int):
        self.n = n
        self.out = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] % self.n:
            raise ValueError(f"Cannot merge {self.n} positions in shape {tuple(x.shape)}")
        batch, time, channels = x.shape
        result = x.reshape(batch, time // self.n, channels * self.n)
        self.out = result.squeeze(1) if result.shape[1] == 1 else result
        return self.out

    def parameters(self) -> list[torch.Tensor]:
        return []


class Sequential:
    def __init__(self, layers: list):
        self.layers = layers
        self.out = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return x

    def parameters(self) -> list[torch.Tensor]:
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def train(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True

    def eval(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False
