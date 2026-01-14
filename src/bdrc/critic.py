# src/bdrc/critic.py

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class CriticConfig:
    state_dim: int = 12
    hidden_sizes: Sequence[int] = (64, 64)


class ValueNetwork(nn.Module):
    def __init__(self, cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg

        layers = []
        last_dim = cfg.state_dim
        for h in cfg.hidden_sizes:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (B, state_dim)

        Returns:
            values: (B,) 标量 value
        """
        v = self.net(states)
        return v.squeeze(-1)
