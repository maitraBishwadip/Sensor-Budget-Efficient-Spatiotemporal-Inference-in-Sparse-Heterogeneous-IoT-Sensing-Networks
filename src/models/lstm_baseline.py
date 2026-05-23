"""LSTM baseline that replicates the B.Tech-thesis architecture.

Per-station forecaster: input is a [history, F] sequence per station; output
is a single-step PM2.5 prediction. Used as both the source-only and the TL
baseline so the numbers map onto thesis Tables 5.1 / 5.2.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """A small, station-agnostic LSTM that ingests [B, H, F] and emits [B, 1].

    The model itself is station-blind — when used on a city, we apply it
    independently to every station's sequence and concatenate the outputs.
    This matches the thesis pipeline exactly.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, F] -> [B, 1]"""
        out, _ = self.lstm(x)
        last = out[:, -1, :]            # [B, hidden_dim]
        return self.head(last)          # [B, 1]
