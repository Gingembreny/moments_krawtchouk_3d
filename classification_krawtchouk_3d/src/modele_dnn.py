from __future__ import annotations

import torch
from torch import nn


class DNNMomentsTchebichef(nn.Module):
    """
    Réseau fully-connected inspiré de l'article.

    Input -> 110 -> 175 -> 240 -> 175 -> 125 -> Softmax implicite via CrossEntropyLoss.
    """

    def __init__(self, dimension_entree: int, nombre_classes: int, dropout: float = 0.15):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(dimension_entree, 110),
            nn.BatchNorm1d(110),
            nn.ELU(),
            nn.Dropout(dropout),

            nn.Linear(110, 175),
            nn.BatchNorm1d(175),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(175, 240),
            nn.BatchNorm1d(240),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(240, 175),
            nn.BatchNorm1d(175),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(175, 125),
            nn.BatchNorm1d(125),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(125, nombre_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reseau(x)
