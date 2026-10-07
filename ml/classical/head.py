"""Classical classifier head: Linear -> ReLU -> Dropout -> Linear(7). Softmax is applied at inference."""
from __future__ import annotations

import torch
from torch import nn

from ml.labels import NUM_CLASSES


class ClassicalHead(nn.Module):
    """Model A. Input: 2048-d ResNet50 features. Output: 7 logits."""

    def __init__(self, in_features: int = 2048, hidden: int = 256, dropout: float = 0.3,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_features, hidden), nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.out(self.dropout(self.proj(features)))


class ClassicalResNet50(nn.Module):
    """Backbone + head on raw images (used for fine-tuning and live inference)."""

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))
