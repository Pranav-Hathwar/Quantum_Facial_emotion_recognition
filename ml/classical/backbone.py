"""Pretrained ResNet50 backbone (Chapter 12 step c): ImageNet transfer learning.

feature_extractor : every backbone layer frozen (BatchNorm kept in eval) -> pure feature extractor
fine_tuning       : layer4 (default) or the whole network is unfrozen and trained with a low LR
The ImageNet classifier (fc) is removed: the backbone outputs the 2048-d pooled feature vector.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

FEATURE_DIM = 2048
_WEIGHTS = {"v1": "IMAGENET1K_V1", "v2": "IMAGENET1K_V2"}


class ResNet50Backbone(nn.Module):
    feature_dim = FEATURE_DIM

    def __init__(self, pretrained: bool = True, mode: str = "feature_extractor",
                 weights: str = "v1", unfreeze: str = "layer4"):
        super().__init__()
        if mode not in {"feature_extractor", "fine_tuning"}:
            raise ValueError("mode must be 'feature_extractor' or 'fine_tuning'")
        if unfreeze not in {"layer4", "all"}:
            raise ValueError("unfreeze must be 'layer4' or 'all'")
        w = models.ResNet50_Weights[_WEIGHTS[weights]] if pretrained else None
        net = models.resnet50(weights=w)
        net.fc = nn.Identity()
        self.net, self.mode, self.pretrained, self.weights_name = net, mode, pretrained, weights
        self._set_trainable(mode, unfreeze)

    def _set_trainable(self, mode: str, unfreeze: str) -> None:
        for p in self.net.parameters():
            p.requires_grad = False
        if mode == "fine_tuning":
            target = self.net if unfreeze == "all" else self.net.layer4
            for p in target.parameters():
                p.requires_grad = True

    @property
    def is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.parameters())

    def train(self, mode: bool = True):  # frozen backbone must keep BatchNorm running statistics fixed
        super().train(mode and not self.is_frozen)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_frozen:
            with torch.no_grad():
                return self.net(x)
        return self.net(x)


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))
