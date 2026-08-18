"""Models.

FLamby's baseline is an ImageNet-pretrained EfficientNet-B0 with the final
fully-connected layer replaced by an 8-way head. We use torchvision's
EfficientNet-B0 (same architecture, no extra dependency).

Three uses:
  - build_finetune_model(): full fine-tuning (Phase 4+).
  - Backbone(): frozen feature extractor for Phase 1 caching (1280-d penultimate).
  - LinearProbe(): logistic head trained on cached features (Phases 2-3).
"""

import torch
from torch import nn

NUM_CLASSES = 8
FEATURE_DIM = 1280  # EfficientNet-B0 penultimate width


def build_finetune_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


class Backbone(nn.Module):
    """Frozen EfficientNet-B0 feature extractor: images -> 1280-d vectors."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_b0(weights=weights)
        self.features = m.features
        self.avgpool = m.avgpool
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(self.features(x)), 1)


class LinearProbe(nn.Module):
    """Logistic regression head over cached features."""

    def __init__(self, in_dim: int = FEATURE_DIM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
