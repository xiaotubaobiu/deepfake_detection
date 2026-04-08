import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Encoder(nn.Module):
    """ResNet18 backbone that outputs 512d features.

    Removes the final FC layer. Face and context share weights within the same domain.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)

        Returns:
            (B, 512)
        """
        h = self.features(x)
        return h.flatten(1)
