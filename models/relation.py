import torch
import torch.nn as nn


class RelationModule(nn.Module):
    """Computes relation features from face and context embeddings.

    r = [z_face, z_context, |z_face - z_context|, z_face * z_context]
    """

    def __init__(self, feat_dim: int = 512):
        super().__init__()
        self.feat_dim = feat_dim
        self.out_dim = feat_dim * 4

    def forward(self, z_face: torch.Tensor, z_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_face: (B, feat_dim)
            z_context: (B, feat_dim)

        Returns:
            (B, feat_dim * 4)
        """
        return torch.cat([
            z_face,
            z_context,
            torch.abs(z_face - z_context),
            z_face * z_context,
        ], dim=1)
