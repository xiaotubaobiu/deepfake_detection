import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss on relation embeddings."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (B, D) embeddings from view 1
            z2: (B, D) embeddings from view 2

        Returns:
            scalar loss
        """
        batch_size = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)

        similarity = torch.matmul(representations, representations.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=z1.device, dtype=torch.bool)
        similarity.masked_fill_(mask, -1e9)

        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z1.device)

        loss = F.cross_entropy(similarity, labels)
        return loss
