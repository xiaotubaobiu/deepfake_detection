import torch
import torch.nn as nn

from models.encoder import ResNet18Encoder
from models.relation import RelationModule


class DualDomainDetector(nn.Module):
    """Dual-domain face-context relation detector.

    Modes:
      - 'rgb':     M1 — RGB only, single-domain relation, CE loss
      - 'freq':    M2 — Freq only, single-domain relation, CE loss
      - 'dual':    M3 — RGB + Freq dual-domain fusion, CE loss
      - 'dual_cl': M4 — RGB + Freq dual-domain fusion, CE + InfoNCE loss
    """

    def __init__(self, mode: str = 'dual_cl', feat_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.mode = mode

        use_rgb = mode in ('rgb', 'dual', 'dual_cl')
        use_freq = mode in ('freq', 'dual', 'dual_cl')
        self.use_rgb = use_rgb
        self.use_freq = use_freq

        if use_rgb:
            self.encoder_rgb = ResNet18Encoder(pretrained=pretrained)
        if use_freq:
            self.encoder_freq = ResNet18Encoder(pretrained=pretrained)

        self.relation = RelationModule(feat_dim=feat_dim)

        n_domains = int(use_rgb) + int(use_freq)
        relation_dim = feat_dim * 4 * n_domains

        self.classifier = nn.Sequential(
            nn.Linear(relation_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

        self.use_contrastive = mode == 'dual_cl'
        if self.use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(relation_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )

    def forward(self, batch: dict, return_features: bool = False):
        """
        Args:
            batch: dict with keys 'face_rgb', 'context_rgb', 'face_freq', 'context_freq'
            return_features: if True, return dict with logits, relation, projection

        Returns:
            logits (B, 2) if return_features=False
            dict with 'logits', 'relation', 'projection' if return_features=True
        """
        relations = []

        if self.use_rgb:
            z_face_rgb = self.encoder_rgb(batch['face_rgb'])
            z_ctx_rgb = self.encoder_rgb(batch['context_rgb'])
            r_rgb = self.relation(z_face_rgb, z_ctx_rgb)
            relations.append(r_rgb)

        if self.use_freq:
            z_face_freq = self.encoder_freq(batch['face_freq'])
            z_ctx_freq = self.encoder_freq(batch['context_freq'])
            r_freq = self.relation(z_face_freq, z_ctx_freq)
            relations.append(r_freq)

        r = torch.cat(relations, dim=1)
        logits = self.classifier(r)

        if return_features:
            result = {
                'logits': logits,
                'relation': r,
            }
            if self.use_contrastive:
                result['projection'] = self.projection_head(r)
            return result

        return logits
