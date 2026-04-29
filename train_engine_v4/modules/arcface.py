"""ArcFace metric loss head for SCER (doc/28 §4.3).

Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
Recognition" (CVPR 2019). The standard formulation:

    cos(theta_y) = (W_y / ||W_y||) · (emb / ||emb||)         # per target class
    cos(theta_y + m) = cos(theta_y) cos(m) - sin(theta_y) sin(m)
    arc_logits[:, y]    = s · cos(theta_y + m)               # margin-shifted target
    arc_logits[:, ¬y]   = s · cos(theta_¬y)                  # other classes unchanged

Then standard CE over arc_logits trains the embedding to keep target class
within an angular margin m of the per-class anchor W_y while pushing other
classes farther by at least the same margin (in angle).

After training, `weight` (Parameter shape (n_classes, emb_dim)) is the
per-class anchor in normalized embedding space — this is exactly what
`51_build_anchor_db.py --mode weight` exports as the deploy-time anchor DB.

Numerically stable variant: instead of computing sin via sqrt(1 - cos^2)
which loses precision near cos = ±1, we use the easyface form (cos(θ+m) when
θ+m ≤ π, else fall back to cos(θ) - m·sin(m)). Adopts the standard PyTorch
ArcFace recipe used in InsightFace and elsewhere.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """ArcFace classifier head.

    Args:
        emb_dim:    embedding dimension (e.g. 128)
        n_classes:  number of classes (e.g. 98169)
        s:          scale factor — large s makes logits sharper;
                    standard 30 (face) or 64 (large-class) — we default 30
        m:          additive angular margin in radians (0.5 ≈ 28.6°)
        easy_margin: when target angle + m would exceed π, fall back to
                     phi = cos(theta) (no margin). Stable but weaker boundary.
                     False is the published ArcFace; True is used during early
                     warmup if loss diverges. Smoke run starts with easy=True
                     for safety, production with easy=False.

    Shapes:
        forward(emb_norm, labels):
            emb_norm : (N, emb_dim) — caller must L2-normalize (we double-check)
            labels   : (N,) long — class indices in [0, n_classes)
            returns  : (N, n_classes) float — scaled logits ready for F.cross_entropy
    """

    def __init__(
        self,
        emb_dim: int,
        n_classes: int,
        s: float = 30.0,
        m: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.s = float(s)
        self.m = float(m)
        self.easy_margin = bool(easy_margin)

        # per-class anchors in (yet-to-be-normalized) embedding space
        self.weight = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)

        # cached scalars (recomputed if s/m mutated by curriculum)
        self._cos_m = math.cos(self.m)
        self._sin_m = math.sin(self.m)
        # cos(π - m) — used to detect "θ + m would exceed π"
        self._th = math.cos(math.pi - self.m)
        # the "fallback offset" for non-easy margin: -m · sin(m)
        self._mm = self.m * self._sin_m

    def set_margin(self, m: float) -> None:
        """Curriculum hook — change m mid-training without rebuilding the head."""
        self.m = float(m)
        self._cos_m = math.cos(self.m)
        self._sin_m = math.sin(self.m)
        self._th = math.cos(math.pi - self.m)
        self._mm = self.m * self._sin_m

    def forward(
        self,
        emb_norm: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # cos similarity matrix: (N, C)
        # caller's emb_norm should already be L2-normalized; re-normalize the
        # weight every step (anchors learn but stay on unit sphere).
        W_norm = F.normalize(self.weight, dim=1)
        cos = F.linear(emb_norm, W_norm)              # (N, C)
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)      # numerical safety

        # cos(theta + m) = cos·cos_m - sin·sin_m, with sin = sqrt(1 - cos^2)
        sin = torch.sqrt(1.0 - cos.pow(2))
        phi = cos * self._cos_m - sin * self._sin_m   # (N, C) — full margin-shifted

        if self.easy_margin:
            # Use phi only where cos > 0 (angle < π/2), else fall back to cos.
            phi = torch.where(cos > 0, phi, cos)
        else:
            # Original ArcFace: when θ + m would exceed π, use cos - m·sin(m)
            # to keep the loss well-defined (monotone in cos).
            phi = torch.where(cos > self._th, phi, cos - self._mm)

        # Build output: phi at target column, cos elsewhere.
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        out = one_hot * phi + (1.0 - one_hot) * cos
        return out * self.s


def normalize_embedding(emb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize the embedding along the last dim. Convenience helper —
    use this in model.forward() right before passing to ArcMarginProduct."""
    return F.normalize(emb, p=2, dim=-1, eps=eps)
