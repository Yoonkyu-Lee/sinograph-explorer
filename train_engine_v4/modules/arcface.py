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
        """Math-parity refactor (doc/30 §2.4 plan E): only compute phi for the
        target column instead of full (N, C). 5 (N,C) buffers → 1 (N,C) + 3 (N,1).

        Memory: at batch=640, n_class=98169 the original allocates
            cos, sin, phi, one_hot, out  →  5 × 640×98169×fp32 ≈ 1.2 GB temp
        New path:
            cos                                           (N, C)
            cos_t, sin_t, phi_t (target column only)      (N, 1)
            scatter on cos clone → out                    (N, C)
        ≈ 240 MB instead.

        Numerically identical: gather→arithmetic→scatter has same fp32 ops as
        the mask-multiply path on the target column; non-target columns are
        bit-identical (cos value passes through unchanged).
        """
        # cos similarity matrix: (N, C). Caller's emb_norm should already be
        # L2-normalized; re-normalize the weight every step (anchors learn but
        # stay on unit sphere).
        W_norm = F.normalize(self.weight, dim=1)
        cos = F.linear(emb_norm, W_norm)                                # (N, C)
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)                        # safety

        # Target-column-only phi computation
        labels_2d = labels.view(-1, 1)                                  # (N, 1)
        cos_t = cos.gather(1, labels_2d)                                # (N, 1)
        sin_t = torch.sqrt(1.0 - cos_t.pow(2))                          # (N, 1)
        phi_t = cos_t * self._cos_m - sin_t * self._sin_m                # (N, 1)

        if self.easy_margin:
            phi_t = torch.where(cos_t > 0, phi_t, cos_t)
        else:
            phi_t = torch.where(cos_t > self._th, phi_t, cos_t - self._mm)

        # Replace target column in cos with phi_t. scatter is out-of-place
        # (autograd-friendly). Non-target columns pass through identically.
        # Under AMP autocast, math ops can promote phi_t to fp32; force back
        # to cos.dtype for scatter (which requires self.dtype == src.dtype).
        out = cos.scatter(1, labels_2d, phi_t.to(cos.dtype))            # (N, C)
        return out * self.s


def normalize_embedding(emb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize the embedding along the last dim. Convenience helper —
    use this in model.forward() right before passing to ArcMarginProduct."""
    return F.normalize(emb, p=2, dim=-1, eps=eps)
