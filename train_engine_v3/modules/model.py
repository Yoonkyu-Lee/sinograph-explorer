"""Multi-head ResNet-18 for Level A structure-aware training.

Architecture (doc/19 §3):

    image (N,3,H,W) → resnet18 backbone → 512-d feature (GAP)
        ├── char_head           Linear(512 → n_class)   CE   w=1.0
        ├── radical_head        Linear(512 → 214)       CE   w=0.2
        ├── total_strokes_head  Linear(512 → 1)         MSE  w=0.1
        ├── residual_head       Linear(512 → 1)         MSE  w=0.1
        └── idc_head            Linear(512 → 12)        CE   w=0.2

Backbone is torchvision resnet18 with the final `fc` removed; 512-d output
is broadcast to all heads. Input is (N, 3, H, W) — H = W = 128 or 192 works
equally (avgpool collapses spatial → 1×1 regardless).

Aux heads are `nn.Linear` on the shared feature. At inference time the
char-only path is exported (see `build_char_only_state_dict`) so deployed
binary has the same cost as v2 baseline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

NUM_RADICALS = 214          # Kangxi 214-way
NUM_IDC = 12                # ⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻
FEAT_DIM = 512              # resnet18 avgpool output

HEAD_NAMES = ("char", "radical", "total_strokes", "residual_strokes", "ids_top_idc")


class MultiHeadResNet18(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_radicals: int = NUM_RADICALS,
        num_idc: int = NUM_IDC,
    ):
        super().__init__()
        backbone = resnet18(weights=None)
        # strip the default fc — we'll add heads manually on the 512-d feature
        self.feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.char_head = nn.Linear(self.feat_dim, num_classes)
        self.radical_head = nn.Linear(self.feat_dim, num_radicals)
        self.total_strokes_head = nn.Linear(self.feat_dim, 1)
        self.residual_head = nn.Linear(self.feat_dim, 1)
        self.idc_head = nn.Linear(self.feat_dim, num_idc)

        self.num_classes = num_classes
        self.num_radicals = num_radicals
        self.num_idc = num_idc

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.backbone(x)  # (N, 512)
        return {
            "char": self.char_head(f),
            "radical": self.radical_head(f),
            "total_strokes": self.total_strokes_head(f).squeeze(-1),
            "residual_strokes": self.residual_head(f).squeeze(-1),
            "ids_top_idc": self.idc_head(f),
        }

    @torch.no_grad()
    def forward_char_only(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-time fast path: backbone + char head, no aux."""
        f = self.backbone(x)
        return self.char_head(f)


def build_model(name: str, num_classes: int) -> MultiHeadResNet18:
    """Mirrors v2 `build_model` signature — name is retained for config parity
    but only "resnet18" is supported (Level A plan explicitly uses resnet18)."""
    if name == "resnet18":
        return MultiHeadResNet18(num_classes=num_classes)
    raise ValueError(f"unknown model: {name!r}")


def build_char_only_state_dict(full_state: dict) -> dict:
    """Strip aux-head weights from a full state_dict. Use before exporting
    to ONNX so the deployed binary only carries backbone + char head."""
    keep_prefixes = ("backbone.", "char_head.")
    return {k: v for k, v in full_state.items() if k.startswith(keep_prefixes)}
