"""SCER (Structure-Conditioned Embedding Recognition) — Phase 2 model.

Doc reference: doc/28 §4 (architecture), doc/24 §6 (original SCER proposal).

Key shape and difference vs v3:

    v3 MultiHeadResNet18 (deploy = 50 MB FC):
        backbone (ResNet-18, 11M) → feat (512)
            ├── char_head        Linear(512, 98169)   ← 50 MB FC
            ├── radical_head     Linear(512, 214)
            ├── total_strokes    Linear(512, 1)
            ├── residual         Linear(512, 1)
            └── idc_head         Linear(512, 12)

    v4 SCERModel (deploy = ~12 MB, char_head dropped):
        backbone (warm-start from v3 best.pt) → feat (512)
            ├── radical_head     (kept)
            ├── total_strokes    (kept)
            ├── residual         (kept)
            ├── idc_head         (kept)
            ├── embedding_head   Linear(512, 128) → L2 normalize
            │     └── arc_classifier (ArcMarginProduct, weight (98169, 128))
            │             ↑ training-time anchor learner; weight = anchor DB
            └── char_head        Linear(512, 98169)  ← training-only warmup
                                                       NOT exported to deploy

At inference time we only use backbone + 4 structure heads + embedding head.
char_head and arc_classifier.weight stay on the host but are stripped from
the TFLite export. arc_classifier.weight is exported separately as
deploy_pi/export/scer_anchor_db.npy (51_build_anchor_db.py).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

from .arcface import ArcMarginProduct, normalize_embedding

NUM_RADICALS = 214
NUM_IDC = 12
FEAT_DIM = 512
EMB_DIM = 128            # SCER embedding dimension (doc/28 §4)


class SCERModel(nn.Module):
    """Structure-Conditioned Embedding Recognition model.

    Args:
        num_classes:    char head + arc classifier output size (98169)
        emb_dim:        embedding dimension (128 default; can grow to 256
                        if 128 turns out to under-separate 98k classes)
        num_radicals:   214 (Kangxi)
        num_idc:        12
        arc_s, arc_m:   ArcFace scale and margin (curriculum may mutate m)
        easy_margin:    True for smoke (safe), False for production
    """

    def __init__(
        self,
        num_classes: int,
        emb_dim: int = EMB_DIM,
        num_radicals: int = NUM_RADICALS,
        num_idc: int = NUM_IDC,
        arc_s: float = 30.0,
        arc_m: float = 0.5,
        easy_margin: bool = False,
    ):
        super().__init__()
        backbone = resnet18(weights=None)
        self.feat_dim = backbone.fc.in_features              # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Structure heads (identical to v3, kept for warm-start compatibility
        # AND for the structure soft filter at inference time)
        self.radical_head = nn.Linear(self.feat_dim, num_radicals)
        self.total_strokes_head = nn.Linear(self.feat_dim, 1)
        self.residual_head = nn.Linear(self.feat_dim, 1)
        self.idc_head = nn.Linear(self.feat_dim, num_idc)

        # Embedding head — the new piece (doc/28 §4.1)
        self.embedding_head = nn.Linear(self.feat_dim, emb_dim)
        self.arc_classifier = ArcMarginProduct(
            emb_dim=emb_dim, n_classes=num_classes,
            s=arc_s, m=arc_m, easy_margin=easy_margin,
        )

        # Training-only warmup signal — dropped at deploy. Same shape as v3
        # char_head so v3 best.pt would in principle warm-start it too, but
        # we leave it random-init since the curriculum down-weights it after
        # epoch 3 anyway.
        self.char_head = nn.Linear(self.feat_dim, num_classes)

        self.num_classes = num_classes
        self.num_radicals = num_radicals
        self.num_idc = num_idc
        self.emb_dim = emb_dim

    # ------------------------------------------------------------------
    # forward (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            x:      (N, 3, H, W)
            labels: (N,) long — required to compute arc_logits with margin.
                    If None, arc_logits is omitted (eval / inference path).
        Returns dict with keys:
            char, radical, total_strokes, residual_strokes, ids_top_idc,
            embedding (N, emb_dim, L2-normalized),
            arc_logits (N, num_classes) — present only if labels provided
        """
        feat = self.backbone(x)                                 # (N, 512)
        emb = self.embedding_head(feat)                         # (N, D)
        emb_norm = normalize_embedding(emb)                     # L2 norm

        out = {
            "char": self.char_head(feat),
            "radical": self.radical_head(feat),
            "total_strokes": self.total_strokes_head(feat).squeeze(-1),
            "residual_strokes": self.residual_head(feat).squeeze(-1),
            "ids_top_idc": self.idc_head(feat),
            "embedding": emb_norm,
        }
        if labels is not None:
            out["arc_logits"] = self.arc_classifier(emb_norm, labels)
        return out

    # ------------------------------------------------------------------
    # forward (deploy / inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_inference(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inference path — returns only what deploy needs.

        Returns:
            radical          (N, 214)
            total_strokes    (N,)
            residual_strokes (N,)
            ids_top_idc      (N, 12)
            embedding        (N, emb_dim, L2-normalized)
        """
        feat = self.backbone(x)
        emb_norm = normalize_embedding(self.embedding_head(feat))
        return {
            "radical": self.radical_head(feat),
            "total_strokes": self.total_strokes_head(feat).squeeze(-1),
            "residual_strokes": self.residual_head(feat).squeeze(-1),
            "ids_top_idc": self.idc_head(feat),
            "embedding": emb_norm,
        }

    # ------------------------------------------------------------------
    # warm-start from v3 best.pt
    # ------------------------------------------------------------------

    def load_v3_backbone(self, v3_ckpt_path: str | Path,
                          load_structure_heads: bool = True) -> dict:
        """Warm-start backbone (and optionally structure heads) from v3 best.pt.

        Embedding head, ArcFace classifier, and (intentionally) char_head
        are left at random init.

        Args:
            v3_ckpt_path: path to train_engine_v3/out/15_t5_light_v2/best.pt
            load_structure_heads: also load radical/total/residual/idc heads.
                                  v3 trained them already (rad 71%, idc 94%)
                                  so this gives a strong warm start. Default
                                  True.
        Returns:
            stats dict: {loaded_backbone, loaded_struct, missing, unexpected}
        """
        v3_ckpt_path = Path(v3_ckpt_path)
        state = torch.load(str(v3_ckpt_path), map_location="cpu",
                           weights_only=False)
        sd = state["model"] if "model" in state else state
        sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}

        # backbone
        backbone_sd = {
            k[len("backbone."):]: v
            for k, v in sd.items()
            if k.startswith("backbone.")
        }
        backbone_msg = self.backbone.load_state_dict(backbone_sd, strict=True)

        loaded_struct = []
        if load_structure_heads:
            head_map = {
                "radical_head": self.radical_head,
                "total_strokes_head": self.total_strokes_head,
                "residual_head": self.residual_head,
                "idc_head": self.idc_head,
            }
            for v3_name, v4_module in head_map.items():
                w_key = f"{v3_name}.weight"
                b_key = f"{v3_name}.bias"
                if w_key in sd and b_key in sd:
                    v4_module.load_state_dict(
                        {"weight": sd[w_key], "bias": sd[b_key]}, strict=True,
                    )
                    loaded_struct.append(v3_name)

        return {
            "loaded_backbone": True,
            "loaded_struct": loaded_struct,
            "missing": list(backbone_msg.missing_keys),
            "unexpected": list(backbone_msg.unexpected_keys),
            "v3_ckpt": str(v3_ckpt_path),
        }

    # ------------------------------------------------------------------
    # backbone freeze toggle (curriculum stage 1)
    # ------------------------------------------------------------------

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze (False) or unfreeze (True) the backbone. Used by the
        curriculum: epoch 1-3 freeze (head-only learning) → epoch 4 unfreeze."""
        for p in self.backbone.parameters():
            p.requires_grad = trainable


# ----------------------------------------------------------------------
# factory
# ----------------------------------------------------------------------


def build_scer(
    name: str,
    num_classes: int,
    emb_dim: int = EMB_DIM,
    arc_s: float = 30.0,
    arc_m: float = 0.5,
    easy_margin: bool = False,
) -> SCERModel:
    """Mirrors v3 build_model signature. Only 'resnet18' supported in v4."""
    if name != "resnet18":
        raise ValueError(f"unknown model: {name!r} (v4 only supports resnet18)")
    return SCERModel(
        num_classes=num_classes, emb_dim=emb_dim,
        arc_s=arc_s, arc_m=arc_m, easy_margin=easy_margin,
    )


def build_deploy_state_dict(full_state: dict) -> dict:
    """Strip training-only weights (char_head, arc_classifier) from a full
    SCERModel state_dict before exporting deploy artifacts."""
    drop_prefixes = ("char_head.", "arc_classifier.")
    return {k: v for k, v in full_state.items()
            if not k.startswith(drop_prefixes)}
