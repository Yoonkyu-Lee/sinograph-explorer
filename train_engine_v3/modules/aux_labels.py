"""Aux-labels sidecar loader (doc/19 §4 / §5.3).

Loads `aux_labels.npz` (produced by `sinograph_canonical_v3/scripts/
50_export_aux_labels.py`) into GPU tensors once at startup. Exposes
`get_aux(char_y_batch)` returning per-sample aux labels as a tuple, keyed
on the primary char label index.

Per-class storage is roughly 7 bytes (10 k class → 78 KB, 76 k class →
548 KB). The full table lives on GPU at all times.

Missing labels use sentinel values:
    radical_idx / total_strokes / residual_strokes : -1 (int16)
    ids_top_idc                                     : -1 (int8)
The companion `valid_mask` (uint8[N_class, 4]) holds the same info as a
boolean table; the training loop uses it to zero-out aux loss where the
label is absent.

Hash check: the sidecar embeds a uint64 hash of the `class_index.json`
snapshot it was built from. The loader re-hashes the loader-side
class_index.json and refuses to start on mismatch — protects against
silent label-index drift when corpora / class lists are regenerated.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


# IDC character → int mapping (doc/19 §4.2). Order matches the standard
# Unicode Ideographic Description Characters block U+2FF0..U+2FFB.
IDC_CHARS = "⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻"
IDC_MAP: dict[str, int] = {c: i for i, c in enumerate(IDC_CHARS)}


def class_index_hash(class_index: dict[str, int]) -> int:
    """Stable uint64 hash of class_index dict. Used to detect drift
    between sidecar and corpus class_index.json."""
    # canonicalize — sort by class_idx to make ordering deterministic
    ordered = sorted(class_index.items(), key=lambda kv: kv[1])
    blob = json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    h = hashlib.blake2b(blob, digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)


@dataclass
class AuxBatch:
    """Per-sample aux labels for one forward pass."""
    radical: torch.Tensor           # (B,)   long
    total: torch.Tensor             # (B,)   float
    residual: torch.Tensor          # (B,)   float
    idc: torch.Tensor               # (B,)   long
    valid: torch.Tensor             # (B, 4) bool  (radical / total / residual / idc)


class AuxTable:
    """Full class → aux table, on a specified device."""

    def __init__(
        self,
        radical: torch.Tensor,          # (N,) long
        total: torch.Tensor,            # (N,) float
        residual: torch.Tensor,         # (N,) float
        idc: torch.Tensor,              # (N,) long
        valid: torch.Tensor,            # (N, 4) bool
        n_class: int,
    ):
        self.radical = radical
        self.total = total
        self.residual = residual
        self.idc = idc
        self.valid = valid
        self.n_class = n_class

    @classmethod
    def from_npz(
        cls,
        npz_path: str | Path,
        expected_class_index: dict[str, int] | None = None,
        device: str | torch.device = "cpu",
    ) -> "AuxTable":
        data = np.load(str(npz_path))
        try:
            radical_i16 = data["radical_idx"]            # int16
            total_i16 = data["total_strokes"]            # int16
            residual_i16 = data["residual_strokes"]      # int16
            idc_i8 = data["ids_top_idc"]                 # int8
            valid_u8 = data["valid_mask"]                # uint8[N, 4]
            embedded_hash = int(data["class_index_hash"].item()) \
                if "class_index_hash" in data.files else None
        finally:
            data.close()

        if expected_class_index is not None and embedded_hash is not None:
            got = class_index_hash(expected_class_index)
            if got != embedded_hash:
                raise RuntimeError(
                    f"aux_labels.npz class_index_hash mismatch "
                    f"(sidecar={embedded_hash!r}, current class_index={got!r}). "
                    f"Re-run sinograph_canonical_v3/scripts/50_export_aux_labels.py "
                    f"against the current class_index.json."
                )

        n = int(radical_i16.shape[0])
        # int16 → long for CE gather; -1 sentinel preserved
        radical = torch.from_numpy(radical_i16.astype(np.int64)).to(device)
        idc = torch.from_numpy(idc_i8.astype(np.int64)).to(device)
        total = torch.from_numpy(total_i16.astype(np.float32)).to(device)
        residual = torch.from_numpy(residual_i16.astype(np.float32)).to(device)
        valid = torch.from_numpy(valid_u8.astype(np.bool_)).to(device)

        return cls(radical, total, residual, idc, valid, n_class=n)

    def to(self, device: str | torch.device) -> "AuxTable":
        return AuxTable(
            radical=self.radical.to(device),
            total=self.total.to(device),
            residual=self.residual.to(device),
            idc=self.idc.to(device),
            valid=self.valid.to(device),
            n_class=self.n_class,
        )

    def get_aux(self, char_y: torch.Tensor) -> AuxBatch:
        """Index the table by per-sample char label (B,) → AuxBatch."""
        return AuxBatch(
            radical=self.radical[char_y],
            total=self.total[char_y],
            residual=self.residual[char_y],
            idc=self.idc[char_y],
            valid=self.valid[char_y],
        )
