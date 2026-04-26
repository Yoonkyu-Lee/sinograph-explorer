"""Stage 2 manifest-based dataset.

Stage 1 (synth_engine_v3) writes `corpus_manifest.jsonl` next to PNGs.
Each row has at least: filename, notation, picked_source.
We treat `notation` (e.g. "U+9451") as the class label.

Two decode backends:
  - "pil"   (default): PIL.Image.open. Max compat. Slower per image.
  - "tvio":  torchvision.io.read_image. ~2-3x faster for PNG decode on CPU,
             returns uint8 tensor directly (no PIL roundtrip).
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


def load_manifest(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_class_index(manifest_rows, max_classes=None):
    notations = sorted({r["notation"] for r in manifest_rows})
    if max_classes is not None:
        notations = notations[:max_classes]
    return {n: i for i, n in enumerate(notations)}


def split_rows(rows, class_index, val_ratio=0.1, val_sources=None, seed=0):
    rows = [r for r in rows if r["notation"] in class_index]
    if val_sources:
        vs = set(val_sources)
        train = [r for r in rows if r.get("picked_source") not in vs]
        val = [r for r in rows if r.get("picked_source") in vs]
    else:
        rng = random.Random(seed)
        rows_shuffled = rows[:]
        rng.shuffle(rows_shuffled)
        n_val = int(len(rows_shuffled) * val_ratio)
        val = rows_shuffled[:n_val]
        train = rows_shuffled[n_val:]
    return train, val


class CorpusDataset(Dataset):
    def __init__(self, rows, class_index, image_root, transform, decode="pil"):
        """transform may be None — dataset then returns a raw uint8 tensor
        [3, H, W] (only valid with decode="tvio"). Used when resize/normalize
        are deferred to GPU to reduce CPU load and inter-process pickle size.
        """
        self.rows = rows
        self.class_index = class_index
        self.image_root = Path(image_root)
        self.transform = transform
        if decode not in ("pil", "tvio"):
            raise ValueError(f"unknown decode backend: {decode}")
        if transform is None and decode != "tvio":
            raise ValueError("transform=None requires decode='tvio' (need tensor output)")
        self.decode = decode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = self.image_root / r["filename"]
        if self.decode == "pil":
            img = Image.open(path).convert("RGB")
        else:
            img = read_image(str(path), mode=ImageReadMode.RGB)  # uint8 tensor [3,H,W]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.class_index[r["notation"]]
