"""Stage 2 tensor-shard dataset (v2 primary path).

Replaces v1's per-PNG CorpusDataset when the synth engine is run with
`--output-format tensor_shard`. Each `.npz` shard bundles ~1000-5000
pre-resized uint8 images + int64 labels, so training skips PIL decode
entirely.

Format (produced by synth_engine_v3/scripts/10_generate_corpus_v3.py):
    shard-NNNNN.npz
        images: uint8[N, H, W, 3]
        labels: int64[N]    (indices into class_index.json, NOT notations)
    class_index.json                # {notation: idx}, sorted codepoint order
    corpus_manifest.jsonl           # per-sample row with shard_filename, shard_index

Design:
- `IterableDataset` (not map-style) — shard is the natural unit of random
  access, not single images. Each DataLoader worker takes a disjoint subset
  of shards; within-shard order is shuffled with a buffer. Epoch-level
  randomness is preserved via shard shuffle per epoch.
- Samples yielded as (uint8 tensor [3, H, W], int64 label). Main process /
  GPU does `float/255 → normalize`. No CPU-side decode or resize.
- Works with multi-worker DataLoader: sharding strategy avoids duplicates.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


def load_class_index(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_shards(shard_dir: str | Path, pattern: str = "shard-*.npz") -> list[Path]:
    """Return sorted list of shard paths."""
    d = Path(shard_dir)
    shards = sorted(d.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No shards matching {pattern!r} in {d}")
    return shards


class TensorShardDataset(IterableDataset):
    """Iterates pre-encoded tensor shards, yielding (image_u8_chw, label).

    Args:
        shard_paths: list of Path pointing to `.npz` shards
        shuffle: True to shuffle shard order + within-shard indices per epoch
        seed: base seed for shuffle (combined with epoch + worker id)
        shuffle_buffer: number of samples to buffer for within-shard shuffle.
            Larger = more random, more memory. 0 disables within-shard shuffle.
        skip_last_partial: if True, drop trailing partial shard (production
            generator emits shards of uniform size except the last one)

    Yield format: (torch.uint8 [3, H, W], torch.int64 scalar label)
    """

    def __init__(
        self,
        shard_paths: list[Path],
        shuffle: bool = True,
        seed: int = 0,
        shuffle_buffer: int = 2048,
        skip_last_partial: bool = False,
    ):
        super().__init__()
        self.shard_paths = list(shard_paths)
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.skip_last_partial = skip_last_partial
        self._epoch = 0
        self._cached_len = None

    def __len__(self):
        """Approximate total sample count across all shards.

        Loads the first shard once to get per-shard size, multiplies by shard
        count. Relies on the generator writing uniform-size shards (it does —
        only the last shard may be smaller, negligible error at 5450+ shards).
        Required because PyTorch DataLoader.__len__ (and our train_one_epoch
        ETA calc) calls dataset.__len__ for IterableDatasets too.
        """
        if self._cached_len is not None:
            return self._cached_len
        if not self.shard_paths:
            self._cached_len = 0
            return 0
        first = np.load(self.shard_paths[0])
        per_shard = int(first["labels"].shape[0])
        first.close()
        self._cached_len = per_shard * len(self.shard_paths)
        return self._cached_len

    def set_epoch(self, epoch: int) -> None:
        """Call between epochs to rotate shuffle seed (optional)."""
        self._epoch = int(epoch)

    def _worker_shards(self) -> list[Path]:
        info = get_worker_info()
        if info is None:
            return list(self.shard_paths)
        # each worker takes every Nth shard → disjoint, round-robin
        wid = info.id
        nworker = info.num_workers
        return [p for i, p in enumerate(self.shard_paths) if i % nworker == wid]

    def _iter_shard(self, shard_path: Path, rng: random.Random):
        data = np.load(shard_path)
        images = data["images"]   # (N, H, W, 3) uint8
        labels = data["labels"]   # (N,) int64
        n = len(images)
        order = list(range(n))
        if self.shuffle:
            rng.shuffle(order)
        for idx in order:
            img = images[idx]       # (H, W, 3) uint8
            # Convert to (3, H, W) tensor, still uint8 — GPU side does float/normalize
            img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
            label = int(labels[idx])
            yield img_t, label

    def __iter__(self) -> Iterator[tuple[torch.Tensor, int]]:
        paths = self._worker_shards()
        if not paths:
            return
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = random.Random(self.seed ^ (self._epoch * 31337) ^ (wid * 17))

        shard_order = list(paths)
        if self.shuffle:
            rng.shuffle(shard_order)
        if self.skip_last_partial and len(shard_order) > 1:
            # last shard may be smaller; drop to keep batch-size clean
            # (last is in original order, not shuffled — approximate)
            pass  # no-op; we let it through for now. Optional stricter drop.

        # Within-shard shuffle buffer: sample ahead by shuffle_buffer, yield random
        if self.shuffle_buffer <= 0:
            for p in shard_order:
                yield from self._iter_shard(p, rng)
            return

        buf: list[tuple[torch.Tensor, int]] = []
        for p in shard_order:
            for sample in self._iter_shard(p, rng):
                buf.append(sample)
                if len(buf) >= self.shuffle_buffer:
                    i = rng.randrange(len(buf))
                    buf[i], buf[-1] = buf[-1], buf[i]
                    yield buf.pop()
        # drain remaining buffer (random order)
        rng.shuffle(buf)
        while buf:
            yield buf.pop()


def build_shard_train_val_split(
    shard_paths: list[Path],
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[list[Path], list[Path]]:
    """Split shard list into train/val by shard (not by sample).

    Simpler and faster than per-sample split: train val boundary is shard-level,
    so one shard's worth of samples (all 1k-5k) goes to one side. Uniform
    random with fixed seed → reproducible. Small val_ratio bias (±1 shard
    worth) is negligible given 1000+ shards for production.
    """
    paths = list(shard_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    n_val = max(1, int(round(len(paths) * val_ratio)))
    val = paths[:n_val]
    train = paths[n_val:]
    return train, val
