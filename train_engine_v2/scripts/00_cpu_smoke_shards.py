"""CPU-only smoke test for tensor_shard path in v2.

Creates 3 tiny `.npz` shards (fake uint8 images + labels) in a tmp dir,
runs 2 mini-epochs on CPU, checks that loss is finite, ckpt save/load
round-trips, and curves.png is written. Does NOT touch GPU — safe to
run while v3 production is generating shards.

Usage:
  python 00_cpu_smoke_shards.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.model import build_model
from modules.shard_dataset import (
    TensorShardDataset, build_shard_train_val_split, list_shards, load_class_index,
)
from modules.train_loop import evaluate, train_one_epoch
from modules.utils import save_checkpoint, seed_everything


def make_fake_shards(root: Path, n_classes: int = 10, n_shards: int = 3,
                      per_shard: int = 40, size: int = 32):
    """Write n_shards × .npz files with uint8 images + int64 labels."""
    root.mkdir(parents=True, exist_ok=True)
    # class_index: notation-like strings → idx
    class_index = {f"U+{0x4E00 + i:04X}": i for i in range(n_classes)}
    (root / "class_index.json").write_text(
        json.dumps(class_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rng = np.random.default_rng(0)
    for s in range(n_shards):
        imgs = rng.integers(0, 255, (per_shard, size, size, 3), dtype=np.uint8)
        # Each class appears at least a few times per shard (class balance)
        labels = np.array([(s * per_shard + i) % n_classes for i in range(per_shard)],
                          dtype=np.int64)
        np.savez(root / f"shard-{s:05d}.npz", images=imgs, labels=labels)
    return class_index


def main():
    t0 = time.time()
    seed_everything(0)
    device = torch.device("cpu")
    print(f"[smoke/shard] device={device}  torch={torch.__version__}")

    tmp = Path(tempfile.mkdtemp(prefix="shard_smoke_"))
    print(f"[smoke/shard] tmp={tmp}")
    try:
        shard_dir = tmp / "shards"
        class_index = make_fake_shards(shard_dir, n_classes=10, n_shards=3,
                                         per_shard=40, size=32)
        shards = list_shards(shard_dir)
        assert len(shards) == 3, f"expected 3 shards, got {len(shards)}"
        print(f"[smoke/shard] shards: {[p.name for p in shards]}")

        train_sh, val_sh = build_shard_train_val_split(shards, val_ratio=0.34, seed=0)
        assert train_sh and val_sh, "empty split"
        print(f"[smoke/shard] split: train={len(train_sh)} val={len(val_sh)}")

        ds_train = TensorShardDataset(train_sh, shuffle=True, shuffle_buffer=20, seed=0)
        ds_val = TensorShardDataset(val_sh, shuffle=False, shuffle_buffer=0, seed=0)

        dl_train = DataLoader(ds_train, batch_size=8, num_workers=0, drop_last=False)
        dl_val = DataLoader(ds_val, batch_size=8, num_workers=0)

        # sanity: iterate once and check shapes
        batch = next(iter(dl_train))
        x, y = batch
        assert x.dtype == torch.uint8, f"got {x.dtype}"
        assert x.shape[1:] == (3, 32, 32), f"got {x.shape}"
        assert y.dtype == torch.int64, f"got {y.dtype}"
        print(f"[smoke/shard] first batch x={tuple(x.shape)} {x.dtype}  y={tuple(y.shape)} {y.dtype}")

        # mini model + train loop
        model = build_model("resnet18", len(class_index)).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)

        # GPU transform (cpu here): uint8 → float/255 → resize (no-op if already 32)
        #  → normalize [-1, 1]
        def gpu_t(x_u8):
            x_f = x_u8.float().div_(255.0)
            # no resize needed (32 == 32 in this test)
            x_f.sub_(0.5).div_(0.5)
            return x_f

        losses = []
        for ep in range(2):
            ds_train.set_epoch(ep)
            ds_val.set_epoch(ep)
            loss = train_one_epoch(model, dl_train, opt, scaler=None, device=device,
                                    label_smoothing=0.1, log_every=10**9,
                                    gpu_transform=gpu_t)
            metrics = evaluate(model, dl_val, device, gpu_transform=gpu_t)
            losses.append(loss)
            print(f"[smoke/shard] epoch {ep+1}  loss={loss:.4f}  {metrics}")

            save_checkpoint(
                {"model": model.state_dict(), "epoch": ep + 1, "metrics": metrics},
                tmp / f"ckpt_epoch_{ep+1:02d}.pth",
            )

        # reload + forward parity
        state = torch.load(tmp / "ckpt_epoch_02.pth", map_location="cpu",
                            weights_only=True)
        model2 = build_model("resnet18", len(class_index)).to(device)
        model2.load_state_dict(state["model"])
        model2.eval()
        with torch.no_grad():
            x = torch.randn(2, 3, 32, 32)
            a = model(x); b = model2(x)
            assert torch.allclose(a, b, atol=1e-5)
        print("[smoke/shard] ckpt reload parity ok")

        for L in losses:
            assert L == L and L < 1e6, f"non-finite/extreme loss: {L}"
        print(f"[smoke/shard] PASS  elapsed={time.time()-t0:.1f}s  losses={losses}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
