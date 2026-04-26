"""CPU-only end-to-end smoke test for train_engine_v1.

Creates tiny synthetic PNGs + manifest, runs 2 mini epochs on CPU,
saves and reloads a checkpoint. Does not touch GPU (safe to run while
synth_engine_v3 is generating images on the GPU).

Usage:
  python 00_cpu_smoke.py
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
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.dataset import (
    CorpusDataset, build_class_index, load_manifest, split_rows,
)
from modules.model import build_model
from modules.train_loop import evaluate, train_one_epoch
from modules.utils import (
    plot_curves, save_checkpoint, save_class_index, seed_everything,
)


def make_fake_corpus(root: Path, n_classes: int = 8, per_class: int = 6, size: int = 48):
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "corpus_manifest.jsonl"
    rows = []
    rng = np.random.default_rng(0)
    for c in range(n_classes):
        cp = 0x4E00 + c
        notation = f"U+{cp:04X}"
        for i in range(per_class):
            arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            fn = f"{c * per_class + i:06d}_{notation}_fake-{i % 2}.png"
            Image.fromarray(arr).save(root / fn)
            rows.append({
                "idx": c * per_class + i,
                "char": chr(cp),
                "notation": notation,
                "block": "CJK_Unified",
                "base_source_kind": "font",
                "picked_source": f"fake-{i % 2}",
                "tag": f"fake-{i % 2}",
                "seed": 0,
                "filename": fn,
            })
    with open(manifest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return manifest


def main():
    t0 = time.time()
    seed_everything(0)
    device = torch.device("cpu")
    print(f"[smoke] device={device}  torch={torch.__version__}")

    tmp = Path(tempfile.mkdtemp(prefix="train_smoke_"))
    print(f"[smoke] tmp={tmp}")
    try:
        manifest_path = make_fake_corpus(tmp / "corpus", n_classes=8, per_class=6, size=48)
        rows = load_manifest(manifest_path)
        class_index = build_class_index(rows)
        assert len(class_index) == 8, f"expected 8 classes, got {len(class_index)}"
        print(f"[smoke] classes={len(class_index)} rows={len(rows)}")

        out_dir = tmp / "out"
        save_class_index(class_index, out_dir / "class_index.json")
        train_rows, val_rows = split_rows(rows, class_index, val_ratio=0.2, seed=0)
        print(f"[smoke] train/val={len(train_rows)}/{len(val_rows)}")

        t_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        t_val = transforms.Compose([
            transforms.Resize(32), transforms.CenterCrop(32),
            transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
        ds_train = CorpusDataset(train_rows, class_index, tmp / "corpus", t_train)
        ds_val = CorpusDataset(val_rows, class_index, tmp / "corpus", t_val)
        dl_train = DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=0)
        dl_val = DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=0)

        model = build_model("resnet18", len(class_index)).to(device)
        opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)

        losses = []
        for ep in range(2):
            loss = train_one_epoch(model, dl_train, opt, scaler=None,
                                   device=device, label_smoothing=0.1, log_every=10**9)
            metrics = evaluate(model, dl_val, device)
            print(f"[smoke] epoch {ep+1}  loss={loss:.4f}  {metrics}")
            losses.append(loss)
            # write per-epoch metrics for plot_curves
            with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as mf:
                mf.write(json.dumps({"epoch": ep + 1, "loss": loss, **metrics}) + "\n")
            save_checkpoint(
                {"model": model.state_dict(), "epoch": ep + 1, "metrics": metrics},
                out_dir / f"ckpt_epoch_{ep+1:02d}.pth",
            )

        # reload + forward equality check
        ckpt_path = out_dir / "ckpt_epoch_02.pth"
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model2 = build_model("resnet18", len(class_index)).to(device)
        model2.load_state_dict(state["model"])
        model2.eval()
        with torch.no_grad():
            x = torch.randn(2, 3, 32, 32)
            a = model(x); b = model2(x)
            assert torch.allclose(a, b, atol=1e-5), "ckpt reload mismatch"
        print("[smoke] ckpt reload ok")

        plot_curves(out_dir / "metrics.jsonl", out_dir / "curves.png")
        assert (out_dir / "curves.png").exists()
        print(f"[smoke] curves.png written ({(out_dir / 'curves.png').stat().st_size} bytes)")

        for L in losses:
            assert L == L and L < 1e6, f"loss went non-finite/extreme: {L}"
        print(f"[smoke] PASS  elapsed={time.time()-t0:.1f}s  losses={losses}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
