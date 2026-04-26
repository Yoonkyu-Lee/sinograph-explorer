"""Stage 2 evaluation CLI.

Loads a trained checkpoint + its saved class_index, re-splits the manifest
with the same seed, and reports top-1/top-5 on the held-out split.
Optionally computes family-aware accuracy if a canonical_v2 SQLite path
is passed (see modules/family_eval.py).

Usage:
  python 21_eval.py --ckpt .../ckpt_epoch_20.pth \
                    --class-index .../class_index.json \
                    --config .../resnet18_t1_full.yaml \
                    [--family-db .../canonical_v2.sqlite]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.dataset import CorpusDataset, load_manifest, split_rows
from modules.model import build_model
from modules.train_loop import evaluate
from modules.utils import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--class-index", required=True)
    ap.add_argument("--config", required=True,
                    help="same yaml used for training (for data paths + split seed)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--family-db", default=None,
                    help="path to sinograph_canonical_v2/canonical_v2.sqlite")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    class_index = json.load(open(args.class_index, encoding="utf-8"))
    assert isinstance(class_index, dict)
    print(f"[eval] classes={len(class_index)}  ckpt={args.ckpt}")

    seed_everything(args.seed)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )
    print(f"[eval] device={device}")

    rows = load_manifest(cfg["data"]["manifest"])
    _, val_rows = split_rows(
        rows, class_index,
        val_ratio=cfg["data"]["val_ratio"],
        val_sources=cfg["data"].get("val_sources"),
        seed=args.seed,
    )
    print(f"[eval] val rows={len(val_rows):,}")

    input_size = cfg["model"]["input_size"]
    t_val = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    ds_val = CorpusDataset(val_rows, class_index, cfg["data"]["image_root"], t_val)
    dl_val = DataLoader(
        ds_val, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(cfg["model"]["name"], len(class_index)).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    metrics = evaluate(model, dl_val, device)
    print(f"[eval] topk: {metrics}")

    if args.family_db:
        from modules.family_eval import family_aware_accuracy
        fam = family_aware_accuracy(
            model, dl_val, device,
            class_index=class_index,
            canonical_db_path=args.family_db,
        )
        print(f"[eval] family: {fam}")
        metrics = {**metrics, **{f"family_{k}": v for k, v in fam.items()}}

    out_path = Path(args.ckpt).with_suffix(".eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
