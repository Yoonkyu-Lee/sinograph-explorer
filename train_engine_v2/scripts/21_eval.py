"""Stage 2 evaluation CLI (v2 — png + tensor_shard).

Branches on config.data.format:
  - png           : re-split manifest with same seed, load PNG val set
  - tensor_shard  : re-split shard list with same seed, load .npz val shards

Reports top-1 / top-5 + (optional) family-aware accuracy.

Usage:
  # auto from a run dir (uses best.pth + class_index.json + embedded config)
  python 21_eval.py --run-dir .../out/03_v3r_prod_t1 \
                    [--family-db .../canonical_v2.sqlite]

  # explicit
  python 21_eval.py --ckpt .../best.pth --class-index .../class_index.json \
                    --config .../resnet18_t1_v3r_full.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.dataset import CorpusDataset, load_manifest, split_rows
from modules.model import build_model
from modules.shard_dataset import (
    TensorShardDataset, build_shard_train_val_split, list_shards,
)
from modules.train_loop import evaluate
from modules.utils import seed_everything


def _make_gpu_transform(input_size):
    def _t(x):
        x = x.float().div_(255.0)
        if x.shape[-1] != input_size or x.shape[-2] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear",
                              align_corners=False, antialias=True)
        x.sub_(0.5).div_(0.5)
        return x
    return _t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None,
                    help="auto-pick best.pth + class_index.json + embedded config")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--class-index", default=None)
    ap.add_argument("--config", default=None,
                    help="optional if ckpt has embedded config")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--family-db", default=None,
                    help="path to sinograph_canonical_v2/canonical_v2.sqlite")
    args = ap.parse_args()

    if args.run_dir:
        rd = Path(args.run_dir)
        best = rd / "best.pth"
        if not best.exists():
            epoch_ckpts = sorted(rd.glob("ckpt_epoch_*.pth"))
            if not epoch_ckpts:
                raise SystemExit(f"no ckpts in {rd}")
            best = epoch_ckpts[-1]
        args.ckpt = args.ckpt or str(best)
        ci = rd / "class_index.json"
        if ci.exists():
            args.class_index = args.class_index or str(ci)
    if not args.ckpt or not args.class_index:
        ap.error("must provide --run-dir or both --ckpt + --class-index")

    class_index = json.load(open(args.class_index, encoding="utf-8"))
    assert isinstance(class_index, dict)
    print(f"[eval] classes={len(class_index)}  ckpt={args.ckpt}")

    # config: embedded (preferred) or external yaml
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "config" in state:
        cfg = state["config"]
    elif args.config:
        cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    else:
        raise SystemExit("ckpt has no embedded config and --config not given")

    seed_everything(args.seed)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )
    print(f"[eval] device={device}")

    data_format = cfg["data"].get("format", "png")
    input_size = cfg["model"]["input_size"]

    # --- build val dataset/loader ----------------------------------------
    if data_format == "tensor_shard":
        shard_dir = Path(cfg["data"]["shard_dir"])
        shards = list_shards(shard_dir)
        _, val_shards = build_shard_train_val_split(
            shards, val_ratio=cfg["data"]["val_ratio"], seed=args.seed,
        )
        print(f"[eval] shard_dir={shard_dir}  val_shards={len(val_shards)}")
        ds_val = TensorShardDataset(
            val_shards, shuffle=False, seed=args.seed, shuffle_buffer=0,
        )
        dl_val = DataLoader(
            ds_val, batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
            pin_memory=(device.type == "cuda"),
        )
        gpu_tx = _make_gpu_transform(input_size)
    else:
        rows = load_manifest(cfg["data"]["manifest"])
        _, val_rows = split_rows(
            rows, class_index,
            val_ratio=cfg["data"]["val_ratio"],
            val_sources=cfg["data"].get("val_sources"),
            seed=args.seed,
        )
        print(f"[eval] val rows={len(val_rows):,}")
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
        gpu_tx = None  # val pipeline already normalized

    # --- model ------------------------------------------------------------
    model = build_model(cfg["model"]["name"], len(class_index)).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    metrics = evaluate(model, dl_val, device, gpu_transform=gpu_tx)
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
