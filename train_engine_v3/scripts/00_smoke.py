"""End-to-end smoke for train_engine_v3 Level A.

Validates:
    (a) TensorShardDataset + DataLoader roundtrip on real production shards
    (b) AuxTable loads from aux_labels.npz with class_index_hash check
    (c) MultiHeadResNet18 forward produces 5 head outputs
    (d) Multi-task loss computes + backward runs under AMP
    (e) One-epoch loop + evaluate() returns sensible per-head metrics

Config: configs/resnet18_level_a_smoke.yaml. Uses first N shards
(`data.max_shards`) to keep runtime ≤ 2 min on an RTX GPU.

Usage:
    python train_engine_v3/scripts/00_smoke.py \
      --config train_engine_v3/configs/resnet18_level_a_smoke.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from train_engine_v3.modules.aux_labels import AuxTable     # noqa: E402
from train_engine_v3.modules.model import build_model       # noqa: E402
from train_engine_v3.modules.shard_dataset import (         # noqa: E402
    TensorShardDataset, build_shard_train_val_split, list_shards,
)
from train_engine_v3.modules.train_loop import (            # noqa: E402
    LossWeights, evaluate, train_one_epoch,
)


def make_gpu_transform(input_size: int):
    def _transform(x):
        x = x.float().div_(255.0)
        if x.shape[-1] != input_size or x.shape[-2] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear",
                               align_corners=False, antialias=True)
        x.sub_(0.5).div_(0.5)
        return x
    return _transform


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "smoke.log"
    print(f"[smoke] out_dir = {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device = {device}")

    # ----- data: shards + class_index + aux_labels -----
    shard_dir = Path(cfg["data"]["shard_dir"])
    class_index_path = shard_dir / "class_index.json"
    aux_path = Path(cfg["data"]["aux_labels"])

    with open(class_index_path, "r", encoding="utf-8") as f:
        class_index = json.load(f)
    n_class = len(class_index)
    print(f"[smoke] class_index {class_index_path.name}  n_class={n_class}")

    aux = AuxTable.from_npz(aux_path, expected_class_index=class_index, device=device)
    print(f"[smoke] aux_labels {aux_path.name} loaded, n={aux.n_class}")

    all_shards = list_shards(shard_dir)
    max_shards = int(cfg["data"].get("max_shards", 0) or 0)
    if max_shards > 0:
        all_shards = all_shards[:max_shards]
    print(f"[smoke] using {len(all_shards)} shards")

    train_paths, val_paths = build_shard_train_val_split(
        all_shards, val_ratio=float(cfg["data"]["val_ratio"]), seed=0,
    )
    print(f"[smoke] split  train={len(train_paths)} shards  val={len(val_paths)} shards")

    ds_train = TensorShardDataset(train_paths, shuffle=True, seed=0, shuffle_buffer=1024)
    ds_val = TensorShardDataset(val_paths, shuffle=False, seed=0, shuffle_buffer=0)
    dl_train = DataLoader(ds_train,
                           batch_size=cfg["train"]["batch_size"],
                           num_workers=cfg["train"]["num_workers"],
                           pin_memory=True, drop_last=True, persistent_workers=True)
    dl_val = DataLoader(ds_val,
                         batch_size=cfg["train"]["batch_size"],
                         num_workers=max(1, cfg["train"]["num_workers"] // 2),
                         pin_memory=True, drop_last=False, persistent_workers=True)

    # ----- model + optim -----
    model = build_model(cfg["model"]["name"], num_classes=n_class).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] model {cfg['model']['name']}  params={n_params/1e6:.2f} M")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler(device="cuda") if (cfg["train"]["amp"] and device == "cuda") else None

    # ----- loss weights -----
    w = cfg.get("loss_weights", {}) or {}
    weights = LossWeights(
        char=float(w.get("char", 1.0)),
        radical=float(w.get("radical", 0.2)),
        total=float(w.get("total", 0.1)),
        residual=float(w.get("residual", 0.1)),
        idc=float(w.get("idc", 0.2)),
    )

    gpu_transform = make_gpu_transform(cfg["model"]["input_size"])

    # ----- 1 epoch -----
    t_start = time.time()
    avg_loss = train_one_epoch(
        model=model, loader=dl_train,
        optimizer=optimizer, scaler=scaler, device=device,
        aux_table=aux, weights=weights,
        log_every=10, sysmon=None, gpu_transform=gpu_transform,
    )
    t_train = time.time() - t_start
    print(f"[smoke] epoch avg_loss = {avg_loss:.4f}   ({t_train:.1f}s)")

    t0 = time.time()
    metrics = evaluate(
        model=model, loader=dl_val, device=device,
        aux_table=aux, topk=(1, 5), gpu_transform=gpu_transform,
    )
    t_eval = time.time() - t0
    print(f"[smoke] eval  ({t_eval:.1f}s)")
    for k, v in metrics.items():
        print(f"           {k:>24s}  = {v:.4f}")

    # dump result for record
    result = {
        "config": args.config,
        "n_class": n_class,
        "shards_used": len(all_shards),
        "train_avg_loss": avg_loss,
        "train_sec": t_train,
        "eval_sec": t_eval,
        "metrics": metrics,
    }
    (out_dir / "smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[smoke] wrote {out_dir / 'smoke_result.json'}")


if __name__ == "__main__":
    main()
