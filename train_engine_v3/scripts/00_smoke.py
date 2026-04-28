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
sys.path.insert(0, str(REPO / "synth_engine_v3" / "scripts" / "cuda_raster"))

from streaming_log import setup_logging                      # noqa: E402

from train_engine_v3.modules.aux_labels import AuxTable     # noqa: E402
from train_engine_v3.modules.model import build_model       # noqa: E402
from train_engine_v3.modules.shard_dataset import (         # noqa: E402
    TensorShardDataset, build_shard_train_val_split,
    build_stratified_val_split, list_shards,
)
from train_engine_v3.modules.train_loop import (            # noqa: E402
    LossWeights, evaluate, train_one_epoch,
)
try:
    from train_engine_v3.modules.sysmon import SysMon       # noqa: E402
except Exception:
    SysMon = None


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

    def _resolve(p: str) -> Path:
        """Resolve config path: absolute as-is, relative against REPO root.
        Lets the same yaml work in WSL (/mnt/d/...) and Windows (d:\\...) by
        favoring relative paths in configs."""
        path = Path(p)
        return path if path.is_absolute() else (REPO / path)

    out_dir = _resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "smoke.log")
    print(f"[smoke] out_dir = {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device = {device}")

    # ----- data: shards + class_index + aux_labels -----
    shard_dir = _resolve(cfg["data"]["shard_dir"])
    class_index_path = shard_dir / "class_index.json"
    aux_path = _resolve(cfg["data"]["aux_labels"])

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

    val_per_shard = int(cfg["data"].get("val_per_shard", 0) or 0)
    if val_per_shard > 0:
        # STRATIFIED split — same shards for both, disjoint sample slices.
        # Required for codepoint-sorted shards so char distribution matches
        # between train and val (otherwise char/top1 ≡ 0).
        ds_train, ds_val = build_stratified_val_split(
            all_shards, val_per_shard=val_per_shard, seed=0, shuffle_buffer=1024,
        )
        print(f"[smoke] STRATIFIED split  shards={len(all_shards)}  "
              f"val_per_shard={val_per_shard}  "
              f"val~{val_per_shard * len(all_shards):,} samples  "
              f"train~{(5000 - val_per_shard) * len(all_shards):,} samples")
    else:
        # Legacy shard-level split (use only when shard contents are char-uniform)
        train_paths, val_paths = build_shard_train_val_split(
            all_shards, val_ratio=float(cfg["data"]["val_ratio"]), seed=0,
        )
        print(f"[smoke] SHARD-level split  train={len(train_paths)} shards  val={len(val_paths)} shards")
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

    # ----- TG-1 optimization toggles (all default off → matches TG-0 baseline) -----
    cudnn_benchmark = bool(cfg["train"].get("cudnn_benchmark", False))
    channels_last = bool(cfg["train"].get("channels_last", False))
    amp_dtype_str = str(cfg["train"].get("amp_dtype", "fp16")).lower()
    # YAML parses bare `off` as boolean False; normalize back to "off".
    compile_raw = cfg["train"].get("compile", "off")
    if compile_raw is False or compile_raw is None: compile_raw = "off"
    if compile_raw is True: compile_raw = "default"
    compile_mode = str(compile_raw).lower()
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print(f"[smoke] cudnn.benchmark = True")

    if amp_dtype_str == "fp16":
        amp_dtype = torch.float16
        amp_uses_grad_scaler = True
    elif amp_dtype_str == "bf16":
        amp_dtype = torch.bfloat16
        amp_uses_grad_scaler = False    # bf16 has fp32 dynamic range; no scaler
    else:
        amp_dtype = torch.float16
        amp_uses_grad_scaler = True
    print(f"[smoke] amp_dtype = {amp_dtype_str}  (grad_scaler={amp_uses_grad_scaler})")
    print(f"[smoke] channels_last = {channels_last}")
    print(f"[smoke] compile = {compile_mode}")

    # ----- model + optim -----
    model = build_model(cfg["model"]["name"], num_classes=n_class).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] model {cfg['model']['name']}  params={n_params/1e6:.2f} M")

    if compile_mode != "off":
        try:
            model = torch.compile(model, mode=compile_mode)
            print(f"[smoke] torch.compile mode={compile_mode}  (first epoch will include compile warmup)")
        except Exception as e:
            print(f"[smoke] torch.compile failed, falling back to eager: {e}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    use_amp = bool(cfg["train"].get("amp", True)) and (device == "cuda")
    scaler = torch.amp.GradScaler(device="cuda") if (use_amp and amp_uses_grad_scaler) else None

    # ----- LR scheduler (warmup + cosine) -----
    n_epochs = int(cfg["train"].get("epochs", 1))
    sched_kind = str(cfg["train"].get("scheduler", "off")).lower()
    warmup_epochs = float(cfg["train"].get("warmup_epochs", 0))
    steps_per_epoch = max(1, len(dl_train))
    total_steps = max(1, steps_per_epoch * n_epochs)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    if sched_kind == "cosine":
        import math as _math
        def _lr_lambda(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + _math.cos(_math.pi * min(1.0, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        print(f"[smoke] scheduler: cosine warmup={warmup_epochs}ep  base_lr={cfg['train']['lr']}  total_steps={total_steps}")
    else:
        scheduler = None
        print(f"[smoke] scheduler: off (constant lr={cfg['train']['lr']})")

    # ----- Checkpoint resume (if last.pt exists) -----
    ckpt_save = bool(cfg["train"].get("ckpt_save", False))
    last_ckpt = out_dir / "last.pt"
    start_epoch = 1
    best_metric_value = -1.0
    best_metric_key = str(cfg["train"].get("ckpt_best_by", "char/top1"))
    if ckpt_save and last_ckpt.exists() and bool(cfg["train"].get("resume", True)):
        print(f"[smoke] resuming from {last_ckpt.name}")
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ck["model"])
        except RuntimeError:
            # Handle compiled model state_dict prefix mismatch
            sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
            model.load_state_dict(sd)
        optimizer.load_state_dict(ck["optimizer"])
        if scaler is not None and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        if scheduler is not None and ck.get("scheduler") is not None:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_metric_value = float(ck.get("best_metric_value", -1.0))
        print(f"[smoke] resumed at epoch {start_epoch}, best {best_metric_key}={best_metric_value:.4f}")

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

    sysmon = None
    if SysMon is not None:
        try:
            sysmon = SysMon()
            print(f"[smoke] sysmon enabled — GPU util / VRAM / RAM logged each progress")
        except Exception as e:
            print(f"[smoke] sysmon unavailable: {e}")

    # ----- N epochs (epoch 1 = warmup, epoch 2+ = steady state) -----
    eval_every = int(cfg.get("eval", {}).get("every_epoch", 1))   # eval cadence
    epoch_train_losses: list[float] = []
    epoch_train_secs: list[float] = []
    metrics: dict = {}
    t_eval = 0.0
    for ep in range(start_epoch, n_epochs + 1):
        ds_train.set_epoch(ep)
        print(f"[smoke] === epoch {ep}/{n_epochs} ===")
        t_start = time.time()
        avg_loss = train_one_epoch(
            model=model, loader=dl_train,
            optimizer=optimizer, scaler=scaler, device=device,
            aux_table=aux, weights=weights,
            log_every=10, sysmon=sysmon, gpu_transform=gpu_transform,
            amp_dtype=amp_dtype, channels_last=channels_last,
            scheduler=scheduler,
        )
        t_train = time.time() - t_start
        epoch_train_losses.append(avg_loss)
        epoch_train_secs.append(t_train)
        print(f"[smoke] epoch {ep} avg_loss = {avg_loss:.4f}   ({t_train:.1f}s)")

        # Eval cadence: only eval when (ep % eval_every == 0) OR last epoch.
        do_eval = (ep % eval_every == 0) or (ep == n_epochs)
        if do_eval:
            t0 = time.time()
            metrics = evaluate(
                model=model, loader=dl_val, device=device,
                aux_table=aux, topk=(1, 5), gpu_transform=gpu_transform,
                amp_dtype=amp_dtype, channels_last=channels_last, use_amp=use_amp,
            )
            t_eval = time.time() - t0
            print(f"[smoke] eval epoch {ep} ({t_eval:.1f}s)")
            for k, v in metrics.items():
                print(f"           {k:>24s}  = {v:.4f}")
        else:
            print(f"[smoke] skipping eval at epoch {ep} (eval_every={eval_every})")

        # ----- Checkpoint save (every epoch — last.pt always; best.pt only when evaluated) -----
        if ckpt_save:
            ck_state = {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "metrics": metrics,           # may be stale if eval skipped this epoch
                "train_avg_loss": avg_loss,
                "best_metric_value": best_metric_value,  # updated below if applicable
                "best_metric_key": best_metric_key,
            }
            tmp = out_dir / "last.pt.tmp"
            torch.save(ck_state, tmp)
            tmp.replace(out_dir / "last.pt")
            print(f"[smoke] ckpt last.pt written (epoch {ep})")
            if do_eval:
                cur_metric = float(metrics.get(best_metric_key, -1.0))
                if cur_metric > best_metric_value:
                    best_metric_value = cur_metric
                    ck_state["best_metric_value"] = cur_metric
                    tmp = out_dir / "best.pt.tmp"
                    torch.save(ck_state, tmp)
                    tmp.replace(out_dir / "best.pt")
                    print(f"[smoke] ckpt best.pt updated ({best_metric_key}={cur_metric:.4f})")

    # dump result for record. epoch_train_secs[-1] = last (steady) epoch wall time.
    result = {
        "config": args.config,
        "n_class": n_class,
        "shards_used": len(all_shards),
        "epochs": n_epochs,
        "epoch_train_losses": epoch_train_losses,
        "epoch_train_secs": epoch_train_secs,
        "last_epoch_train_sec": epoch_train_secs[-1] if epoch_train_secs else 0.0,
        "eval_sec": t_eval,
        "metrics": metrics,
    }
    (out_dir / "smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[smoke] wrote {out_dir / 'smoke_result.json'}")


if __name__ == "__main__":
    main()
