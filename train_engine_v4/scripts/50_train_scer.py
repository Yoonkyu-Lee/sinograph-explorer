"""SCER production training (doc/28 §5.2).

Differences from `00_smoke.py`:
    - LR scheduler (cosine + warmup_epochs)
    - Checkpoint resume (last.pt) + best.pt tracking
    - Result file: train_result.json (ckpt history snapshot)

Otherwise identical: same modules, same SCERModel, same train_one_epoch with
§4.4 / §4.5 curriculum + non-finite guard, same realtime log spec (§9).

Config:   train_engine_v4/configs/scer_production.yaml
Runtime:  ~17.5 hours on RTX 4080 Laptop GPU (10 epochs × 3927 shards × batch 640)

Usage:
    python train_engine_v4/scripts/50_train_scer.py \\
        --config train_engine_v4/configs/scer_production.yaml
"""
from __future__ import annotations

import argparse
import json
import math
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

from streaming_log import setup_logging                          # noqa: E402

# v3 modules — reused (doc/28 §3.2)
from train_engine_v3.modules.aux_labels import AuxTable          # noqa: E402
from train_engine_v3.modules.shard_dataset import (              # noqa: E402
    TensorShardDataset, build_shard_train_val_split,
    build_stratified_val_split, list_shards,
)
try:
    from train_engine_v3.modules.sysmon import SysMon            # noqa: E402
except Exception:
    SysMon = None

# v4 SCER modules
from train_engine_v4.modules.model import build_scer            # noqa: E402
from train_engine_v4.modules.train_loop import (                 # noqa: E402
    LossWeights, evaluate, print_epoch_end, print_epoch_start,
    schedule, train_one_epoch,
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

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (REPO / path)

    out_dir = _resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "run.log")
    print(f"[scer-prod] out_dir = {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[scer-prod] device = {device}")

    # ----- data -----
    shard_dir = _resolve(cfg["data"]["shard_dir"])
    class_index_path = shard_dir / "class_index.json"
    aux_path = _resolve(cfg["data"]["aux_labels"])

    with open(class_index_path, "r", encoding="utf-8") as f:
        class_index = json.load(f)
    n_class = len(class_index)
    print(f"[scer-prod] class_index {class_index_path.name}  n_class={n_class}")

    aux = AuxTable.from_npz(aux_path, expected_class_index=class_index, device=device)
    print(f"[scer-prod] aux_labels loaded, n={aux.n_class}")

    all_shards = list_shards(shard_dir)
    max_shards = int(cfg["data"].get("max_shards", 0) or 0)
    if max_shards > 0:
        all_shards = all_shards[:max_shards]
    print(f"[scer-prod] using {len(all_shards)} shards")

    n_epochs_cfg = int(cfg["train"].get("epochs", 1))
    eval_every_cfg = int(cfg.get("eval", {}).get("every_epoch", 1))
    eval_disabled = eval_every_cfg > n_epochs_cfg

    val_per_shard = int(cfg["data"].get("val_per_shard", 0) or 0)
    if eval_disabled:
        ds_train = TensorShardDataset(all_shards, shuffle=True, seed=0,
                                       shuffle_buffer=1024)
        ds_val = None
        print(f"[scer-prod] EVAL-DISABLED  all {len(all_shards)} shards used")
    elif val_per_shard > 0:
        ds_train, ds_val = build_stratified_val_split(
            all_shards, val_per_shard=val_per_shard, seed=0, shuffle_buffer=1024,
        )
        print(f"[scer-prod] STRATIFIED split  shards={len(all_shards)}  "
              f"val_per_shard={val_per_shard}")
    else:
        train_paths, val_paths = build_shard_train_val_split(
            all_shards, val_ratio=float(cfg["data"]["val_ratio"]), seed=0,
        )
        print(f"[scer-prod] SHARD-level split  "
              f"train={len(train_paths)}  val={len(val_paths)}")
        ds_train = TensorShardDataset(train_paths, shuffle=True, seed=0,
                                       shuffle_buffer=1024)
        ds_val = TensorShardDataset(val_paths, shuffle=False, seed=0,
                                     shuffle_buffer=0)

    dl_train = DataLoader(ds_train,
                           batch_size=cfg["train"]["batch_size"],
                           num_workers=cfg["train"]["num_workers"],
                           pin_memory=True, drop_last=True,
                           persistent_workers=True)
    dl_val = None
    if ds_val is not None:
        dl_val = DataLoader(ds_val,
                             batch_size=cfg["train"]["batch_size"],
                             num_workers=max(1, cfg["train"]["num_workers"] // 2),
                             pin_memory=True, drop_last=False,
                             persistent_workers=True)

    # ----- AMP / channels_last -----
    cudnn_benchmark = bool(cfg["train"].get("cudnn_benchmark", False))
    channels_last = bool(cfg["train"].get("channels_last", False))
    amp_dtype_str = str(cfg["train"].get("amp_dtype", "fp16")).lower()
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print(f"[scer-prod] cudnn.benchmark = True")
    if amp_dtype_str == "bf16":
        amp_dtype = torch.bfloat16
        amp_uses_grad_scaler = False
    else:
        amp_dtype = torch.float16
        amp_uses_grad_scaler = True
    print(f"[scer-prod] amp_dtype = {amp_dtype_str}  "
          f"(grad_scaler={amp_uses_grad_scaler})")

    # ----- SCER model + warm-start -----
    model_cfg = cfg["model"]
    model = build_scer(
        name=model_cfg["name"],
        num_classes=n_class,
        emb_dim=int(model_cfg.get("emb_dim", 128)),
        arc_s=float(model_cfg.get("arc_s", 30.0)),
        arc_m=float(model_cfg.get("arc_m", 0.5)),
        easy_margin=bool(model_cfg.get("easy_margin", False)),
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[scer-prod] model={model_cfg['name']}  emb_dim={model.emb_dim}  "
          f"params={n_params/1e6:.2f} M")

    ws = cfg.get("warm_start") or {}
    if ws.get("v3_ckpt"):
        v3_ckpt = _resolve(ws["v3_ckpt"])
        load_struct = bool(ws.get("load_structure_heads", True))
        print(f"[scer-prod] warm-start: {v3_ckpt}  load_struct={load_struct}")
        stats = model.load_v3_backbone(v3_ckpt, load_structure_heads=load_struct)
        print(f"[scer-prod]   backbone OK  struct={stats['loaded_struct']}")
    else:
        print(f"[scer-prod] no warm-start (random init)")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    use_amp = bool(cfg["train"].get("amp", True)) and (device == "cuda")
    scaler = (torch.amp.GradScaler(device="cuda")
              if (use_amp and amp_uses_grad_scaler) else None)

    # ----- LR scheduler (warmup + cosine) -----
    n_epochs = int(cfg["train"].get("epochs", 1))
    sched_kind = str(cfg["train"].get("scheduler", "off")).lower()
    warmup_epochs = float(cfg["train"].get("warmup_epochs", 0))
    steps_per_epoch = max(1, len(dl_train))
    total_steps = steps_per_epoch * n_epochs
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    if sched_kind == "cosine":
        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        print(f"[scer-prod] scheduler: cosine warmup={warmup_epochs}ep  "
              f"base_lr={cfg['train']['lr']}  total_steps={total_steps}")
    else:
        scheduler = None
        print(f"[scer-prod] scheduler: off (constant lr={cfg['train']['lr']})")

    # ----- Checkpoint resume -----
    ckpt_save = bool(cfg["train"].get("ckpt_save", True))
    last_ckpt = out_dir / "last.pt"
    start_epoch = 1
    best_metric_value = -1.0
    best_metric_key = str(cfg["train"].get("ckpt_best_by", "emb/top1"))
    nan_count_total = 0                              # cumulative across run + resume
    if ckpt_save and last_ckpt.exists() and bool(cfg["train"].get("resume", True)):
        print(f"[scer-prod] resuming from {last_ckpt.name}")
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scaler is not None and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        if scheduler is not None and ck.get("scheduler") is not None:
            scheduler.load_state_dict(ck["scheduler"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_metric_value = float(ck.get("best_metric_value", -1.0))
        nan_count_total = int(ck.get("nan_count_total", 0))
        print(f"[scer-prod] resumed at epoch {start_epoch}, "
              f"best {best_metric_key}={best_metric_value:.4f}, "
              f"prior nan_total={nan_count_total}")

    # ----- loss weights -----
    w = cfg.get("loss_weights", {}) or {}
    weights = LossWeights(
        char=float(w.get("char", 1.0)),
        embedding=float(w.get("embedding", 1.0)),
        radical=float(w.get("radical", 0.2)),
        total=float(w.get("total", 0.1)),
        residual=float(w.get("residual", 0.1)),
        idc=float(w.get("idc", 0.2)),
    )

    gpu_transform = make_gpu_transform(model_cfg["input_size"])

    sysmon = None
    if SysMon is not None:
        try:
            sysmon = SysMon()
            print(f"[scer-prod] sysmon enabled")
        except Exception as e:
            print(f"[scer-prod] sysmon unavailable: {e}")

    log_every = int(cfg["train"].get("log_every", 50))
    window_steps = int(cfg["train"].get("window_steps", 50))
    eval_every = int(cfg.get("eval", {}).get("every_epoch", 1))

    # Codex review #1 — boundary checkpoints. Curriculum changes on epochs
    # 4 and 8 (warmup→transition, transition→fine). Save the *previous* epoch
    # state as an immutable rollback anchor before each boundary so a degraded
    # transition epoch does not destroy our last healthy training state.
    boundary_anchor_epochs = set(
        int(e) for e in cfg["train"].get("boundary_anchor_epochs", [3, 7])
    )

    epoch_train_losses: list[float] = []
    epoch_train_secs: list[float] = []
    epoch_nan_counts: list[int] = []
    metrics: dict = {}
    t_eval = 0.0
    nan_window = None                                # sliding-window carry-over (§4.5)
    ckpt_every_steps = int(cfg["train"].get("ckpt_every_steps", 5000))

    def _save_step_ckpt(step_in_epoch: int, ep_num: int) -> None:
        """Intra-epoch atomic last.pt save (Codex review #3). Captures full
        state needed to resume mid-epoch."""
        st = {
            "epoch": ep_num - 1,         # store as "completed up to ep-1"; resume goes to ep
            "step_in_epoch": int(step_in_epoch),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "best_metric_value": best_metric_value,
            "best_metric_key": best_metric_key,
            "n_class": n_class,
            "emb_dim": model.emb_dim,
            "nan_count_total": nan_count_total,
        }
        tmp = out_dir / "last.pt.tmp"
        torch.save(st, tmp)
        tmp.replace(out_dir / "last.pt")
        print(f"[scer-prod] step-ckpt last.pt written "
              f"(epoch {ep_num} step {step_in_epoch})", flush=True)

    for ep in range(start_epoch, n_epochs + 1):
        ds_train.set_epoch(ep)
        print_epoch_start(ep, n_epochs, model=model)         # applies §4.4 curriculum

        t_start = time.time()
        ep_out = train_one_epoch(
            model=model, loader=dl_train,
            optimizer=optimizer, scaler=scaler, device=device,
            aux_table=aux, weights=weights, epoch=ep,
            log_every=log_every, sysmon=sysmon,
            gpu_transform=gpu_transform,
            window_steps=window_steps,
            amp_dtype=amp_dtype, channels_last=channels_last,
            scheduler=scheduler,
            nan_count_in=nan_count_total,
            nan_window_in=nan_window,
            ckpt_callback=lambda s, e=ep: _save_step_ckpt(s, e),
            ckpt_every_steps=ckpt_every_steps,
        )
        avg_loss = ep_out["avg_loss"]
        nan_count_total = ep_out["nan_count_total"]
        nan_window = ep_out["nan_window"]            # carry to next epoch
        epoch_nan_counts.append(ep_out["nan_count_epoch"])
        t_train = time.time() - t_start
        epoch_train_losses.append(avg_loss)
        epoch_train_secs.append(t_train)

        do_eval = (ep % eval_every == 0) and (dl_val is not None)
        val_metrics: dict = {}
        if do_eval:
            t0 = time.time()
            val_metrics = evaluate(
                model=model, loader=dl_val, device=device,
                aux_table=aux, topk=(1, 5),
                gpu_transform=gpu_transform,
                amp_dtype=amp_dtype, channels_last=channels_last,
                use_amp=use_amp,
            )
            t_eval = time.time() - t0
            metrics = val_metrics
        print_epoch_end(ep, t_train + (t_eval if do_eval else 0.0),
                        avg_loss, val_metrics)
        if do_eval:
            for k, v in val_metrics.items():
                print(f"           {k:>24s}  = {v:.4f}")

        # ----- ckpt save (last.pt every epoch, best.pt when metric improves) -----
        if ckpt_save:
            ck_state = {
                "epoch": ep,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "metrics": metrics,
                "train_avg_loss": avg_loss,
                "best_metric_value": best_metric_value,
                "best_metric_key": best_metric_key,
                "n_class": n_class,
                "emb_dim": model.emb_dim,
                "nan_count_total": nan_count_total,    # carry across resume
            }
            tmp = out_dir / "last.pt.tmp"
            torch.save(ck_state, tmp)
            tmp.replace(out_dir / "last.pt")
            print(f"[scer-prod] ckpt last.pt written (epoch {ep}, "
                  f"nan_total={nan_count_total})", flush=True)

            # Boundary anchor: keep an *immutable* checkpoint for rollback
            # (Codex review #1 — pre-transition state preserved).
            if ep in boundary_anchor_epochs:
                anchor = out_dir / f"epoch_{ep:03d}.pt"
                tmp = out_dir / f"epoch_{ep:03d}.pt.tmp"
                torch.save(ck_state, tmp)
                tmp.replace(anchor)
                print(f"[scer-prod] BOUNDARY anchor {anchor.name} written "
                      f"(curriculum boundary; do not overwrite)", flush=True)

            if do_eval:
                cur_metric = float(metrics.get(best_metric_key, -1.0))
                if cur_metric > best_metric_value:
                    best_metric_value = cur_metric
                    ck_state["best_metric_value"] = cur_metric
                    tmp = out_dir / "best.pt.tmp"
                    torch.save(ck_state, tmp)
                    tmp.replace(out_dir / "best.pt")
                    print(f"[scer-prod] ckpt best.pt updated "
                          f"({best_metric_key}={cur_metric:.4f})", flush=True)

    # ----- result snapshot -----
    result = {
        "config": args.config,
        "n_class": n_class,
        "shards_used": len(all_shards),
        "epochs": n_epochs,
        "epoch_train_losses": epoch_train_losses,
        "epoch_train_secs": epoch_train_secs,
        "last_epoch_train_sec": (epoch_train_secs[-1]
                                  if epoch_train_secs else 0.0),
        "eval_sec": t_eval,
        "metrics": metrics,
        "best_metric_value": best_metric_value,
        "best_metric_key": best_metric_key,
        "nan_count_total": nan_count_total,           # Codex #2 — cumulative
        "epoch_nan_counts": epoch_nan_counts,
    }
    (out_dir / "train_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[scer-prod] wrote {out_dir / 'train_result.json'}")


if __name__ == "__main__":
    main()
