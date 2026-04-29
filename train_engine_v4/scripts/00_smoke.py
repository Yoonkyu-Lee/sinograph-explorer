"""End-to-end smoke for train_engine_v4 SCER (doc/28 §5.1).

Validates:
    (a) shard + aux_labels load (uses v3 modules: shard_dataset / aux_labels /
        sysmon — never duplicated, doc/28 §3.2)
    (b) v3 best.pt warm-start: backbone + 4 structure heads load OK,
        embedding head + arc_classifier + char_head random init
    (c) SCER forward returns 'embedding' (L2-norm) and 'arc_logits'
    (d) Joint loss with curriculum (α, ε) backprop runs under AMP
    (e) Realtime log spec (doc/28 §9.1) prints to console with flush=True
    (f) End-of-epoch eval reports both char/* and emb/*

Config:   train_engine_v4/configs/scer_smoke.yaml
Runtime:  ~30-60 s on RTX 4080 Laptop GPU (4 shards, 1 epoch, batch 256)

Usage:
    python train_engine_v4/scripts/00_smoke.py \\
        --config train_engine_v4/configs/scer_smoke.yaml
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

from streaming_log import setup_logging                          # noqa: E402

# v3 modules — reused as-is (doc/28 §3.2)
from train_engine_v3.modules.aux_labels import AuxTable          # noqa: E402
from train_engine_v3.modules.shard_dataset import (              # noqa: E402
    TensorShardDataset, build_shard_train_val_split,
    build_stratified_val_split, list_shards,
)
try:
    from train_engine_v3.modules.sysmon import SysMon            # noqa: E402
except Exception:
    SysMon = None

# v4 SCER modules (new)
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
    ap.add_argument(
        "--inject-nan-step", type=int, default=None,
        help="Debug: inject NaN loss at this step (1-indexed) of epoch 1 to "
             "exercise the §4.5 guard. Useful for sanity-testing pre-trigger.",
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def _resolve(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (REPO / path)

    out_dir = _resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "smoke.log")
    print(f"[scer-smoke] out_dir = {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[scer-smoke] device = {device}")

    # ----- data: shards + class_index + aux_labels (v3 corpus, unchanged) -----
    shard_dir = _resolve(cfg["data"]["shard_dir"])
    class_index_path = shard_dir / "class_index.json"
    aux_path = _resolve(cfg["data"]["aux_labels"])

    with open(class_index_path, "r", encoding="utf-8") as f:
        class_index = json.load(f)
    n_class = len(class_index)
    print(f"[scer-smoke] class_index {class_index_path.name}  n_class={n_class}")

    aux = AuxTable.from_npz(aux_path, expected_class_index=class_index, device=device)
    print(f"[scer-smoke] aux_labels {aux_path.name} loaded, n={aux.n_class}")

    all_shards = list_shards(shard_dir)
    max_shards = int(cfg["data"].get("max_shards", 0) or 0)
    if max_shards > 0:
        all_shards = all_shards[:max_shards]
    print(f"[scer-smoke] using {len(all_shards)} shards")

    # Codex review §4.5 #3: if config schedules no eval (eval.every_epoch >
    # n_epochs), use ALL shards for training and skip val construction entirely.
    # Otherwise the advertised shard count is misleading.
    n_epochs_cfg = int(cfg["train"].get("epochs", 1))
    eval_every_cfg = int(cfg.get("eval", {}).get("every_epoch", 1))
    eval_disabled = eval_every_cfg > n_epochs_cfg

    val_per_shard = int(cfg["data"].get("val_per_shard", 0) or 0)
    if eval_disabled:
        ds_train = TensorShardDataset(all_shards, shuffle=True, seed=0,
                                       shuffle_buffer=1024)
        ds_val = None
        print(f"[scer-smoke] EVAL-DISABLED  all {len(all_shards)} shards "
              f"used for training (no val split)")
    elif val_per_shard > 0:
        ds_train, ds_val = build_stratified_val_split(
            all_shards, val_per_shard=val_per_shard, seed=0, shuffle_buffer=1024,
        )
        print(f"[scer-smoke] STRATIFIED split  shards={len(all_shards)}  "
              f"val_per_shard={val_per_shard}")
    else:
        train_paths, val_paths = build_shard_train_val_split(
            all_shards, val_ratio=float(cfg["data"]["val_ratio"]), seed=0,
        )
        print(f"[scer-smoke] SHARD-level split  "
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

    # ----- AMP / channels_last toggles -----
    cudnn_benchmark = bool(cfg["train"].get("cudnn_benchmark", False))
    channels_last = bool(cfg["train"].get("channels_last", False))
    amp_dtype_str = str(cfg["train"].get("amp_dtype", "fp16")).lower()
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        print(f"[scer-smoke] cudnn.benchmark = True")

    if amp_dtype_str == "bf16":
        amp_dtype = torch.bfloat16
        amp_uses_grad_scaler = False
    else:
        amp_dtype = torch.float16
        amp_uses_grad_scaler = True
    print(f"[scer-smoke] amp_dtype = {amp_dtype_str}  "
          f"(grad_scaler={amp_uses_grad_scaler})")

    # ----- SCER model + warm-start from v3 best.pt -----
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
    print(f"[scer-smoke] model={model_cfg['name']}  emb_dim={model.emb_dim}  "
          f"arc_s={model.arc_classifier.s}  arc_m={model.arc_classifier.m}  "
          f"easy={model.arc_classifier.easy_margin}  "
          f"params={n_params/1e6:.2f} M")

    ws = cfg.get("warm_start") or {}
    if ws.get("v3_ckpt"):
        v3_ckpt = _resolve(ws["v3_ckpt"])
        load_struct = bool(ws.get("load_structure_heads", True))
        print(f"[scer-smoke] warm-start: {v3_ckpt}  load_struct={load_struct}")
        stats = model.load_v3_backbone(v3_ckpt, load_structure_heads=load_struct)
        print(f"[scer-smoke]   backbone OK  struct_loaded={stats['loaded_struct']}")
    else:
        print(f"[scer-smoke] no warm-start (random init backbone)")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    use_amp = bool(cfg["train"].get("amp", True)) and (device == "cuda")
    scaler = (torch.amp.GradScaler(device="cuda")
              if (use_amp and amp_uses_grad_scaler) else None)

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
            print(f"[scer-smoke] sysmon enabled")
        except Exception as e:
            print(f"[scer-smoke] sysmon unavailable: {e}")

    # ----- N epochs -----
    n_epochs = int(cfg["train"].get("epochs", 1))
    log_every = int(cfg["train"].get("log_every", 10))
    window_steps = int(cfg["train"].get("window_steps", 20))
    eval_every = int(cfg.get("eval", {}).get("every_epoch", 1))

    epoch_train_losses: list[float] = []
    epoch_train_secs: list[float] = []
    epoch_nan_counts: list[int] = []                     # this-epoch NaN events per epoch
    metrics: dict = {}
    t_eval = 0.0
    nan_count_total = 0                                  # cumulative across run (Codex #2)
    for ep in range(1, n_epochs + 1):
        ds_train.set_epoch(ep)
        print_epoch_start(ep, n_epochs, model=model)         # applies curriculum

        # Inject NaN only on epoch 1 (sanity test for §4.5 guard)
        inject_step = args.inject_nan_step if ep == 1 else None

        t_start = time.time()
        ep_out = train_one_epoch(
            model=model, loader=dl_train,
            optimizer=optimizer, scaler=scaler, device=device,
            aux_table=aux, weights=weights, epoch=ep,
            log_every=log_every, sysmon=sysmon,
            gpu_transform=gpu_transform,
            window_steps=window_steps,
            amp_dtype=amp_dtype, channels_last=channels_last,
            scheduler=None,
            inject_nan_step=inject_step,
            nan_count_in=nan_count_total,
        )
        avg_loss = ep_out["avg_loss"]
        nan_count_total = ep_out["nan_count_total"]
        epoch_nan_counts.append(ep_out["nan_count_epoch"])
        t_train = time.time() - t_start
        epoch_train_losses.append(avg_loss)
        epoch_train_secs.append(t_train)

        # Codex review §4.5 #3: respect config's eval.every_epoch literally —
        # do NOT force eval on the last epoch. Throughput configs set
        # eval_every=99 to skip eval entirely; that promise must hold.
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

    # dump result
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
        "schedule_phase": schedule(n_epochs)[5],
        "nan_count_total": nan_count_total,              # Codex #2
        "epoch_nan_counts": epoch_nan_counts,
    }
    (out_dir / "smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[scer-smoke] wrote {out_dir / 'smoke_result.json'}")


if __name__ == "__main__":
    main()
