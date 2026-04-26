"""Stage 2 training CLI — run with --config.

Usage:
  python 20_train.py --config ../configs/resnet18_t1_pilot.yaml
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.dataset import (
    CorpusDataset, build_class_index, load_manifest, split_rows,
)
from modules.model import build_model
from modules.shard_dataset import (
    TensorShardDataset, build_shard_train_val_split, list_shards, load_class_index,
)
from modules.sysmon import SysMon
from modules.train_loop import evaluate, train_one_epoch
from modules.utils import (
    Tee, plot_curves, save_checkpoint, save_class_index, seed_everything,
)


def make_gpu_transform(input_size):
    """Returns a callable that takes a uint8 tensor [B, 3, H, W] on device
    and returns a float32 tensor [B, 3, input_size, input_size] normalized
    to mean=0.5, std=0.5. Fuses resize + normalize + dtype conversion on
    GPU, so CPU workers only have to decode + IPC as uint8 (4x less data).
    No RandomCrop — Stage 1 augment already provides geometric variation.
    """
    def _transform(x):
        x = x.float().div_(255.0)
        if x.shape[-1] != input_size or x.shape[-2] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear",
                              align_corners=False, antialias=True)
        x.sub_(0.5).div_(0.5)
        return x
    return _transform


def build_transforms(input_size, decode="pil", output_uint8=False):
    """output_uint8=True: skip ToTensor/Normalize, emit uint8 tensor only.
    Caller must do float/255 + normalize downstream (usually on GPU)."""
    if decode == "pil":
        base_train = [
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size, padding=4),
        ]
        base_val = [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
        ]
        if output_uint8:
            t_train = transforms.Compose(base_train + [transforms.PILToTensor()])
            t_val = transforms.Compose(base_val + [transforms.PILToTensor()])
        else:
            tail = [transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
            t_train = transforms.Compose(base_train + tail)
            t_val = transforms.Compose(base_val + tail)
    else:  # "tvio": input is uint8 tensor [3,H,W]
        base_train = [
            transforms.Resize(input_size, antialias=True),
            transforms.RandomCrop(input_size, padding=4),
        ]
        base_val = [
            transforms.Resize(input_size, antialias=True),
            transforms.CenterCrop(input_size),
        ]
        if output_uint8:
            t_train = transforms.Compose(base_train)
            t_val = transforms.Compose(base_val)
        else:
            tail = [
                transforms.ConvertImageDtype(torch.float32),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
            t_train = transforms.Compose(base_train + tail)
            t_val = transforms.Compose(base_val + tail)
    return t_train, t_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    # --- tuning overrides (override the yaml without editing it) ---
    ap.add_argument("--run-tag", default=None,
                    help="suffix for out_dir: e.g. 'w8_bs256' -> out/<base>__w8_bs256/")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None,
                    help="override config.train.lr. Linear scaling rule: "
                         "batch 2x → lr 2x. bs256=0.2, bs512=0.4, bs1024=0.8 (aggressive).")
    ap.add_argument("--max-classes", type=int, default=None)
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--amp", choices=["on", "off"], default=None)
    ap.add_argument("--prefetch-factor", type=int, default=None)
    ap.add_argument("--decode", choices=["pil", "tvio"], default=None,
                    help="image decode backend (overrides cfg.data.decode)")
    ap.add_argument("--gpu-transforms", choices=["on", "off"], default=None,
                    help="defer resize+normalize to GPU side (forces decode=tvio, "
                         "drops RandomCrop). Cuts CPU load and IPC size 4x.")
    ap.add_argument("--data-format", choices=["png", "tensor_shard"], default=None,
                    help="png: per-file PNG corpus (legacy v1). "
                         "tensor_shard: .npz shards from synth engine --output-format tensor_shard.")
    ap.add_argument("--resume-from", default=None,
                    help="training run dir to resume from. Picks the latest "
                         "ckpt_epoch_NN.pth, restores model/optimizer/scheduler "
                         "state, re-opens metrics.jsonl in append mode, "
                         "and starts at epoch N+1. Overrides cfg.out_dir.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    # apply overrides onto cfg BEFORE touching out_dir / loaders
    if args.epochs is not None:         cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:     cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:    cfg["train"]["num_workers"] = args.num_workers
    if args.lr is not None:             cfg["train"]["lr"] = args.lr
    if args.max_classes is not None:    cfg["data"]["max_classes"] = args.max_classes
    if args.input_size is not None:     cfg["model"]["input_size"] = args.input_size
    if args.amp is not None:            cfg["train"]["amp"] = (args.amp == "on")
    if args.prefetch_factor is not None: cfg["train"]["prefetch_factor"] = args.prefetch_factor
    if args.decode is not None:         cfg["data"]["decode"] = args.decode
    if args.gpu_transforms is not None: cfg["data"]["gpu_transforms"] = (args.gpu_transforms == "on")
    if args.data_format is not None:    cfg["data"]["format"] = args.data_format
    # decode default: pil. gpu_transforms works with both decode backends —
    # PIL path keeps resize on CPU, tvio path defers resize to GPU.
    cfg["data"].setdefault("decode", "pil")
    cfg["data"].setdefault("format", "png")  # backward compat

    # --resume-from overrides out_dir: we write to the SAME directory as the
    # run being resumed. Otherwise apply --run-tag suffix as usual.
    if args.resume_from:
        out_dir = Path(args.resume_from)
        if not out_dir.exists():
            raise SystemExit(f"--resume-from: directory not found: {out_dir}")
    else:
        out_dir = Path(cfg["out_dir"])
        if args.run_tag:
            out_dir = out_dir.parent / f"{out_dir.name}__{args.run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    Tee(out_dir / "run.log").install()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}  torch={torch.__version__}")
    print(f"[config] {args.config}")
    print(f"[out_dir] {out_dir}")
    print(f"[overrides] format={cfg['data']['format']} epochs={cfg['train']['epochs']} "
          f"batch={cfg['train']['batch_size']} lr={cfg['train']['lr']} "
          f"workers={cfg['train']['num_workers']} amp={cfg['train'].get('amp')} "
          f"max_classes={cfg['data'].get('max_classes')} input={cfg['model']['input_size']} "
          f"prefetch={cfg['train'].get('prefetch_factor')} decode={cfg['data'].get('decode','pil')} "
          f"gpu_tx={bool(cfg['data'].get('gpu_transforms'))}")

    # --- data ------------------------------------------------------------
    data_format = cfg["data"]["format"]

    if data_format == "tensor_shard":
        # v2 primary path: pre-encoded uint8 shards from synth engine.
        shard_dir = Path(cfg["data"]["shard_dir"])
        class_index = load_class_index(shard_dir / "class_index.json")
        shard_paths = list_shards(shard_dir)
        print(f"[data] shard_dir: {shard_dir}")
        print(f"[data] shards: {len(shard_paths):,}   classes: {len(class_index):,}")
        save_class_index(class_index, out_dir / "class_index.json")

        train_shards, val_shards = build_shard_train_val_split(
            shard_paths, val_ratio=cfg["data"]["val_ratio"], seed=args.seed,
        )
        print(f"[data] train/val shards: {len(train_shards):,} / {len(val_shards):,}")

        ds_train = TensorShardDataset(
            train_shards, shuffle=True, seed=args.seed, shuffle_buffer=2048,
        )
        ds_val = TensorShardDataset(
            val_shards, shuffle=False, seed=args.seed, shuffle_buffer=0,
        )
        # GPU-side transform: uint8 → float/255 → (optional resize) → normalize.
        # Resize only runs if shard's input size != model's input_size.
        gpu_transform = make_gpu_transform(cfg["model"]["input_size"])

    else:  # "png" — legacy v1 path (per-file PNG + manifest)
        rows = load_manifest(cfg["data"]["manifest"])
        print(f"[data] manifest rows: {len(rows):,}")
        class_index = build_class_index(rows, cfg["data"].get("max_classes"))
        print(f"[data] classes: {len(class_index):,}")
        save_class_index(class_index, out_dir / "class_index.json")

        train_rows, val_rows = split_rows(
            rows, class_index,
            val_ratio=cfg["data"]["val_ratio"],
            val_sources=cfg["data"].get("val_sources"),
            seed=args.seed,
        )
        print(f"[data] train/val: {len(train_rows):,} / {len(val_rows):,}")

        decode = cfg["data"].get("decode", "pil")
        gpu_tx = bool(cfg["data"].get("gpu_transforms"))
        image_root = cfg["data"]["image_root"]
        if gpu_tx:
            t_train, t_val = build_transforms(cfg["model"]["input_size"],
                                              decode=decode, output_uint8=True)
            ds_train = CorpusDataset(train_rows, class_index, image_root, t_train, decode=decode)
            ds_val = CorpusDataset(val_rows, class_index, image_root, t_val, decode=decode)
            gpu_transform = make_gpu_transform(cfg["model"]["input_size"])
        else:
            t_train, t_val = build_transforms(cfg["model"]["input_size"], decode=decode)
            ds_train = CorpusDataset(train_rows, class_index, image_root, t_train, decode=decode)
            ds_val = CorpusDataset(val_rows, class_index, image_root, t_val, decode=decode)
            gpu_transform = None

    dl_kwargs = {
        "batch_size": cfg["train"]["batch_size"],
        "num_workers": cfg["train"]["num_workers"],
        "pin_memory": True,
        "persistent_workers": cfg["train"]["num_workers"] > 0,
    }
    if cfg["train"].get("prefetch_factor") and cfg["train"]["num_workers"] > 0:
        dl_kwargs["prefetch_factor"] = cfg["train"]["prefetch_factor"]

    if data_format == "tensor_shard":
        # IterableDataset — shuffle is handled inside, don't pass shuffle=True
        # to DataLoader (it'd error). drop_last still valid: last batch of each
        # worker may be partial.
        dl_train = DataLoader(ds_train, drop_last=True, **dl_kwargs)
        dl_val = DataLoader(ds_val, **dl_kwargs)
    else:
        dl_train = DataLoader(ds_train, shuffle=True, drop_last=True, **dl_kwargs)
        dl_val = DataLoader(ds_val, shuffle=False, **dl_kwargs)

    # --- model / optim ---
    model = build_model(cfg["model"]["name"], len(class_index)).to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
        nesterov=True,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])
    scaler = torch.amp.GradScaler("cuda") if cfg["train"].get("amp") else None
    sysmon = SysMon() if device.type == "cuda" else None

    # --- resume --------------------------------------------------------------
    start_epoch = 0
    initial_best_top1 = -1.0
    initial_best_epoch = -1
    if args.resume_from:
        ckpts = sorted(Path(args.resume_from).glob("ckpt_epoch_*.pth"))
        if not ckpts:
            raise SystemExit(f"--resume-from: no ckpt_epoch_*.pth in {args.resume_from}")
        resume_ckpt = ckpts[-1]
        print(f"[resume] loading {resume_ckpt}")
        state = torch.load(resume_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            opt.load_state_dict(state["optimizer"])
        if "scheduler" in state and state["scheduler"] is not None:
            sched.load_state_dict(state["scheduler"])
        else:
            # pre-resume-feature ckpts have no scheduler state — step it N times
            for _ in range(state["epoch"]):
                sched.step()
        start_epoch = int(state["epoch"])

        # If user explicitly overrode --lr at resume time, reset the cosine
        # schedule's base_lrs + optimizer param_groups to the new value. Without
        # this, scheduler.load_state_dict() restored the old base_lr and the
        # next sched.step() would continue the OLD cosine trajectory, silently
        # ignoring the new lr. Typical use: switch batch 512 → 1024 mid-training
        # with linear scaling lr 0.4 → 0.8.
        if args.lr is not None:
            new_lr = float(args.lr)
            for group in opt.param_groups:
                group["lr"] = new_lr
            # CosineAnnealingLR stores base_lrs list; override so future
            # sched.step() uses the new base.
            if hasattr(sched, "base_lrs"):
                sched.base_lrs = [new_lr for _ in sched.base_lrs]
            # Force _last_lr to reflect the override for accurate first print
            if hasattr(sched, "_last_lr"):
                sched._last_lr = [new_lr for _ in sched._last_lr]
            print(f"[resume] --lr {new_lr} override applied: optimizer.lr + scheduler.base_lrs reset")

        print(f"[resume] resuming at epoch {start_epoch + 1}/{cfg['train']['epochs']}  "
              f"(lr will be {opt.param_groups[0]['lr']:.5f})")

        # restore best tracking from metrics.jsonl (the running best across prior epochs)
        prev_metrics = out_dir / "metrics.jsonl"
        if prev_metrics.exists():
            with open(prev_metrics, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    top1 = row.get("top1")
                    if top1 is not None and float(top1) > initial_best_top1:
                        initial_best_top1 = float(top1)
                        initial_best_epoch = int(row["epoch"])
            if initial_best_epoch > 0:
                print(f"[resume] prior best val top-1 = {initial_best_top1:.4f} @ epoch {initial_best_epoch}")

    # --- loop ---
    # Best-ckpt tracking — highest val top-1 across epochs is copied to
    # `best.pth`. Tie-breaker: earlier epoch wins. Useful for overfit runs
    # (final epoch may be worse than mid-training).
    best_top1 = initial_best_top1
    best_epoch = initial_best_epoch

    metrics_path = out_dir / "metrics.jsonl"
    with open(metrics_path, "a", encoding="utf-8") as mf:
        for epoch in range(start_epoch, cfg["train"]["epochs"]):
            lr = opt.param_groups[0]["lr"]
            print(f"\n== epoch {epoch+1}/{cfg['train']['epochs']} (lr={lr:.5f}) ==")
            # reseed shard shuffle per-epoch for IterableDataset paths
            if data_format == "tensor_shard":
                ds_train.set_epoch(epoch)
                ds_val.set_epoch(epoch)
            t0 = time.time()
            loss = train_one_epoch(
                model, dl_train, opt, scaler, device,
                label_smoothing=cfg["train"].get("label_smoothing", 0.0),
                sysmon=sysmon, gpu_transform=gpu_transform,
            )
            sched.step()
            metrics = {}
            if (epoch + 1) % cfg["eval"].get("every_epoch", 1) == 0:
                metrics = evaluate(model, dl_val, device, gpu_transform=gpu_transform)
            peaks = sysmon.peaks() if sysmon is not None else {}
            row = {
                "epoch": epoch + 1,
                "loss": loss,
                "wall_s": time.time() - t0,
                **metrics,
                **peaks,
            }
            print(f"  {row}")
            mf.write(json.dumps(row) + "\n")
            mf.flush()

            ckpt_state = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "epoch": epoch + 1,
                "metrics": row,
                "config": cfg,
            }
            ckpt_path = out_dir / f"ckpt_epoch_{epoch+1:02d}.pth"
            save_checkpoint(ckpt_state, ckpt_path)

            # Best-ckpt mirror: if val top-1 improved, save a second copy as
            # best.pth. Writing a separate file (not symlink) so Windows and
            # downstream scripts always see a valid ckpt without needing
            # developer mode / admin for symlink creation.
            top1 = metrics.get("top1")
            if top1 is not None and top1 > best_top1:
                best_top1 = float(top1)
                best_epoch = epoch + 1
                save_checkpoint(ckpt_state, out_dir / "best.pth")
                print(f"  [best] new best val top-1 = {best_top1:.4f} @ epoch {best_epoch}")
    plot_curves(metrics_path, out_dir / "curves.png")
    summary = {"best_top1": best_top1, "best_epoch": best_epoch,
               "final_epoch": cfg["train"]["epochs"]}
    (out_dir / "best.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] out_dir={out_dir}")
    if best_epoch > 0:
        print(f"[best] val top-1 = {best_top1:.4f} @ epoch {best_epoch}  (→ best.pth, best.json)")


if __name__ == "__main__":
    main()
