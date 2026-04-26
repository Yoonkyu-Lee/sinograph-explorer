import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


class Tee:
    """Duplicate writes to stdout and a log file. Used to capture training
    stdout to out_dir/run.log without blocking terminal output."""

    def __init__(self, log_path, mode="a"):
        self.log = open(log_path, mode, encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, msg):
        self.stdout.write(msg)
        self.log.write(msg)
        # explicit flush-on-newline so `tail -f` / Get-Content -Wait sees lines
        # within milliseconds instead of whenever the OS buffer happens to
        # cycle. buffering=1 on text-mode file mostly does this but flushing
        # stdout too ensures real-time terminal output under wrapped python.
        if "\n" in msg:
            self.stdout.flush()
            self.log.flush()

    def flush(self):
        self.stdout.flush()
        self.log.flush()

    def install(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log.close()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_class_index(class_index, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(class_index, f, ensure_ascii=False, indent=2)


def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def plot_curves(metrics_path, out_png):
    """Read metrics.jsonl and save loss + top-1 curve."""
    import matplotlib.pyplot as plt

    rows = []
    with open(metrics_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return

    epochs = [r["epoch"] for r in rows]
    losses = [r["loss"] for r in rows]
    top1 = [r.get("top1") for r in rows]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, losses, "b-o", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="b")
    if any(t is not None for t in top1):
        ax2 = ax1.twinx()
        ax2.plot(
            [e for e, t in zip(epochs, top1) if t is not None],
            [t for t in top1 if t is not None],
            "r-s", label="val top-1",
        )
        ax2.set_ylabel("val top-1", color="r")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
