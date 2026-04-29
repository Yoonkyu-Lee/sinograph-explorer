"""Live training monitor — stock-chart style (run while training is running).

Tails the SCER `run.log` and updates a 3-panel matplotlib chart every N
seconds without canvas flicker (uses persistent line artists + set_data,
not ax.clear() + replot).

  Panel 1  Loss (total) + per-component overlay
  Panel 2  Throughput — cum + win img/s
  Panel 3  LR + curriculum α/ε

Status bar (suptitle) reports current step / ETA / nan rate / GPU util.

Usage:
    python train_engine_v4/scripts/60_live_monitor.py \\
        --log train_engine_v4/out/16_scer_v1/run.log \\
        --refresh 3
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"] = "#fafafa"
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.alpha"] = 0.25
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["toolbar"] = "None"            # less overhead


# ----------------------------------------------------------------------
# Log parsing
# ----------------------------------------------------------------------

RE_STEP = re.compile(
    r"\[(?P<ts>\d\d:\d\d:\d\d)\]\s+"
    r"step\s+(?P<step>\d+)/(?P<total>\d+)\s+"
    r"loss=(?P<loss>[-\d.]+)\s+"
    r"char=(?P<char>[-\d.]+)\s+"
    r"arc=(?P<arc>[-\d.]+)\s+"
    r"rad=(?P<rad>[-\d.]+)\s+"
    r"tot=(?P<tot>[-\d.]+)\s+"
    r"res=(?P<res>[-\d.]+)\s+"
    r"idc=(?P<idc>[-\d.]+)\s+"
    r"α=(?P<alpha>[-\d.]+)\s+"
    r"ε=(?P<eps>[-\d.]+)\s+"
    r"lr=(?P<lr>[-\d.eE+]+)\s+"
    r"\(cum=(?P<cum>[-\d.]+)\s+win=(?P<win>[-\d.]+)\s+img/s,\s+"
    r"t=(?P<t>\d+)s,\s+eta=(?P<eta>\d+)s\)"
    r"(?:.*?gpu=(?P<gpu>\d+)%)?"
    r"(?:.*?vram_dev=(?P<vram>[\d.]+)/[\d.]+GB)?"
)
RE_EPOCH_START = re.compile(
    r"=== Epoch (?P<ep>\d+)/(?P<n>\d+) ===.*?"
    r"α=(?P<alpha>[-\d.]+)\s+ε=(?P<eps>[-\d.]+)\s+m=(?P<m>[-\d.]+)\s+"
    r"easy_margin=(?P<easy>\w+)\s+backbone_trainable=(?P<freeze>\w+)\s+"
    r"\(phase:\s*(?P<phase>\w+)\)"
)
RE_GUARD = re.compile(
    r"\[(?P<ts>\d\d:\d\d:\d\d)\]\s+\[guard\]\s+non-finite\s+(?P<kind>LOSS|GRAD)\s+"
    r"at\s+step\s+(?P<step>\d+).*?nan_count=(?P<nan>\d+).*?"
    r"(?:rate=(?P<rate>[\d.]+)%)?"
)
RE_ANCHOR = re.compile(r"BOUNDARY anchor (?P<file>epoch_\d+\.pt)")


@dataclass
class State:
    BUF = 50000
    steps: deque = field(default_factory=lambda: deque(maxlen=50000))
    loss_total: deque = field(default_factory=lambda: deque(maxlen=50000))
    loss_char: deque = field(default_factory=lambda: deque(maxlen=50000))
    loss_arc: deque = field(default_factory=lambda: deque(maxlen=50000))
    loss_rad: deque = field(default_factory=lambda: deque(maxlen=50000))
    loss_tot: deque = field(default_factory=lambda: deque(maxlen=50000))
    loss_res: deque = field(default_factory=lambda: deque(maxlen=50000))
    loss_idc: deque = field(default_factory=lambda: deque(maxlen=50000))
    cum_rate: deque = field(default_factory=lambda: deque(maxlen=50000))
    win_rate: deque = field(default_factory=lambda: deque(maxlen=50000))
    lr: deque = field(default_factory=lambda: deque(maxlen=50000))
    alpha: deque = field(default_factory=lambda: deque(maxlen=50000))
    eps: deque = field(default_factory=lambda: deque(maxlen=50000))
    last_ts: str = ""
    last_step: int = 0           # within-epoch (for status display)
    last_global: int = 0         # global = (ep-1)*total + step
    last_total: int = 0          # per-epoch step count
    last_eta: int = 0
    last_t: int = 0
    last_gpu: int = 0
    last_vram: float = 0.0
    last_phase: str = "?"
    last_m: float = 0.0
    last_freeze: str = "?"
    cur_epoch: int = 0
    n_epochs: int = 0
    nan_count: int = 0
    nan_rate: float = 0.0
    epoch_starts: list = field(default_factory=list)         # (global_step, ep, phase)
    guard_events: list = field(default_factory=list)         # (global_step, kind)
    anchor_events: list = field(default_factory=list)


class LogTailer:
    def __init__(self, log_path: Path, state: State):
        self.path = log_path
        self.state = state
        self.pos = 0

    def poll(self) -> int:
        if not self.path.exists():
            return 0
        new = 0
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(self.pos)
            for line in f:
                self._parse(line.rstrip("\n"))
                new += 1
            self.pos = f.tell()
        return new

    def _parse(self, line: str) -> None:
        s = self.state
        m = RE_STEP.search(line)
        if m:
            per_epoch_step = int(m.group("step"))
            per_epoch_total = int(m.group("total"))
            ep = max(s.cur_epoch, 1)                         # default 1 if banner not seen
            global_step = (ep - 1) * per_epoch_total + per_epoch_step
            s.steps.append(global_step)
            s.loss_total.append(float(m.group("loss")))
            s.loss_char.append(float(m.group("char")))
            s.loss_arc.append(float(m.group("arc")))
            s.loss_rad.append(float(m.group("rad")))
            s.loss_tot.append(float(m.group("tot")))
            s.loss_res.append(float(m.group("res")))
            s.loss_idc.append(float(m.group("idc")))
            s.cum_rate.append(float(m.group("cum")))
            s.win_rate.append(float(m.group("win")))
            s.lr.append(float(m.group("lr")))
            s.alpha.append(float(m.group("alpha")))
            s.eps.append(float(m.group("eps")))
            s.last_ts = m.group("ts")
            s.last_step = per_epoch_step
            s.last_global = global_step
            s.last_total = per_epoch_total
            s.last_t = int(m.group("t"))
            s.last_eta = int(m.group("eta"))
            if m.group("gpu"):
                s.last_gpu = int(m.group("gpu"))
            if m.group("vram"):
                s.last_vram = float(m.group("vram"))
            return
        m = RE_EPOCH_START.search(line)
        if m:
            ep = int(m.group("ep"))
            s.cur_epoch = ep
            s.n_epochs = int(m.group("n"))
            s.last_phase = m.group("phase")
            s.last_m = float(m.group("m"))
            s.last_freeze = m.group("freeze")
            # global step at epoch start = (ep-1) * per_epoch_total
            per_epoch_total = s.last_total or 1
            global_step_at_start = (ep - 1) * per_epoch_total
            s.epoch_starts.append((global_step_at_start, ep, s.last_phase))
            return
        m = RE_GUARD.search(line)
        if m:
            s.nan_count = int(m.group("nan"))
            if m.group("rate"):
                s.nan_rate = float(m.group("rate"))
            per_epoch_step = int(m.group("step"))
            ep = max(s.cur_epoch, 1)
            global_step = (ep - 1) * (s.last_total or 1) + per_epoch_step
            s.guard_events.append((global_step, m.group("kind")))
            return
        m = RE_ANCHOR.search(line)
        if m:
            s.anchor_events.append((s.last_global, m.group("file")))


# ----------------------------------------------------------------------
# Persistent-artist plotting (no flicker)
# ----------------------------------------------------------------------

class LiveChart:
    """Owns persistent matplotlib artists. Update via .update(state) each
    tick — only set_data on existing lines, no ax.clear() (which would
    flash the canvas)."""

    def __init__(self):
        self.fig, axes = plt.subplots(
            3, 1, figsize=(12, 8.5), sharex=True,
            gridspec_kw={"height_ratios": [3, 2, 1.5], "hspace": 0.18},
        )
        try:
            self.fig.canvas.manager.set_window_title("SCER live monitor")
        except Exception:
            pass
        self.ax_loss, self.ax_thr, self.ax_lr = axes
        self.ax_alpha = self.ax_lr.twinx()

        # ---- Loss panel ----
        self.l_total, = self.ax_loss.plot([], [], color="#000000", linewidth=1.6,
                                            label="total")
        self.l_char,  = self.ax_loss.plot([], [], color="#888888", linewidth=0.8,
                                            alpha=0.65, label="char")
        self.l_arc,   = self.ax_loss.plot([], [], color="#dd5555", linewidth=0.8,
                                            alpha=0.75, label="arc")
        self.l_rad,   = self.ax_loss.plot([], [], color="#5588dd", linewidth=0.7,
                                            alpha=0.55, label="rad")
        self.l_tot,   = self.ax_loss.plot([], [], color="#bbbb33", linewidth=0.6,
                                            alpha=0.45, label="tot_strk")
        self.l_res,   = self.ax_loss.plot([], [], color="#cc99cc", linewidth=0.6,
                                            alpha=0.45, label="res_strk")
        self.l_idc,   = self.ax_loss.plot([], [], color="#33aa66", linewidth=0.6,
                                            alpha=0.45, label="idc")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.85)

        # ---- Throughput panel ----
        self.l_cum, = self.ax_thr.plot([], [], color="#0066cc", linewidth=1.0,
                                         label="cum")
        self.l_win, = self.ax_thr.plot([], [], color="#22aa22", linewidth=0.9,
                                         alpha=0.85, label="win")
        self.ax_thr.set_ylabel("Throughput (img/s)")
        self.legend_thr = self.ax_thr.legend(loc="lower right", fontsize=8,
                                              framealpha=0.85)

        # ---- LR + curriculum panel ----
        self.l_lr,    = self.ax_lr.plot([], [], color="#000000", linewidth=1.0,
                                         label="lr")
        self.l_alpha, = self.ax_alpha.plot([], [], color="#cc6600", linewidth=0.8,
                                             linestyle="--", label="α")
        self.l_eps,   = self.ax_alpha.plot([], [], color="#aa00aa", linewidth=0.8,
                                             linestyle=":", label="ε")
        self.ax_alpha.set_ylim(-0.05, 1.15)
        self.ax_alpha.set_ylabel("α / ε", fontsize=9)
        self.ax_lr.set_ylabel("LR")
        self.ax_lr.set_xlabel("Global step")
        self.ax_lr.legend(loc="upper left", fontsize=8, framealpha=0.85)
        self.ax_alpha.legend(loc="upper right", fontsize=8, framealpha=0.85)

        # Event markers — collected as scatter PathCollections (one per frame)
        self.guard_scatter = self.ax_loss.scatter(
            [], [], marker="x", color="#cc0000", s=40, zorder=5)
        self.anchor_scatter = self.ax_loss.scatter(
            [], [], marker="*", color="#d4a017", s=70, zorder=5)
        # Vertical lines for epoch boundaries — drawn lazily
        self._epoch_vlines = []                        # list[Line2D] across all 3 axes

        self.suptitle = self.fig.suptitle(
            "SCER live monitor — waiting for first step…",
            fontsize=10, y=0.995,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])

    def update(self, s: State) -> None:
        if not s.steps:
            return
        x = list(s.steps)

        # ---- Loss ----
        self.l_total.set_data(x, list(s.loss_total))
        self.l_char.set_data(x, list(s.loss_char))
        self.l_arc.set_data(x, list(s.loss_arc))
        self.l_rad.set_data(x, list(s.loss_rad))
        self.l_tot.set_data(x, list(s.loss_tot))
        self.l_res.set_data(x, list(s.loss_res))
        self.l_idc.set_data(x, list(s.loss_idc))
        # Update legend label of total to show current value
        self.l_total.set_label(f"total = {s.loss_total[-1]:.3f}")
        self.ax_loss.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.85)
        ymax_loss = max(max(s.loss_total),
                         max(s.loss_arc) if s.loss_arc else 0,
                         max(s.loss_char) if s.loss_char else 0) * 1.05
        self.ax_loss.set_xlim(x[0], x[-1])
        self.ax_loss.set_ylim(0, ymax_loss)

        # ---- Throughput ----
        self.l_cum.set_data(x, list(s.cum_rate))
        self.l_win.set_data(x, list(s.win_rate))
        self.l_cum.set_label(f"cum = {s.cum_rate[-1]:.0f}")
        self.l_win.set_label(f"win = {s.win_rate[-1]:.0f}")
        self.ax_thr.legend(loc="lower right", fontsize=8, framealpha=0.85)
        ymax_thr = max(max(s.win_rate), max(s.cum_rate)) * 1.1
        self.ax_thr.set_ylim(0, ymax_thr)

        # ---- LR + α/ε ----
        self.l_lr.set_data(x, list(s.lr))
        self.l_alpha.set_data(x, list(s.alpha))
        self.l_eps.set_data(x, list(s.eps))
        self.l_lr.set_label(f"lr = {s.lr[-1]:.4g}")
        self.ax_lr.legend(loc="upper left", fontsize=8, framealpha=0.85)
        ymax_lr = max(s.lr) * 1.1
        self.ax_lr.set_ylim(0, max(ymax_lr, 1e-4))

        # ---- Event markers (scatter) ----
        if s.guard_events:
            xs = [step for step, _ in s.guard_events
                  if x[0] <= step <= x[-1]]
            ys = [ymax_loss * 0.95] * len(xs)
            self.guard_scatter.set_offsets(list(zip(xs, ys)) if xs else [[0, 0]])
            self.guard_scatter.set_visible(bool(xs))
        if s.anchor_events:
            xs = [step for step, _ in s.anchor_events
                  if x[0] <= step <= x[-1]]
            ys = [ymax_loss * 0.92] * len(xs)
            self.anchor_scatter.set_offsets(list(zip(xs, ys)) if xs else [[0, 0]])
            self.anchor_scatter.set_visible(bool(xs))

        # ---- Epoch boundary vertical lines ----
        # Add lines for any new epoch_start that doesn't yet have one
        existing = len(self._epoch_vlines) // 3            # 3 axes per epoch
        new_starts = s.epoch_starts[existing:]
        for step, ep, phase in new_starts:
            for ax in (self.ax_loss, self.ax_thr, self.ax_lr):
                vl = ax.axvline(step, color="#888", linestyle="--",
                                  linewidth=0.8, alpha=0.55, zorder=1)
                self._epoch_vlines.append(vl)

        # ---- Suptitle / status ----
        epoch_progress = s.last_step / max(s.last_total, 1) * 100.0
        global_total = s.last_total * s.n_epochs if s.n_epochs else 1
        global_progress = s.last_global / max(global_total, 1) * 100.0
        eta_min = s.last_eta / 60.0
        if s.last_step > 0 and s.last_total > 0:
            full_eta_sec = ((s.n_epochs - s.cur_epoch)
                            * (s.last_t / max(s.last_step, 1))
                            * s.last_total)
            full_eta_h = (s.last_eta + full_eta_sec) / 3600.0
        else:
            full_eta_h = 0.0
        title = (
            f"SCER Production — Epoch {s.cur_epoch}/{s.n_epochs} "
            f"({s.last_phase}, m={s.last_m:.2f}, freeze={s.last_freeze})  ┆  "
            f"epoch {s.last_step}/{s.last_total} ({epoch_progress:.0f}%)  "
            f"run {global_progress:.1f}%  ┆  "
            f"ETA epoch {eta_min:.1f}m / total ~{full_eta_h:.1f}h  ┆  "
            f"nan {s.nan_count} cum  rate {s.nan_rate:.2f}%  ┆  "
            f"GPU {s.last_gpu}%  VRAM {s.last_vram:.2f} GB  ┆  "
            f"upd {s.last_ts}"
        )
        self.suptitle.set_text(title)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to run.log")
    ap.add_argument("--refresh", type=float, default=3.0,
                    help="Refresh interval in seconds (default 3)")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"[60_live] waiting for log file: {log_path}", file=sys.stderr)

    state = State()
    tailer = LogTailer(log_path, state)
    chart = LiveChart()

    def _frame(_):
        tailer.poll()
        chart.update(state)

    interval_ms = int(args.refresh * 1000)
    anim = FuncAnimation(                                                # noqa: F841
        chart.fig, _frame,
        interval=interval_ms,
        cache_frame_data=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
