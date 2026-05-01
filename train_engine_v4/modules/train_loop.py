"""SCER training loop — Phase 2 (doc/28 §4-5).

Extends v3's train_loop.py pattern:
  - Same NVTX-instrumented stages (h2d / gpu_transform / aux_gather / forward
    / loss / backward / optim)
  - Same rolling-window throughput + cumulative throughput + ETA
  - Same flush=True realtime logging (doc/28 §9 — game-stopper requirement)
  - Same masked CE / masked SmoothL1 for structure heads

Adds:
  - ArcFace term in the joint loss
  - Curriculum schedule(epoch) → (alpha_char, eps_embedding)
  - `arc=...` in the per-step log line
  - `α=... ε=...` in the per-step log line (transition tracking)
  - Embedding-pipeline oracle eval (`emb/top1`, `emb/top5`) using the
    ArcFace classifier's per-class anchors as a sanity DB

Per doc/28 §9, the realtime log spec is:

  [HH:MM:SS] step {i+1}/{total}  loss={avg:.4f}  \\
    char={l_char:.3f} arc={l_arc:.3f} rad={l_rad:.3f} \\
    tot={l_tot:.3f} res={l_res:.3f} idc={l_idc:.3f}  \\
    α={alpha:.2f} ε={eps:.2f}  lr={lr:.4g}  \\
    (cum={cum:.0f} win={win:.0f} img/s, t={t:.0f}s, eta={eta:.0f}s) {sysmon}
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# NVTX: enable only when SCER_NVTX=1 in env (e.g. for nsys profiling).
# Default off — saves a few % per step on production runs (no try/except + no
# CUDA range_push/pop syscalls in the hot path). Always-on context manager
# overhead is small but real at ~10 ranges × ~30k steps × ~10 epochs.
_NVTX_ENABLED = os.environ.get("SCER_NVTX", "0") == "1"

# Reuse v3 helpers where identical — single source of truth for these utilities
from train_engine_v3.modules.aux_labels import AuxTable                # noqa: F401
from train_engine_v3.modules.sysmon import format_snapshot


# ----------------------------------------------------------------------
# Stability constants (doc/28 §4.5 — rate-based sliding window)
# ----------------------------------------------------------------------
#
# Why rate-based instead of cumulative count: a 17h × 300k step production
# run with 0.05% AMP-underflow rate (normal, GradScaler auto-recovers) hits
# a small cumulative absolute count quickly without indicating any actual
# instability. Sanity (200 step) at 2% rate also looks fine. So abort must
# scale with run length.
#
# Sliding window of the most recent NAN_WINDOW_SIZE steps; if the rate
# inside the window exceeds NAN_RATE_ABORT, abort. This is robust to step
# count and catches *systemic* instability (sustained failure to make
# progress), not transient AMP underflow events.

NAN_WINDOW_SIZE = 1000           # rolling window of last N steps
NAN_RATE_ABORT = 0.05             # 5% (50 nan-events / 1000 steps) → abort
NAN_RATE_WARN = 0.01              # 1% (10 / 1000 steps) → log a warning
GRAD_CLIP_MAX_NORM = 10.0         # clip global grad norm; also enables nonfinite detection


@contextmanager
def _nvtx(name: str):
    """NVTX range — gated by SCER_NVTX env var (default off).

    Production: env unset → context manager is a near-no-op (just yield).
    Profile:    SCER_NVTX=1 → CUDA NVTX range_push/pop emitted for nsys.
    """
    if _NVTX_ENABLED:
        try:
            torch.cuda.nvtx.range_push(name)
        except Exception:
            pass
        try:
            yield
        finally:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
    else:
        yield


# ----------------------------------------------------------------------
# Loss weights and curriculum (doc/28 §4.4)
# ----------------------------------------------------------------------


@dataclass
class LossWeights:
    """Joint loss component weights. char and embedding are scaled by the
    curriculum schedule per epoch; structure weights are static (kept at
    v3 values)."""
    char: float = 1.0           # base — scaled by alpha(epoch)
    embedding: float = 1.0      # base — scaled by eps(epoch)
    radical: float = 0.2
    total: float = 0.1
    residual: float = 0.1
    idc: float = 0.2


def schedule(
    epoch: int,
) -> tuple[float, float, float, bool, bool, str]:
    """Returns (alpha_char, eps_embedding, arc_m, easy_margin,
                backbone_trainable, phase_label) for a 1-indexed epoch.

    doc/28 §4.4 — joint curriculum (CE/ArcFace weight + ArcFace margin +
    backbone freeze). Throughput run with easy_margin=False + m=0.5 from
    random-init head diverged (loss 13.3 → 19.5); this schedule ramps m and
    only unfreezes the backbone after the new heads have aligned.

      Warmup     epoch 1-3   α=1.0 ε=0.1  m=0.3  easy=True   freeze backbone
      Transition epoch 4-7   α=0.5 ε=0.5  m=0.4  easy=False  unfreeze
      Fine       epoch 8+    α=0.1 ε=1.0  m=0.5  easy=False  unfreeze
    """
    if epoch <= 3:
        return (1.0, 0.1, 0.3, True,  False, "warmup")
    if epoch <= 7:
        return (0.5, 0.5, 0.4, False, True,  "transition")
    return     (0.1, 1.0, 0.5, False, True,  "fine")


# ----------------------------------------------------------------------
# Loss components (mostly mirroring v3 — kept inline for clarity)
# ----------------------------------------------------------------------


def _masked_ce(logits: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return logits.new_zeros(())
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    per_sample = F.cross_entropy(logits, safe_target, reduction="none")
    return (per_sample * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return pred.new_zeros(())
    per_sample = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0)
    return (per_sample * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def compute_scer_loss(
    out: dict[str, torch.Tensor],
    y_char: torch.Tensor,
    aux,                                  # AuxBatch
    weights: LossWeights,
    alpha: float,
    eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint loss: alpha·CE(char) + eps·CE(arc) + structure terms.

    Args:
        out:    SCERModel.forward output (dict; needs 'arc_logits' + 'char' +
                structure heads)
        y_char: (N,) long
        aux:    AuxBatch with valid mask
        weights: static base weights
        alpha:  current epoch's char weight (curriculum)
        eps:    current epoch's embedding weight (curriculum)
    """
    l_char = F.cross_entropy(out["char"], y_char, label_smoothing=0.1)
    l_arc = F.cross_entropy(out["arc_logits"], y_char)
    l_rad = _masked_ce(out["radical"], aux.radical, aux.valid[:, 0])
    l_tot = _masked_smooth_l1(out["total_strokes"], aux.total, aux.valid[:, 1])
    l_res = _masked_smooth_l1(out["residual_strokes"], aux.residual, aux.valid[:, 2])
    l_idc = _masked_ce(out["ids_top_idc"], aux.idc, aux.valid[:, 3])

    total = (
        (alpha * weights.char) * l_char
        + (eps * weights.embedding) * l_arc
        + weights.radical * l_rad
        + weights.total * l_tot
        + weights.residual * l_res
        + weights.idc * l_idc
    )
    parts = {
        "loss": total.item(),
        "l_char": l_char.item(),
        "l_arc": l_arc.item(),
        "l_radical": l_rad.item(),
        "l_total": l_tot.item(),
        "l_residual": l_res.item(),
        "l_idc": l_idc.item(),
        "alpha": alpha,
        "eps": eps,
    }
    return total, parts


# ----------------------------------------------------------------------
# Epoch banners (doc/28 §9.3)
# ----------------------------------------------------------------------


def print_epoch_start(epoch: int, total_epochs: int, model=None) -> None:
    """Print epoch banner and apply curriculum to model (if given).

    Side effects when model is given (doc/28 §4.4):
      - model.arc_classifier.set_margin(m)
      - model.arc_classifier.easy_margin = e
      - model.set_backbone_trainable(backbone_trainable)
    """
    alpha, eps, m, easy, backbone_trainable, phase = schedule(epoch)
    ts = time.strftime("%H:%M:%S")

    extra = ""
    if model is not None:
        model.arc_classifier.set_margin(m)
        model.arc_classifier.easy_margin = easy
        model.set_backbone_trainable(backbone_trainable)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        extra = (
            f"  trainable={n_train/1e6:.2f}M / total={n_total/1e6:.2f}M"
        )

    print(
        f"[{ts}] === Epoch {epoch}/{total_epochs} ===  "
        f"schedule: α={alpha:.2f} ε={eps:.2f} m={m:.2f} "
        f"easy_margin={easy} backbone_trainable={backbone_trainable}  "
        f"(phase: {phase}){extra}",
        flush=True,
    )


def print_epoch_end(epoch: int, dt_seconds: float, train_loss: float,
                    val: dict) -> None:
    ts = time.strftime("%H:%M:%S")
    pieces = [
        f"[{ts}] epoch {epoch} done in {dt_seconds:.0f}s",
        f"train_loss={train_loss:.4f}",
    ]
    if val:
        v = []
        for k in ("char/top1", "char/top5", "emb/top1", "emb/top5",
                  "radical/top1", "idc/top1"):
            if k in val:
                v.append(f"{k}={val[k]:.3f}")
        if "total_strokes/mae" in val:
            v.append(f"stroke_mae={val['total_strokes/mae']:.2f}")
        pieces.append("val: " + " ".join(v))
    print("  ".join(pieces), flush=True)


# ----------------------------------------------------------------------
# Train one epoch (doc/28 §9.1 realtime log spec)
# ----------------------------------------------------------------------


def train_one_epoch(
    model, loader, optimizer, scaler, device,
    aux_table: AuxTable,
    weights: LossWeights,
    epoch: int,
    log_every: int = 50,
    sysmon=None,
    gpu_transform=None,
    window_steps: int = 50,
    amp_dtype: torch.dtype = torch.float16,
    channels_last: bool = False,
    scheduler=None,
    inject_nan_step: int | None = None,    # debug: force NaN at this step (1-indexed)
    nan_count_in: int = 0,                  # run-level cumulative count entering this epoch
    nan_window_in: deque | None = None,    # run-level sliding window entering this epoch
    ckpt_callback=None,                     # optional: fn(step_in_epoch) → None
    ckpt_every_steps: int = 5000,           # call ckpt_callback every N optimizer steps
):
    """Single SCER train epoch — same skeleton as v3, with arc + curriculum.

    Returns: dict {
        'avg_loss':        float,
        'nan_count_total': int,    # run-level cumulative count after this epoch
        'nan_count_epoch': int,    # events that occurred in this epoch alone
        'nan_window':      deque,  # run-level sliding window (carry to next epoch)
    }

    Stability (doc/28 §4.5):
      - Pre-backward: if loss is non-finite, skip backward+step
      - Post-backward: clip with error_if_nonfinite=True; on RuntimeError skip step
      - Sliding-window rate-based abort: track last NAN_WINDOW_SIZE step outcomes
        (each step is either OK=0 or skipped=1); if count of 1s in the window
        exceeds NAN_WINDOW_SIZE × NAN_RATE_ABORT → abort.
        This catches *systemic* instability (sustained inability to make progress)
        instead of normal AMP underflow events that GradScaler auto-recovers from.
    """
    alpha, eps, _m, _easy, _freeze, _phase = schedule(epoch)

    model.train()
    if sysmon is not None:
        sysmon.reset_peaks()

    sum_loss = 0.0
    sum_n = 0
    t0 = time.time()
    use_amp = scaler is not None
    nan_count = int(nan_count_in)                   # cumulative across run (informational)
    nan_count_epoch = 0                              # this epoch alone
    if nan_window_in is None:
        nan_window: deque = deque(maxlen=NAN_WINDOW_SIZE)
    else:
        nan_window = nan_window_in
    nan_abort_threshold = int(NAN_WINDOW_SIZE * NAN_RATE_ABORT)
    nan_warn_threshold = int(NAN_WINDOW_SIZE * NAN_RATE_WARN)
    last_warn_at = -10000                           # step index of last warn (debounce)

    win_buf: deque = deque(maxlen=window_steps)
    t_prev = t0

    with _nvtx("train_epoch"):
        for i, (x, y) in enumerate(loader):
            with _nvtx("h2d"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
            with _nvtx("gpu_transform"):
                if gpu_transform is not None:
                    x = gpu_transform(x)
                if channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
            with _nvtx("aux_gather"):
                aux = aux_table.get_aux(y)

            optimizer.zero_grad(set_to_none=True)
            with _nvtx("forward"):
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    out = model(x, labels=y)
            with _nvtx("loss"):
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    loss, parts = compute_scer_loss(
                        out, y, aux, weights, alpha=alpha, eps=eps,
                    )

            # Optional NaN injection for guard sanity test (doc/28 §4.5)
            if inject_nan_step is not None and (i + 1) == inject_nan_step:
                print(
                    f"[guard-test] injecting NaN into loss at step {i+1}",
                    flush=True,
                )
                loss = loss * float("nan")

            # ----- pre-backward: loss must be finite -----
            if not torch.isfinite(loss):
                nan_count += 1
                nan_count_epoch += 1
                nan_window.append(1)
                ts = time.strftime("%H:%M:%S")
                rate_in_window = sum(nan_window) / max(len(nan_window), 1)
                print(
                    f"[{ts}] [guard] non-finite LOSS at step {i+1}  "
                    f"char={parts['l_char']:.3f} arc={parts['l_arc']:.3f} "
                    f"rad={parts['l_radical']:.3f} tot={parts['l_total']:.3f} "
                    f"res={parts['l_residual']:.3f} idc={parts['l_idc']:.3f} "
                    f"(nan_count={nan_count} cum, "
                    f"rate={rate_in_window:.3%} of last {len(nan_window)})",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                if (len(nan_window) >= NAN_WINDOW_SIZE
                        and sum(nan_window) >= nan_abort_threshold):
                    if ckpt_callback is not None:
                        try:
                            ckpt_callback(i + 1)             # save before raise
                        except Exception as ckpt_err:
                            print(f"[guard] pre-abort ckpt failed: {ckpt_err}",
                                  flush=True)
                    raise RuntimeError(
                        f"aborted: non-finite-step rate {rate_in_window:.1%} "
                        f"in last {NAN_WINDOW_SIZE} steps exceeds "
                        f"abort threshold {NAN_RATE_ABORT:.1%}"
                    )
                continue                             # skip backward + step

            with _nvtx("backward"):
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # ----- post-backward: gradients must be finite -----
            if scaler is not None:
                scaler.unscale_(optimizer)           # unscale BEFORE clipping
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP_MAX_NORM,
                    error_if_nonfinite=True,
                )
            except RuntimeError:
                nan_count += 1
                nan_count_epoch += 1
                nan_window.append(1)
                ts = time.strftime("%H:%M:%S")
                rate_in_window = sum(nan_window) / max(len(nan_window), 1)
                print(
                    f"[{ts}] [guard] non-finite GRAD at step {i+1}  "
                    f"loss={parts['loss']:.3f} char={parts['l_char']:.3f} "
                    f"arc={parts['l_arc']:.3f} "
                    f"(nan_count={nan_count} cum, "
                    f"rate={rate_in_window:.3%} of last {len(nan_window)})",
                    flush=True,
                )
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.update()                  # let scaler back off scale
                if (len(nan_window) >= NAN_WINDOW_SIZE
                        and sum(nan_window) >= nan_abort_threshold):
                    if ckpt_callback is not None:
                        try:
                            ckpt_callback(i + 1)             # save before raise
                        except Exception as ckpt_err:
                            print(f"[guard] pre-abort ckpt failed: {ckpt_err}",
                                  flush=True)
                    raise RuntimeError(
                        f"aborted: non-finite-step rate {rate_in_window:.1%} "
                        f"in last {NAN_WINDOW_SIZE} steps exceeds "
                        f"abort threshold {NAN_RATE_ABORT:.1%}"
                    )
                continue                             # skip optimizer.step

            with _nvtx("optim"):
                if scaler is not None:
                    scaler.step(optimizer)           # already unscaled — fast path
                    scaler.update()
                else:
                    optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # Successful step — record OK in sliding window
            nan_window.append(0)

            # Periodic warn at sub-abort but elevated rate (1%+)
            if (len(nan_window) >= NAN_WINDOW_SIZE
                    and sum(nan_window) >= nan_warn_threshold
                    and (i - last_warn_at) > NAN_WINDOW_SIZE):
                rate = sum(nan_window) / NAN_WINDOW_SIZE
                ts = time.strftime("%H:%M:%S")
                print(
                    f"[{ts}] [guard-warn] non-finite-step rate {rate:.2%} "
                    f"in last {NAN_WINDOW_SIZE} steps (warn threshold "
                    f"{NAN_RATE_WARN:.1%}; abort {NAN_RATE_ABORT:.1%})",
                    flush=True,
                )
                last_warn_at = i

            bs = x.size(0)
            sum_loss += parts["loss"] * bs
            sum_n += bs

            t_now = time.time()
            win_buf.append((t_now - t_prev, bs))
            t_prev = t_now

            if (i + 1) % log_every == 0:
                elapsed = t_now - t0
                cum_rate = sum_n / max(elapsed, 1e-6)
                win_t = sum(dt for dt, _ in win_buf)
                win_n = sum(n for _, n in win_buf)
                win_rate = win_n / max(win_t, 1e-6)
                steps_left = max(len(loader) - (i + 1), 0)
                eta = steps_left * (elapsed / max(i + 1, 1))
                ts = time.strftime("%H:%M:%S")
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"[{ts}] step {i+1}/{len(loader)}  "
                    f"loss={sum_loss/sum_n:.4f}  "
                    f"char={parts['l_char']:.3f} arc={parts['l_arc']:.3f} "
                    f"rad={parts['l_radical']:.3f} "
                    f"tot={parts['l_total']:.3f} res={parts['l_residual']:.3f} "
                    f"idc={parts['l_idc']:.3f}  "
                    f"α={alpha:.2f} ε={eps:.2f}  "
                    f"lr={lr:.4g}  "
                    f"(cum={cum_rate:.0f} win={win_rate:.0f} img/s, "
                    f"t={elapsed:.0f}s, eta={eta:.0f}s)"
                )
                if sysmon is not None:
                    msg += "  " + format_snapshot(sysmon.snapshot())
                print(msg, flush=True)
                sys.stdout.flush()

            # Periodic intra-epoch checkpoint (Codex review #3 — protect against
            # mid-epoch abort losing all progress)
            if (ckpt_callback is not None
                    and ckpt_every_steps > 0
                    and (i + 1) % ckpt_every_steps == 0):
                try:
                    ckpt_callback(i + 1)
                except Exception as ckpt_err:
                    print(f"[ckpt] step-level save failed at step {i+1}: "
                          f"{ckpt_err}", flush=True)
    return {
        "avg_loss": sum_loss / max(sum_n, 1),
        "nan_count_total": nan_count,
        "nan_count_epoch": nan_count_epoch,
        "nan_window": nan_window,                    # carry to next epoch
    }


# ----------------------------------------------------------------------
# Evaluate (doc/28 §9.3 — char/top1, char/top5, emb/top1, emb/top5, structure)
# ----------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model, loader, device,
    aux_table: AuxTable,
    topk: tuple = (1, 5),
    gpu_transform=None,
    amp_dtype: torch.dtype = torch.float16,
    channels_last: bool = False,
    use_amp: bool = True,
):
    """Eval — reports both char_head ranking and embedding ranking.

    Embedding ranking ('emb/top1', 'emb/top5'): uses the ArcFace classifier's
    per-class weight as the anchor DB (doc/28 §6.1 'weight' mode). For each
    sample we compute cos(emb, W) over all 98169 classes and rank. This
    measures whether the embedding head actually clusters into the per-class
    anchors. After training this becomes the deploy-time NN search.
    """
    model.eval()
    n = 0
    char_correct = {k: 0 for k in topk}
    emb_correct = {k: 0 for k in topk}
    rad_correct = 0
    rad_valid_n = 0
    idc_correct = 0
    idc_valid_n = 0
    total_mae_sum = 0.0
    total_mae_n = 0
    resid_mae_sum = 0.0
    resid_mae_n = 0
    k_max = max(topk)

    # Pre-normalize anchors once (assumes ArcMarginProduct semantics)
    W = model.arc_classifier.weight.detach()                  # (C, D)
    W_norm = F.normalize(W, dim=1)                            # (C, D)

    with _nvtx("evaluate"):
        for x, y in loader:
            with _nvtx("eval_h2d"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if gpu_transform is not None:
                    x = gpu_transform(x)
                if channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                aux = aux_table.get_aux(y)
            with _nvtx("eval_forward"):
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    out = model(x, labels=None)               # no margin, just emb + char + struct

            # char top-k (legacy — to compare against v3 baseline)
            _, pred = out["char"].topk(k_max, dim=1)
            match = pred == y.unsqueeze(1)
            for k in topk:
                char_correct[k] += match[:, :k].any(dim=1).sum().item()

            # embedding top-k (cosine sim vs ArcFace anchors)
            emb_norm = out["embedding"]                        # already L2 normalized
            cos = emb_norm @ W_norm.t()                       # (N, C)
            _, emb_pred = cos.topk(k_max, dim=1)
            emb_match = emb_pred == y.unsqueeze(1)
            for k in topk:
                emb_correct[k] += emb_match[:, :k].any(dim=1).sum().item()

            # structure heads (masked)
            rad_mask = aux.valid[:, 0]
            if rad_mask.any():
                rad_pred = out["radical"].argmax(dim=1)
                rad_correct += ((rad_pred == aux.radical) & rad_mask).sum().item()
                rad_valid_n += rad_mask.sum().item()

            idc_mask = aux.valid[:, 3]
            if idc_mask.any():
                idc_pred = out["ids_top_idc"].argmax(dim=1)
                idc_correct += ((idc_pred == aux.idc) & idc_mask).sum().item()
                idc_valid_n += idc_mask.sum().item()

            t_mask = aux.valid[:, 1]
            if t_mask.any():
                diff = (out["total_strokes"] - aux.total).abs()
                total_mae_sum += (diff * t_mask.float()).sum().item()
                total_mae_n += t_mask.sum().item()
            r_mask = aux.valid[:, 2]
            if r_mask.any():
                diff = (out["residual_strokes"] - aux.residual).abs()
                resid_mae_sum += (diff * r_mask.float()).sum().item()
                resid_mae_n += r_mask.sum().item()

            n += y.size(0)

    out_dict = {f"char/top{k}": char_correct[k] / max(n, 1) for k in topk}
    out_dict.update({f"emb/top{k}": emb_correct[k] / max(n, 1) for k in topk})
    out_dict["radical/top1"] = rad_correct / max(rad_valid_n, 1)
    out_dict["idc/top1"] = idc_correct / max(idc_valid_n, 1)
    out_dict["total_strokes/mae"] = total_mae_sum / max(total_mae_n, 1)
    out_dict["residual_strokes/mae"] = resid_mae_sum / max(resid_mae_n, 1)
    return out_dict
