"""Multi-task train / eval loop (doc/19 §3, §5.3).

train_one_epoch:
    L = CE(char) + 0.2·CE(radical) + 0.1·MSE(total) + 0.1·MSE(residual)
      + 0.2·CE(idc)

Each aux head is masked out per-sample using `aux.valid[:, head_idx]` so
classes without canonical annotation don't corrupt the gradient. The char
head is always active (every sample has a char label).

Logged metrics (every `log_every` steps):
    char/top1, char/top5   — primary
    radical/top1           — aux sanity
    total_mae, resid_mae   — regression aux
    idc/top1               — layout aux
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .aux_labels import AuxTable
from .sysmon import format_snapshot


# Loss weights (doc/19 §3). Single source of truth — override via config at
# call site if needed (train driver passes LossWeights in config).
@dataclass
class LossWeights:
    char: float = 1.0
    radical: float = 0.2
    total: float = 0.1
    residual: float = 0.1
    idc: float = 0.2


def _masked_ce(logits: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged over samples where mask=True. Target values at
    mask=False positions are ignored (we clamp them to 0 to avoid gather
    errors, then zero out the loss). Returns 0 if mask is all-False."""
    if not mask.any():
        return logits.new_zeros(())
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    per_sample = F.cross_entropy(logits, safe_target, reduction="none")
    return (per_sample * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 (Huber β=1.0) — behaves like MSE for small errors and L1 for
    large ones. doc/19 §3 specifies MSE for stroke heads but raw MSE explodes
    on the 1..84 target range (peak loss ≈ target² ≈ 7k), dwarfing the CE
    terms. Smooth-L1 keeps the numerical scale O(target) and is the standard
    choice for stroke-count regression. Semantically equivalent to the doc."""
    if not mask.any():
        return pred.new_zeros(())
    per_sample = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0)
    return (per_sample * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def compute_multitask_loss(
    logits: dict[str, torch.Tensor],
    y_char: torch.Tensor,
    aux,                              # AuxBatch
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_char = F.cross_entropy(logits["char"], y_char, label_smoothing=0.1)
    l_rad = _masked_ce(logits["radical"], aux.radical,  aux.valid[:, 0])
    l_tot = _masked_smooth_l1(logits["total_strokes"],    aux.total,    aux.valid[:, 1])
    l_res = _masked_smooth_l1(logits["residual_strokes"], aux.residual, aux.valid[:, 2])
    l_idc = _masked_ce(logits["ids_top_idc"], aux.idc, aux.valid[:, 3])

    total = (
        weights.char * l_char
        + weights.radical * l_rad
        + weights.total * l_tot
        + weights.residual * l_res
        + weights.idc * l_idc
    )
    parts = {
        "loss": total.item(),
        "l_char": l_char.item(),
        "l_radical": l_rad.item(),
        "l_total": l_tot.item(),
        "l_residual": l_res.item(),
        "l_idc": l_idc.item(),
    }
    return total, parts


def train_one_epoch(
    model, loader, optimizer, scaler, device,
    aux_table: AuxTable,
    weights: LossWeights,
    log_every: int = 50,
    sysmon=None,
    gpu_transform=None,
):
    model.train()
    if sysmon is not None:
        sysmon.reset_peaks()

    sum_loss = 0.0
    sum_n = 0
    t0 = time.time()
    use_amp = scaler is not None

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if gpu_transform is not None:
            x = gpu_transform(x)
        aux = aux_table.get_aux(y)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss, parts = compute_multitask_loss(logits, y, aux, weights)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        sum_loss += parts["loss"] * bs
        sum_n += bs

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = sum_n / max(elapsed, 1e-6)
            steps_left = max(len(loader) - (i + 1), 0)
            eta = steps_left * (elapsed / max(i + 1, 1))
            ts = time.strftime("%H:%M:%S")
            msg = (
                f"[{ts}] step {i+1}/{len(loader)}  "
                f"loss={sum_loss/sum_n:.4f}  "
                f"char={parts['l_char']:.3f} rad={parts['l_radical']:.3f} "
                f"tot={parts['l_total']:.3f} res={parts['l_residual']:.3f} "
                f"idc={parts['l_idc']:.3f}  "
                f"({rate:.1f} img/s, t={elapsed:.0f}s, eta={eta:.0f}s)"
            )
            if sysmon is not None:
                msg += "  " + format_snapshot(sysmon.snapshot())
            print(msg, flush=True)
    return sum_loss / max(sum_n, 1)


@torch.no_grad()
def evaluate(
    model, loader, device,
    aux_table: AuxTable,
    topk: tuple = (1, 5),
    gpu_transform=None,
):
    model.eval()
    n = 0
    char_correct = {k: 0 for k in topk}
    rad_correct = 0
    rad_valid_n = 0
    idc_correct = 0
    idc_valid_n = 0
    total_mae_sum = 0.0
    total_mae_n = 0
    resid_mae_sum = 0.0
    resid_mae_n = 0
    k_max = max(topk)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if gpu_transform is not None:
            x = gpu_transform(x)
        aux = aux_table.get_aux(y)
        logits = model(x)

        # char top-k
        _, pred = logits["char"].topk(k_max, dim=1)
        match = pred == y.unsqueeze(1)
        for k in topk:
            char_correct[k] += match[:, :k].any(dim=1).sum().item()

        # radical top-1 (masked)
        rad_mask = aux.valid[:, 0]
        if rad_mask.any():
            rad_pred = logits["radical"].argmax(dim=1)
            rad_correct += ((rad_pred == aux.radical) & rad_mask).sum().item()
            rad_valid_n += rad_mask.sum().item()

        # idc top-1 (masked)
        idc_mask = aux.valid[:, 3]
        if idc_mask.any():
            idc_pred = logits["ids_top_idc"].argmax(dim=1)
            idc_correct += ((idc_pred == aux.idc) & idc_mask).sum().item()
            idc_valid_n += idc_mask.sum().item()

        # stroke MAE (masked)
        t_mask = aux.valid[:, 1]
        if t_mask.any():
            diff = (logits["total_strokes"] - aux.total).abs()
            total_mae_sum += (diff * t_mask.float()).sum().item()
            total_mae_n += t_mask.sum().item()
        r_mask = aux.valid[:, 2]
        if r_mask.any():
            diff = (logits["residual_strokes"] - aux.residual).abs()
            resid_mae_sum += (diff * r_mask.float()).sum().item()
            resid_mae_n += r_mask.sum().item()

        n += y.size(0)

    out = {f"char/top{k}": char_correct[k] / max(n, 1) for k in topk}
    out["radical/top1"] = rad_correct / max(rad_valid_n, 1)
    out["idc/top1"] = idc_correct / max(idc_valid_n, 1)
    out["total_strokes/mae"] = total_mae_sum / max(total_mae_n, 1)
    out["residual_strokes/mae"] = resid_mae_sum / max(resid_mae_n, 1)
    return out
