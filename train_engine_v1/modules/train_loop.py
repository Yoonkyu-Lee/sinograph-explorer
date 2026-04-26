import time

import torch
import torch.nn.functional as F

from .sysmon import format_snapshot


def train_one_epoch(model, loader, optimizer, scaler, device,
                    label_smoothing=0.1, log_every=50, sysmon=None,
                    gpu_transform=None):
    """If `gpu_transform` is provided, it's applied to each batch after
    moving to device. Used for deferred resize/normalize on GPU side."""
    model.train()
    total_loss = 0.0
    total_n = 0
    t0 = time.time()
    use_amp = scaler is not None
    if sysmon is not None:
        sysmon.reset_peaks()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if gpu_transform is not None:
            x = gpu_transform(x)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = total_n / elapsed
            steps_left = len(loader) - (i + 1)
            eta = steps_left * (elapsed / (i + 1))
            ts = time.strftime("%H:%M:%S")
            msg = (
                f"[{ts}] step {i+1}/{len(loader)}  "
                f"loss={total_loss/total_n:.4f}  "
                f"({rate:.1f} img/s, t={elapsed:.0f}s, eta={eta:.0f}s)"
            )
            if sysmon is not None:
                snap = sysmon.snapshot()
                msg += "  " + format_snapshot(snap)
            print(msg)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device, topk=(1, 5), gpu_transform=None):
    model.eval()
    n = 0
    correct = {k: 0 for k in topk}
    k_max = max(topk)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if gpu_transform is not None:
            x = gpu_transform(x)
        logits = model(x)
        _, pred = logits.topk(k_max, dim=1)
        match = pred == y.unsqueeze(1)
        for k in topk:
            correct[k] += match[:, :k].any(dim=1).sum().item()
        n += y.size(0)
    return {f"top{k}": correct[k] / max(n, 1) for k in topk}
