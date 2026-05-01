"""ArcFace gather/scatter refactor parity test.

Implements the OLD (5-buffer) form inline and compares against the NEW
(gather/scatter) form. Both should produce identical logits up to fp32
accumulation order error (< 1e-5).

Usage:
    python train_engine_v4/scripts/test_arcface_parity.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from train_engine_v4.modules.arcface import ArcMarginProduct       # noqa: E402


def _old_forward(arc: ArcMarginProduct, emb_norm, labels):
    """Reproduce the pre-refactor forward (5 N×C buffers)."""
    W_norm = F.normalize(arc.weight, dim=1)
    cos = F.linear(emb_norm, W_norm)
    cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    sin = torch.sqrt(1.0 - cos.pow(2))
    phi = cos * arc._cos_m - sin * arc._sin_m
    if arc.easy_margin:
        phi = torch.where(cos > 0, phi, cos)
    else:
        phi = torch.where(cos > arc._th, phi, cos - arc._mm)
    one_hot = torch.zeros_like(cos)
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)
    out = one_hot * phi + (1.0 - one_hot) * cos
    return out * arc.s


def run_case(name, *, n=8, c=100, d=128, s=30.0, m=0.5, easy_margin=False):
    torch.manual_seed(0)
    arc = ArcMarginProduct(emb_dim=d, n_classes=c, s=s, m=m,
                            easy_margin=easy_margin)
    arc.eval()

    emb = torch.randn(n, d)
    emb = F.normalize(emb, dim=1)
    labels = torch.randint(0, c, (n,))

    with torch.no_grad():
        new = arc.forward(emb, labels)
        old = _old_forward(arc, emb, labels)

    diff = (new - old).abs()
    max_d = float(diff.max())
    mean_d = float(diff.mean())
    ok = max_d < 1e-4
    flag = "✓" if ok else "✗"
    print(f"  {flag} {name:35s}  max={max_d:.2e}  mean={mean_d:.2e}  "
          f"shape={tuple(new.shape)}  {'PASS' if ok else 'FAIL'}")

    # Sanity: gradient parity through both paths
    emb_g = emb.clone().requires_grad_(True)
    arc_new = ArcMarginProduct(emb_dim=d, n_classes=c, s=s, m=m,
                                easy_margin=easy_margin)
    arc_new.load_state_dict(arc.state_dict())
    out_new = arc_new(emb_g, labels)
    grad_target = torch.randn_like(out_new)
    (out_new * grad_target).sum().backward()
    grad_new = emb_g.grad.clone()
    weight_grad_new = arc_new.weight.grad.clone()

    emb_g2 = emb.clone().requires_grad_(True)
    arc_old = ArcMarginProduct(emb_dim=d, n_classes=c, s=s, m=m,
                                easy_margin=easy_margin)
    arc_old.load_state_dict(arc.state_dict())
    out_old = _old_forward(arc_old, emb_g2, labels)
    (out_old * grad_target).sum().backward()
    grad_old = emb_g2.grad.clone()
    weight_grad_old = arc_old.weight.grad.clone()

    grad_max = float((grad_new - grad_old).abs().max())
    wgrad_max = float((weight_grad_new - weight_grad_old).abs().max())
    grad_ok = grad_max < 1e-4 and wgrad_max < 1e-4
    flag2 = "✓" if grad_ok else "✗"
    print(f"  {flag2} {name:35s}  emb_grad max={grad_max:.2e}  "
          f"weight_grad max={wgrad_max:.2e}  {'PASS' if grad_ok else 'FAIL'}")

    return ok and grad_ok


def main():
    print("=" * 70)
    print("ArcFace gather/scatter parity test")
    print("=" * 70)
    cases = [
        ("easy_margin=False, m=0.5",
         dict(n=8, c=100, d=128, s=30.0, m=0.5, easy_margin=False)),
        ("easy_margin=True,  m=0.3",
         dict(n=8, c=100, d=128, s=30.0, m=0.3, easy_margin=True)),
        ("easy_margin=False, m=0.4 (transition)",
         dict(n=8, c=200, d=128, s=30.0, m=0.4, easy_margin=False)),
        ("large batch, n=64",
         dict(n=64, c=500, d=128, s=30.0, m=0.5, easy_margin=False)),
    ]
    all_ok = True
    for name, kw in cases:
        ok = run_case(name, **kw)
        all_ok = all_ok and ok
    print("=" * 70)
    print("✅ ALL PASS" if all_ok else "❌ FAIL — gather/scatter math differs")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
