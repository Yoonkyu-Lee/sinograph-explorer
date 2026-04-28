"""Predict on real test images with char-only + multi-head fusion (doc/19 §6.8).

Loads a trained best.pt + class_index + aux_labels.npz, splits each test
image into character cells (assuming horizontal arrangement, ~square cells),
and reports top-k under two scoring rules:

  (1) char-only:    argmax over char head logits
  (2) fusion:       char + α·log p_radical(rad(c)) + β·log p_idc(idc(c))
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from train_engine_v3.modules.aux_labels import AuxTable           # noqa: E402
from train_engine_v3.modules.model import build_model             # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def estimate_char_count(img: Image.Image, char_aspect: float = 0.85) -> int:
    w, h = img.size
    return max(1, round(w / (h * char_aspect)))


def split_image(img: Image.Image, n: int) -> list[Image.Image]:
    w, h = img.size
    cell_w = w // n
    cells = []
    for i in range(n):
        x0 = i * cell_w
        x1 = (i + 1) * cell_w if i < n - 1 else w
        cells.append(img.crop((x0, 0, x1, h)))
    return cells


def preprocess(cell: Image.Image, input_size: int = 128) -> torch.Tensor:
    cell = cell.convert("RGB")
    w, h = cell.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), color=(255, 255, 255))
    canvas.paste(cell, ((side - w) // 2, (side - h) // 2))
    canvas = canvas.resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(canvas, dtype=np.uint8).copy().transpose(2, 0, 1)
    t = torch.from_numpy(arr).unsqueeze(0).float().div_(255.0)
    t.sub_(0.5).div_(0.5)
    return t


def topk_char(logits: torch.Tensor, idx_to_char: list[str], k: int):
    probs = F.softmax(logits, dim=-1)
    vals, idx = probs.topk(k)
    return [(int(i), idx_to_char[int(i)], float(v)) for i, v in zip(idx, vals)]


def fuse_logits(z_char, z_rad, z_idc, rad_lookup, idc_lookup,
                valid_rad, valid_idc, alpha, beta):
    log_p_char = F.log_softmax(z_char, dim=-1)
    log_p_rad = F.log_softmax(z_rad, dim=-1)
    log_p_idc = F.log_softmax(z_idc, dim=-1)
    safe_rad = rad_lookup.clamp(min=0)
    safe_idc = idc_lookup.clamp(min=0)
    rad_term = log_p_rad[safe_rad]
    idc_term = log_p_idc[safe_idc]
    rad_term = torch.where(valid_rad, rad_term, torch.zeros_like(rad_term))
    idc_term = torch.where(valid_idc, idc_term, torch.zeros_like(idc_term))
    return log_p_char + alpha * rad_term + beta * idc_term


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--class-index", required=True)
    ap.add_argument("--aux-labels", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--char-aspect", type=float, default=0.85)
    ap.add_argument("--n-chars", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[predict] device = {device}")

    class_index = json.loads(Path(args.class_index).read_text(encoding="utf-8"))
    n_class = len(class_index)
    idx_to_char = [""] * n_class
    for notation, idx in class_index.items():
        cp = int(notation[2:], 16)
        idx_to_char[int(idx)] = chr(cp)
    print(f"[predict] n_class = {n_class}")

    model = build_model("resnet18", num_classes=n_class).to(device).eval()
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ck["model"]
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    print(f"[predict] loaded {args.ckpt} (epoch {ck.get('epoch')}, "
          f"best {ck.get('best_metric_key')}={ck.get('best_metric_value', 0):.4f})")

    aux = AuxTable.from_npz(args.aux_labels, expected_class_index=class_index, device=device)
    rad_lookup = aux.radical
    idc_lookup = aux.idc
    valid_rad = aux.valid[:, 0].to(torch.bool)
    valid_idc = aux.valid[:, 3].to(torch.bool)
    print(f"[predict] aux table loaded; rad valid {valid_rad.sum().item()}/{n_class}, "
          f"idc valid {valid_idc.sum().item()}/{n_class}")
    print(f"[predict] fusion: alpha (rad) = {args.alpha}, beta (idc) = {args.beta}")

    image_paths = sorted(Path().glob(args.images))
    if not image_paths:
        print(f"[predict] no images matched {args.images!r}", file=sys.stderr)
        sys.exit(1)
    print(f"[predict] {len(image_paths)} images found\n")

    # Track aggregate hit rates if filenames are single-char (= ground truth)
    stats = {"char_top1": 0, "char_top5": 0, "fusion_top1": 0, "fusion_top5": 0,
             "total": 0, "single_char_gt": 0}

    for path in image_paths:
        img = Image.open(path)
        n_chars = args.n_chars if args.n_chars > 0 else estimate_char_count(img, args.char_aspect)
        cells = split_image(img, n_chars)
        # Ground truth from filename stem (works only when filename is the char itself)
        gt = path.stem if len(path.stem) == 1 and n_chars == 1 else None
        gt_str = f" gt={gt}" if gt else ""
        print("=" * 80)
        print(f"  IMAGE  {path.name}    size={img.size}    n_chars={n_chars}{gt_str}")
        print("=" * 80)

        for ci, cell in enumerate(cells):
            x = preprocess(cell, args.input_size).to(device)
            with torch.no_grad():
                logits = model(x)
                z_char = logits["char"][0]
                z_rad = logits["radical"][0]
                z_idc = logits["ids_top_idc"][0]
                z_total = float(logits["total_strokes"][0])
                z_residual = float(logits["residual_strokes"][0])

            char_only = topk_char(z_char, idx_to_char, args.topk)
            fused = fuse_logits(z_char, z_rad, z_idc, rad_lookup, idc_lookup,
                                 valid_rad, valid_idc, args.alpha, args.beta)
            f_vals, f_idx = fused.topk(args.topk)
            fusion_topk = [(int(i), idx_to_char[int(i)], float(v)) for i, v in zip(f_idx, f_vals)]

            rad_vals, rad_idx = F.softmax(z_rad, dim=-1).topk(3)
            idc_vals, idc_idx = F.softmax(z_idc, dim=-1).topk(3)

            if n_chars > 1:
                print(f"  --- cell {ci+1}/{n_chars} ---")
            print(f"    aux: rad top-3 = {[(int(i)+1, f'{float(v):.2f}') for i,v in zip(rad_idx, rad_vals)]}")
            print(f"    aux: idc top-3 = {[(int(i), f'{float(v):.2f}') for i,v in zip(idc_idx, idc_vals)]}  "
                  f"(0=⿰ 1=⿱ 2=⿲ 3=⿳ 4=⿴ 5=⿵ 6=⿶ 7=⿷ 8=⿸ 9=⿹ 10=⿺ 11=⿻)")
            print(f"    aux: total_strokes={z_total:.1f}  residual={z_residual:.1f}")

            # Per-cell hit detection (only if filename = single-char ground truth)
            cot1 = cot5 = fst1 = fst5 = "  "
            if gt is not None and ci == 0:
                stats["single_char_gt"] += 1
                co_chars = [c for _, c, _ in char_only]
                fu_chars = [c for _, c, _ in fusion_topk]
                if co_chars and co_chars[0] == gt:
                    stats["char_top1"] += 1; cot1 = "✓ "
                if gt in co_chars:
                    stats["char_top5"] += 1; cot5 = "✓ "
                if fu_chars and fu_chars[0] == gt:
                    stats["fusion_top1"] += 1; fst1 = "✓ "
                if gt in fu_chars:
                    stats["fusion_top5"] += 1; fst5 = "✓ "

            print(f"    char-only top-{args.topk}:  {cot1}top-1   {cot5}top-{args.topk}")
            for i, c, p in char_only:
                mark = "  ←GT" if (gt and c == gt) else ""
                print(f"      U+{ord(c):05X}  {c}   p={p:.4f}    (idx {i}){mark}")
            print(f"    fusion top-{args.topk} (α={args.alpha}, β={args.beta}):  {fst1}top-1   {fst5}top-{args.topk}")
            for i, c, p in fusion_topk:
                mark = "  ←GT" if (gt and c == gt) else ""
                print(f"      U+{ord(c):05X}  {c}   score={p:.3f}    (idx {i}){mark}")
            print()
            stats["total"] += 1

    # Aggregate report
    if stats["single_char_gt"] > 0:
        n = stats["single_char_gt"]
        print("=" * 80)
        print(f"  AGGREGATE — {n} single-char images with filename ground truth")
        print("=" * 80)
        print(f"  char-only  top-1: {stats['char_top1']:>3d}/{n}  ({100*stats['char_top1']/n:5.1f}%)")
        print(f"  char-only  top-{args.topk}: {stats['char_top5']:>3d}/{n}  ({100*stats['char_top5']/n:5.1f}%)")
        print(f"  fusion     top-1: {stats['fusion_top1']:>3d}/{n}  ({100*stats['fusion_top1']/n:5.1f}%)")
        print(f"  fusion     top-{args.topk}: {stats['fusion_top5']:>3d}/{n}  ({100*stats['fusion_top5']/n:5.1f}%)")
        gain = stats['fusion_top5'] - stats['char_top5']
        print(f"  fusion gain over char-only (top-{args.topk}): +{gain} ({100*gain/n:+5.1f}pp)")


if __name__ == "__main__":
    main()
