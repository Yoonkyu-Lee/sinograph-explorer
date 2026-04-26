"""Side-by-side comparison: EasyOCR vs our trained model.

EasyOCR is a mainstream open-source OCR (https://github.com/JaidedAI/EasyOCR)
with CJK support. It's a reasonable proxy for "what commodity OCR does"
without requiring cloud APIs. We run it on the same test images we've been
using and compare against our v2 checkpoint.

Focus cases where we expect EasyOCR to struggle:
  - Korean-specific hanja (畓, 媤) — not Chinese/Japanese curriculum
  - CJK Ext B (𤴡) — too rare for most OCR
  - Low-quality signs / OOD styles
  - Variant characters (斈)

Usage:
  python 32_easyocr_compare.py \
    --run-dir .../out/03_v3r_prod_t1 \
    --images train_engine_v1/test_img/*.png \
    [--lang-list ch_tra,ch_sim,ja,ko,en]
"""
from __future__ import annotations

import argparse
import glob
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from modules.model import build_model


def _parse_expected_codepoint(filename: str) -> str | None:
    """Extract expected codepoint from filename. Patterns we handle:
      - 24D21.png                → U+24D21
      - crop_sign_mu.png         → None (no codepoint in name)
      - 시집 시.png              → None (infer manually by caller)
    Caller can pass --expected dict for non-codepoint filenames.
    """
    import re
    m = re.search(r"([0-9A-Fa-f]{4,6})\.(?:png|jpg|jpeg|webp)$", filename)
    if m:
        return f"U+{m.group(1).upper()}"
    return None


def predict_ours(model, class_index_list, device, img_pil, input_size) -> list[tuple[str, float]]:
    """Run our classifier, return top-5 as [(notation, prob), ...]."""
    cpu_t = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.PILToTensor(),
    ])
    x = cpu_t(img_pil).unsqueeze(0).to(device)
    x = x.float().div_(255.0).sub_(0.5).div_(0.5)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        top_p, top_i = probs[0].topk(5)
    return [(class_index_list[i], float(p))
            for p, i in zip(top_p.cpu().tolist(), top_i.cpu().tolist())]


def predict_easyocr(readers: list, img_pil: Image.Image) -> list[str]:
    """Return EasyOCR's text guesses across multiple readers (each for a
    different lang set, since EasyOCR restricts combinations like ch_tra).
    Flat list, duplicates removed (preserving order).

    We pass a numpy array rather than a file path because EasyOCR uses
    cv2.imread internally, which fails on non-ASCII Windows paths (Korean
    filenames like '학.png'). PIL handles them fine."""
    # BGR for OpenCV (EasyOCR's internal expectation)
    arr = np.array(img_pil.convert("RGB"))[:, :, ::-1].copy()
    seen = set()
    out = []
    for r in readers:
        try:
            results = r.readtext(arr, detail=0, paragraph=False)
        except Exception as e:
            results = [f"<ERROR: {e}>"]
        for s in results:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def family_of(cp: str, con: sqlite3.Connection) -> set[str]:
    """Return variant family for a codepoint. Singleton set if none."""
    row = con.execute(
        "SELECT family_members_json FROM variant_components WHERE codepoint=?",
        (cp,),
    ).fetchone()
    if row and row[0]:
        return set(json.loads(row[0]))
    return {cp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="our trained model's run directory (contains best.pth + class_index.json)")
    ap.add_argument("--images", nargs="+", required=True,
                    help="glob(s) of images to test")
    ap.add_argument("--lang-groups", default="ch_sim+en;ch_tra+en;ja+en;ko+en",
                    help="EasyOCR reader groups separated by ';' (e.g. 'ch_sim+en;ko+en'). "
                         "EasyOCR restricts combinations (ch_tra can only pair with en), "
                         "so we run one reader per group and union their outputs.")
    ap.add_argument("--family-db", default=None,
                    help="canonical_v2.sqlite for family-aware comparison")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default=None,
                    help="write results as .jsonl (default: print only)")
    # manual expected codepoints (filename → codepoint). Use when filename
    # doesn't embed the codepoint. Format: "path1=U+4FDD,path2=U+7553,..."
    ap.add_argument("--expected", default="",
                    help="filename=U+XXXX pairs (comma-sep, substring match)")
    args = ap.parse_args()

    # Parse expected mapping
    expected_map = {}
    if args.expected:
        for pair in args.expected.split(","):
            pair = pair.strip()
            if "=" in pair:
                name, cp = pair.split("=", 1)
                expected_map[name.strip()] = cp.strip()

    # Expand globs
    image_paths = []
    for g in args.images:
        image_paths.extend(sorted(glob.glob(g)))
    image_paths = [Path(p) for p in image_paths if Path(p).is_file()]
    if not image_paths:
        raise SystemExit("no images found")
    print(f"[compare] {len(image_paths)} images")

    # Our model
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "best.pth"
    if not ckpt_path.exists():
        ckpt_path = sorted(run_dir.glob("ckpt_epoch_*.pth"))[-1]
    class_index = json.load(open(run_dir / "class_index.json", encoding="utf-8"))
    class_index_list = sorted(class_index.keys(), key=lambda k: class_index[k])
    num_classes = len(class_index)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state["config"]
    model = build_model(cfg["model"]["name"], num_classes).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    input_size = cfg["model"]["input_size"]
    print(f"[compare] our model: {ckpt_path.name}  classes={num_classes}  input={input_size}")

    # EasyOCR — one reader per lang group (EasyOCR restricts combinations)
    import easyocr
    groups = [g.strip() for g in args.lang_groups.split(";") if g.strip()]
    parsed_groups = [[x.strip() for x in g.split("+")] for g in groups]
    print(f"[compare] EasyOCR groups={parsed_groups}  (first run downloads ~100MB)")
    readers = []
    for langs in parsed_groups:
        print(f"  loading reader for {langs}")
        readers.append(easyocr.Reader(langs, gpu=(device.type == "cuda"), verbose=False))

    # Family DB
    con_family = sqlite3.connect(args.family_db) if args.family_db else None

    # Run comparisons
    out_fp = open(args.out, "w", encoding="utf-8") if args.out else None
    rows = []

    for p in image_paths:
        # expected codepoint
        expected = _parse_expected_codepoint(p.name)
        if expected is None:
            for key, cp in expected_map.items():
                if key in p.name:
                    expected = cp
                    break

        img_pil = Image.open(p).convert("RGB")

        # ours
        t0 = time.perf_counter()
        top5 = predict_ours(model, class_index_list, device, img_pil, input_size)
        t_ours = (time.perf_counter() - t0) * 1000

        # easyocr
        t0 = time.perf_counter()
        ez = predict_easyocr(readers, img_pil)
        t_ez = (time.perf_counter() - t0) * 1000

        # analysis
        ours_top1_cp = top5[0][0]
        ours_top1_ch = chr(int(ours_top1_cp[2:], 16)) if ours_top1_cp.startswith("U+") else "?"
        ours_top5_chars = [(chr(int(cp[2:], 16)) if cp.startswith("U+") else "?", p)
                            for cp, p in top5]

        # verdicts
        if expected:
            expected_ch = chr(int(expected[2:], 16))
            ours_correct = (top5[0][0] == expected)
            # family-aware: ok if top-1 is in expected's family
            if con_family:
                fam = family_of(expected, con_family)
                ours_family = (top5[0][0] in fam)
            else:
                ours_family = ours_correct
            # easyocr correct if expected char appears anywhere in its output
            ez_correct = any(expected_ch in t for t in ez) if ez else False
        else:
            expected_ch = "?"
            ours_correct = None
            ours_family = None
            ez_correct = None

        row = {
            "image": p.name,
            "expected": expected,
            "expected_ch": expected_ch,
            "ours_top1_cp": ours_top1_cp,
            "ours_top1_ch": ours_top1_ch,
            "ours_top1_prob": top5[0][1],
            "ours_top5": [{"cp": cp, "ch": ch, "prob": prob} for (cp, prob), (ch, _) in zip(top5, ours_top5_chars)],
            "ours_correct": ours_correct,
            "ours_family_correct": ours_family,
            "ours_ms": t_ours,
            "easyocr_output": ez,
            "easyocr_correct": ez_correct,
            "easyocr_ms": t_ez,
        }
        rows.append(row)
        if out_fp:
            out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"\n[{p.name}]  expected: {expected} '{expected_ch}'")
        print(f"  ours   : '{ours_top1_ch}' ({ours_top1_cp})  prob={top5[0][1]*100:.1f}%  "
              f"correct={ours_correct}  family_ok={ours_family}  ({t_ours:.0f}ms)")
        top5_str = '  '.join(f"{ch}({prob*100:.1f}%)" for ch, prob in ours_top5_chars)
        print(f"  top-5  : {top5_str}")
        print(f"  easyocr: {ez}  correct={ez_correct}  ({t_ez:.0f}ms)")

    if out_fp:
        out_fp.close()
        print(f"\n[compare] wrote {args.out}")

    # summary
    with_expected = [r for r in rows if r["expected"]]
    if with_expected:
        ours_hits = sum(1 for r in with_expected if r["ours_correct"])
        ours_family_hits = sum(1 for r in with_expected if r["ours_family_correct"])
        ez_hits = sum(1 for r in with_expected if r["easyocr_correct"])
        n = len(with_expected)
        print(f"\n=== Summary (n={n} with expected codepoint) ===")
        print(f"  Ours top-1 exact : {ours_hits}/{n} ({100*ours_hits/n:.0f}%)")
        print(f"  Ours family match: {ours_family_hits}/{n} ({100*ours_family_hits/n:.0f}%)")
        print(f"  EasyOCR correct  : {ez_hits}/{n} ({100*ez_hits/n:.0f}%)")


if __name__ == "__main__":
    main()
