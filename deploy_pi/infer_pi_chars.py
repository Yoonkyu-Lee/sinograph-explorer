"""SCER inference on real character images (Pi/Coral).

Reads PNG files where filename = ground-truth character (e.g. 三.png).
Runs the full SCER deploy pipeline:
    image → INT8 forward → embedding → cosine NN over anchor DB → top-5 chars

Compares CPU vs Coral on the same images. Reports per-image:
    GT char | top-1 pred | top-5 chars | rank of GT in full table | latency

Usage:
    python deploy_pi/infer_pi_chars.py \\
        --image-dir ~/ece479/test \\
        --tflite-cpu scer/scer_int8.tflite \\
        --tflite-coral scer/scer_int8_edgetpu.tflite \\
        --anchors scer/scer_anchor_db.npy \\
        --class-index scer/class_index.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from ai_edge_litert.interpreter import Interpreter, load_delegate


def char_to_codepoint_key(c: str) -> str:
    return f"U+{ord(c):04X}"


def codepoint_key_to_char(k: str) -> str:
    """Convert 'U+XXXX' string to the character itself."""
    if not k.startswith("U+"):
        return "?"
    try:
        return chr(int(k[2:], 16))
    except (ValueError, OverflowError):
        return "?"


def quantize_input(x_uint8: np.ndarray, in_d) -> np.ndarray:
    f = x_uint8.astype(np.float32) / 255.0
    f = (f - 0.5) / 0.5
    scale, zp = in_d["quantization"]
    return np.round(f / scale + zp).clip(-128, 127).astype(np.int8)


def dequantize_emb(y_int8: np.ndarray, out_d) -> np.ndarray:
    scale, zp = out_d["quantization"]
    y = (y_int8.astype(np.float32) - zp) * scale
    norm = np.linalg.norm(y, axis=-1, keepdims=True).clip(min=1e-8)
    return y / norm


def find_emb_output(interp):
    for d in interp.get_output_details():
        if list(d["shape"][-1:]) == [128]:
            return d
    raise RuntimeError("no 128-d output")


def load_image_for_model(path: Path, input_size: int = 128) -> np.ndarray:
    """Load PNG → RGB 128×128 uint8 NHWC (1, H, W, 3)."""
    img = Image.open(path).convert("RGB")
    if img.size != (input_size, input_size):
        img = img.resize((input_size, input_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)            # (H, W, 3)
    return arr[None, ...]                           # (1, H, W, 3)


def run_one(interp, in_d, emb_d, x_q: np.ndarray, anchors: np.ndarray,
             topk: int = 5):
    """Single forward + cosine NN. Returns (top5_idx, top5_sim, fwd_ms)."""
    t0 = time.perf_counter()
    interp.set_tensor(in_d["index"], x_q)
    interp.invoke()
    emb_int8 = interp.get_tensor(emb_d["index"])
    fwd_ms = (time.perf_counter() - t0) * 1000

    emb = dequantize_emb(emb_int8, emb_d)           # (1, 128) L2-norm
    sims = emb @ anchors.T                           # (1, C)
    top_idx = np.argsort(-sims, axis=1)[0, :topk]
    top_sim = sims[0, top_idx]
    return top_idx, top_sim, fwd_ms


def find_rank(emb: np.ndarray, anchors: np.ndarray, gt_idx: int) -> int:
    """Where does gt_idx rank when sorted by cosine sim? 0-indexed."""
    sims = emb @ anchors.T                           # (1, C)
    rank = (sims[0] > sims[0, gt_idx]).sum()         # # of classes with higher sim
    return int(rank)


def setup_interp(model_path: Path, use_coral: bool):
    delegates = [load_delegate("libedgetpu.so.1")] if use_coral else []
    interp = Interpreter(model_path=str(model_path),
                          experimental_delegates=delegates)
    interp.allocate_tensors()
    return interp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--tflite-cpu", default=None)
    ap.add_argument("--tflite-coral", default=None)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--class-index", required=True)
    ap.add_argument("--input-size", type=int, default=128)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    print(f"[infer] image_dir = {args.image_dir}")

    class_index = json.loads(Path(args.class_index).read_text(encoding="utf-8"))
    idx_to_key = {v: k for k, v in class_index.items()}            # int → "U+XXXX"
    print(f"[infer] class_index n={len(class_index)}")

    anchors = np.load(args.anchors)                                # (C, D) fp32
    print(f"[infer] anchors {anchors.shape}")

    # collect images
    img_dir = Path(args.image_dir).expanduser()
    pngs = sorted(img_dir.glob("*.png"))
    print(f"[infer] {len(pngs)} PNG files in {img_dir}")

    interp_cpu = None
    interp_coral = None
    in_d_cpu = emb_d_cpu = None
    in_d_coral = emb_d_coral = None
    if args.tflite_cpu:
        interp_cpu = setup_interp(Path(args.tflite_cpu), use_coral=False)
        in_d_cpu = interp_cpu.get_input_details()[0]
        emb_d_cpu = find_emb_output(interp_cpu)
        # Warmup
        x0 = quantize_input(np.zeros((1, args.input_size, args.input_size, 3),
                                       dtype=np.uint8), in_d_cpu)
        for _ in range(3):
            interp_cpu.set_tensor(in_d_cpu["index"], x0); interp_cpu.invoke()
    if args.tflite_coral:
        interp_coral = setup_interp(Path(args.tflite_coral), use_coral=True)
        in_d_coral = interp_coral.get_input_details()[0]
        emb_d_coral = find_emb_output(interp_coral)
        x0 = quantize_input(np.zeros((1, args.input_size, args.input_size, 3),
                                       dtype=np.uint8), in_d_coral)
        for _ in range(3):
            interp_coral.set_tensor(in_d_coral["index"], x0); interp_coral.invoke()

    # Header
    cpu_correct1 = cpu_correct5 = 0
    coral_correct1 = coral_correct5 = 0
    cpu_t_sum = coral_t_sum = 0.0
    n = 0

    print()
    if interp_cpu and interp_coral:
        print(f"{'GT':>3}  {'CPU top-5 (sim)':<60}  {'Coral top-5 (sim)':<60}  rank_CPU rank_Coral")
    elif interp_cpu:
        print(f"{'GT':>3}  {'CPU top-5 (sim)':<70}  rank")
    print("-" * 140)

    for png in pngs:
        gt_char = png.stem
        gt_key = char_to_codepoint_key(gt_char)
        gt_idx = class_index.get(gt_key, None)
        if gt_idx is None:
            print(f"  {gt_char}  [GT not in class_index, skip]")
            continue
        n += 1

        x_uint8 = load_image_for_model(png, input_size=args.input_size)

        cpu_pred_str = ""
        coral_pred_str = ""
        cpu_rank = coral_rank = -1

        if interp_cpu:
            x_q = quantize_input(x_uint8, in_d_cpu)
            top_idx, top_sim, fwd_ms = run_one(interp_cpu, in_d_cpu, emb_d_cpu,
                                                  x_q, anchors, topk=args.topk)
            cpu_t_sum += fwd_ms
            top_chars = [codepoint_key_to_char(idx_to_key[int(i)])
                         for i in top_idx]
            cpu_pred_str = " ".join(f"{c}({s:.2f})"
                                    for c, s in zip(top_chars, top_sim))
            # Use the same emb for rank
            emb_int8 = interp_cpu.get_tensor(emb_d_cpu["index"])
            emb = dequantize_emb(emb_int8, emb_d_cpu)
            cpu_rank = find_rank(emb, anchors, gt_idx)
            if int(top_idx[0]) == gt_idx:
                cpu_correct1 += 1
            if gt_idx in top_idx:
                cpu_correct5 += 1

        if interp_coral:
            x_q = quantize_input(x_uint8, in_d_coral)
            top_idx, top_sim, fwd_ms = run_one(interp_coral, in_d_coral, emb_d_coral,
                                                  x_q, anchors, topk=args.topk)
            coral_t_sum += fwd_ms
            top_chars = [codepoint_key_to_char(idx_to_key[int(i)])
                         for i in top_idx]
            coral_pred_str = " ".join(f"{c}({s:.2f})"
                                      for c, s in zip(top_chars, top_sim))
            emb_int8 = interp_coral.get_tensor(emb_d_coral["index"])
            emb = dequantize_emb(emb_int8, emb_d_coral)
            coral_rank = find_rank(emb, anchors, gt_idx)
            if int(top_idx[0]) == gt_idx:
                coral_correct1 += 1
            if gt_idx in top_idx:
                coral_correct5 += 1

        if interp_cpu and interp_coral:
            print(f"  {gt_char}  {cpu_pred_str:<60}  {coral_pred_str:<60}  "
                  f"{cpu_rank:>5}    {coral_rank:>5}")
        elif interp_cpu:
            print(f"  {gt_char}  {cpu_pred_str:<70}  {cpu_rank}")

    print()
    print("=" * 70)
    if interp_cpu and n > 0:
        print(f"CPU      top-1 = {cpu_correct1}/{n} ({cpu_correct1/n*100:.1f}%)  "
              f"top-5 = {cpu_correct5}/{n} ({cpu_correct5/n*100:.1f}%)  "
              f"avg fwd = {cpu_t_sum/n:.2f} ms")
    if interp_coral and n > 0:
        print(f"Coral    top-1 = {coral_correct1}/{n} ({coral_correct1/n*100:.1f}%)  "
              f"top-5 = {coral_correct5}/{n} ({coral_correct5/n*100:.1f}%)  "
              f"avg fwd = {coral_t_sum/n:.2f} ms")


if __name__ == "__main__":
    main()
