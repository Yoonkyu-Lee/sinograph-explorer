"""Stage 1 CPU benchmark — Commodity OCR vs v3 vs v4 SCER.

Runs all three model families on the same N PNG images, then prints a
single tabular comparison: per-image rows + summary footer with top-1,
top-5 (where applicable), avg latency, model size, and architectural
notes.

Designed to be the headline demo bench. Output is plain-text, copy-pastable.

Reuse: ai_edge_litert.Interpreter for both v3 and v4 (Pi has only this).
v3 uses a 98k-class FC head (single-stage); v4 uses 128-d embedding head
+ cosine NN over a precomputed anchor DB (open-set, can grow without
retraining).

The Commodity OCR is pluggable via ocr_adapters.py — to add a new one
just drop a class in there and append to DEFAULT_ADAPTERS.

Usage (Pi):
    ~/venv-ocr/bin/python ~/ece479/demo/bench_cpu_three.py \\
        --image-dir ~/ece479/test \\
        --v3-tflite ~/ece479/lab_v3/v3_char_full_integer_quant.tflite \\
        --v3-class-index ~/ece479/lab_v3/class_index.json \\
        --v4-tflite ~/ece479/scer/scer_int8_v20.tflite \\
        --v4-anchors ~/ece479/scer/scer_anchor_db_v20.npy \\
        --v4-class-index ~/ece479/scer/class_index.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Pi-side: ai_edge_litert is the only available runtime
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from ai_edge_litert.interpreter import Interpreter

# Local
sys.path.insert(0, str(Path(__file__).parent))
from ocr_adapters import resolve_adapter_spec, OCRAdapter  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# Image preprocessing — matches train pipeline
# =============================================================================

def preprocess_for_model(img_path: Path, input_size: int = 128) -> np.ndarray:
    """Load → RGB on white bg → resize 128×128 → uint8 NHWC (1, H, W, 3).

    Returns uint8 because per-model quantization happens after.
    """
    img = Image.open(img_path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (input_size, input_size):
        img = img.resize((input_size, input_size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)[None, ...]  # (1, H, W, 3)


def quantize_input(x_uint8: np.ndarray, in_d) -> np.ndarray:
    """uint8 NHWC → int8 quantized via model's input scale/zp.

    Matches train normalization: x/255 → (x-0.5)/0.5 → [-1, 1].
    Then quantize: round(x/scale + zp) clipped to int8.
    """
    f = x_uint8.astype(np.float32) / 255.0
    f = (f - 0.5) / 0.5
    scale, zp = in_d["quantization"]
    return np.round(f / scale + zp).clip(-128, 127).astype(np.int8)


# =============================================================================
# v3 single-stage FC classifier — ONNX FP32 (INT8 deploy was a known blocker;
# we run the working FP32 artifact for honest baseline comparison)
# =============================================================================

class V3Backend:
    """v3 = backbone + 98k-class FC. ONNX FP32 (NCHW). Forward → argmax."""
    name = "v3"
    notes = "98k-FC, ONNX FP32 (INT8 deploy failed, see doc/24)"

    def __init__(self, onnx_path: Path, class_index_path: Path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.in_name = self.sess.get_inputs()[0].name
        self.classes = json.loads(class_index_path.read_text(encoding="utf-8"))
        self.idx_to_key = {v: k for k, v in self.classes.items()}
        self.size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        # Warmup
        x0 = np.zeros((1, 3, 128, 128), dtype=np.float32)
        for _ in range(3):
            self.sess.run(None, {self.in_name: x0})

    def predict(self, img_uint8: np.ndarray, topk: int = 5
                ) -> Tuple[List[str], float]:
        # uint8 NHWC → float [-1,1] NCHW
        x = img_uint8.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = np.transpose(x, (0, 3, 1, 2)).copy()
        t0 = time.perf_counter()
        logits = self.sess.run(None, {self.in_name: x})[0][0]
        dt = (time.perf_counter() - t0) * 1000.0
        top_idx = np.argsort(-logits)[:topk]
        chars = [_key_to_char(self.idx_to_key.get(int(i), "?"))
                 for i in top_idx]
        return chars, dt


# =============================================================================
# v4 SCER: backbone + 128d emb + cosine NN over anchor DB
# =============================================================================

class V4Backend:
    """v4 SCER = backbone + 128d L2-norm embedding + cosine NN."""
    name = "v4"
    notes = "embedding + cosine NN, open-set"

    def __init__(self, tflite_path: Path, anchors_path: Path,
                 class_index_path: Path):
        self.interp = Interpreter(model_path=str(tflite_path))
        self.interp.allocate_tensors()
        self.in_d = self.interp.get_input_details()[0]
        self.emb_d = _find_emb_output(self.interp)
        self.anchors = np.load(anchors_path)             # (C, 128) fp32 L2-norm
        self.classes = json.loads(class_index_path.read_text(encoding="utf-8"))
        self.idx_to_key = {v: k for k, v in self.classes.items()}
        self.size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        # Warmup
        x0 = np.zeros((1, 128, 128, 3), dtype=np.uint8)
        x_q = quantize_input(x0, self.in_d)
        for _ in range(3):
            self.interp.set_tensor(self.in_d["index"], x_q)
            self.interp.invoke()

    def predict(self, img_uint8: np.ndarray, topk: int = 5
                ) -> Tuple[List[str], float, float]:
        """Returns (top-k chars, fwd_ms, cos_nn_ms)."""
        x_q = quantize_input(img_uint8, self.in_d)
        t0 = time.perf_counter()
        self.interp.set_tensor(self.in_d["index"], x_q)
        self.interp.invoke()
        emb_int8 = self.interp.get_tensor(self.emb_d["index"])
        fwd_ms = (time.perf_counter() - t0) * 1000.0

        scale, zp = self.emb_d["quantization"]
        emb = (emb_int8.astype(np.float32) - zp) * scale
        norm = np.linalg.norm(emb, axis=-1, keepdims=True).clip(min=1e-8)
        emb = emb / norm

        t0 = time.perf_counter()
        sims = emb @ self.anchors.T                      # (1, C)
        top_idx = np.argsort(-sims, axis=1)[0, :topk]
        nn_ms = (time.perf_counter() - t0) * 1000.0

        chars = [_key_to_char(self.idx_to_key.get(int(i), "?"))
                 for i in top_idx]
        return chars, fwd_ms, nn_ms


def _find_emb_output(interp):
    """Locate the (1,128) L2-norm embedding among SCER's 5 outputs."""
    for d in interp.get_output_details():
        if list(d["shape"][-1:]) == [128]:
            return d
    raise RuntimeError("v4 SCER: no 128-d embedding output found")


def _key_to_char(k: str) -> str:
    if not k.startswith("U+"):
        return "?"
    try:
        return chr(int(k[2:], 16))
    except Exception:
        return "?"


# =============================================================================
# Bench loop + table formatting
# =============================================================================

def fmt_pred(pred: str, gt: str) -> str:
    """Render '三 ✓' or '一 ✗' or '"" ✗'."""
    if pred == gt:
        return f"{pred} ✓"
    if not pred:
        return '"" ✗'
    return f"{pred} ✗"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--v3-onnx", required=True,
                    help="v3 ONNX FP32 model (NCHW float input)")
    ap.add_argument("--v3-class-index", required=True)
    ap.add_argument("--v4-tflite", required=True)
    ap.add_argument("--v4-anchors", required=True)
    ap.add_argument("--v4-class-index", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--commodity", default="all",
                    help="adapter spec: 'all' | 'tesseract' | 'easyocr' | "
                         "comma-separated ids (e.g. tesseract,easyocr-ja). "
                         "See ocr_adapters.ADAPTER_SPECS / ADAPTER_GROUPS.")
    ap.add_argument("--limit", type=int, default=0,
                    help="limit to first N images (0 = all)")
    args = ap.parse_args()

    img_dir = Path(args.image_dir).expanduser()
    pngs = sorted(img_dir.glob("*.png"))
    if args.limit > 0:
        pngs = pngs[: args.limit]
    if not pngs:
        sys.exit(f"no PNGs in {img_dir}")

    print(f"[bench] {len(pngs)} images from {img_dir}")
    print(f"[bench] loading models...")

    # Build commodity adapters per spec.
    adapters: List[OCRAdapter] = []
    for aid, cls, kwargs in resolve_adapter_spec(args.commodity):
        try:
            print(f"[bench]   loading commodity: {aid} ...", flush=True)
            adapters.append(cls(**kwargs))
            print(f"[bench]     {adapters[-1].name} ✓")
        except Exception as e:
            print(f"[bench]     {aid} init failed — {e}")

    # v3 + v4
    print(f"[bench]   v3 ONNX FP32 (single-stage 98k-FC)...")
    v3 = V3Backend(Path(args.v3_onnx), Path(args.v3_class_index))
    print(f"[bench]     {v3.size_mb:.1f} MB")
    print(f"[bench]   v4 SCER INT8 TFLite (emb + cosine NN)...")
    v4 = V4Backend(Path(args.v4_tflite), Path(args.v4_anchors),
                   Path(args.v4_class_index))
    print(f"[bench]     {v4.size_mb:.1f} MB,  anchors {v4.anchors.shape}")

    # Warmup commodity adapters on first image
    if adapters and pngs:
        for a in adapters:
            try:
                a.warmup(pngs[0], n=1)
            except Exception:
                pass

    # ============ Per-image table ============
    headers = ["GT"] + [a.name for a in adapters] + ["v3 top-1", "v4 top-1"]
    if args.topk > 1:
        headers.append(f"v3 top-{args.topk}")
        headers.append(f"v4 top-{args.topk}")

    col_widths = [max(4, len(h) + 2) for h in headers]
    # Bump v3/v4 top-k columns to fit topk chars × ~3 each
    if args.topk > 1:
        col_widths[-2] = max(col_widths[-2], args.topk * 3 + 2)
        col_widths[-1] = max(col_widths[-1], args.topk * 3 + 2)

    def line(cells):
        out = []
        for i, c in enumerate(cells):
            out.append(f"{str(c):<{col_widths[i]}}")
        return "  ".join(out)

    print()
    print("=" * 100)
    print(line(headers))
    print("-" * 100)

    # Stats accumulators
    n = 0
    correct_top1: Dict[str, int] = {a.name: 0 for a in adapters}
    correct_topk: Dict[str, int] = {a.name: 0 for a in adapters}
    correct_top1["v3"] = correct_top1["v4"] = 0
    correct_topk["v3"] = correct_topk["v4"] = 0
    lat_sum: Dict[str, float] = {a.name: 0.0 for a in adapters}
    lat_sum["v3"] = 0.0
    lat_sum["v4_fwd"] = 0.0
    lat_sum["v4_nn"] = 0.0

    for png in pngs:
        gt = png.stem
        if not gt:
            continue
        img_uint8 = preprocess_for_model(png)

        cells = [gt]
        # commodity adapters → top-1 cell only (top-k almost always == top-1)
        for a in adapters:
            try:
                preds, ms = a.recognize_topk(png, k=args.topk)
            except Exception:
                preds, ms = [], 0.0
            top1 = preds[0] if preds else ""
            cells.append(fmt_pred(top1, gt))
            lat_sum[a.name] += ms
            if top1 == gt:
                correct_top1[a.name] += 1
            if gt in preds:
                correct_topk[a.name] += 1

        # v3 — top-1 + top-k tracked
        v3_chars, v3_ms = v3.predict(img_uint8, topk=args.topk)
        cells.append(fmt_pred(v3_chars[0], gt))
        lat_sum["v3"] += v3_ms
        if v3_chars[0] == gt:
            correct_top1["v3"] += 1
        if gt in v3_chars:
            correct_topk["v3"] += 1

        # v4 — top-1 + top-k tracked
        v4_chars, v4_fwd, v4_nn = v4.predict(img_uint8, topk=args.topk)
        cells.append(fmt_pred(v4_chars[0], gt))
        lat_sum["v4_fwd"] += v4_fwd
        lat_sum["v4_nn"] += v4_nn
        if v4_chars[0] == gt:
            correct_top1["v4"] += 1
        if gt in v4_chars:
            correct_topk["v4"] += 1

        # top-k strings appended last (in same order as headers)
        if args.topk > 1:
            cells.append("".join(v3_chars[: args.topk]))
            cells.append("".join(v4_chars[: args.topk]))

        print(line(cells))
        n += 1

    print("-" * 100)
    print()
    if n == 0:
        return

    # ============ Summary footer ============
    print("=" * 100)
    print(f"  SUMMARY  (n = {n})")
    print("=" * 100)

    # 4-row block: top-1, top-k, latency, model size
    rows = []
    cols = [a.name for a in adapters] + ["v3", "v4"]
    # Note for top-k row: commodity adapters that returned only [top1] will
    # have correct_topk == correct_top1 — that's the honest "no native top-k"
    # signal. v3 + v4 are model-native top-k.
    rows.append(("top-1 acc",
                 [f"{correct_top1[a.name] / n * 100:5.1f}%" for a in adapters]
                 + [f"{correct_top1['v3'] / n * 100:5.1f}%",
                    f"{correct_top1['v4'] / n * 100:5.1f}%"]))
    rows.append((f"top-{args.topk} acc",
                 [(f"{correct_topk[a.name] / n * 100:5.1f}%"
                   if correct_topk[a.name] != correct_top1[a.name]
                   else f"{correct_topk[a.name] / n * 100:5.1f}%¹")
                  for a in adapters]
                 + [f"{correct_topk['v3'] / n * 100:5.1f}%",
                    f"{correct_topk['v4'] / n * 100:5.1f}%"]))
    rows.append(("avg lat",
                 [f"{lat_sum[a.name] / n:6.1f}ms" for a in adapters]
                 + [f"{lat_sum['v3'] / n:6.1f}ms",
                    f"{(lat_sum['v4_fwd'] + lat_sum['v4_nn']) / n:6.1f}ms"]))
    rows.append(("model size",
                 [f"{a.install_size_mb:>5.0f}MB" for a in adapters]
                 + [f"{v3.size_mb:>5.1f}MB", f"{v4.size_mb:>5.1f}MB"]))

    label_w = max(len(r[0]) for r in rows) + 2
    col_w = max(11, max(len(c) for c in cols) + 2)
    header_line = " " * label_w + "  ".join(f"{c:<{col_w}}" for c in cols)
    print(header_line)
    print(" " * label_w + "  ".join("-" * col_w for _ in cols))
    for label, vals in rows:
        print(f"{label:<{label_w}}" + "  ".join(f"{v:<{col_w}}" for v in vals))

    print()
    if "v4_fwd" in lat_sum and n > 0:
        print(f"  v4 latency split: forward {lat_sum['v4_fwd']/n:.1f}ms  +  "
              f"cosine NN {lat_sum['v4_nn']/n:.1f}ms (over "
              f"{v4.anchors.shape[0]:,} anchors)")
    if any(correct_topk[a.name] == correct_top1[a.name] for a in adapters):
        print(f"  ¹ commodity OCR engines without native top-k: top-{args.topk} "
              f"score equals top-1 (engine returns 1 hypothesis)")
    print()
    print("  Notes:")
    label_w = max(len(a.name) for a in adapters) + 2 if adapters else 12
    for a in adapters:
        print(f"    {a.name:<{label_w}} {a.notes}")
    print(f"    {'v3':<{label_w}} {v3.notes}")
    print(f"    {'v4':<{label_w}} {v4.notes}")
    print()


if __name__ == "__main__":
    main()
