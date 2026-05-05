"""Stage 3 — Pi Camera capture → auto-crop → v4 SCER → top-5.

Pipeline:
  picamera2 still capture (1280×720) → BGR ndarray
  → grayscale → adaptive threshold (Gaussian, INV)
  → external contours, area > 200
  → bounding box of largest contour → square pad with 20px margin
  → resize 128×128 (INTER_AREA)
  → v4 SCER forward → 128d emb → cosine NN over anchor DB → top-5

Run on Pi:
  ~/venv-ocr/bin/python ~/ece479/demo/capture_predict.py            # one-shot
  ~/venv-ocr/bin/python ~/ece479/demo/capture_predict.py --loop      # repeat on Enter

Saves last capture + last crop to /tmp/cap_orig.jpg and /tmp/cap_crop.png
for offline review.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from ai_edge_litert.interpreter import Interpreter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# Auto-crop — assumes dark glyph on light paper
# =============================================================================

def _binarize(img_bgr: np.ndarray, *, morph_kernel: int = 3,
              morph_iter: int = 1) -> np.ndarray:
    """Adaptive threshold + morphological close → binary mask of dark ink."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=21, C=8,
    )
    if morph_iter > 0:
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iter,
        )
    return binary


def _square_pad_crop(img_bgr: np.ndarray, bbox, pad_px: int = 20):
    """Take an x,y,w,h bbox and produce a square crop centered on it.

    Returns (cropped_bgr, (cx0, cy0, side)) — crop window coords for overlay.
    """
    x, y, w, h = bbox
    side = max(w, h) + pad_px * 2
    cx, cy = x + w // 2, y + h // 2
    H, W = img_bgr.shape[:2]
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(W, x0 + side)
    y1 = min(H, y0 + side)
    if x1 - x0 < side:
        x0 = max(0, x1 - side)
    if y1 - y0 < side:
        y0 = max(0, y1 - side)
    return img_bgr[y0:y1, x0:x1].copy(), (x0, y0, side)


def auto_crop_largest_contour(
    img_bgr: np.ndarray, *, min_area: int = 200, pad_px: int = 20,
):
    """Single-char mode: largest external contour. Returns (crop_bgr, bbox)."""
    binary = _binarize(img_bgr, morph_kernel=3, morph_iter=1)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    if not contours:
        return None, None

    biggest = max(contours, key=cv2.contourArea)
    bbox = cv2.boundingRect(biggest)
    crop, _ = _square_pad_crop(img_bgr, bbox, pad_px=pad_px)
    return crop, bbox


def auto_detect_multi_chars(
    img_bgr: np.ndarray, *,
    min_area: int = 500,
    morph_kernel: int = 9,      # bigger → merges strokes within a char
    morph_iter: int = 2,
    pad_px: int = 20,
    line_y_tol: float = 0.5,    # share-a-line if y-distance < tol × median height
):
    """Multi-char mode: find every character-sized contour, return one crop
    per character in reading order (line-by-line, then left-to-right).

    The morph_close kernel must be aggressive enough to merge sub-strokes
    into one blob per character (else 三 becomes 3 contours), but not so
    aggressive it bridges adjacent characters.

    Returns list of (bbox, square_crop_bgr) — possibly empty.
    """
    # Aggressive close to merge strokes inside a character.
    binary = _binarize(img_bgr, morph_kernel=morph_kernel, morph_iter=morph_iter)
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    # Filter: must be character-sized (not noise speckle, not whole-page blob)
    H, W = img_bgr.shape[:2]
    page_area = H * W
    contours = [c for c in contours
                if min_area < cv2.contourArea(c) < page_area * 0.5]
    if not contours:
        return []

    bboxes = [cv2.boundingRect(c) for c in contours]

    # Robust median height for line grouping.
    heights = np.array([h for _, _, _, h in bboxes])
    med_h = float(np.median(heights))
    line_h = max(med_h * line_y_tol, 1.0)

    # Sort by reading order: y-bin, then x.
    bboxes_sorted = sorted(
        bboxes, key=lambda b: (int((b[1] + b[3] / 2) / line_h), b[0])
    )

    out = []
    for bbox in bboxes_sorted:
        crop, _ = _square_pad_crop(img_bgr, bbox, pad_px=pad_px)
        out.append((bbox, crop))
    return out


# =============================================================================
# v4 SCER inference (mirrors infer_pi_chars.py)
# =============================================================================

class V4SCER:
    def __init__(self, tflite_path: Path, anchors_path: Path,
                 class_index_path: Path):
        self.interp = Interpreter(model_path=str(tflite_path))
        self.interp.allocate_tensors()
        self.in_d = self.interp.get_input_details()[0]
        self.emb_d = next(d for d in self.interp.get_output_details()
                          if list(d["shape"][-1:]) == [128])
        self.anchors = np.load(anchors_path)
        cls = json.loads(class_index_path.read_text(encoding="utf-8"))
        self.idx_to_key = {v: k for k, v in cls.items()}
        # Warmup
        x0 = np.zeros(tuple(self.in_d["shape"]), dtype=self.in_d["dtype"])
        for _ in range(3):
            self.interp.set_tensor(self.in_d["index"], x0)
            self.interp.invoke()

    def predict(self, patch_rgb_uint8: np.ndarray, topk: int = 5):
        x = patch_rgb_uint8.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        scale, zp = self.in_d["quantization"]
        x_q = np.round(x / scale + zp).clip(-128, 127).astype(np.int8)[None, ...]
        t0 = time.perf_counter()
        self.interp.set_tensor(self.in_d["index"], x_q)
        self.interp.invoke()
        emb_int8 = self.interp.get_tensor(self.emb_d["index"])
        fwd_ms = (time.perf_counter() - t0) * 1000.0

        s, zp = self.emb_d["quantization"]
        emb = (emb_int8.astype(np.float32) - zp) * s
        norm = np.linalg.norm(emb, axis=-1, keepdims=True).clip(min=1e-8)
        emb = emb / norm

        t0 = time.perf_counter()
        sims = emb @ self.anchors.T
        top_idx = np.argsort(-sims, axis=1)[0, :topk]
        nn_ms = (time.perf_counter() - t0) * 1000.0

        chars = []
        for i in top_idx:
            k = self.idx_to_key.get(int(i), "?")
            try:
                chars.append(chr(int(k[2:], 16)))
            except Exception:
                chars.append("?")
        return chars, sims[0, top_idx], fwd_ms, nn_ms


# =============================================================================
# Camera capture loop
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default=str(Path.home() / "ece479/scer/scer_int8_v20.tflite"))
    ap.add_argument("--anchors", default=str(Path.home() / "ece479/scer/scer_anchor_db_v20.npy"))
    ap.add_argument("--class-index", default=str(Path.home() / "ece479/scer/class_index.json"))
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--loop", action="store_true",
                    help="capture loop — press Enter to capture, q+Enter to quit")
    ap.add_argument("--save-dir", default=str(Path.home() / "ece479/tmp"),
                    help="directory for saved capture artifacts")
    ap.add_argument("--save-orig", default=None,
                    help="override original-image path (default: <save-dir>/cap_orig.jpg)")
    ap.add_argument("--save-crop", default=None,
                    help="override cropped-patch path (default: <save-dir>/cap_crop.png)")
    ap.add_argument("--preview", action="store_true", default=True,
                    help="show live preview on Pi's local monitor (default on)")
    ap.add_argument("--no-preview", dest="preview", action="store_false")
    ap.add_argument("--preview-x", type=int, default=100)
    ap.add_argument("--preview-y", type=int, default=100)
    ap.add_argument("--preview-w", type=int, default=800)
    ap.add_argument("--preview-h", type=int, default=600)
    # Focus / zoom — Camera Module 3 (imx708) supports AF.
    ap.add_argument("--af-mode", choices=["continuous", "auto", "manual"],
                    default="continuous",
                    help="continuous=re-focus always, auto=focus once on trigger, "
                         "manual=fixed via --lens-position")
    ap.add_argument("--af-range", choices=["normal", "macro", "full"],
                    default="full",
                    help="macro for close-up paper (~7-30cm), full = anywhere")
    ap.add_argument("--lens-position", type=float, default=None,
                    help="manual focus in diopters (1/m). 0=infinity, "
                         "~10=10cm. Only when --af-mode=manual.")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="digital zoom factor (1.0=no zoom, 2.0=2× crop). "
                         "Trades resolution for narrower FoV.")
    # Multi-character detection
    ap.add_argument("--multi", action="store_true",
                    help="detect ALL characters in the frame (not just "
                         "the largest contour). Sorted in reading order.")
    ap.add_argument("--multi-min-area", type=int, default=500,
                    help="ignore contours smaller than this many pixels")
    ap.add_argument("--multi-morph-kernel", type=int, default=9,
                    help="morphological close kernel — larger merges strokes "
                         "inside a character; too large bridges neighbors")
    args = ap.parse_args()

    # Resolve + create save dir
    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_orig = Path(args.save_orig) if args.save_orig else save_dir / "cap_orig.jpg"
    save_crop = Path(args.save_crop) if args.save_crop else save_dir / "cap_crop.png"

    print("[capture] loading v4 SCER...")
    scer = V4SCER(Path(args.tflite), Path(args.anchors),
                  Path(args.class_index))
    print(f"[capture]   anchors {scer.anchors.shape}")

    print("[capture] starting camera...")
    # Live preview env: when SSH'd in, neither DISPLAY nor WAYLAND_DISPLAY
    # is auto-set. Detect which display server the Pi user is running
    # (RPi OS Bookworm/Trixie defaults to Wayland/labwc; older / lite
    # imgs use X11) and inject the matching env. User overrides take
    # precedence (just export DISPLAY / WAYLAND_DISPLAY before invoking).
    import os
    if args.preview:
        if (not os.environ.get("WAYLAND_DISPLAY") and
                not os.environ.get("DISPLAY")):
            uid = os.getuid()
            rt = Path(f"/run/user/{uid}")
            wayland_sock = rt / "wayland-0"
            if wayland_sock.exists():
                os.environ["XDG_RUNTIME_DIR"] = str(rt)
                os.environ["WAYLAND_DISPLAY"] = "wayland-0"
                # qtwayland5 plugin enables Qt-on-Wayland.
                # Force the platform — without this, Qt defaults to xcb
                # (X11) and fails to connect on a Wayland-only Pi.
                os.environ.setdefault("QT_QPA_PLATFORM", "wayland")
            else:
                os.environ["DISPLAY"] = ":0"
                os.environ.setdefault(
                    "XAUTHORITY", str(Path.home() / ".Xauthority")
                )

    from picamera2 import Picamera2
    cam = Picamera2()
    # preview_configuration is suitable for both preview + capture_array.
    cam.configure(cam.create_preview_configuration(
        main={"size": (args.width, args.height)},
    ))

    preview_started = False
    preview_kind = None
    if args.preview:
        from picamera2 import Preview
        # Prefer QT (software) — works on Wayland without an OpenGL/EGL
        # context; QTGL requires wayland-egl which often isn't usable
        # from a remote SSH session. DRM is a last-resort full-screen
        # framebuffer takeover that conflicts with the desktop.
        for kind, label in [(Preview.QT, "QT-SW"),
                            (Preview.QTGL, "QTGL"),
                            (Preview.DRM, "DRM")]:
            try:
                if kind == Preview.DRM:
                    cam.start_preview(kind)
                else:
                    cam.start_preview(
                        kind,
                        x=args.preview_x, y=args.preview_y,
                        width=args.preview_w, height=args.preview_h,
                    )
                preview_started = True
                preview_kind = label
                target = (os.environ.get("WAYLAND_DISPLAY")
                          or os.environ.get("DISPLAY") or "unknown")
                print(f"[capture] live preview on {target} "
                      f"({label} {args.preview_w}×{args.preview_h})")
                break
            except Exception as e:
                last_err = e
                continue
        if not preview_started:
            print(f"[capture] preview unavailable: {last_err}")
            print(f"[capture] running headless — saved artifacts at {save_dir}")

    # Apply focus + zoom controls before starting the stream.
    from libcamera import controls
    af_mode_map = {
        "continuous": controls.AfModeEnum.Continuous,
        "auto":       controls.AfModeEnum.Auto,
        "manual":     controls.AfModeEnum.Manual,
    }
    af_range_map = {
        "normal": controls.AfRangeEnum.Normal,
        "macro":  controls.AfRangeEnum.Macro,
        "full":   controls.AfRangeEnum.Full,
    }

    def apply_zoom(z: float):
        """Center-crop the sensor at zoom factor z. z=1.0 disables crop."""
        if z is None or abs(z - 1.0) < 1e-3:
            # full sensor
            fW, fH = cam.camera_properties["PixelArraySize"]
            cam.set_controls({"ScalerCrop": (0, 0, fW, fH)})
            return
        fW, fH = cam.camera_properties["PixelArraySize"]
        cw, ch = int(fW / z), int(fH / z)
        cx, cy = (fW - cw) // 2, (fH - ch) // 2
        cam.set_controls({"ScalerCrop": (cx, cy, cw, ch)})

    initial_ctrls = {"AfMode": af_mode_map[args.af_mode]}
    if args.af_mode != "manual":
        initial_ctrls["AfRange"] = af_range_map[args.af_range]
    if args.af_mode == "manual" and args.lens_position is not None:
        initial_ctrls["LensPosition"] = args.lens_position
    cam.set_controls(initial_ctrls)
    if args.af_mode == "auto":
        # trigger one-shot AF on startup
        cam.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})

    cam.start()
    apply_zoom(args.zoom)
    # Auto-exposure + AF converge
    time.sleep(1.0)
    print(f"[capture] AfMode={args.af_mode}  AfRange={args.af_range}  "
          f"zoom={args.zoom:.2f}×  lens={args.lens_position}")

    # Mutable state so interactive 'M' / 'S' commands can flip detection mode
    state = {"multi": args.multi}

    def predict_one(crop_bgr):
        patch = cv2.resize(crop_bgr, (128, 128), interpolation=cv2.INTER_AREA)
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        return patch, scer.predict(patch_rgb, topk=5)

    def annotate(annotated, bbox, label, idx=None, color=(0, 255, 0)):
        x, y, w, h = bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        # Crop-window outline
        side = max(w, h) + 40
        cx, cy = x + w // 2, y + h // 2
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        cv2.rectangle(annotated, (x0, y0), (x0 + side, y0 + side),
                      (255, 0, 255), 1)
        # Label (use ASCII/index — cv2.putText doesn't render CJK)
        text = f"#{idx}" if idx is not None else label
        cv2.putText(annotated, text, (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    def one_shot():
        img_rgb = cam.capture_array()
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        annotated = img_bgr.copy()

        if state["multi"]:
            detections = auto_detect_multi_chars(
                img_bgr,
                min_area=args.multi_min_area,
                morph_kernel=args.multi_morph_kernel,
            )
            if not detections:
                cv2.imwrite(str(save_orig), img_bgr)
                print("  [warn] no characters found")
                print(f"  saved: {save_orig}")
                return

            print(f"  detected {len(detections)} character(s):")
            # Build a matrix of crops, batch-write per-char crops, annotate
            per_char_crops_dir = save_dir / "cap_chars"
            per_char_crops_dir.mkdir(parents=True, exist_ok=True)
            # Clear stale per-char crops
            for old in per_char_crops_dir.glob("char_*.png"):
                old.unlink()

            total_fwd = total_nn = 0.0
            for i, (bbox, crop) in enumerate(detections):
                patch, (chars, sims, fwd, nn) = predict_one(crop)
                total_fwd += fwd
                total_nn += nn
                cv2.imwrite(
                    str(per_char_crops_dir / f"char_{i:02d}.png"), patch
                )
                annotate(annotated, bbox, chars[0], idx=i, color=(0, 255, 0))
                top5 = " ".join(f"{c}({s:.2f})" for c, s in zip(chars, sims))
                x, y, w, h = bbox
                print(f"    #{i:02d}  bbox={x},{y} {w}×{h}  '{chars[0]}'  "
                      f"top-5: {top5}")

            cv2.imwrite(str(save_orig), annotated)
            print(f"  total: fwd={total_fwd:.1f}ms  nn={total_nn:.1f}ms  "
                  f"({len(detections)} chars)")
            print(f"  saved: {save_orig}  +  {per_char_crops_dir}/char_*.png")
            return

        # Single-char mode (default)
        cropped_bgr, bbox = auto_crop_largest_contour(img_bgr)
        if cropped_bgr is None:
            cv2.imwrite(str(save_orig), img_bgr)
            print("  [warn] no character found — try better lighting / clearer paper")
            print(f"  saved: {save_orig}")
            return

        patch, (chars, sims, fwd, nn) = predict_one(cropped_bgr)
        cv2.imwrite(str(save_crop), patch)
        annotate(annotated, bbox, chars[0], color=(0, 255, 0))
        cv2.imwrite(str(save_orig), annotated)

        x, y, w, h = bbox
        print(f"  bbox={x},{y} {w}×{h}  fwd={fwd:.1f}ms  nn={nn:.1f}ms")
        print(f"  top-5: {' '.join(f'{c}({s:.2f})' for c, s in zip(chars, sims))}")
        print(f"  saved: {save_orig}  +  {save_crop}")

    def help_msg():
        print("  commands:")
        print("    <Enter>  capture (single-char or multi mode per current state)")
        print("    M        toggle multi-character detection (current: "
              f"{'ON' if state['multi'] else 'OFF'})")
        print("    f        trigger autofocus once")
        print("    z N      digital zoom = N× (1.0 = full FoV)")
        print("    p N      manual lens position N diopters (0=∞, ~10=10cm)")
        print("    a        switch AF back to continuous")
        print("    m        switch AF to macro range (close-up paper)")
        print("    h        this help")
        print("    q        quit")

    try:
        if args.loop:
            print("[capture] loop mode")
            help_msg()
            while True:
                raw = input("> ").strip()
                # Preserve case for the multi toggle (M vs lowercase m for macro)
                cmd_case_sensitive = raw
                line = raw.lower()
                if line in ("q", "quit", "exit"):
                    break
                if line == "h":
                    help_msg()
                    continue
                if cmd_case_sensitive == "M":
                    state["multi"] = not state["multi"]
                    print(f"  [multi] detection mode = "
                          f"{'ON (multi-char)' if state['multi'] else 'OFF (single largest)'}")
                    continue
                if line == "f":
                    cam.set_controls({"AfTrigger": controls.AfTriggerEnum.Start})
                    print("  [af] one-shot AF triggered")
                    time.sleep(0.6)
                    continue
                if line == "a":
                    cam.set_controls({"AfMode": af_mode_map["continuous"]})
                    print("  [af] continuous mode")
                    continue
                if line == "m":
                    cam.set_controls({
                        "AfMode": af_mode_map["continuous"],
                        "AfRange": af_range_map["macro"],
                    })
                    print("  [af] continuous + macro range")
                    continue
                if line.startswith("z"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            z = float(parts[1])
                            apply_zoom(z)
                            print(f"  [zoom] {z:.2f}×")
                            time.sleep(0.3)
                        except ValueError:
                            print("  usage: z <factor>")
                    else:
                        print("  usage: z <factor>")
                    continue
                if line.startswith("p"):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            lp = float(parts[1])
                            cam.set_controls({
                                "AfMode": af_mode_map["manual"],
                                "LensPosition": lp,
                            })
                            print(f"  [lens] manual {lp:.2f} diopters")
                            time.sleep(0.3)
                        except ValueError:
                            print("  usage: p <diopters>")
                    else:
                        print("  usage: p <diopters>")
                    continue
                # default: empty line (or any unknown) → capture
                if line == "":
                    t0 = time.perf_counter()
                    one_shot()
                    print(f"  total wall: {(time.perf_counter()-t0)*1000:.0f}ms")
                    print()
                else:
                    print(f"  unknown command '{line}' — type 'h' for help")
        else:
            one_shot()
    finally:
        if preview_started:
            try:
                cam.stop_preview()
            except Exception:
                pass
        cam.stop()
        cam.close()


if __name__ == "__main__":
    main()
