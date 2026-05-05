# Pi-side Demo Setup

End-to-end procedure for reproducing the live demo on a Raspberry Pi 5
(8 GB), Pi Camera Module 3, Coral USB Accelerator, Pi OS Bookworm 64-bit.

## Pi directory layout (target)

```
~/ece479/
├── demo/                                  ← this folder, scp'd from repo (deploy_pi/demo/)
│   ├── ocr_adapters.py
│   ├── bench_cpu_three.py
│   ├── capture_predict.py
│   ├── run_stage1.sh
│   ├── run_stage2.sh
│   └── run_stage3.sh
├── scer/                                  ← v4 SCER deploy artifacts
│   ├── scer_int8_v20.tflite               (~11.5 MB) — INT8 TFLite
│   ├── scer_int8_v20_edgetpu.tflite       (~11.6 MB) — Edge TPU compiled
│   ├── scer_anchor_db_v20.npy             (~50 MB)   — (98169, 128) anchor matrix
│   ├── class_index.json                   (~2 MB)    — index → unicode codepoint
│   ├── val_pack_1000.npz                  (~19 MB)   — held-out 1000-pack for accuracy check
│   ├── infer_pi_chars.py
│   ├── eval_pi_scer.py
│   └── bench_scer_pi.py
├── lab_v3/                                ← v3 baseline for Pareto comparison
│   ├── v3_char.onnx                       (~235 MB) — FP32 single-stage classifier
│   ├── v3_char_full_integer_quant.tflite  (~63 MB)  — INT8 (deploy broken, kept for proof)
│   └── class_index.json
└── test/                                  ← evaluation PNG set (38 images currently)
```

## 1 — System packages (apt)

```bash
sudo apt update
sudo apt install -y \
    python3-pip python3-venv python3-dev \
    libcamera-apps rpicam-apps python3-picamera2 python3-libcamera \
    python3-pyqt5 python3-pyqt5.sip qtwayland5 \
    python3-opencv \
    libedgetpu1-std \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-jpn \
    tesseract-ocr-chi-tra tesseract-ocr-chi-sim
```

Why each:
- `libedgetpu1-std` — Edge TPU userspace driver (use `libedgetpu1-max` for
  higher clock if you have heatsinking).
- `python3-picamera2` / `python3-libcamera` — Pi Camera Python bindings;
  these are **apt-only**, no pip wheel.
- `python3-pyqt5` + `qtwayland5` — Stage 3 live preview (Pi OS Bookworm
  uses Wayland / `labwc`, not X11).
- `python3-opencv` — ARM-optimized OpenCV; faster than the pip wheel.
- `tesseract-ocr-*` — language data files for the Tesseract adapter.

## 2 — Python venv

The venv must inherit system site-packages so it can see `picamera2` and
`libcamera`:

```bash
python3 -m venv --system-site-packages ~/venv-ocr
~/venv-ocr/bin/pip install --upgrade pip wheel
~/venv-ocr/bin/pip install -r ~/ece479/demo/requirements_pi.txt
```

`requirements_pi.txt` lives at [`deploy_pi/requirements_pi.txt`](../requirements_pi.txt)
in the repo — `scp` it to the Pi alongside `demo/`.

Notes:
- `tflite-runtime` has no Python 3.13 wheel. We use `ai-edge-litert==2.1.3`
  (Google's successor) instead. The `Interpreter` import in the demo
  scripts handles both via try-except.
- `manga-ocr` pulls in `torch` + `transformers` (~1.5 GB). Skip these
  packages from `requirements_pi.txt` if you are not running the Manga
  adapter.
- `paddleocr` 3.x SIGSEGVs on ARM64 + Python 3.13 (paddlepaddle's
  `SaveOrLoadPirParameters`). The adapter guards against this and raises a
  clear error — install only on x86_64 or accept the graceful skip.

## 3 — Copy artifacts from dev (`scp`)

From a development machine where the model has been trained, ported, and
quantized:

```bash
# Demo scripts
scp -r deploy_pi/demo/ pi@<pi-ip>:~/ece479/demo/

# v4 SCER deploy artifacts
scp deploy_pi/export/scer_int8_v20.tflite               pi@<pi-ip>:~/ece479/scer/
scp deploy_pi/export/scer_int8_v20_edgetpu.tflite       pi@<pi-ip>:~/ece479/scer/
scp deploy_pi/export/scer_anchor_db_v20.npy             pi@<pi-ip>:~/ece479/scer/
scp deploy_pi/export/class_index.json                   pi@<pi-ip>:~/ece479/scer/
scp deploy_pi/{infer_pi_chars,eval_pi_scer,bench_scer_pi}.py pi@<pi-ip>:~/ece479/scer/

# v3 baseline (for Stage 1 Pareto comparison)
scp deploy_pi/export/v3_char.onnx                       pi@<pi-ip>:~/ece479/lab_v3/
scp deploy_pi/export/class_index.json                   pi@<pi-ip>:~/ece479/lab_v3/

# Test images
scp -r test/ pi@<pi-ip>:~/ece479/test/
```

## 4 — Smoke test

```bash
ssh pi "~/venv-ocr/bin/python -c 'import ai_edge_litert, numpy, PIL, cv2, picamera2; print(\"ok\")'"
ssh pi "lsusb | grep -iE '18d1|global'"          # Coral detected: 18d1:9302
ssh pi "rpicam-hello --list-cameras | head -3"   # Camera 3 detected: imx708
```

## 5 — Run the demo

```bash
ssh pi "~/ece479/demo/run_stage1.sh"        # Commodity OCR + v3 + v4 bench (CPU)
ssh pi "~/ece479/demo/run_stage2.sh"        # v4 SCER on 20 PNGs (CPU + Coral if available)
ssh -t pi "~/ece479/demo/run_stage3.sh"     # Live Pi Camera capture loop
```

See `doc/33_DEMO.md` for talking points, expected output, and the
9-line core message.

## Optional — environment variables

For the Google Cloud Vision adapter:

```bash
echo 'export GOOGLE_VISION_API_KEY=AIza...' >> ~/.profile
```

`run_stage1.sh` sources `~/.profile` so non-interactive SSH inherits it.
The adapter is excluded from the default `--commodity all` group; opt in
with `--commodity gvision-cloud` after billing is enabled on the GCP
project.

## Known issues

- **Coral delegate segfault on Python 3.13 + ai-edge-litert 2.x.** Hardware
  is fine. `run_stage2.sh` probes Coral first and falls back to CPU-only
  with a cited prior-measurement table if the segfault recurs. To run
  Coral live, create a separate Python 3.10 venv with the legacy
  `tflite-runtime` 2.5 + `pycoral` stack.
- **`paddleocr` ARM64 incompatibility.** As above — guard in adapter
  raises a clear message; bench skips gracefully.
- **Camera autofocus can stick at infinity.** Press `f` in the Stage 3
  loop to re-trigger AF, `m` for macro range, or `p <0..15>` to set lens
  position manually.
