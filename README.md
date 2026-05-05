# Sinograph Explorer — Universal CJK OCR for the Edge

> **TL;DR** — A 98,169-class Chinese / Japanese / Korean ideograph recognizer
> that runs in **11 MB of INT8 weights at 25 ms/char on a Raspberry Pi 5
> CPU** (24.7 ms with a Coral USB Edge TPU). End-to-end ML stack: PyTorch
> training on RTX 4080 → Keras port → TFLite INT8 quantization → Edge TPU
> compile → live inference on a Raspberry Pi 5 + Pi Camera Module 3.

ECE 479 *Hardware for Cognitive Computing* (UIUC, Spring 2026), Lab 3 final
project. Maintained by **Yoonkyu Lee** (`yoonguri21@gmail.com`).

---

## Headline Numbers

| Metric | Value | Comment |
| --- | ---: | --- |
| Character vocabulary | **98,169** | Universal CJK: Japanese kanji + Traditional & Simplified Chinese hanzi + Korean hanja + Unihan SMP Ext B / C / D |
| Deployable model size | **11 MB INT8** | 4× smaller than the 235 MB FP32 baseline; 40× smaller than Manga-OCR's 440 MB transformer |
| Pi CPU forward latency | **10.1 ms** | INT8 TFLite via `ai-edge-litert`, batch 1 |
| Pi CPU end-to-end | **25 ms / char** (40 char/s) | forward + 128-d cosine NN over 98 K anchors |
| Pi + Coral end-to-end | **24.7 ms / char** | `41 / 41` ops mapped to Edge TPU, 7.6 MiB on-chip cache hit |
| INT8 quantization loss | **0.00 pp** | Keras-native `tf.lite.TFLiteConverter`, 300-sample MinMax calibration |
| In-distribution top-1 (1 K val pack) | **96.90 %** | identical CPU and Coral |
| Real-world test top-1 / top-5 (n=38) | **81.6 % / 92.1 %** | mixed handwritten + printed + Unihan Ext B |

### Compared to commodity OCR engines on the same 38-image set

| Engine | Top-1 | Latency (Pi) | Disk | Coverage |
| --- | ---: | ---: | ---: | --- |
| Tesseract 5 (`jpn+chi_tra+chi_sim`) | 37 % | 427 ms | 50 MB | multi-lang, single-char mode |
| EasyOCR `ja` | 53 % | 38 ms | 70 MB | Japanese silo |
| EasyOCR `ch_tra` | 66 % | 445 ms | 70 MB | Traditional silo |
| EasyOCR `ch_sim` | 37 % | 62 ms | 70 MB | Simplified silo |
| cnocr (PP-OCRv5 ONNX) | 50 % | **13 ms** | 100 MB | Chinese specialist (latency winner) |
| Manga-OCR (ViT + GPT-2) | 66 % | 788 ms | **440 MB** | Japanese specialist (commodity accuracy winner) |
| **v4 SCER (this repo)** | **81.6 %** | **25 ms** | **11 MB** | **all CJK regions, single model, native top-k** |

v4 sits on the Pareto frontier across **size × latency × accuracy × coverage**
— smaller than every commodity engine, faster than all but cnocr, and more
accurate than every one.

---

## Tech Stack

`PyTorch 2.11+cu128` · `CUDA 12.8` · `RTX 4080 Laptop` · `ONNX 1.14` ·
`TensorFlow 2.15 (Keras)` · `TFLite INT8` · `Edge TPU Compiler 16.0` ·
`Coral USB Accelerator` · `Raspberry Pi 5 (ARM64, Pi OS Bookworm)` ·
`Pi Camera Module 3 (imx708) / picamera2 / libcamera` ·
`OpenCV 4` · `NumPy 2` · `Python 3.13 / ai-edge-litert 2.1` ·
`Tesseract / EasyOCR / PaddleOCR / cnocr / Manga-OCR / Google Cloud Vision`
(competitive baselines)

Hosts: Windows 11 (training) · WSL Ubuntu (Edge TPU compile) · Raspberry Pi 5
(deployment & demo).

---

## The Engineering Story

The starting baseline — a single-stage **ResNet-18 → 98,169-way `nn.Linear`
classifier** (v3) — does not deploy to a Coral Edge TPU. Two reasons that
only show up at the deploy boundary:

1. **INT8 quantization corrupts the FC head.** The 50 MB classifier has a
   wide weight distribution; after symmetric INT8 calibration every input
   collapses to a single output token (`⺀`).
2. **The Coral USB Accelerator only has 8 MiB of on-chip SRAM.** Even after
   a clean compile, 87 % of the 59 MB v3 model has to stream over USB, so
   the matrix-multiply acceleration is hidden behind PCIe latency.

**The pivot — SCER (Structure-Conditioned Embedding Recognition).** Instead
of a classifier head, train a 128-d L2-normalized embedding with **ArcFace**
(additive angular margin) loss. At inference, replace `argmax(logits)` with a
**cosine NN search** over a precomputed `(98169, 128)` anchor matrix that
lives outside the deployed graph.

The deployed model becomes `backbone + 128-d embedding head`, dropping the
50 MB FC plus the training-only ArcFace classifier. Result: **11 MB INT8,
69 % on-chip cache hit, 41 / 41 ops mapped, no quantization accuracy loss**.

This also gives the system **open-set classification for free** — a new
character is added by appending one row to `scer_anchor_db.npy`, **no
retraining**.

```
v3 (single-stage classifier) ── 235 MB FP32 / 59 MB INT8 / 87% off-chip ── ❌ deploy
                          │
                          ▼ pivot: trade FC for embedding + cosine NN
                          │
v4 SCER (open-set metric)  ── 11 MB INT8 / 69% on-chip / 0.00pp INT8 loss ── ✅ 25 ms/char
```

### Architecture (simplified)

```
            ┌────────────────────────────────────────────────┐
   image ──▶│  ResNet-18 backbone   (warm-start from v3)     │── feat (512)
   128×128  └──┬───┬───┬───┬───┬─────────────────────────────┘
              │   │   │   │   │
              │   │   │   │   └─ idc head           (12)         ┐  structure
              │   │   │   └───── residual strokes   (1, MSE)     │  heads, kept
              │   │   └───────── total strokes      (1, MSE)     │  for soft
              │   └───────────── radical head       (214, CE)    │  re-ranking
              ▼
         embedding head   Linear(512 → 128) → L2-norm
              ▼
        ┌──── train time ────────────┐  ┌──── inference ────────┐
        │ ArcFace classifier         │  │ cosine NN over        │
        │   weight (98169, 128)      │  │ anchor_db.npy         │
        │   margin m, scale s        │  │ (98169, 128)          │
        │   → softmax CE             │  │ → top-k characters    │
        └────────────────────────────┘  └───────────────────────┘
```

The four auxiliary structure heads (radical, total strokes, residual
strokes, IDC top operator) survive into deployment for use as a soft
**multi-stage filter**: candidates can be narrowed from 98 K → ~50 by
matching predicted Kangxi radical and stroke count *before* running the
cosine NN, which folds CPU latency from 13 ms → ~1 ms.

---

## Deployment Pipeline

```
PyTorch best.pt (RTX 4080)
       │  scripts/40_port_pytorch_to_keras.py
       ▼
Keras (FP32)            ── parity check vs PT, all heads max-abs-diff < 1e-5
       │  scripts/41_export_keras_tflite.py    (300-sample MinMax calibration)
       ▼
TFLite INT8             ── 11.5 MB, INT8 in/out, embedding L2-norm preserved
       │  scripts/43_eval_int8_accuracy.py
       ▼  (PT FP32 vs TFLite INT8: 0.00 pp top-1 loss on 1000-pack)
       │
       │  edgetpu_compiler -s scer_int8.tflite       (WSL Ubuntu 22.04, v16.0)
       ▼
Edge TPU TFLite         ── 11.0 MB, 41/41 ops mapped, 7.58 MiB on-chip cache
       │  scp → Pi 5
       ▼
Pi 5 + Coral USB        ── ai-edge-litert 2.1 + libedgetpu1-std
                          live: Pi Camera 3 → adaptive threshold → contour
                          crop → 128² → INT8 forward → 128-d embedding →
                          cosine NN over scer_anchor_db.npy → top-5 chars
```

Every stage has a verification gate (numerical parity, INT8 accuracy delta,
TPU op coverage, end-to-end latency). The full record is in
[`doc/`](doc/).

---

## Repository Layout

```
sinograph-explorer/
├── README.md                          ← you are here
├── CLAUDE.md                          ← project rules / conventions
├── doc/                               ← phase-by-phase design docs (33 files)
│   ├── 18_FINAL_PRESENTATION.md
│   ├── 24_DEPLOY_BLOCKERS_AND_V4_PLAN.md     ← why v3 → v4
│   ├── 28_PHASE2_SCER_PLAN.md                ← architecture spec
│   ├── 32_PHASE3_4_REDO_RESULTS.md           ← deploy verification
│   └── 33_DEMO.md                            ← demo run sheet
├── db_mining/                         ← Unihan / IDS / e-hanja / KanjiVG mining
├── db_src/                            ← raw character source data (gitignored)
├── sinograph_canonical_v[1-3]/        ← canonicalized SQLite DB build pipelines
├── synth_engine_v[1-3]/               ← synthetic-data generators
│   └── synth_engine_v3/               ← final: 102 K classes × 200 samples = 20.4 M
├── train_engine_v[1-4]/
│   └── train_engine_v4/               ← SCER (this is the production model)
│       ├── modules/
│       │   ├── model.py               ← SCERModel, multi-head + embedding
│       │   ├── arcface.py             ← ArcMarginProduct (additive margin)
│       │   ├── train_loop.py
│       │   └── keras_scer.py          ← PT → Keras port
│       └── scripts/
│           ├── 50_train_scer.py
│           ├── 51_build_anchor_db.py
│           ├── 40_port_pytorch_to_keras.py
│           ├── 41_export_keras_tflite.py     ← INT8 quantize
│           ├── 43_eval_int8_accuracy.py
│           └── 52_eval_scer_pipeline.py
├── deploy_pi/
│   ├── export/                        ← model artifacts (TFLite, ONNX, anchors)
│   ├── infer_pi.py                    ← Pi reference inference
│   ├── infer_pi_chars.py              ← 20 PNG sweep
│   ├── eval_pi_scer.py                ← 1000 val-pack accuracy
│   ├── bench_scer_pi.py               ← latency micro-bench
│   └── demo/                          ← live demo infrastructure
│       ├── ocr_adapters.py            ← Tesseract / EasyOCR / Paddle / cnocr / Manga / GVision adapters
│       ├── bench_cpu_three.py         ← unified CPU bench (commodity vs v3 vs v4)
│       ├── capture_predict.py         ← Pi Camera live capture + auto-crop + v4 inference
│       ├── recompile_edgetpu_with_summary.sh   ← WSL TPU compile cache split proof
│       └── run_stage{1,2,3}.sh        ← single-command demo wrappers
├── test/                              ← 38 hand-picked CJK test images
└── unihan_graph_viewer/               ← (Tauri) auxiliary character explorer
```

---

## Notable Engineering Details

- **Custom synthetic data generator (20.4 M samples).** GPU-accelerated
  font-rendering pipeline that tiles 102 K characters across 200 augmented
  variants per class, with stroke perturbation, blur, noise, and elastic
  distortion. See `synth_engine_v3/`.
- **Multi-host build chain.** Training and Keras port on Windows + CUDA;
  Edge TPU compile on WSL Ubuntu (the `edgetpu_compiler` is Linux-only and
  the `tflite-runtime` Coral SDK was last updated in 2022, so it requires
  exact Python / TF / `libedgetpu` version pinning); inference on Raspberry
  Pi OS Bookworm 64-bit.
- **PyTorch → Keras parity.** Hand-ported each ResNet block, including a
  fix for `stride > 1` asymmetric padding (PyTorch's `nn.Conv2d(stride=2)`
  pads top-left, Keras' `Conv2D(strides=2, padding='same')` pads bottom-right
  — produces a 1-pixel feature shift). Verified all 5 output heads match
  within `1e-5` max-abs-diff.
- **Open-set deployment.** New characters are added by computing the
  embedding of a single template image and `np.append`-ing one row to
  `scer_anchor_db.npy`. No retraining, no model rebuild.
- **Live Pi Camera demo.** Wayland-aware preview (Pi OS Bookworm uses
  `labwc`, not X11), continuous autofocus + macro mode + digital zoom for
  Pi Camera 3, OpenCV adaptive threshold + largest-contour auto-crop,
  multi-character mode with morphological closing + reading-order sort.

---

## Quickstart

### Pi-side inference (after `scp`-ing the artifacts)

```bash
~/venv-ocr/bin/python ~/ece479/scer/infer_pi_chars.py \
    --tflite ~/ece479/scer/scer_int8_v20.tflite \
    --anchors ~/ece479/scer/scer_anchor_db_v20.npy \
    --class-index ~/ece479/scer/class_index.json \
    --image-dir ~/ece479/test
```

### Live demo (3-stage)

```bash
ssh pi "~/ece479/demo/run_stage1.sh"   # CPU bench: 7 commodity OCR + v3 + v4
ssh pi "~/ece479/demo/run_stage2.sh"   # CPU vs Coral on 20 PNG
ssh -t pi "~/ece479/demo/run_stage3.sh"   # Pi Camera live capture loop
```

### Reproduce training

```bash
# Windows + CUDA 12.8
.venv\Scripts\python.exe train_engine_v4\scripts\50_train_scer.py \
    --config train_engine_v4/configs/scer_v1.yaml \
    --out train_engine_v4/out/16_scer_v1
.venv\Scripts\python.exe train_engine_v4\scripts\51_build_anchor_db.py \
    --ckpt train_engine_v4/out/16_scer_v1/best.pt
```

---

## Documentation Index

The full design record is in [`doc/`](doc/). Highlights for reviewers:

- [`doc/18_FINAL_PRESENTATION.md`](doc/18_FINAL_PRESENTATION.md) — project overview, motivation, narrative
- [`doc/19_TRAIN_ENGINE_V3_PLAN.md`](doc/19_TRAIN_ENGINE_V3_PLAN.md) — multi-head ResNet-18 baseline
- [`doc/24_DEPLOY_BLOCKERS_AND_V4_PLAN.md`](doc/24_DEPLOY_BLOCKERS_AND_V4_PLAN.md) — INT8 corruption + Edge TPU SRAM analysis, root cause + pivot
- [`doc/27_PHASE1_RESULTS.md`](doc/27_PHASE1_RESULTS.md) — PyTorch → Keras port verification
- [`doc/28_PHASE2_SCER_PLAN.md`](doc/28_PHASE2_SCER_PLAN.md) — SCER architecture spec (ArcFace + embedding + anchor DB)
- [`doc/29_PHASE2_RESULTS.md`](doc/29_PHASE2_RESULTS.md) — training metrics, ablations
- [`doc/32_PHASE3_4_REDO_RESULTS.md`](doc/32_PHASE3_4_REDO_RESULTS.md) — INT8 quantization + Edge TPU compile + Pi/Coral latency
- [`doc/33_DEMO.md`](doc/33_DEMO.md) — final demo run sheet, talking points, Pareto comparison

---

## About the author

Yoonkyu Lee — UIUC ECE, Spring 2026. Interested in **edge ML, embedded
systems, and on-device cognitive computing**: shrinking large models so
they fit and run on small, cheap, low-power hardware without giving up
accuracy. This repo is a single-person, end-to-end execution of that
question — from raw character database mining, through synthetic data
generation, model design, training on consumer GPU, multi-target
quantization, hardware accelerator deployment, and live camera demo —
without sacrificing accuracy at any stage.

Open to **deep learning / ML platform / GPU systems / IoT / embedded ML**
roles. Reach out: [`yoonguri21@gmail.com`](mailto:yoonguri21@gmail.com).
