#!/usr/bin/env bash
# Stage 1 demo wrapper — Commodity OCR + v3 + v4 SCER on 20 PNGs (CPU only).
#
# Run from dev:
#   ssh yoonkyu2@192.168.1.16 "~/ece479/demo/run_stage1.sh"
set -euo pipefail

# SSH non-interactive shells don't auto-source ~/.profile — pull it in so
# user-exported env vars (e.g. GOOGLE_VISION_API_KEY) reach this script.
[ -f "$HOME/.profile" ] && . "$HOME/.profile"

PYTHON=~/venv-ocr/bin/python
DEMO=~/ece479/demo

exec "$PYTHON" "$DEMO/bench_cpu_three.py" \
    --image-dir       ~/ece479/test \
    --v3-onnx         ~/ece479/lab_v3/v3_char.onnx \
    --v3-class-index  ~/ece479/lab_v3/class_index.json \
    --v4-tflite       ~/ece479/scer/scer_int8_v20.tflite \
    --v4-anchors      ~/ece479/scer/scer_anchor_db_v20.npy \
    --v4-class-index  ~/ece479/scer/class_index.json \
    --topk            5 \
    --commodity       all \
    "$@"
