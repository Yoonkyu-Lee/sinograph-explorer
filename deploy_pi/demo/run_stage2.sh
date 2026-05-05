#!/usr/bin/env bash
# Stage 2b demo wrapper — v4 SCER on 20 PNGs.
#
# Tries CPU + Coral first; if Coral segfaults (ai-edge-litert 2.x + Py 3.13
# known issue), falls back to CPU-only and prints prior-measurement note.
#
# Run from dev:
#   ssh yoonkyu2@192.168.1.16 "~/ece479/demo/run_stage2.sh"
set -uo pipefail

PYTHON=~/venv-ocr/bin/python
SCER=~/ece479/scer

CORAL_OK=1

# Try Coral first (capture segfault → fallback)
if "$PYTHON" -c "
from ai_edge_litert.interpreter import Interpreter, load_delegate
import numpy as np
i = Interpreter(model_path='$SCER/scer_int8_v20_edgetpu.tflite',
                experimental_delegates=[load_delegate('libedgetpu.so.1')])
i.allocate_tensors()
in_d = i.get_input_details()[0]
i.set_tensor(in_d['index'], np.zeros(tuple(in_d['shape']), dtype=in_d['dtype']))
i.invoke()
" 2>/dev/null
then
    CORAL_OK=1
else
    CORAL_OK=0
fi

if [[ $CORAL_OK -eq 1 ]]; then
    echo "[stage2] Coral live ok — running CPU + Coral side-by-side"
    exec "$PYTHON" "$SCER/infer_pi_chars.py" \
        --image-dir    ~/ece479/test \
        --tflite-cpu   "$SCER/scer_int8_v20.tflite" \
        --tflite-coral "$SCER/scer_int8_v20_edgetpu.tflite" \
        --anchors      "$SCER/scer_anchor_db_v20.npy" \
        --class-index  "$SCER/class_index.json" \
        --topk         5 \
        "$@"
else
    echo "[stage2] Coral live UNAVAILABLE (ai-edge-litert/Py3.13 segfault)"
    echo "[stage2] running CPU live; Coral results from prior verified measurement"
    echo
    "$PYTHON" "$SCER/infer_pi_chars.py" \
        --image-dir    ~/ece479/test \
        --tflite-cpu   "$SCER/scer_int8_v20.tflite" \
        --anchors      "$SCER/scer_anchor_db_v20.npy" \
        --class-index  "$SCER/class_index.json" \
        --topk         5 \
        "$@"
    echo
    echo "============================================================"
    echo "  Prior verified measurement (2026-05-01, doc/32):"
    echo "============================================================"
    echo "  Pi CPU INT8     : forward 14.65 ms,  end-to-end 28.52 ms"
    echo "  Pi Coral INT8   : forward 11.23 ms,  end-to-end 24.70 ms"
    echo "                    (12% faster, 0pp accuracy loss)"
    echo "  20 PNG accuracy : both = 19/20 (95%) top-1, 20/20 (100%) top-5"
    echo "  Source          : ~/ece479/scer/scer_bench_pi.json"
    echo "                  : ~/ece479/scer/eval_pi_scer.json (was Coral run)"
    echo
fi
