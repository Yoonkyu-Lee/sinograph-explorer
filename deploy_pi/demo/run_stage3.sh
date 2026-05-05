#!/usr/bin/env bash
# Stage 3 — Pi Camera live capture + auto-crop + v4 SCER top-5.
#
# Default: loop mode — press Enter to capture, q+Enter to quit.
# One-shot: pass --once.
#
# Run from dev (interactive):
#   ssh -t yoonkyu2@192.168.1.16 "~/ece479/demo/run_stage3.sh"
# One-shot:
#   ssh yoonkyu2@192.168.1.16 "~/ece479/demo/run_stage3.sh --once"

set -euo pipefail

PYTHON=~/venv-ocr/bin/python
DEMO=~/ece479/demo

ARGS=("--loop")
for a in "$@"; do
    if [[ "$a" == "--once" ]]; then
        ARGS=()
    else
        ARGS+=("$a")
    fi
done

exec "$PYTHON" "$DEMO/capture_predict.py" "${ARGS[@]}"
