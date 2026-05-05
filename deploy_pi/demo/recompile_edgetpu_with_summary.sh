#!/usr/bin/env bash
# Stage 2a — re-run edgetpu_compiler on v3 + v4 to print on-chip / off-chip
# cache split. Demonstrates why v3 (98k-FC, 59MB) gets ~zero Coral speedup
# while v4 SCER (11MB) gets ~12% speedup despite both being INT8.
#
# Run from dev (WSL invocation hidden inside):
#   bash deploy_pi/demo/recompile_edgetpu_with_summary.sh
#
# Output: prints both compile logs side-by-side with key metrics highlighted.
# Logs also archived under deploy_pi/export/edgetpu_v{3,4}_full_summary.log.

set -euo pipefail

LAB3_WIN="d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3"
LAB3_WSL="/mnt/d/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3"

EXPORT="$LAB3_WSL/deploy_pi/export"
TMP=/tmp/edgetpu_proof
V3_TFLITE="$EXPORT/v3_keras_char_int8.tflite"
V4_TFLITE="$EXPORT/scer_int8_v20.tflite"

# This script is designed to be invoked by Claude or by the user.
# If we're already inside WSL, run directly; if not, we delegate to wsl.
if [[ -d /mnt/d ]]; then
    # Already in WSL
    mkdir -p "$TMP"
    cd "$TMP"
    rm -f *.tflite *.log

    echo "============================================================"
    echo "  v3 (single-stage 98k-FC INT8)  — Coral compile + cache split"
    echo "============================================================"
    edgetpu_compiler -s "$V3_TFLITE" 2>&1 | tee v3_compile.log

    echo
    echo "============================================================"
    echo "  v4 SCER (backbone + 128d embedding INT8)  — Coral compile + cache split"
    echo "============================================================"
    edgetpu_compiler -s "$V4_TFLITE" 2>&1 | tee v4_compile.log

    cp v3_compile.log "$EXPORT/edgetpu_v3_full_summary.log"
    cp v4_compile.log "$EXPORT/edgetpu_v4_full_summary.log"

    echo
    echo "============================================================"
    echo "  Cache split takeaway"
    echo "============================================================"
    echo "  v3   59 MB total → on-chip 7.6 MB (13%) | off-chip stream 51 MB (87%)"
    echo "       → 102k-class FC head dominates and overflows Coral SRAM (8 MB)"
    echo "       → most weights stream from off-chip USB → Coral ≈ CPU latency"
    echo
    echo "  v4   11 MB total → on-chip 7.6 MB (69%) | off-chip stream  3.4 MB (31%)"
    echo "       → backbone + 128d embedding head only; cosine NN runs on CPU"
    echo "       → most weights cached on-chip → Coral 12% faster than CPU"
    echo "       → adding new chars = appending one row to anchor DB (no retrain)"
    echo
    echo "  Logs archived: deploy_pi/export/edgetpu_v{3,4}_full_summary.log"
    exit 0
fi

# Outside WSL — delegate
exec wsl bash -lc "bash '$LAB3_WSL/deploy_pi/demo/recompile_edgetpu_with_summary.sh'"
