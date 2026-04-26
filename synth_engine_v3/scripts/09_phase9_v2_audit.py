"""Phase 9 audit — scan ALL v2 YAML configs + check every layer/op is
registered in v3. Where a config is fully covered, render a small batch to
confirm it actually runs.

Exit status: prints a covered/missing table; any missing handler is a real gap
that must be filled before Phase 10.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from mask_adapter import CANVAS, batch_render_from_spec
from pipeline_gpu import GPUContext, finalize_center_crop, run_pipeline, tensor_to_pil_batch, REGISTRY
import style_gpu    # noqa: F401
import augment_gpu  # noqa: F401


V2_CONFIGS_ROOT = Path(
    "d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/synth_engine_v2/configs"
)
TEST_CHARS = ["鑑", "學", "媤", "畓", "裡"]


def collect_requirements(cfg: dict) -> set[str]:
    need: set[str] = set()
    for s in cfg.get("style", []) or []:
        n = s.get("layer")
        if n:
            need.add(n)
    for a in cfg.get("augment", []) or []:
        n = a.get("op")
        if n:
            need.add(f"augment.{n}")
    return need


def try_render(cfg: dict, seed: int, device: str = "cuda") -> tuple[bool, str]:
    """Try to render a small batch through this config. Returns (ok, note)."""
    try:
        chars = TEST_CHARS * 2
        rng_np = np.random.default_rng(seed)
        base_spec = cfg.get("base_source", {"kind": "font"})
        mask_t, tags, kinds = batch_render_from_spec(chars, base_spec, rng=rng_np)
        valid = [i for i, t in enumerate(tags) if t is not None]
        if not valid:
            return True, "no base source covers test chars (skipped)"
        mask_t_v = mask_t[valid].to(device)
        chars_v = [chars[i] for i in valid]
        kinds_v = [kinds[i] for i in valid]
        gen = torch.Generator(device=device).manual_seed(seed)
        canvas = torch.ones(len(valid), 3, CANVAS, CANVAS, device=device)
        ctx = GPUContext(canvas=canvas, mask=mask_t_v, rng=gen,
                         chars=chars_v, source_kinds=kinds_v, device=device)
        ctx = run_pipeline(ctx, cfg)
        final = finalize_center_crop(ctx.canvas)
        torch.cuda.synchronize()
        return True, f"rendered {final.shape[0]} samples"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    known = set(REGISTRY.keys())
    all_yamls = sorted(V2_CONFIGS_ROOT.rglob("*.yaml"))
    print(f"found {len(all_yamls)} v2 YAML configs under {V2_CONFIGS_ROOT}")
    print()

    required_union: set[str] = set()
    results: list[tuple[str, str, str]] = []  # (rel, missing_str, render_note)

    for y in all_yamls:
        rel = str(y.relative_to(V2_CONFIGS_ROOT))
        try:
            with open(y, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            results.append((rel, f"YAML PARSE: {e}", ""))
            continue
        need = collect_requirements(cfg)
        required_union |= need
        missing = sorted(n for n in need if n not in known)
        if missing:
            results.append((rel, ", ".join(missing), "(not attempted)"))
            continue
        ok, note = try_render(cfg, args.seed)
        results.append((rel, "", ("OK " + note) if ok else ("FAIL " + note)))

    covered = sum(1 for _, m, _ in results if not m)
    print(f"=== coverage: {covered}/{len(results)} configs have ALL layers/ops registered in v3 ===")
    print()
    for rel, miss, note in results:
        tag = "MISS" if miss else "OK  "
        print(f"  [{tag}] {rel:55s}  missing: {miss or '-'}  |  {note}")
    print()
    # Union report
    all_missing = sorted(n for n in required_union if n not in known)
    if all_missing:
        print(f"UNION of missing registrations across all configs ({len(all_missing)}):")
        for n in all_missing:
            print(f"   - {n}")
    else:
        print("NO missing registrations — v3 covers every layer/op referenced by v2 YAML configs.")
    print()
    # Any FAIL (runtime)?
    runtime_fails = [r for r in results if r[2].startswith("FAIL")]
    if runtime_fails:
        print(f"runtime failures ({len(runtime_fails)}):")
        for rel, _, note in runtime_fails:
            print(f"   - {rel}: {note}")
        sys.exit(1)
    else:
        print("no runtime failures")


if __name__ == "__main__":
    main()
