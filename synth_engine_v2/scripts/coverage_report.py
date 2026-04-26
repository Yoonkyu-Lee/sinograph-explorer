"""
Phase 4 — coverage report across stroke-source backends.

Computes, for each character ever seen in at least one source, which backends
can supply per-stroke variation for it. Useful to:
  - Quantify the union/intersection of MakeMeAHanzi vs e-hanja
  - Break down e-hanja-only gains by Unicode block (Ext A, Ext B, Compat, …)
  - Generate a manifest the engine can consult to decide source fallback

Output
------
    synth_engine_v2/out/coverage_report.json
      {
        "mmh_count": N,
        "ehanja_count": M,
        "intersection": K,
        "union": U,
        "mmh_only": ...,
        "ehanja_only": ...,
        "ehanja_only_by_block": { "CJK_Unified": …, "Ext_B_SMP": …, … }
      }

    synth_engine_v2/out/coverage_per_char.jsonl
      one record per char in the union:
        {"char": "鑑", "cp": 38481, "sources": ["mmh", "ehanja"]}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path("d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3")
OUT_DIR = ROOT / "synth_engine_v2" / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MMH_GRAPHICS = ROOT / "db_src" / "MAKEMEAHANZI" / "graphics.txt"
EHANJA_MANIFEST = ROOT / "db_src" / "e-hanja_online" / "strokes_animated.jsonl"
KANJIVG_MANIFEST = ROOT / "db_src" / "KanjiVG" / "strokes_kanjivg.jsonl"


def load_mmh_chars(path: Path) -> set[str]:
    chars: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            c = rec.get("character")
            if c:
                chars.add(c)
    return chars


def load_ehanja_chars(path: Path) -> set[str]:
    chars: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            c = rec.get("char")
            if c:
                chars.add(c)
    return chars


BLOCKS = [
    ("CJK_Unified",       0x4E00,  0x9FFF),
    ("CJK_Ext_A",         0x3400,  0x4DBF),
    ("CJK_Compat",        0xF900,  0xFAFF),
    ("CJK_Compat_Supp",   0x2F800, 0x2FA1F),
    ("Radicals_Supp",     0x2E80,  0x2EFF),
    ("Kangxi_Radicals",   0x2F00,  0x2FDF),
    ("Ext_B_SMP",         0x20000, 0x2A6DF),
    ("Ext_C_SMP",         0x2A700, 0x2B73F),
    ("Ext_D_SMP",         0x2B740, 0x2B81F),
    ("Ext_E_SMP",         0x2B820, 0x2CEAF),
]


def block_of(cp: int) -> str:
    for name, lo, hi in BLOCKS:
        if lo <= cp <= hi:
            return name
    return "Other"


def bucket(chars):
    out: dict[str, int] = {}
    for c in chars:
        b = block_of(ord(c))
        out[b] = out.get(b, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def load_kanjivg_chars(path: Path) -> set[str]:
    chars: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            c = rec.get("char")
            if c:
                chars.add(c)
    return chars


def main():
    print("loading sources...")
    mmh = load_mmh_chars(MMH_GRAPHICS)
    ehanja = load_ehanja_chars(EHANJA_MANIFEST)
    kanjivg = load_kanjivg_chars(KANJIVG_MANIFEST)

    union = mmh | ehanja | kanjivg

    kanjivg_only_all = kanjivg - mmh - ehanja
    ehanja_only_all = ehanja - mmh - kanjivg
    mmh_only_all = mmh - ehanja - kanjivg

    summary = {
        "mmh_count": len(mmh),
        "ehanja_count": len(ehanja),
        "kanjivg_count": len(kanjivg),
        "union": len(union),
        "mmh_only": len(mmh_only_all),
        "ehanja_only": len(ehanja_only_all),
        "kanjivg_only": len(kanjivg_only_all),
        "pairwise": {
            "mmh_ehanja_intersect": len(mmh & ehanja),
            "mmh_kanjivg_intersect": len(mmh & kanjivg),
            "ehanja_kanjivg_intersect": len(ehanja & kanjivg),
            "all_three": len(mmh & ehanja & kanjivg),
        },
        "mmh_only_by_block": bucket(mmh_only_all),
        "ehanja_only_by_block": bucket(ehanja_only_all),
        "kanjivg_only_by_block": bucket(kanjivg_only_all),
    }

    summary_path = OUT_DIR / "coverage_report.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    per_char_path = OUT_DIR / "coverage_per_char.jsonl"
    with per_char_path.open("w", encoding="utf-8") as f:
        for c in sorted(union, key=lambda x: ord(x)):
            sources = []
            if c in mmh: sources.append("mmh")
            if c in ehanja: sources.append("ehanja")
            if c in kanjivg: sources.append("kanjivg")
            rec = {"char": c, "cp": ord(c), "sources": sources}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print("=" * 60)
    print("coverage summary (3-source union)")
    print("=" * 60)
    print(f"MakeMeAHanzi:        {len(mmh):>6,}")
    print(f"e-hanja animated:    {len(ehanja):>6,}")
    print(f"KanjiVG:             {len(kanjivg):>6,}")
    print(f"union (all three):   {len(union):>6,}  ({len(union)/len(mmh):.2f}x MMH)")
    print()
    print("pairwise intersections:")
    print(f"  MMH ∩ e-hanja:     {summary['pairwise']['mmh_ehanja_intersect']:>6,}")
    print(f"  MMH ∩ KanjiVG:     {summary['pairwise']['mmh_kanjivg_intersect']:>6,}")
    print(f"  e-hanja ∩ KanjiVG: {summary['pairwise']['ehanja_kanjivg_intersect']:>6,}")
    print(f"  all three:         {summary['pairwise']['all_three']:>6,}")
    print()
    print(f"MMH-only (not in the other two):        {len(mmh_only_all):>6,}")
    for k, v in summary["mmh_only_by_block"].items():
        print(f"    {k:<22} {v:>6,}")
    print()
    print(f"e-hanja-only (not in the other two):    {len(ehanja_only_all):>6,}")
    for k, v in summary["ehanja_only_by_block"].items():
        print(f"    {k:<22} {v:>6,}")
    print()
    print(f"KanjiVG-only (not in the other two):    {len(kanjivg_only_all):>6,}")
    for k, v in summary["kanjivg_only_by_block"].items():
        print(f"    {k:<22} {v:>6,}")
    print()
    print(f"wrote {summary_path}")
    print(f"wrote {per_char_path}  ({len(union):,} records)")


if __name__ == "__main__":
    main()
