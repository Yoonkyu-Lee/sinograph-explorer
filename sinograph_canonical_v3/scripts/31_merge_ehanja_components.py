"""Merge e-hanja `shape.components` (flat list) into canonical_v3
characters_ids as cross-check / confidence signal.

Adds columns:
  ehanja_components_json  TEXT  — e-hanja's immediate-children list ([金, 監])
  ehanja_agreement        TEXT  — classification:
     'unanimous'         — e-hanja matches all IDS sources present
     'matches_multi'     — matches 2 of 3 IDS sources
     'matches_chise'     — matches CHISE only
     'matches_cjkvi'     — matches cjkvi only
     'matches_babelstone'— matches BabelStone only
     'matches_primary'   — matches the selected primary_ids
     'disagree_all'      — matches NO IDS source
     'ids_atomic'        — IDS is atomic, e-hanja decomposes
     'ehanja_absent'     — e-hanja has no entry or no components
  ehanja_aligned_sources  TEXT  — JSON list of IDS sources matching e-hanja
                                  (['chise','cjkvi'] etc.)

NOT a training label — used as confidence multiplier for primary_ids and
(optionally) as signal to re-select primary from cjkvi when e-hanja aligns
with cjkvi (per user observation that e-hanja ≈ cjkvi canonical form).

Usage:
  python sinograph_canonical_v3/scripts/31_merge_ehanja_components.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path


def normalize_char(s: str) -> str:
    """NFKC normalization maps CJK Compatibility Ideographs (U+F900-U+FAFF,
    U+2F800-U+2FA1F) to their canonical CJK Unified equivalents. Example:
    e-hanja '金' U+F90A → '金' U+91D1. Without this, multiset comparison
    between e-hanja components and IDS children spuriously fails on ~5,600
    chars (7.4% of e-hanja rows)."""
    return unicodedata.normalize("NFKC", s)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3")
DB = ROOT / "sinograph_canonical_v3/out/ids_merged.sqlite"
EHANJA_DETAIL = ROOT / "db_src/e-hanja_online/detail.jsonl"

IDC_ARITY: dict[str, int] = {
    chr(c): 2 for c in range(0x2FF0, 0x3000)
}
# Override 3-arity IDCs
IDC_ARITY["\u2ff2"] = 3  # ⿲
IDC_ARITY["\u2ff3"] = 3  # ⿳


def is_idc(ch: str) -> bool:
    return len(ch) == 1 and ch in IDC_ARITY


def parse_ids_unit(s: str, i: int) -> tuple[str, int]:
    if i >= len(s):
        return "", i
    first = s[i]
    if first == "&":
        end = s.find(";", i)
        if end != -1:
            return s[i:end + 1], end + 1
        return first, i + 1
    if first == "{":
        end = s.find("}", i)
        if end != -1:
            return s[i:end + 1], end + 1
        return first, i + 1
    if not is_idc(first):
        return first, i + 1
    arity = IDC_ARITY[first]
    j = i + 1
    for _ in range(arity):
        if j >= len(s):
            break
        _, j = parse_ids_unit(s, j)
    return s[i:j], j


def top_children(ids: str) -> list[str]:
    if not ids:
        return []
    if not is_idc(ids[0]):
        return [ids]
    arity = IDC_ARITY[ids[0]]
    children, i = [], 1
    for _ in range(arity):
        if i >= len(ids):
            break
        unit, i = parse_ids_unit(ids, i)
        children.append(unit)
    return children


def load_ehanja() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(EHANJA_DETAIL, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cp = f"U+{d['cp']:04X}"
            sh = d.get("shape") or {}
            comps = sh.get("components") or []
            chars = [normalize_char(c.get("char")) for c in comps if c.get("char")]
            if chars:
                out[cp] = chars
    return out


def classify(eh_comps: list[str], sources_ids: dict[str, str],
              primary_src: str) -> tuple[str, list[str]]:
    """Return (agreement_category, aligned_sources_list)."""
    eh_multiset = Counter(eh_comps)
    aligned = []
    for name, ids in sources_ids.items():
        if not ids:
            continue
        src_children = [normalize_char(c) for c in top_children(ids)]
        if Counter(src_children) == eh_multiset:
            aligned.append(name)

    # Is primary's IDS atomic while e-hanja decomposes?
    primary_ids_str = sources_ids.get(primary_src, "")
    primary_is_atomic = (primary_ids_str and not is_idc(primary_ids_str[0]))
    if primary_is_atomic and len(eh_comps) >= 2:
        return "ids_atomic", aligned

    n = len(aligned)
    if n == 0:
        return "disagree_all", aligned
    if n == len(sources_ids):  # matches all present sources
        return "unanimous", aligned
    if n >= 2:
        return "matches_multi", aligned
    # exactly 1 match
    return f"matches_{aligned[0]}", aligned


def main() -> None:
    if not DB.exists():
        raise SystemExit(f"canonical_v3 DB not found: {DB}. Run 30 first.")

    print("[load] e-hanja components...")
    eh = load_ehanja()
    print(f"  {len(eh):,} cps with shape.components")

    con = sqlite3.connect(DB)

    # Add columns (idempotent)
    existing = {row[1] for row in con.execute("PRAGMA table_info(characters_ids)")}
    for col, typ in [
        ("ehanja_components_json", "TEXT"),
        ("ehanja_agreement", "TEXT"),
        ("ehanja_aligned_sources", "TEXT"),
    ]:
        if col not in existing:
            con.execute(f"ALTER TABLE characters_ids ADD COLUMN {col} {typ}")
    con.commit()

    # Load primary per cp
    rows = con.execute(
        "SELECT codepoint, primary_ids, primary_source "
        "FROM characters_ids"
    ).fetchall()
    print(f"[load] canonical_v3 rows: {len(rows):,}")

    # Load each IDS source directly — the DB's ids_alternates_json omits
    # sources whose IDS happens to equal primary, which would bias our
    # per-source alignment counts. Re-reading raw files gives us the true
    # "does source X have this IDS for this cp?" signal.
    def load_primary(path: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                out[d["codepoint"]] = d["ids"]
        return out

    print("[load] 3 IDS sources for per-source comparison...")
    babel = load_primary(ROOT / "db_src/babelstone_ids/ids_primary.jsonl")
    chise = load_primary(ROOT / "db_src/chise_ids/ids_primary.jsonl")
    cjkvi = load_primary(ROOT / "db_src/cjkvi_ids/ids_primary.jsonl")
    print(f"  babelstone={len(babel):,} chise={len(chise):,} cjkvi={len(cjkvi):,}")

    def src_ids_map(cp: str, primary_src: str, primary_ids: str) -> dict[str, str]:
        m: dict[str, str] = {}
        if cp in chise: m["chise"] = chise[cp]
        if cp in cjkvi: m["cjkvi"] = cjkvi[cp]
        if cp in babel: m["babelstone"] = babel[cp]
        return m

    cat_counts: Counter = Counter()
    updates: list[tuple[str, str, str, str]] = []
    for cp, primary_ids, primary_src in rows:
        eh_comps = eh.get(cp)
        if not eh_comps:
            updates.append((None, "ehanja_absent", json.dumps([]), cp))
            cat_counts["ehanja_absent"] += 1
            continue
        sources = src_ids_map(cp, primary_src, primary_ids)
        cat, aligned = classify(eh_comps, sources, primary_src)
        cat_counts[cat] += 1
        updates.append((
            json.dumps(eh_comps, ensure_ascii=False),
            cat,
            json.dumps(aligned, ensure_ascii=False),
            cp,
        ))

    print(f"[update] writing {len(updates):,} rows...")
    con.executemany(
        "UPDATE characters_ids SET ehanja_components_json=?, "
        "ehanja_agreement=?, ehanja_aligned_sources=? WHERE codepoint=?",
        updates,
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_eh_agreement "
                 "ON characters_ids(ehanja_agreement)")
    con.commit()
    con.close()

    print()
    print("=" * 60)
    print("ehanja_agreement distribution:")
    print("=" * 60)
    total = sum(cat_counts.values())
    for cat, n in cat_counts.most_common():
        print(f"  {cat:<25} {n:>7,}  ({100*n/total:.1f}%)")

    # Quick sanity: show aligned_sources breakdown for informative cats
    print()
    con = sqlite3.connect(DB)
    print("aligned_sources breakdown (only non-absent cases):")
    rows2 = con.execute(
        "SELECT ehanja_aligned_sources, COUNT(*) FROM characters_ids "
        "WHERE ehanja_agreement NOT IN ('ehanja_absent') "
        "GROUP BY ehanja_aligned_sources ORDER BY COUNT(*) DESC LIMIT 12"
    ).fetchall()
    for al, n in rows2:
        print(f"  {al:<40} {n:>6,}")

    # Per-IDS source alignment rate (among e-hanja-present cps)
    print()
    print("Per-source alignment with e-hanja:")
    for src in ("chise", "cjkvi", "babelstone"):
        n = con.execute(
            "SELECT COUNT(*) FROM characters_ids "
            "WHERE ehanja_aligned_sources LIKE ?",
            (f'%"{src}"%',)
        ).fetchone()[0]
        print(f"  {src:<12} : {n:,}")
    con.close()


if __name__ == "__main__":
    main()
