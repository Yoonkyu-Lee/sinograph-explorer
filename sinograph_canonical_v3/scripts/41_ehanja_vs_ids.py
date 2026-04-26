"""Compare e-hanja `shape.components` (flat component list) with the
top-level children extracted from each IDS source's decomposition tree.

Questions answered:
 1. How often do e-hanja's components match each IDS source?
 2. When IDS sources disagree (Section 6.2 structure_only cases), which
    source's component naming does e-hanja match?

Method:
 - For each e-hanja detail entry (75,669 cps), extract the
   `shape.components[*].char` list. This is e-hanja's "immediate children"
   decomposition.
 - For each IDS source (BabelStone / CHISE / cjkvi / merged primary),
   parse IDS string into top-level children (recursing through nested IDS
   to collect *immediate* children of the root IDC, not all leaves).
 - Compare as multisets.

Output:
 - Overall agreement % per source
 - Per-category breakdown: agreement for unanimous vs structure_only vs …
 - Sample disagreements where e-hanja matches ONE source but not the others
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3")
DB_SRC = ROOT / "db_src"
DB_V3 = ROOT / "sinograph_canonical_v3/out/ids_merged.sqlite"

IDC_ARITY: dict[str, int] = {
    "\u2ff0": 2,  # ⿰ 좌우
    "\u2ff1": 2,  # ⿱ 상하
    "\u2ff2": 3,  # ⿲ 좌중우
    "\u2ff3": 3,  # ⿳ 상중하
    "\u2ff4": 2,  # ⿴ 둘러싸기
    "\u2ff5": 2,  # ⿵ 위-둘러
    "\u2ff6": 2,  # ⿶ 아래-둘러
    "\u2ff7": 2,  # ⿷ 좌-둘러
    "\u2ff8": 2,  # ⿸ 좌상-둘러
    "\u2ff9": 2,  # ⿹ 우상-둘러
    "\u2ffa": 2,  # ⿺ 좌하-둘러
    "\u2ffb": 2,  # ⿻ 중첩
    "\u2ffc": 2,  # ⿼ 반사
    "\u2ffd": 2,  # ⿽ 회전
    "\u2ffe": 2,  # ⿾ 거꾸로
    "\u2fff": 2,  # ⿿ 부분포함
}


def is_idc(ch: str) -> bool:
    return len(ch) == 1 and ch in IDC_ARITY


def parse_ids_unit(s: str, i: int) -> tuple[str, int]:
    """Parse one 'unit' starting at s[i]. Returns (unit_str, next_i)."""
    if i >= len(s):
        return "", i

    first = s[i]
    # &CDP-XXXX; entity reference (CHISE)
    if first == "&":
        end = s.find(";", i)
        if end != -1:
            return s[i:end + 1], end + 1
        return first, i + 1
    # BabelStone's {nn} numeric placeholder for un-encoded component
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
    """Return the list of immediate children of the root IDC.
    If IDS is atomic (no leading IDC), return [ids]."""
    if not ids:
        return []
    first = ids[0]
    if not is_idc(first):
        return [ids]
    arity = IDC_ARITY[first]
    children = []
    i = 1
    for _ in range(arity):
        unit, i = parse_ids_unit(ids, i)
        children.append(unit)
    return children


def children_to_leaves(children: list[str]) -> list[str]:
    """Flatten recursively. From top-level children that may themselves be
    nested IDS sub-trees, recurse to gather all leaf (non-IDC) chars.
    Used when comparing at leaf level instead of immediate-children level."""
    leaves = []
    for c in children:
        if not c:
            continue
        if is_idc(c[0]):
            leaves.extend(children_to_leaves(top_children(c)))
        else:
            leaves.append(c)
    return leaves


def load_ehanja_components() -> dict[str, list[str]]:
    """Returns {cp: [comp_char, ...]} from e-hanja shape.components."""
    out: dict[str, list[str]] = {}
    with open(DB_SRC / "e-hanja_online/detail.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cp = f"U+{d['cp']:04X}"
            sh = d.get("shape") or {}
            comps = sh.get("components") or []
            chars = [c.get("char") for c in comps if c.get("char")]
            if chars:
                out[cp] = chars
    return out


def load_ids_source(name: str) -> dict[str, str]:
    path = {
        "babelstone": DB_SRC / "babelstone_ids/ids_primary.jsonl",
        "chise":      DB_SRC / "chise_ids/ids_primary.jsonl",
        "cjkvi":      DB_SRC / "cjkvi_ids/ids_primary.jsonl",
    }[name]
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["codepoint"]] = d["ids"]
    return out


def load_merged_primary() -> dict[str, tuple[str, str, str]]:
    """Returns {cp: (primary_ids, primary_source, agreement_level)}."""
    out: dict[str, tuple[str, str, str]] = {}
    con = sqlite3.connect(DB_V3)
    for row in con.execute(
        "SELECT codepoint, primary_ids, primary_source, agreement_level "
        "FROM characters_ids"
    ):
        out[row[0]] = (row[1], row[2], row[3])
    con.close()
    return out


def compare(eh_comps: list[str], ids_str: str, mode: str = "immediate") -> bool:
    """Compare e-hanja components with IDS tree decomposition.
    mode='immediate': compare with immediate children of root IDC
    mode='leaves': compare with all leaf characters (recursive)
    """
    children = top_children(ids_str)
    if mode == "immediate":
        ids_comps = children
    else:
        ids_comps = children_to_leaves(children)
    # Strip atomic singletons: if ids_str is atomic and eh has no comps, match
    return Counter(eh_comps) == Counter(ids_comps)


def main() -> None:
    print("[load] e-hanja components...")
    eh = load_ehanja_components()
    print(f"  {len(eh):,} cps with shape.components")

    print("[load] IDS sources + merged primary...")
    babel = load_ids_source("babelstone")
    chise = load_ids_source("chise")
    cjkvi = load_ids_source("cjkvi")
    merged = load_merged_primary()
    print(f"  babelstone={len(babel):,}  chise={len(chise):,}  cjkvi={len(cjkvi):,}")
    print(f"  merged primary = {len(merged):,}")

    # Universe: cps present in e-hanja AND in at least one IDS source
    ids_universe = set(babel) | set(chise) | set(cjkvi)
    shared = set(eh) & ids_universe
    print(f"\n[compare] e-hanja ∩ (IDS union): {len(shared):,} cps")

    # Per-source agreement (immediate children)
    print("\n" + "=" * 72)
    print("Agreement: e-hanja components == IDS source's immediate children")
    print("=" * 72)
    print(f"{'source':<14} {'available':>10} {'matched':>10} {'rate':>8}")
    print("-" * 72)
    for name, src in [("babelstone", babel), ("chise", chise),
                       ("cjkvi", cjkvi), ("merged.primary",
                        {k: v[0] for k, v in merged.items()})]:
        inter = shared & set(src)
        n_match = sum(1 for cp in inter if compare(eh[cp], src[cp]))
        rate = 100 * n_match / len(inter) if inter else 0
        print(f"{name:<14} {len(inter):>10,} {n_match:>10,} {rate:>7.2f}%")

    # The user's intuition: e-hanja preferentially matches cjkvi's canonical form?
    # Compute per-cp "who does e-hanja agree with?"
    print("\n" + "=" * 72)
    print("Per-codepoint agreement pattern (e-hanja matches which source?)")
    print("=" * 72)
    pattern_counts: Counter = Counter()
    mismatch_examples: dict[str, list[tuple[str, str, list[str], str]]] = {}
    for cp in shared:
        eh_c = eh[cp]
        results = {}
        for name, src in [("babelstone", babel), ("chise", chise),
                           ("cjkvi", cjkvi)]:
            if cp in src:
                results[name] = compare(eh_c, src[cp])
        # Build pattern key: which sources agree with e-hanja
        agreeing = sorted([n for n, r in results.items() if r])
        missing = sorted(set(("babelstone", "chise", "cjkvi")) - set(results))
        key_parts = []
        if agreeing:
            key_parts.append("agree={" + "+".join(agreeing) + "}")
        else:
            key_parts.append("agree={none}")
        if missing:
            key_parts.append("absent={" + "+".join(missing) + "}")
        key = " ".join(key_parts)
        pattern_counts[key] += 1
        # Save a few examples per pattern
        if key not in mismatch_examples or len(mismatch_examples[key]) < 3:
            mismatch_examples.setdefault(key, []).append((cp, "".join(eh_c),
                eh_c, {n: src[cp] for n, src in [("babelstone", babel), ("chise", chise), ("cjkvi", cjkvi)] if cp in src}))

    for key, cnt in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / len(shared)
        print(f"  {cnt:>7,} ({pct:>5.2f}%)  {key}")

    # Show a few "cjkvi-only match" examples (user's intuition)
    print("\n" + "=" * 72)
    print("Examples where e-hanja matches cjkvi only (user's intuition check):")
    print("=" * 72)
    examples = [(k, v) for k, v in mismatch_examples.items()
                if "agree={cjkvi}" == k.split(" ")[0].replace("agree=", "agree=")
                and k.startswith("agree={cjkvi}")]
    shown = 0
    for key, ex_list in examples:
        if not key.startswith("agree={cjkvi}"):
            continue
        for cp, eh_str, eh_list, ids_map in ex_list[:3]:
            print(f"\n  {cp}  '{chr(int(cp[2:], 16))}'")
            print(f"    e-hanja components : {eh_list}")
            for s in ("babelstone", "chise", "cjkvi"):
                if s in ids_map:
                    ch = top_children(ids_map[s])
                    mark = "✓" if Counter(eh_list) == Counter(ch) else "✗"
                    print(f"    {s:<12} {ids_map[s]:<20} children={ch}  {mark}")
            shown += 1
            if shown >= 8:
                break
        if shown >= 8:
            break

    # Same for chise-only and babelstone-only
    for target in ("chise", "babelstone"):
        print("\n" + "=" * 72)
        print(f"Examples where e-hanja matches {target} only:")
        print("=" * 72)
        shown = 0
        for key, ex_list in mismatch_examples.items():
            if key != f"agree={{{target}}}":
                continue
            for cp, eh_str, eh_list, ids_map in ex_list[:3]:
                print(f"\n  {cp}  '{chr(int(cp[2:], 16))}'")
                print(f"    e-hanja components : {eh_list}")
                for s in ("babelstone", "chise", "cjkvi"):
                    if s in ids_map:
                        ch = top_children(ids_map[s])
                        mark = "✓" if Counter(eh_list) == Counter(ch) else "✗"
                        print(f"    {s:<12} {ids_map[s]:<20} children={ch}  {mark}")
                shown += 1
                if shown >= 5:
                    break
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
