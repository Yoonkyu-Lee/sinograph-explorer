"""Re-select primary_ids based on e-hanja alignment priority.

Prior rule (30_build_ids_table.py): CHISE > cjkvi > BabelStone (coverage
priority). Majority rule in 2-vs-1 cases.

New rule (this script): if e-hanja is present and any IDS source aligns
with e-hanja's `shape.components`, switch primary to that source (priority
among aligned sources: cjkvi > chise > babelstone). This overrides the
majority-rule even when cjkvi is the minority (e.g., 𤨒 where
CHISE+BabelStone = `⿰𤣩恩` but cjkvi+e-hanja = `⿰王恩`).

e-hanja-absent rows are untouched (stick with coverage priority; this is
where CHISE's Ext J 4,298 entries live — none of which overlap e-hanja
anyway).

Updates in place:
  primary_ids
  primary_source
  ids_top_idc
  ids_alternates_json  (recomputed to exclude new primary)
  has_cdp_ref

Writes stats to out/reselect_primary_stats.json.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import unicodedata
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3")
DB = ROOT / "sinograph_canonical_v3/out/ids_merged.sqlite"
STATS = ROOT / "sinograph_canonical_v3/out/reselect_primary_stats.json"

ALIGN_PRIORITY = ("cjkvi", "chise", "babelstone")  # new priority

IDC_ARITY: dict[str, int] = {chr(c): 2 for c in range(0x2FF0, 0x3000)}
IDC_ARITY["\u2ff2"] = 3
IDC_ARITY["\u2ff3"] = 3


def is_idc(ch: str) -> bool:
    return len(ch) == 1 and ch in IDC_ARITY


def top_idc(ids: str) -> str:
    if not ids:
        return "(empty)"
    return ids[0] if is_idc(ids[0]) else "(leaf)"


def has_cdp(ids: str) -> bool:
    return "&CDP-" in ids


def load_primary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["codepoint"]] = d["ids"]
    return out


def load_babelstone_full(path: Path) -> dict[str, list[dict]]:
    """BabelStone has multi-alternate with regions. Return primary + list of
    {ids, regions} per cp. Used to populate ids_alternates_json."""
    out: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids_list = d.get("ids") or []
            regions = d.get("regions") or []
            out[d["codepoint"]] = [
                {"ids": ids, "regions": regions[i] if i < len(regions) else []}
                for i, ids in enumerate(ids_list)
            ]
    return out


def main() -> None:
    if not DB.exists():
        raise SystemExit(f"DB not found: {DB}")

    print("[load] 3 IDS sources...")
    chise = load_primary(ROOT / "db_src/chise_ids/ids_primary.jsonl")
    cjkvi = load_primary(ROOT / "db_src/cjkvi_ids/ids_primary.jsonl")
    babel = load_primary(ROOT / "db_src/babelstone_ids/ids_primary.jsonl")
    babel_full = load_babelstone_full(ROOT / "db_src/babelstone_ids/ids.jsonl")
    print(f"  chise={len(chise):,} cjkvi={len(cjkvi):,} babel={len(babel):,}")

    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT codepoint, primary_ids, primary_source, ehanja_aligned_sources "
        "FROM characters_ids"
    ).fetchall()
    print(f"[load] rows: {len(rows):,}")

    updates = []
    n_changed = 0
    change_by_new_src: Counter = Counter()
    no_change: Counter = Counter()

    for cp, old_primary, old_src, aligned_json in rows:
        aligned = json.loads(aligned_json) if aligned_json else []
        # Decide new primary source
        new_src = None
        for cand in ALIGN_PRIORITY:
            if cand in aligned:
                new_src = cand
                break
        if not new_src:
            no_change["no_ehanja_alignment"] += 1
            continue  # keep current primary (no e-hanja alignment info)

        src_ids = {"chise": chise, "cjkvi": cjkvi, "babelstone": babel}[new_src]
        new_primary = src_ids.get(cp)
        if not new_primary:
            # aligned source lacks entry — shouldn't happen, but safe guard
            no_change["aligned_source_no_ids"] += 1
            continue

        if new_src == old_src and new_primary == old_primary:
            no_change["already_matches"] += 1
            continue

        # Rebuild alternates (all non-primary variants from all 3 sources)
        alternates = []
        # CHISE
        if cp in chise and (new_src != "chise" or chise[cp] != new_primary):
            alternates.append({"source": "chise", "ids": chise[cp]})
        # cjkvi
        if cp in cjkvi and (new_src != "cjkvi" or cjkvi[cp] != new_primary):
            alternates.append({"source": "cjkvi", "ids": cjkvi[cp]})
        # BabelStone (with regions + multi-alternate)
        for alt in babel_full.get(cp, []):
            if new_src == "babelstone" and alt["ids"] == new_primary:
                continue
            alternates.append({
                "source": "babelstone",
                "ids": alt["ids"],
                "regions": alt["regions"],
            })

        updates.append((
            new_primary, new_src, top_idc(new_primary),
            json.dumps(alternates, ensure_ascii=False),
            1 if has_cdp(new_primary) else 0,
            cp,
        ))
        n_changed += 1
        change_by_new_src[new_src] += 1

    print(f"[update] {n_changed:,} rows changing primary")
    print(f"  by new source: {dict(change_by_new_src)}")
    print(f"  no_change    : {dict(no_change)}")

    con.executemany(
        "UPDATE characters_ids SET primary_ids=?, primary_source=?, "
        "ids_top_idc=?, ids_alternates_json=?, has_cdp_ref=? WHERE codepoint=?",
        updates,
    )
    con.commit()
    con.close()

    out_stats = {
        "rows_total": len(rows),
        "rows_changed": n_changed,
        "rows_unchanged": len(rows) - n_changed,
        "changed_by_new_source": dict(change_by_new_src),
        "unchanged_reasons": dict(no_change),
    }
    STATS.write_text(json.dumps(out_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[write] {STATS}")
    print(json.dumps(out_stats, ensure_ascii=False, indent=2))

    print()
    print("NOTE: rerun 31_merge_ehanja_components.py so ehanja_agreement "
          "reflects the new primary.")


if __name__ == "__main__":
    main()
