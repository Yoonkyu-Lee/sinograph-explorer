"""Phase 1 — build characters_structure table.

For each codepoint in the universe (characters_ids, 103,046), derive:
  - radical_idx (1-214)         from Unihan kRSUnicode primarily
  - total_strokes (integer)     from Unihan kTotalStrokes primarily
  - residual_strokes (integer)  from Unihan kRSUnicode (after '.') primarily

e-hanja used as fallback / cross-check.

Unihan kRSUnicode format examples:
  "167.14"    → radical=167, residual=14
  "167'.10"   → radical=167 (apostrophe = simplified/non-Kangxi form),
                 residual=10
  (a list — take first)

Unihan radical chars in the Kangxi Radicals block (U+2F00-U+2FD5) correspond
to radical numbers 1-214. e-hanja `radical.char` is a literal char — map via
Unihan kRSUnicode of that char's codepoint (all 214 radical chars have
kRSUnicode ending in ".0"), or simply use a static table.

Output: table `characters_structure` in the same DB
(sinograph_canonical_v3/out/ids_merged.sqlite).

Source precedence: Unihan > e-hanja. e-hanja value recorded in
`alt_sources_json` when it disagrees with Unihan.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3")
V2_DB = ROOT / "sinograph_canonical_v2/out/sinograph_canonical_v2.sqlite"
V3_DB = ROOT / "sinograph_canonical_v3/out/ids_merged.sqlite"
STATS = ROOT / "sinograph_canonical_v3/out/structure_stats.json"

RSUNICODE_RE = re.compile(r"(\d+)'?\.(\d+)")


def parse_kRSUnicode(raw) -> tuple[int | None, int | None]:
    """Return (radical_idx, residual_strokes) or (None, None)."""
    if not raw:
        return None, None
    s = raw[0] if isinstance(raw, list) else raw
    m = RSUNICODE_RE.match(s.strip())
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def build_radical_char_to_idx(v2: sqlite3.Connection) -> dict[str, int]:
    """Map a radical character literal to its 1-214 number by walking Unihan
    entries whose kRSUnicode = 'N.0'. Covers both Kangxi Radicals (U+2F00-
    U+2FD5) and CJK Radicals Supplement (U+2E80-U+2EFF variant forms), plus
    the underlying unified codepoints (金 U+91D1 is radical 167, 金 U+2F97
    Kangxi char is also radical 167, etc.)."""
    out: dict[str, int] = {}
    for row in v2.execute(
        "SELECT codepoint, payload_json FROM source_payloads "
        "WHERE source_name='unihan'"
    ):
        d = json.loads(row[1])
        rs_raw = d.get("kRSUnicode")
        if not rs_raw:
            continue
        s = rs_raw[0] if isinstance(rs_raw, list) else rs_raw
        m = RSUNICODE_RE.match(s.strip())
        if not m:
            continue
        if int(m.group(2)) != 0:
            continue  # not a radical self-entry
        rad_idx = int(m.group(1))
        ch = chr(int(row[0][2:], 16))
        out[ch] = rad_idx
    # Also hardcode Kangxi block U+2F00-U+2FD5 → 1..214 (in case some don't
    # have kRSUnicode N.0 entry)
    for i in range(214):
        out[chr(0x2F00 + i)] = i + 1
    return out


def main() -> None:
    v2 = sqlite3.connect(V2_DB)
    v3 = sqlite3.connect(V3_DB)

    # Universe
    universe = {r[0] for r in v3.execute("SELECT codepoint FROM characters_ids")}
    print(f"[universe] {len(universe):,} cp")

    # Build radical char → idx map
    print("[load] Unihan radical char → idx mapping...")
    rad_char_to_idx = build_radical_char_to_idx(v2)
    print(f"  {len(rad_char_to_idx)} radical-form chars → 1-214 idx")

    # Load Unihan kRSUnicode + kTotalStrokes for universe
    print("[load] Unihan kRSUnicode + kTotalStrokes...")
    unihan_rs: dict[str, tuple[int, int]] = {}
    unihan_ts: dict[str, int] = {}
    for row in v2.execute(
        "SELECT codepoint, payload_json FROM source_payloads "
        "WHERE source_name='unihan'"
    ):
        cp = row[0]
        if cp not in universe:
            continue
        d = json.loads(row[1])
        rad_idx, residual = parse_kRSUnicode(d.get("kRSUnicode"))
        if rad_idx is not None:
            unihan_rs[cp] = (rad_idx, residual)
        ts = d.get("kTotalStrokes")
        if ts:
            try:
                unihan_ts[cp] = int(ts[0] if isinstance(ts, list) else ts)
            except ValueError:
                pass
    print(f"  unihan kRSUnicode: {len(unihan_rs):,}")
    print(f"  unihan kTotalStrokes: {len(unihan_ts):,}")

    # Load e-hanja total_strokes + radical_strokes + radical.char
    print("[load] e-hanja total_strokes + radical_strokes + radical.char...")
    eh_total: dict[str, int] = {}
    eh_radstroke: dict[str, int] = {}
    eh_rad_idx: dict[str, int] = {}
    eh_unknown_radicals: Counter = Counter()
    for row in v2.execute(
        "SELECT codepoint, payload_json FROM source_payloads "
        "WHERE source_name='ehanja_online'"
    ):
        cp = row[0]
        if cp not in universe:
            continue
        d = json.loads(row[1])
        det = d.get("detail_raw") or {}
        if det.get("total_strokes") is not None:
            eh_total[cp] = int(det["total_strokes"])
        if det.get("radical_strokes") is not None:
            eh_radstroke[cp] = int(det["radical_strokes"])
        rad_char = (det.get("radical") or {}).get("char")
        if rad_char:
            if rad_char in rad_char_to_idx:
                eh_rad_idx[cp] = rad_char_to_idx[rad_char]
            else:
                eh_unknown_radicals[rad_char] += 1
    print(f"  e-hanja total_strokes : {len(eh_total):,}")
    print(f"  e-hanja radical_strokes: {len(eh_radstroke):,}")
    print(f"  e-hanja radical → idx : {len(eh_rad_idx):,}")
    if eh_unknown_radicals:
        print(f"  e-hanja unknown radical chars (not mappable): "
              f"{sum(eh_unknown_radicals.values())} occurrences across "
              f"{len(eh_unknown_radicals)} distinct chars")
        for ch, n in eh_unknown_radicals.most_common(5):
            print(f"    '{ch}' U+{ord(ch):04X}: {n}")

    # Build structure table
    print("[build] merging...")
    rows_to_insert = []
    stats = Counter()
    disagreement_examples: list = []

    for cp in universe:
        rad_idx = None
        residual = None
        total = None
        sources = {}
        alt_sources = {}

        # Primary: Unihan
        if cp in unihan_rs:
            rad_idx, residual = unihan_rs[cp]
            sources["radical_idx"] = "unihan"
            sources["residual_strokes"] = "unihan"
        # Fallback: e-hanja
        elif cp in eh_rad_idx:
            rad_idx = eh_rad_idx[cp]
            sources["radical_idx"] = "e-hanja"
            stats["radical_fallback_to_ehanja"] += 1

        if cp in unihan_ts:
            total = unihan_ts[cp]
            sources["total_strokes"] = "unihan"
        elif cp in eh_total:
            total = eh_total[cp]
            sources["total_strokes"] = "e-hanja"
            stats["total_strokes_fallback_to_ehanja"] += 1

        # residual fallback: total - eh_radstroke (if Unihan missing but e-hanja has both)
        if residual is None and cp in eh_total and cp in eh_radstroke:
            residual = eh_total[cp] - eh_radstroke[cp]
            if residual >= 0:
                sources["residual_strokes"] = "e-hanja (computed)"
                stats["residual_fallback_to_ehanja"] += 1
            else:
                residual = None

        # Cross-check disagreements
        if cp in unihan_rs and cp in eh_rad_idx:
            u_rad, _ = unihan_rs[cp]
            if u_rad != eh_rad_idx[cp]:
                alt_sources["e-hanja.radical_idx"] = eh_rad_idx[cp]
                stats["radical_disagree_unihan_vs_ehanja"] += 1
                if len(disagreement_examples) < 5:
                    disagreement_examples.append(
                        (cp, chr(int(cp[2:], 16)), "radical",
                         f"unihan={u_rad} ehanja={eh_rad_idx[cp]}"))
        if cp in unihan_ts and cp in eh_total:
            if unihan_ts[cp] != eh_total[cp]:
                alt_sources["e-hanja.total_strokes"] = eh_total[cp]
                stats["total_strokes_disagree_unihan_vs_ehanja"] += 1
                if len(disagreement_examples) < 10:
                    disagreement_examples.append(
                        (cp, chr(int(cp[2:], 16)), "total_strokes",
                         f"unihan={unihan_ts[cp]} ehanja={eh_total[cp]}"))

        # Tally coverage
        if rad_idx is not None: stats["has_radical"] += 1
        if total is not None:   stats["has_total_strokes"] += 1
        if residual is not None: stats["has_residual"] += 1

        rows_to_insert.append((
            cp, rad_idx, total, residual,
            json.dumps(sources, ensure_ascii=False) if sources else None,
            json.dumps(alt_sources, ensure_ascii=False) if alt_sources else None,
        ))

    # Write table
    v3.execute("DROP TABLE IF EXISTS characters_structure")
    v3.execute("""
        CREATE TABLE characters_structure (
            codepoint         TEXT PRIMARY KEY,
            radical_idx       INTEGER,
            total_strokes     INTEGER,
            residual_strokes  INTEGER,
            sources_json      TEXT,
            alt_sources_json  TEXT
        )
    """)
    v3.executemany(
        "INSERT INTO characters_structure VALUES (?, ?, ?, ?, ?, ?)",
        rows_to_insert,
    )
    v3.execute("CREATE INDEX idx_struct_radical ON characters_structure(radical_idx)")
    v3.execute("CREATE INDEX idx_struct_strokes ON characters_structure(total_strokes)")
    v3.commit()

    N = len(universe)
    out_stats = {
        "universe": N,
        "coverage": {
            "radical_idx":      {"count": stats["has_radical"], "pct": round(100*stats["has_radical"]/N, 2)},
            "total_strokes":    {"count": stats["has_total_strokes"], "pct": round(100*stats["has_total_strokes"]/N, 2)},
            "residual_strokes": {"count": stats["has_residual"], "pct": round(100*stats["has_residual"]/N, 2)},
        },
        "fallback_counts": {
            "radical_fallback_to_ehanja":     stats["radical_fallback_to_ehanja"],
            "total_strokes_fallback_to_ehanja": stats["total_strokes_fallback_to_ehanja"],
            "residual_fallback_to_ehanja":    stats["residual_fallback_to_ehanja"],
        },
        "disagreements": {
            "radical_unihan_vs_ehanja":     stats["radical_disagree_unihan_vs_ehanja"],
            "total_strokes_unihan_vs_ehanja": stats["total_strokes_disagree_unihan_vs_ehanja"],
        },
        "disagreement_samples": [
            {"cp": cp, "char": ch, "field": f, "values": v}
            for cp, ch, f, v in disagreement_examples[:10]
        ],
    }
    STATS.write_text(json.dumps(out_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print(json.dumps(out_stats, ensure_ascii=False, indent=2))

    v2.close(); v3.close()


if __name__ == "__main__":
    main()
