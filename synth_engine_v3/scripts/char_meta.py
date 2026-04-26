"""Per-char metadata lookup (canonical_v3.characters_structure).

Loads once per worker; every subsequent call is an O(1) dict hit. Used by
the synth pipeline's stroke-count-aware caps (stroke_weight.dilate cap,
high-stroke font-weight filter) so rendering adapts to character complexity.

Added 2026-04-23.
"""
from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

_DB = (
    Path(__file__).resolve().parents[2]
    / "sinograph_canonical_v3" / "out" / "ids_merged.sqlite"
)


@lru_cache(maxsize=1)
def _load_stroke_table() -> dict[int, int]:
    """codepoint → total_strokes. Empty dict if DB missing (cold / dev env)."""
    if not _DB.exists():
        return {}
    out: dict[int, int] = {}
    con = sqlite3.connect(_DB)
    try:
        cur = con.execute(
            "SELECT codepoint, total_strokes FROM characters_structure "
            "WHERE total_strokes IS NOT NULL"
        )
        for cp_s, ts in cur:
            try:
                cp = int(cp_s[2:], 16)  # "U+XXXX" → int
            except Exception:
                continue
            out[cp] = int(ts)
    finally:
        con.close()
    return out


def get_total_strokes(char: str) -> int:
    """Return total_strokes for `char`, or 0 when unknown."""
    table = _load_stroke_table()
    return table.get(ord(char), 0)


def total_strokes_for(chars) -> list[int]:
    """Vectorized lookup for a batch of chars."""
    table = _load_stroke_table()
    return [table.get(ord(c), 0) for c in chars]


def chars_by_stroke_count() -> dict[int, list[int]]:
    """Inverse index: stroke_count → list of codepoints. Useful for the
    diagnostic grid that samples one char per stroke-count band."""
    table = _load_stroke_table()
    inv: dict[int, list[int]] = {}
    for cp, ts in table.items():
        inv.setdefault(ts, []).append(cp)
    for ts in inv:
        inv[ts].sort()
    return inv
