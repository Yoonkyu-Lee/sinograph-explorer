"""Canonical DB 자전 (dictionary) lookup — codepoint → radical / strokes /
readings / meanings / variant family.

Standalone: takes a codepoint or literal character and prints the full
structured entry. Not yet wired to the OCR model — intended as the "second
half" of the pipeline (model → top-1 codepoint → this lookup).

Usage:
  # by codepoint (U+XXXX or raw hex)
  python 33_canonical_lookup.py --cp U+24A12
  python 33_canonical_lookup.py --cp 24A12

  # by literal char
  python 33_canonical_lookup.py --char 𤨒

  # compact one-line mode (for piping)
  python 33_canonical_lookup.py --char 𤨒 --oneline
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_DB = (Path(__file__).resolve().parents[2]
              / "sinograph_canonical_v2" / "out"
              / "sinograph_canonical_v2.sqlite")


def normalize_cp(s: str) -> str:
    """Accepts 'U+24A12', 'u+24a12', or '24A12' → 'U+24A12'."""
    s = s.strip().upper()
    if s.startswith("U+"):
        s = s[2:]
    int(s, 16)  # validate
    return f"U+{s}"


def char_to_cp(ch: str) -> str:
    return f"U+{ord(ch):04X}"


def cp_to_char(cp: str) -> str:
    return chr(int(cp[2:], 16))


def lookup(con: sqlite3.Connection, cp: str) -> dict | None:
    """Return a structured record for the given codepoint, or None if absent."""
    row = con.execute(
        "SELECT codepoint, character, radical, total_strokes, "
        "canonical_representative, enriched_representative, data_json "
        "FROM characters WHERE codepoint=?", (cp,)
    ).fetchone()
    if not row:
        return None

    cp_, ch, radical, strokes, canon_rep, enriched_rep, data_json = row
    data = json.loads(data_json)
    core = data.get("core", {})
    variants = data.get("variants", {})

    readings = con.execute(
        "SELECT reading_type, value FROM character_readings WHERE codepoint=?",
        (cp,)
    ).fetchall()
    meanings = con.execute(
        "SELECT language, value FROM character_meanings WHERE codepoint=?",
        (cp,)
    ).fetchall()
    presence = con.execute(
        "SELECT unihan, ehanja_online, kanjidic2, makemeahanzi "
        "FROM source_presence WHERE codepoint=?", (cp,)
    ).fetchone()

    # Parse e-hanja source payload for jahun (훈) + dok (음) pairs.
    # Format in tree_raw.getHunum[*].hRead is "훈 음" (Korean convention),
    # e.g. "옥 은" → jahun="옥", dok="은".
    jahun_list: list[str] = []
    dok_list: list[str] = []
    eh = con.execute(
        "SELECT payload_json FROM source_payloads "
        "WHERE codepoint=? AND source_name='ehanja_online'", (cp,)
    ).fetchone()
    if eh:
        try:
            tree = json.loads(eh[0]).get("tree_raw", {})
            for h in tree.get("getHunum", []) or []:
                s = (h.get("hRead") or "").strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) >= 2:
                    jahun_list.append(" ".join(parts[:-1]))
                    dok_list.append(parts[-1])
                else:
                    dok_list.append(parts[0])
        except Exception:
            pass

    vc_row = con.execute(
        "SELECT family_members_json, canonical_family_members_json, "
        "enriched_family_members_json FROM variant_components WHERE codepoint=?",
        (cp,)
    ).fetchone()
    family = json.loads(vc_row[0]) if vc_row and vc_row[0] else [cp]
    canon_family = json.loads(vc_row[1]) if vc_row and vc_row[1] else [cp]
    enriched_family = json.loads(vc_row[2]) if vc_row and vc_row[2] else [cp]

    # Edges: direct variant relationships (traditional↔simplified, spoofing, etc.)
    edges = con.execute(
        "SELECT source_codepoint, target_codepoint, relation, relation_scope, "
        "support_count FROM variant_edges "
        "WHERE source_codepoint=? OR target_codepoint=?",
        (cp, cp)
    ).fetchall()

    return {
        "codepoint": cp_,
        "character": ch,
        "radical": radical,
        "total_strokes": strokes,
        "canonical_representative": canon_rep,
        "enriched_representative": enriched_rep,
        "source_presence": {
            "unihan": bool(presence[0]) if presence else False,
            "ehanja_online": bool(presence[1]) if presence else False,
            "kanjidic2": bool(presence[2]) if presence else False,
            "makemeahanzi": bool(presence[3]) if presence else False,
        },
        "readings_grouped": core.get("readings", {}),
        "korean_dok": dok_list,          # 음독 (sound reading)
        "korean_jahun": jahun_list,      # 자훈 (character gloss)
        "readings_flat": [(t, v) for t, v in readings],
        "meanings": [(lang, v) for lang, v in meanings],
        "definitions_en": core.get("definitions", {}).get("english", []),
        "family_members": family,
        "canonical_family": canon_family,
        "enriched_family": enriched_family,
        "variants_detail": variants,
        "edges": [
            {"src": e[0], "tgt": e[1], "relation": e[2],
             "scope": e[3], "support": e[4]}
            for e in edges
        ],
    }


def fmt_pretty(rec: dict) -> str:
    """Human-friendly multi-line formatting."""
    lines = []
    ch = rec["character"]; cp = rec["codepoint"]
    lines.append("=" * 60)
    lines.append(f"  {ch}   {cp}")
    lines.append("=" * 60)
    lines.append(f"  radical:       {rec['radical']}   total strokes: "
                 f"{rec['total_strokes']}")

    pres = rec["source_presence"]
    flags = [k for k, v in pres.items() if v]
    lines.append(f"  source flags:  {', '.join(flags) if flags else '(none)'}")

    # Readings by language. Korean: prefer e-hanja-derived dok+jahun pair
    # (훈 음) over the raw kHangul reading, since it carries the jahun gloss.
    rg = rec["readings_grouped"]
    for lang_key in ("mandarin", "cantonese"):
        vals = rg.get(lang_key, [])
        if vals:
            lines.append(f"  {lang_key:<14} {', '.join(vals)}")
    if rec["korean_dok"]:
        lines.append(f"  {'korean_dok':<14} {', '.join(rec['korean_dok'])}")
    elif rg.get("korean_hangul"):
        lines.append(f"  {'korean_dok':<14} {', '.join(rg['korean_hangul'])}")
    if rec["korean_jahun"]:
        lines.append(f"  {'korean_jahun':<14} {', '.join(rec['korean_jahun'])}")
    for lang_key in ("japanese_on", "japanese_kun", "vietnamese"):
        vals = rg.get(lang_key, [])
        if vals:
            lines.append(f"  {lang_key:<14} {', '.join(vals)}")

    # Meanings (ko/en/etc.)
    if rec["meanings"]:
        lines.append("  meanings:")
        for lang, v in rec["meanings"]:
            lines.append(f"    [{lang}] {v}")
    if rec["definitions_en"]:
        lines.append(f"  english def:   {'; '.join(rec['definitions_en'])}")

    # Family
    fam = rec["family_members"]
    if len(fam) > 1:
        chars = []
        for m in fam:
            try:
                chars.append(chr(int(m[2:], 16)))
            except Exception:
                chars.append("?")
        lines.append(f"  variant family ({len(fam)}): {' '.join(chars)}")
    else:
        lines.append(f"  variant family: (singleton — no recorded variants)")

    # Enriched family (canonical DB's broadest grouping)
    ef = rec["enriched_family"]
    if ef != fam:
        chars = []
        for m in ef:
            try:
                chars.append(chr(int(m[2:], 16)))
            except Exception:
                chars.append("?")
        lines.append(f"  enriched family ({len(ef)}): {' '.join(chars)}")

    # Edges
    if rec["edges"]:
        lines.append(f"  direct variant edges ({len(rec['edges'])}):")
        for e in rec["edges"][:10]:
            src = cp_to_char(e["src"]) if e["src"].startswith("U+") else "?"
            tgt = cp_to_char(e["tgt"]) if e["tgt"].startswith("U+") else "?"
            lines.append(f"    {src}({e['src']}) --{e['relation']}--> "
                         f"{tgt}({e['tgt']})  [support={e['support']}]")
        if len(rec["edges"]) > 10:
            lines.append(f"    ... ({len(rec['edges']) - 10} more)")

    return "\n".join(lines)


def fmt_oneline(rec: dict) -> str:
    ch = rec["character"]; cp = rec["codepoint"]
    rg = rec["readings_grouped"]
    dok = ",".join(rec["korean_dok"] or rg.get("korean_hangul", []))
    jahun = ",".join(rec["korean_jahun"])
    man = ",".join(rg.get("mandarin", []))
    jon = ",".join(rg.get("japanese_on", []))
    mean = "; ".join(v for _, v in rec["meanings"][:2])
    return (f"{ch}\t{cp}\trad={rec['radical']}\tstrokes={rec['total_strokes']}"
            f"\tdok={dok}\tjahun={jahun}\tman={man}\tjon={jon}\tmean={mean}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cp", help="codepoint (U+XXXX or raw hex)")
    g.add_argument("--char", help="literal character")
    ap.add_argument("--db", default=str(DEFAULT_DB),
                    help=f"path to canonical_v2.sqlite (default: {DEFAULT_DB})")
    ap.add_argument("--oneline", action="store_true",
                    help="tab-separated one-line output")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable JSON output")
    args = ap.parse_args()

    if args.char:
        cp = char_to_cp(args.char)
    else:
        cp = normalize_cp(args.cp)

    con = sqlite3.connect(args.db)
    rec = lookup(con, cp)
    con.close()
    if rec is None:
        print(f"[lookup] {cp} not found in canonical DB")
        sys.exit(1)

    if args.json:
        print(json.dumps(rec, ensure_ascii=False, indent=2))
    elif args.oneline:
        print(fmt_oneline(rec))
    else:
        print(fmt_pretty(rec))


if __name__ == "__main__":
    main()
