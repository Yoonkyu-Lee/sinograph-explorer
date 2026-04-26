from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_ROOT = SCRIPT_DIR.parent
DEFAULT_SQLITE = BUILD_ROOT / "out" / "sinograph_canonical_v1.sqlite"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalize_query_to_codepoint(query: str) -> tuple[str, str]:
    text = query.strip()
    if not text:
        raise ValueError("Please provide a character or a codepoint like U+5B78.")

    upper = text.upper()
    if upper.startswith("U+"):
        codepoint = upper
        try:
            character = chr(int(upper[2:], 16))
        except ValueError as exc:
            raise ValueError(f"Invalid codepoint: {query}") from exc
        return codepoint, character

    if len(text) != 1:
        raise ValueError("Please pass exactly one character or one codepoint like U+5B78.")

    return f"U+{ord(text):04X}", text


def load_record(conn: sqlite3.Connection, codepoint: str, character: str) -> dict[str, Any] | None:
    cur = conn.cursor()
    row = cur.execute(
        """
        SELECT data_json
        FROM characters
        WHERE codepoint = ?
           OR character = ?
        ORDER BY CASE WHEN codepoint = ? THEN 0 ELSE 1 END
        LIMIT 1
        """,
        (codepoint, character, codepoint),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def joined(values: list[str]) -> str:
    if not values:
        return "(none)"
    return ", ".join(values)


def format_codepoint_label(value: str | None) -> str:
    if not value:
        return "(none)"
    if value.startswith("U+"):
        try:
            return f"{chr(int(value[2:], 16))} {value}"
        except ValueError:
            return value
    return value


def format_codepoint_list(values: list[str]) -> str:
    if not values:
        return "(none)"
    return ", ".join(format_codepoint_label(value) for value in values)


def format_source_flags(flags: dict[str, bool]) -> str:
    present = [name for name, value in flags.items() if value]
    return ", ".join(present) if present else "(none)"


def print_summary(record: dict[str, Any]) -> None:
    core = record["core"]
    variants = record["variants"]
    supplementary = record.get("supplementary_variants", {})
    variant_graph = record.get("variant_graph", {})
    structure = record["structure"]
    media = record["media"]
    references = record["references"]

    print(f"Character          : {record['character']}")
    print(f"Unicode codepoint  : {record['codepoint']}")
    print(f"Sources present    : {format_source_flags(record['source_flags'])}")
    print()

    print("Core")
    print(f"  Radical          : {core['radical'] if core['radical'] is not None else '(none)'}")
    print(f"  Total strokes    : {core['total_strokes'] if core['total_strokes'] is not None else '(none)'}")
    print(f"  English          : {joined(core['definitions']['english'])}")
    print(f"  Korean expl.     : {joined(core['definitions']['korean_explanation'])}")
    print(f"  Korean hun       : {joined(core['definitions']['korean_hun'])}")
    print(f"  Mandarin         : {joined(core['readings']['mandarin'])}")
    print(f"  Cantonese        : {joined(core['readings']['cantonese'])}")
    print(f"  Korean hangul    : {joined(core['readings']['korean_hangul'])}")
    print(f"  Korean romanized : {joined(core['readings']['korean_romanized'])}")
    print(f"  Japanese on      : {joined(core['readings']['japanese_on'])}")
    print(f"  Japanese kun     : {joined(core['readings']['japanese_kun'])}")
    print(f"  Vietnamese       : {joined(core['readings']['vietnamese'])}")
    print()

    print("Canonical variants (Unihan backbone)")
    print(f"  Representative   : {format_codepoint_label(variants['representative_form'])}")
    print(f"  Family size      : {len(variants['family_members'])}")
    print(f"  Family members   : {format_codepoint_list(variants['family_members'])}")
    print(f"  Traditional      : {format_codepoint_list(variants['traditional'])}")
    print(f"  Simplified       : {format_codepoint_list(variants['simplified'])}")
    print(f"  Semantic         : {format_codepoint_list(variants['semantic'])}")
    print(f"  Spec. semantic   : {format_codepoint_list(variants['specialized_semantic'])}")
    print(f"  Z-variants       : {format_codepoint_list(variants['z_variants'])}")
    print(f"  Spoofing         : {format_codepoint_list(variants['spoofing'])}")
    print()

    print("Supplementary variants")
    print(f"  e-hanja yakja    : {format_codepoint_list(supplementary.get('ehanja_yakja', []))}")
    print(f"  e-hanja bonja    : {format_codepoint_list(supplementary.get('ehanja_bonja', []))}")
    print(f"  e-hanja simp CN  : {format_codepoint_list(supplementary.get('ehanja_simple_china', []))}")
    print(f"  e-hanja kanji    : {format_codepoint_list(supplementary.get('ehanja_kanji', []))}")
    print(f"  e-hanja dongja   : {format_codepoint_list(supplementary.get('ehanja_dongja', []))}")
    print(f"  e-hanja tongja   : {format_codepoint_list(supplementary.get('ehanja_tongja', []))}")
    print(f"  KANJIDIC2 resolv.: {format_codepoint_list(supplementary.get('kanjidic2_resolved', []))}")
    print()

    print("Variant graph views")
    print(f"  Canonical rep.   : {format_codepoint_label(variant_graph.get('canonical_representative_form'))}")
    print(f"  Canonical size   : {len(variant_graph.get('canonical_family_members', []))}")
    print(f"  Canonical family : {format_codepoint_list(variant_graph.get('canonical_family_members', []))}")
    print(f"  Enriched rep.    : {format_codepoint_label(variant_graph.get('enriched_representative_form'))}")
    print(f"  Enriched size    : {len(variant_graph.get('enriched_family_members', []))}")
    print(f"  Enriched family  : {format_codepoint_list(variant_graph.get('enriched_family_members', []))}")
    print()

    print("Structure")
    print(f"  Decomposition    : {structure['decomposition'] or '(none)'}")
    print(f"  Etymology type   : {structure['etymology_type'] or '(none)'}")
    print(f"  Etymology hint   : {structure['etymology_hint'] or '(none)'}")
    print(f"  Phonetic comp.   : {structure['phonetic_component'] or '(none)'}")
    print(f"  Semantic comp.   : {structure['semantic_component'] or '(none)'}")
    print()

    print("Media")
    print(f"  Stroke paths     : {len(media['stroke_svg_paths'])}")
    print(f"  Stroke medians   : {len(media['stroke_medians'])}")
    print()

    print("References")
    print(f"  Unihan refs      : {', '.join(sorted(references['unihan'].keys())) or '(none)'}")
    print(f"  KANJIDIC2 refs   : {', '.join(sorted(references['kanjidic2'].keys())) or '(none)'}")
    print(f"  e-hanja refs     : {', '.join(sorted(references['ehanja'].keys())) or '(none)'}")
    print()


def main() -> int:
    configure_stdout()

    parser = argparse.ArgumentParser(
        description="Look up one character in the Sinograph Canonical DB v1 SQLite artifact."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="斈",
        help="Single character or codepoint like U+5B78. Default: 斈",
    )
    parser.add_argument(
        "--sqlite",
        type=Path,
        default=DEFAULT_SQLITE,
        help=f"Path to sinograph_canonical_v1.sqlite (default: {DEFAULT_SQLITE})",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the full canonical JSON record after the summary.",
    )
    args = parser.parse_args()

    try:
        codepoint, character = normalize_query_to_codepoint(args.query)
    except ValueError as exc:
        print(exc)
        return 1

    if not args.sqlite.exists():
        print(f"SQLite file not found: {args.sqlite}")
        print("Run build_canonical_db.py first.")
        return 1

    conn = sqlite3.connect(args.sqlite)
    try:
        record = load_record(conn, codepoint, character)
    finally:
        conn.close()

    if record is None:
        print(f"Character not found in canonical DB: {args.query}")
        return 1

    print_summary(record)

    if args.raw:
        print("Raw canonical JSON")
        print("-" * 72)
        print(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
