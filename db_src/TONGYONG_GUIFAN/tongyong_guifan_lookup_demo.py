from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "tongyong_guifan_2013.csv"
BUCKET_RANGES = {
    "一级字表": "1..3500",
    "二级字表": "3501..6500",
    "三级字表": "6501..8105",
}


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalize_query(query: str) -> str:
    text = query.strip()
    if not text:
        raise ValueError("Please provide one character or a codepoint like U+5B78.")
    upper = text.upper()
    if upper.startswith("U+"):
        try:
            return chr(int(upper[2:], 16))
        except ValueError as exc:
            raise ValueError(f"Invalid codepoint: {query}") from exc
    if len(text) != 1:
        raise ValueError("Tongyong Guifan lookup expects exactly one character.")
    return text


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def print_meta(rows: list[dict[str, str]]) -> None:
    bucket_counts = Counter(row["bucket"] for row in rows)
    unique_chars = {row["character"] for row in rows if row.get("character")}

    print("Dataset metadata")
    print("  Official title   : 通用规范汉字表")
    print("  Nature           : PRC standard common-character tier table")
    print(f"  Source file      : {DEFAULT_CSV.name}")
    print(f"  Total rows       : {len(rows):,}")
    print(f"  Unique chars     : {len(unique_chars):,}")
    print("  Fields           : id, character, bucket")
    print("  Bucket ranges    :")
    for bucket, span in BUCKET_RANGES.items():
        print(f"    {bucket:<10} -> {span}")
    print("  Bucket counts    :")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"    {bucket:<10} : {count:,}")
    print()


def print_summary(query_char: str, rows: list[dict[str, str]]) -> None:
    bucket_counts = Counter(row["bucket"] for row in rows)
    match = next((row for row in rows if row["character"] == query_char), None)

    print("Tongyong Guifan 2013 Lookup")
    print(f"Query character    : {query_char}")
    print(f"Unicode codepoint  : U+{ord(query_char):04X}")
    print(f"Dataset rows       : {len(rows):,}")
    print()

    if match is None:
        print("Status             : Not present in Tongyong Guifan")
        return

    row_index = rows.index(match) + 1
    print("Status             : Present")
    print(f"Ordinal id         : {match['id']}")
    print(f"Bucket             : {match['bucket']}")
    print(f"CSV row position   : {row_index}")
    print()

    print("Bucket distribution")
    for bucket, count in sorted(bucket_counts.items()):
        print(f"  {bucket:<10} : {count:,}")
    print()

    print("Interpretation")
    if match["bucket"] == "一级字表":
        meaning = "Tier 1: core/common standard character"
    elif match["bucket"] == "二级字表":
        meaning = "Tier 2: additional common standard character"
    elif match["bucket"] == "三级字表":
        meaning = "Tier 3: standard but comparatively less common character"
    else:
        meaning = "Unknown bucket"
    print(f"  {meaning}")


def main() -> int:
    configure_stdout()

    parser = argparse.ArgumentParser(
        description="Look up one character in the Tongyong Guifan 2013 CSV."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="学",
        help="Single character or codepoint like U+5B66. Default: 学",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to tongyong_guifan_2013.csv (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--show-meta",
        action="store_true",
        help="Print dataset-level metadata described in the manual before the lookup result.",
    )
    args = parser.parse_args()

    try:
        query_char = normalize_query(args.query)
    except ValueError as exc:
        print(exc)
        return 1

    if not args.csv.exists():
        print(f"CSV file not found: {args.csv}")
        return 1

    rows = load_rows(args.csv)
    if args.show_meta:
        print_meta(rows)
    print_summary(query_char, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
