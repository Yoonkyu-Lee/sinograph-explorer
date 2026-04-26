from __future__ import annotations

import argparse
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX = SCRIPT_DIR / "dict_revised_2015_20251229.xlsx"

NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
HEADER_TO_KEY = {
    "字詞名": "headword",
    "詞條別名": "entry_alias",
    "字數": "char_count",
    "字詞號": "entry_id",
    "部首字": "radical_char",
    "總筆畫數": "total_strokes",
    "部首外筆畫數": "non_radical_strokes",
    "多音排序": "polyphonic_order",
    "注音一式": "zhuyin",
    "變體類型 1:變 2:又音 3:語音 4:讀音": "variant_type",
    "變體注音": "variant_zhuyin",
    "漢語拼音": "pinyin",
    "變體漢語拼音": "variant_pinyin",
    "相似詞": "synonyms",
    "相反詞": "antonyms",
    "釋義": "definition",
    "多音參見訊息": "polyphonic_reference",
    "異體字": "variant_characters",
}
DISPLAY_ORDER = [
    ("headword", "Headword"),
    ("entry_alias", "Entry alias"),
    ("char_count", "Character count"),
    ("entry_id", "Entry id"),
    ("radical_char", "Radical"),
    ("total_strokes", "Total strokes"),
    ("non_radical_strokes", "Non-radical strokes"),
    ("polyphonic_order", "Polyphonic order"),
    ("zhuyin", "Zhuyin"),
    ("variant_type", "Variant type"),
    ("variant_zhuyin", "Variant zhuyin"),
    ("pinyin", "Pinyin"),
    ("variant_pinyin", "Variant pinyin"),
    ("synonyms", "Synonyms"),
    ("antonyms", "Antonyms"),
    ("definition", "Definition"),
    ("polyphonic_reference", "Polyphonic reference"),
    ("variant_characters", "Variant characters"),
]
WORKBOOK_FIELDS = list(HEADER_TO_KEY.keys())


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
        raise ValueError("MOE revised dict demo expects exactly one character.")
    return text


def column_index_from_ref(ref: str | None) -> int:
    if not ref:
        return 0
    letters = []
    for char in ref:
        if char.isalpha():
            letters.append(char)
        else:
            break
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return max(index - 1, 0)


def load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for si in root.findall("main:si", NS):
        text = "".join(node.text or "" for node in si.iterfind(".//main:t", NS))
        values.append(text)
    return values


def cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value_node = cell.find("main:v", NS)
    if value_node is None:
        return ""
    raw = value_node.text or ""
    if cell.get("t") == "s":
        return shared_strings[int(raw)]
    return raw


def iter_rows(xlsx_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings = load_shared_strings(zf)
        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows = sheet_root.findall(".//main:sheetData/main:row", NS)

        header: list[str] = []
        parsed_rows: list[dict[str, str]] = []

        for row_idx, row in enumerate(rows):
            values: list[str] = [""] * 18
            for cell in row.findall("main:c", NS):
                col_idx = column_index_from_ref(cell.get("r"))
                if col_idx >= len(values):
                    values.extend([""] * (col_idx - len(values) + 1))
                values[col_idx] = cell_value(cell, shared_strings).strip()

            if row_idx == 0:
                header = values
                continue

            if len(values) < len(header):
                values.extend([""] * (len(header) - len(values)))
            parsed_rows.append(dict(zip(header, values)))

    return header, parsed_rows


def normalize_row(row: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for header, key in HEADER_TO_KEY.items():
        normalized[key] = row.get(header, "").strip()
    return normalized


def print_meta(raw_rows: list[dict[str, str]]) -> None:
    single_char_rows = 0
    single_char_unique: set[str] = set()
    for row in raw_rows:
        headword = row.get("字詞名", "").strip()
        char_count = row.get("字數", "").strip()
        if headword and char_count == "1" and len(headword) == 1:
            single_char_rows += 1
            single_char_unique.add(headword)

    print("Dataset metadata")
    print("  Official title   : 重編國語辭典修訂本")
    print("  Nature           : large lexical workbook mixing single-character and lexical entries")
    print(f"  Source file      : {DEFAULT_XLSX.name}")
    print("  License          : CC BY-ND 3.0 Taiwan")
    print(f"  Workbook rows    : {len(raw_rows):,}")
    print(f"  Single-char rows : {single_char_rows:,}")
    print(f"  Single-char uniq.: {len(single_char_unique):,}")
    print(f"  Column count     : {len(WORKBOOK_FIELDS)}")
    print("  Workbook fields  :")
    for field in WORKBOOK_FIELDS:
        print(f"    - {field}")
    print()


def print_row_block(index: int, row: dict[str, str]) -> None:
    print(f"Entry {index}")
    for key, label in DISPLAY_ORDER:
        value = row.get(key, "")
        print(f"  {label:<20}: {value or '(none)'}")
    print()


def main() -> int:
    configure_stdout()

    parser = argparse.ArgumentParser(
        description="Look up one character in the MOE Revised Dictionary workbook."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="學",
        help="Single character or codepoint like U+5B78. Default: 學",
    )
    parser.add_argument(
        "--xlsx",
        type=Path,
        default=DEFAULT_XLSX,
        help=f"Path to dict_revised_2015_20251229.xlsx (default: {DEFAULT_XLSX})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=8,
        help="Maximum matching rows to print. Default: 8",
    )
    parser.add_argument(
        "--show-meta",
        action="store_true",
        help="Print workbook-level metadata described in the manual before the lookup result.",
    )
    args = parser.parse_args()

    try:
        query_char = normalize_query(args.query)
    except ValueError as exc:
        print(exc)
        return 1

    if not args.xlsx.exists():
        print(f"Workbook not found: {args.xlsx}")
        return 1

    _, raw_rows = iter_rows(args.xlsx)
    if args.show_meta:
        print_meta(raw_rows)

    matches = [normalize_row(row) for row in raw_rows if row.get("字詞名", "").strip() == query_char]

    print("MOE Revised Dictionary Lookup")
    print(f"Query character    : {query_char}")
    print(f"Unicode codepoint  : U+{ord(query_char):04X}")
    print(f"Workbook rows      : {len(raw_rows):,}")
    print(f"Matching entries   : {len(matches)}")
    print()

    if not matches:
        print("Status             : No matching single-headword rows found")
        return 0

    print("Notes")
    print("  This workbook mixes single-character and lexical entries.")
    print("  The demo prints rows where 字詞名 exactly matches the query character.")
    print()

    for idx, row in enumerate(matches[: max(args.max_rows, 1)], start=1):
        print_row_block(idx, row)

    if len(matches) > args.max_rows:
        remaining = len(matches) - args.max_rows
        print(f"... {remaining} additional matching row(s) not shown")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
