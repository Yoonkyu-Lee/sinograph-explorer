from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path


READING_FIELDS = {
    "mandarin": ("kMandarin",),
    "cantonese": ("kCantonese",),
    # Japanese는 kJapanese / kJapaneseOn / kJapaneseKun 중 하나라도 있으면 "있음"으로 집계
    "japanese": ("kJapanese", "kJapaneseOn", "kJapaneseKun"),
    "korean": ("kKorean",),
}

DEFAULT_UNIHAN_DIR = Path(__file__).resolve().parent / "Unihan_txt"


def iter_unihan_txt_files(unihan_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in unihan_dir.iterdir()
        if path.is_file() and path.name.startswith("Unihan_") and path.suffix == ".txt"
    )


def load_unihan_fields(unihan_dir: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = defaultdict(dict)

    for path in iter_unihan_txt_files(unihan_dir):
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    codepoint, field, value = line.split("\t", 2)
                except ValueError:
                    continue

                records[codepoint][field] = value.strip()

    return records


def has_any_field(entry: dict[str, str], field_names: tuple[str, ...]) -> bool:
    return any(entry.get(field_name, "").strip() for field_name in field_names)


def format_ratio(count: int, total: int) -> str:
    ratio = (count / total * 100.0) if total else 0.0
    return f"{count:,} / {total:,} ({ratio:6.2f}%)"


def combo_label(flags: dict[str, bool]) -> str:
    order = ("mandarin", "cantonese", "japanese", "korean")
    active = [name[0].upper() for name in order if flags[name]]
    return "".join(active) if active else "(none)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan all unpacked Unihan_*.txt files and report reading-field coverage "
            "for Mandarin, Cantonese, Japanese (combined), and Korean."
        )
    )
    parser.add_argument(
        "--unihan-dir",
        type=Path,
        default=DEFAULT_UNIHAN_DIR,
        help=f"Path to the unpacked Unihan directory (default: {DEFAULT_UNIHAN_DIR})",
    )
    parser.add_argument(
        "--top-combos",
        type=int,
        default=16,
        help="How many reading-presence combinations to print (default: 16)",
    )
    args = parser.parse_args()

    unihan_dir = args.unihan_dir
    if not unihan_dir.exists():
        raise SystemExit(f"Unihan directory not found: {unihan_dir}")

    records = load_unihan_fields(unihan_dir)
    total = len(records)

    print("=== Unihan Reading Coverage Audit ===")
    print(f"Unihan dir            : {unihan_dir.resolve()}")
    print(f"Total unique codepoints: {total:,}")
    print()

    presence_counts = Counter()
    combo_counts = Counter()
    conditional_counts = Counter()

    for entry in records.values():
        flags = {
            name: has_any_field(entry, field_names)
            for name, field_names in READING_FIELDS.items()
        }

        for name, present in flags.items():
            if present:
                presence_counts[name] += 1

        combo_counts[combo_label(flags)] += 1
        if flags["japanese"] and not flags["korean"]:
            conditional_counts["japanese_present_korean_absent"] += 1
        if flags["korean"] and not flags["japanese"]:
            conditional_counts["korean_present_japanese_absent"] += 1
        if flags["japanese"] and not flags["mandarin"] and not flags["cantonese"]:
            conditional_counts["japanese_only_vs_chinese"] += 1
        if (flags["mandarin"] or flags["cantonese"]) and not flags["japanese"] and not flags["korean"]:
            conditional_counts["chinese_only"] += 1

    print("[Field presence across all Unihan codepoints]")
    for name in ("mandarin", "cantonese", "japanese", "korean"):
        present = presence_counts[name]
        absent = total - present
        print(f"- {name.capitalize():<10} present: {format_ratio(present, total)}")
        print(f"  {name.capitalize():<10} absent : {format_ratio(absent, total)}")
    print()

    print("[Cross-field highlights]")
    print(
        f"- Japanese present & Korean absent : "
        f"{format_ratio(conditional_counts['japanese_present_korean_absent'], total)}"
    )
    print(
        f"- Korean present & Japanese absent : "
        f"{format_ratio(conditional_counts['korean_present_japanese_absent'], total)}"
    )
    print(
        f"- Japanese present (any form)      : {format_ratio(presence_counts['japanese'], total)}"
    )
    print(
        f"- Korean present                   : {format_ratio(presence_counts['korean'], total)}"
    )
    print(
        f"- Japanese present, no Chinese     : {format_ratio(conditional_counts['japanese_only_vs_chinese'], total)}"
    )
    print(
        f"- Chinese present, no JP/KR        : {format_ratio(conditional_counts['chinese_only'], total)}"
    )
    print()

    print("[Reading presence combinations]")
    print("Legend: M=Mandarin, C=Cantonese, J=Japanese(any of kJapanese/kJapaneseOn/kJapaneseKun), K=Korean")
    for label, count in combo_counts.most_common(args.top_combos):
        print(f"- {label:<8} {format_ratio(count, total)}")


if __name__ == "__main__":
    main()
