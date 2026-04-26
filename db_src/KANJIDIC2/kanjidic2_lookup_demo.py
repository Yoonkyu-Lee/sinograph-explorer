from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_XML = Path(__file__).resolve().parent / "KANJIDIC2_xml" / "kanjidic2.xml"


def configure_stdout() -> None:
    """Windows 터미널에서도 CJK 문자가 최대한 깨지지 않게 UTF-8을 강제한다."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def find_character_entry(xml_path: Path, target: str) -> ET.Element | None:
    """KANJIDIC2 XML을 순회하면서 literal이 target인 character 엔트리를 찾는다."""
    context = ET.iterparse(xml_path, events=("end",))
    for _, elem in context:
        if elem.tag == "character":
            literal = elem.findtext("literal")
            if literal == target:
                return elem
            elem.clear()
    return None


def collect_values(parent: ET.Element | None, child_tag: str, attr_name: str | None = None) -> list[str]:
    """같은 이름의 하위 태그를 모두 문자열 목록으로 모은다."""
    if parent is None:
        return []

    values: list[str] = []
    for child in parent.findall(child_tag):
        text = (child.text or "").strip()
        if attr_name is None:
            if text:
                values.append(text)
                continue
            values.append("(empty)")
            continue

        attr_value = child.get(attr_name, "")
        if text:
            values.append(f"{attr_value}: {text}" if attr_value else text)
        else:
            values.append(attr_value if attr_value else "(empty)")
    return values


def print_list_section(title: str, values: list[str]) -> None:
    print(title)
    if not values:
        print("  (none)")
        return
    for value in values:
        print(f"  - {value}")


def print_character_summary(entry: ET.Element) -> None:
    literal = entry.findtext("literal", default="(none)")
    print(f"Character          : {literal}")
    print(f"Unicode codepoint  : U+{ord(literal):04X}" if literal != "(none)" else "Unicode codepoint  : (none)")
    print()

    codepoint = entry.find("codepoint")
    radical = entry.find("radical")
    misc = entry.find("misc")
    dic_number = entry.find("dic_number")
    query_code = entry.find("query_code")
    reading_meaning = entry.find("reading_meaning")
    rmgroup = reading_meaning.find("rmgroup") if reading_meaning is not None else None

    print_list_section("Codepoints", collect_values(codepoint, "cp_value", "cp_type"))
    print()
    print_list_section("Radicals", collect_values(radical, "rad_value", "rad_type"))
    print()

    misc_values: list[str] = []
    for tag in ("grade", "stroke_count", "freq", "jlpt", "rad_name"):
        misc_values.extend([f"{tag}: {value}" for value in collect_values(misc, tag)])
    misc_values.extend([f"variant ({value})" for value in collect_values(misc, "variant", "var_type")])
    print_list_section("Misc", misc_values)
    print()

    print_list_section("Dictionary references", collect_values(dic_number, "dic_ref", "dr_type"))
    print()
    print_list_section("Query codes", collect_values(query_code, "q_code", "qc_type"))
    print()

    reading_values: list[str] = []
    meaning_values: list[str] = []
    if rmgroup is not None:
        reading_values = collect_values(rmgroup, "reading", "r_type")
        meaning_values = collect_values(rmgroup, "meaning", "m_lang")

    print_list_section("Readings", reading_values)
    print()
    print_list_section("Meanings", meaning_values)
    print()

    print_list_section("Nanori", collect_values(reading_meaning, "nanori"))
    print()


def print_raw_xml(entry: ET.Element) -> None:
    print("Raw XML subtree")
    print("-" * 72)
    xml_text = ET.tostring(entry, encoding="unicode")
    print(xml_text)


def main() -> int:
    configure_stdout()

    parser = argparse.ArgumentParser(
        description="Look up one character in KANJIDIC2 and print all available data."
    )
    parser.add_argument(
        "character",
        nargs="?",
        default="斈",
        help="Single character to look up. Default: 斈",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help=f"Path to kanjidic2.xml (default: {DEFAULT_XML})",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Hide the raw XML subtree and print only the summarized fields.",
    )
    args = parser.parse_args()

    target = args.character.strip()
    if not target:
        print("Please provide one character.")
        return 1

    target = target[0]
    if not args.xml.exists():
        print(f"KANJIDIC2 XML not found: {args.xml}")
        return 1

    entry = find_character_entry(args.xml, target)
    if entry is None:
        print(f"Character not found in KANJIDIC2: {target}")
        return 1

    print_character_summary(entry)

    if not args.no_raw:
        print_raw_xml(entry)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
