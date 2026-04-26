from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BUILD_ROOT.parent
DB_SRC_ROOT = REPO_ROOT / "db_src"

UNIHAN_DIR = DB_SRC_ROOT / "Unihan" / "Unihan_txt"
EHANJA_CSV_DIR = DB_SRC_ROOT / "e-hanja" / "ejajeon_csv"
KANJIDIC2_XML = DB_SRC_ROOT / "KANJIDIC2" / "KANJIDIC2_xml" / "kanjidic2.xml"
MAKEMEAHANZI_DICT = DB_SRC_ROOT / "MAKEMEAHANZI" / "dictionary.txt"
MAKEMEAHANZI_GRAPHICS = DB_SRC_ROOT / "MAKEMEAHANZI" / "graphics.txt"

STAGING_DIR = BUILD_ROOT / "staging"
OUT_DIR = BUILD_ROOT / "out"

SOURCE_NAMES = ("unihan", "ehanja", "kanjidic2", "makemeahanzi")

READING_KEYS = (
    "mandarin",
    "cantonese",
    "korean_hangul",
    "korean_romanized",
    "japanese_on",
    "japanese_kun",
    "vietnamese",
)

VARIANT_KEYS = (
    "traditional",
    "simplified",
    "semantic",
    "specialized_semantic",
    "z_variants",
    "spoofing",
)

SUPPLEMENTARY_VARIANT_KEYS = (
    "ehanja_yakja",
    "ehanja_bonja",
    "ehanja_simple_china",
    "ehanja_kanji",
    "ehanja_dongja",
    "ehanja_tongja",
    "kanjidic2_resolved",
)

UNIHAN_VARIANT_FIELD_MAP = {
    "kTraditionalVariant": "traditional",
    "kSimplifiedVariant": "simplified",
    "kSemanticVariant": "semantic",
    "kSpecializedSemanticVariant": "specialized_semantic",
    "kZVariant": "z_variants",
    "kSpoofingVariant": "spoofing",
}

UNIHAN_READING_FIELD_MAP = {
    "kMandarin": "mandarin",
    "kCantonese": "cantonese",
    "kKorean": "korean_hangul",
    "kJapaneseOn": "japanese_on",
    "kJapaneseKun": "japanese_kun",
    "kVietnamese": "vietnamese",
}

IDC_OPERATORS = {
    "⿰",
    "⿱",
    "⿲",
    "⿳",
    "⿴",
    "⿵",
    "⿶",
    "⿷",
    "⿸",
    "⿹",
    "⿺",
    "⿻",
}

VARIANT_CODEPOINT_RE = re.compile(r"U\+[0-9A-F]{4,6}")
KANJIDIC2_RESOLVABLE_VARIANT_TYPES = {"jis208", "jis212", "jis213", "ucs"}


@dataclass
class SourceRecord:
    source_name: str
    character: str
    codepoint: str
    radical: str | None
    total_strokes: int | None
    readings: dict[str, list[str]]
    definitions: dict[str, list[str]]
    variants: dict[str, list[str]]
    supplementary_variants: dict[str, list[str]]
    structure: dict[str, Any]
    media: dict[str, Any]
    references: dict[str, Any]
    source_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "character": self.character,
            "codepoint": self.codepoint,
            "radical": self.radical,
            "total_strokes": self.total_strokes,
            "readings": self.readings,
            "definitions": self.definitions,
            "variants": self.variants,
            "supplementary_variants": self.supplementary_variants,
            "structure": self.structure,
            "media": self.media,
            "references": self.references,
            "source_payload": self.source_payload,
        }


def make_empty_readings() -> dict[str, list[str]]:
    return {key: [] for key in READING_KEYS}


def make_empty_variants() -> dict[str, list[str]]:
    return {key: [] for key in VARIANT_KEYS}


def make_empty_supplementary_variants() -> dict[str, list[str]]:
    return {key: [] for key in SUPPLEMENTARY_VARIANT_KEYS}


def unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def ensure_single_character(character: str) -> str | None:
    text = (character or "").strip()
    if len(text) != 1:
        return None
    return text


def codepoint_from_char(character: str) -> str:
    return f"U+{ord(character):04X}"


def codepoint_to_char(codepoint: str) -> str:
    return chr(int(codepoint[2:], 16))


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_unihan_radical(rs_unicode_values: list[str]) -> str | None:
    for value in rs_unicode_values:
        text = value.strip()
        if not text:
            continue
        first = text.split()[0]
        radical = first.split(".")[0].split("'")[0]
        if radical:
            return radical
    return None


def extract_unihan_variant_codepoints(values: list[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.extend(VARIANT_CODEPOINT_RE.findall(value))
    return unique_preserve(output)


def parse_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def parse_character_targets(raw: str) -> list[str]:
    output: list[str] = []
    for piece in (raw or "").split(","):
        text = piece.strip()
        if not text:
            continue
        single = ensure_single_character(text)
        if single is None:
            continue
        output.append(codepoint_from_char(single))
    return unique_preserve(output)


def build_unihan_records() -> list[SourceRecord]:
    per_codepoint: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for path in sorted(UNIHAN_DIR.glob("Unihan_*.txt")):
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                codepoint, field, value = line.split("\t", 2)
                per_codepoint[codepoint][field].append(value)

    records: list[SourceRecord] = []
    for codepoint in sorted(per_codepoint):
        try:
            character = chr(int(codepoint[2:], 16))
        except ValueError:
            continue

        field_map = per_codepoint[codepoint]
        readings = make_empty_readings()
        for field_name, canonical_key in UNIHAN_READING_FIELD_MAP.items():
            readings[canonical_key] = unique_preserve(field_map.get(field_name, []))

        definitions = {
            "english": unique_preserve(field_map.get("kDefinition", [])),
            "korean_explanation": [],
            "korean_hun": [],
        }

        variants = make_empty_variants()
        for field_name, canonical_key in UNIHAN_VARIANT_FIELD_MAP.items():
            variants[canonical_key] = extract_unihan_variant_codepoints(field_map.get(field_name, []))

        references = {
            "rs_unicode": unique_preserve(field_map.get("kRSUnicode", [])),
            "kangxi": unique_preserve(field_map.get("kKangXi", [])),
            "irg_kangxi": unique_preserve(field_map.get("kIRGKangXi", [])),
            "hanyu": unique_preserve(field_map.get("kHanYu", [])),
            "unihan_core_2020": unique_preserve(field_map.get("kUnihanCore2020", [])),
        }

        strokes = [parse_int(v) for v in field_map.get("kTotalStrokes", [])]
        total_strokes = next((v for v in strokes if v is not None), None)

        record = SourceRecord(
            source_name="unihan",
            character=character,
            codepoint=codepoint,
            radical=parse_unihan_radical(field_map.get("kRSUnicode", [])),
            total_strokes=total_strokes,
            readings=readings,
            definitions=definitions,
            variants=variants,
            supplementary_variants=make_empty_supplementary_variants(),
            structure={},
            media={},
            references=references,
            source_payload={key: unique_preserve(values) for key, values in field_map.items()},
        )
        records.append(record)

    return records


def load_csv_by_key(path: Path, key_field: str, many: bool = False) -> dict[str, Any]:
    output: dict[str, Any] = defaultdict(list) if many else {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get(key_field) or "").strip()
            if not key:
                continue
            if many:
                output[key].append(row)
            else:
                output[key] = row
    return dict(output)


def build_ehanja_records() -> list[SourceRecord]:
    hschool_path = EHANJA_CSV_DIR / "hSchool.csv"
    if not hschool_path.exists():
        return []

    school_com = load_csv_by_key(EHANJA_CSV_DIR / "hSchoolCom.csv", "hanja")
    current = load_csv_by_key(EHANJA_CSV_DIR / "hCur.csv", "hanja")
    theory = load_csv_by_key(EHANJA_CSV_DIR / "hTheory.csv", "hanja")
    roots = load_csv_by_key(EHANJA_CSV_DIR / "hRoot.csv", "hanja", many=True)
    laws = load_csv_by_key(EHANJA_CSV_DIR / "hLaw.csv", "hanja", many=True)
    lengths = load_csv_by_key(EHANJA_CSV_DIR / "hLength.csv", "hanja")

    records: list[SourceRecord] = []
    with hschool_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            character = ensure_single_character(row.get("hanja", ""))
            if not character:
                continue

            root_rows = roots.get(character, [])
            hroot_meanings = unique_preserve([item.get("rMeaning", "") for item in root_rows])
            hroot_sounds = unique_preserve([item.get("rSnd", "") for item in root_rows])
            hschool_sound = unique_preserve([row.get("hSnd", "")])

            readings = make_empty_readings()
            readings["korean_hangul"] = unique_preserve(hschool_sound + hroot_sounds)

            definitions = {
                "english": unique_preserve([row.get("english", "")]),
                "korean_explanation": hroot_meanings,
                "korean_hun": unique_preserve([row.get("hRead", "")]),
            }

            school_com_row = school_com.get(character) or {}
            supplementary_variants = make_empty_supplementary_variants()
            supplementary_variants["ehanja_yakja"] = parse_character_targets(school_com_row.get("yakja", ""))
            supplementary_variants["ehanja_bonja"] = parse_character_targets(school_com_row.get("bonja", ""))
            supplementary_variants["ehanja_simple_china"] = parse_character_targets(school_com_row.get("simpleChina", ""))
            supplementary_variants["ehanja_kanji"] = parse_character_targets(school_com_row.get("kanji", ""))
            supplementary_variants["ehanja_dongja"] = parse_character_targets(school_com_row.get("dongja", ""))
            supplementary_variants["ehanja_tongja"] = parse_character_targets(school_com_row.get("tongja", ""))

            references = {
                "hschool_id": row.get("_id"),
                "busu_id": row.get("busu_Id"),
                "busu2_id": row.get("busu2_Id"),
                "hshape": row.get("hShape"),
                "china_english": row.get("chinaEng"),
                "current": current.get(character),
                "theory": theory.get(character),
                "law": laws.get(character, []),
                "length": lengths.get(character),
                "school_com": school_com.get(character),
            }

            source_payload = {
                "hSchool": row,
                "hSchoolCom": school_com.get(character),
                "hCur": current.get(character),
                "hTheory": theory.get(character),
                "hRoot": root_rows,
                "hLaw": laws.get(character, []),
                "hLength": lengths.get(character),
            }

            record = SourceRecord(
                source_name="ehanja",
                character=character,
                codepoint=codepoint_from_char(character),
                radical=(row.get("busu_Id") or "").strip() or None,
                total_strokes=parse_int(row.get("hTotal")),
                readings=readings,
                definitions=definitions,
                variants=make_empty_variants(),
                supplementary_variants=supplementary_variants,
                structure={},
                media={},
                references=references,
                source_payload=source_payload,
            )
            records.append(record)

    return records


def build_kanjidic2_records() -> list[SourceRecord]:
    parsed_rows: list[dict[str, Any]] = []
    cp_value_maps: dict[str, dict[str, str]] = defaultdict(dict)

    context = ET.iterparse(KANJIDIC2_XML, events=("end",))
    for _, elem in context:
        if elem.tag != "character":
            continue

        character = ensure_single_character(elem.findtext("literal", ""))
        if not character:
            elem.clear()
            continue

        misc = elem.find("misc")
        rm = elem.find("reading_meaning")
        rmgroup = rm.find("rmgroup") if rm is not None else None
        dic_number = elem.find("dic_number")
        query_code = elem.find("query_code")

        readings = make_empty_readings()
        meanings: list[str] = []
        nanori: list[str] = []
        if rmgroup is not None:
            for reading in rmgroup.findall("reading"):
                text = (reading.text or "").strip()
                if not text:
                    continue
                r_type = reading.get("r_type")
                if r_type == "ja_on":
                    readings["japanese_on"].append(text)
                elif r_type == "ja_kun":
                    readings["japanese_kun"].append(text)
                elif r_type == "pinyin":
                    readings["mandarin"].append(text)
                elif r_type == "korean_h":
                    readings["korean_hangul"].append(text)
                elif r_type == "korean_r":
                    readings["korean_romanized"].append(text)
                elif r_type == "vietnam":
                    readings["vietnamese"].append(text)

            for meaning in rmgroup.findall("meaning"):
                lang = meaning.get("m_lang")
                text = (meaning.text or "").strip()
                if not text:
                    continue
                if lang in (None, "", "en"):
                    meanings.append(text)

        if rm is not None:
            nanori = unique_preserve([(node.text or "").strip() for node in rm.findall("nanori")])

        radical = None
        for rad in elem.findall("radical/rad_value"):
            if rad.get("rad_type") == "classical":
                radical = (rad.text or "").strip() or None
                if radical:
                    break

        stroke_counts = [parse_int((node.text or "").strip()) for node in elem.findall("misc/stroke_count")]
        total_strokes = next((value for value in stroke_counts if value is not None), None)

        codepoint_refs: list[dict[str, str]] = []
        for node in elem.findall("codepoint/cp_value"):
            cp_type = node.get("cp_type", "")
            value = (node.text or "").strip()
            if cp_type and value:
                cp_value_maps[cp_type][value] = character
                codepoint_refs.append({"cp_type": cp_type, "value": value})

        variant_refs: list[dict[str, str]] = []
        if misc is not None:
            for node in misc.findall("variant"):
                value = (node.text or "").strip()
                if not value:
                    continue
                variant_refs.append({
                    "var_type": node.get("var_type", ""),
                    "value": value,
                })

        references = {
            "dic_number": {},
            "query_code": [],
            "variant_refs": variant_refs,
            "nanori": nanori,
            "codepoint_refs": codepoint_refs,
        }
        if dic_number is not None:
            for node in dic_number.findall("dic_ref"):
                dr_type = node.get("dr_type", "")
                references["dic_number"].setdefault(dr_type, []).append((node.text or "").strip())
        if query_code is not None:
            for node in query_code.findall("q_code"):
                references["query_code"].append({
                    "qc_type": node.get("qc_type", ""),
                    "skip_misclass": node.get("skip_misclass", ""),
                    "value": (node.text or "").strip(),
                })

        parsed_rows.append(
            {
                "character": character,
                "codepoint": codepoint_from_char(character),
                "radical": radical,
                "total_strokes": total_strokes,
                "readings": {key: unique_preserve(values) for key, values in readings.items()},
                "definitions": {
                    "english": unique_preserve(meanings),
                    "korean_explanation": [],
                    "korean_hun": [],
                },
                "references": references,
                "variant_refs": variant_refs,
            }
        )
        elem.clear()

    records: list[SourceRecord] = []
    for row in parsed_rows:
        resolved_targets: list[str] = []
        resolved_variant_refs: list[dict[str, str]] = []
        unresolved_variant_refs: list[dict[str, str]] = []
        for ref in row["variant_refs"]:
            var_type = ref["var_type"]
            value = ref["value"]
            if var_type not in KANJIDIC2_RESOLVABLE_VARIANT_TYPES:
                unresolved_variant_refs.append(ref)
                continue
            target_char = cp_value_maps.get(var_type, {}).get(value)
            if target_char is None:
                unresolved_variant_refs.append(ref)
                continue
            target_cp = codepoint_from_char(target_char)
            resolved_targets.append(target_cp)
            resolved_variant_refs.append({
                "var_type": var_type,
                "value": value,
                "target_character": target_char,
                "target_codepoint": target_cp,
            })

        supplementary_variants = make_empty_supplementary_variants()
        supplementary_variants["kanjidic2_resolved"] = unique_preserve(resolved_targets)
        row["references"]["resolved_variant_refs"] = resolved_variant_refs
        row["references"]["unresolved_variant_refs"] = unresolved_variant_refs

        records.append(
            SourceRecord(
                source_name="kanjidic2",
                character=row["character"],
                codepoint=row["codepoint"],
                radical=row["radical"],
                total_strokes=row["total_strokes"],
                readings=row["readings"],
                definitions=row["definitions"],
                variants=make_empty_variants(),
                supplementary_variants=supplementary_variants,
                structure={},
                media={},
                references=row["references"],
                source_payload={
                    "variant_refs": row["variant_refs"],
                    "resolved_variant_refs": resolved_variant_refs,
                    "unresolved_variant_refs": unresolved_variant_refs,
                    "dic_number": row["references"]["dic_number"],
                    "query_code": row["references"]["query_code"],
                    "nanori": row["references"]["nanori"],
                    "codepoint_refs": row["references"]["codepoint_refs"],
                },
            )
        )

    return records


def build_makemeahanzi_records() -> list[SourceRecord]:
    dictionary_rows = {
        row["character"]: row
        for row in parse_json_lines(MAKEMEAHANZI_DICT)
        if ensure_single_character(row.get("character", "")) is not None
    }
    graphics_rows = {
        row["character"]: row
        for row in parse_json_lines(MAKEMEAHANZI_GRAPHICS)
        if ensure_single_character(row.get("character", "")) is not None
    }

    records: list[SourceRecord] = []
    for character in sorted(set(dictionary_rows) | set(graphics_rows)):
        drow = dictionary_rows.get(character, {})
        grow = graphics_rows.get(character, {})
        etymology = drow.get("etymology") or {}

        readings = make_empty_readings()
        readings["mandarin"] = unique_preserve(drow.get("pinyin", []) or [])

        definitions = {
            "english": unique_preserve([drow.get("definition", "")]),
            "korean_explanation": [],
            "korean_hun": [],
        }

        records.append(
            SourceRecord(
                source_name="makemeahanzi",
                character=character,
                codepoint=codepoint_from_char(character),
                radical=(drow.get("radical") or "").strip() or None,
                total_strokes=len(grow.get("strokes", []) or []) or None,
                readings=readings,
                definitions=definitions,
                variants=make_empty_variants(),
                supplementary_variants=make_empty_supplementary_variants(),
                structure={
                    "decomposition": drow.get("decomposition"),
                    "matches": drow.get("matches"),
                    "etymology_type": etymology.get("type"),
                    "etymology_hint": etymology.get("hint"),
                    "phonetic_component": etymology.get("phonetic"),
                    "semantic_component": etymology.get("semantic"),
                },
                media={
                    "stroke_svg_paths": grow.get("strokes", []) or [],
                    "stroke_medians": grow.get("medians", []) or [],
                },
                references={},
                source_payload={
                    "dictionary": drow,
                    "graphics": grow,
                },
            )
        )

    return records


def build_source_adapters() -> dict[str, list[SourceRecord]]:
    return {
        "unihan": build_unihan_records(),
        "ehanja": build_ehanja_records(),
        "kanjidic2": build_kanjidic2_records(),
        "makemeahanzi": build_makemeahanzi_records(),
    }


def source_presence_flags(source_map: dict[str, SourceRecord]) -> dict[str, bool]:
    return {source: source in source_map for source in SOURCE_NAMES}


def merge_source_records(source_records: dict[str, list[SourceRecord]]) -> dict[str, dict[str, SourceRecord]]:
    merged: dict[str, dict[str, SourceRecord]] = defaultdict(dict)
    for source_name, records in source_records.items():
        for record in records:
            key = record.codepoint or codepoint_from_char(record.character)
            merged[key][source_name] = record
    return dict(merged)


def build_graph_components(
    adjacency: dict[str, set[str]],
    merged_by_codepoint: dict[str, dict[str, SourceRecord]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    incoming_traditional = Counter()
    outgoing_simplified = Counter()

    for edge in adjacency.get("__canonical_edge_metadata__", []):
        if edge["relation"] == "traditional":
            incoming_traditional[edge["target_codepoint"]] += 1
        if edge["relation"] == "simplified":
            outgoing_simplified[edge["source_codepoint"]] += 1

    components: dict[str, list[str]] = {}
    representative: dict[str, str] = {}
    visited: set[str] = set()

    def metadata_score(cp: str) -> int:
        source_map = merged_by_codepoint.get(cp, {})
        score = len(source_map)
        for record in source_map.values():
            score += sum(1 for values in record.readings.values() if values)
            score += sum(1 for values in record.definitions.values() if values)
            if record.radical:
                score += 1
            if record.total_strokes is not None:
                score += 1
            if record.structure.get("decomposition"):
                score += 1
            if record.media.get("stroke_svg_paths"):
                score += 1
        return score

    def representative_sort_key(cp: str) -> tuple[int, int, str]:
        traditionality = incoming_traditional[cp] + outgoing_simplified[cp]
        return (-traditionality, -metadata_score(cp), cp)

    for start in sorted(cp for cp in adjacency if cp != "__canonical_edge_metadata__"):
        if start in visited:
            continue
        queue = deque([start])
        component: list[str] = []
        visited.add(start)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        component_sorted = sorted(component)
        rep = sorted(component_sorted, key=representative_sort_key)[0]
        for cp in component_sorted:
            components[cp] = component_sorted
            representative[cp] = rep

    return representative, components


def build_variant_views(
    merged_by_codepoint: dict[str, dict[str, SourceRecord]]
) -> tuple[list[dict[str, str]], dict[str, str], dict[str, list[str]], dict[str, str], dict[str, list[str]]]:
    canonical_adjacency: dict[str, set[str]] = defaultdict(set)
    enriched_adjacency: dict[str, set[str]] = defaultdict(set)
    canonical_edge_metadata: list[dict[str, str]] = []
    all_edges: list[dict[str, str]] = []

    def add_edge(
        *,
        source_name: str,
        relation_scope: str,
        relation: str,
        source_codepoint: str,
        target_codepoint: str,
        include_in_canonical: bool,
    ) -> None:
        if not target_codepoint or source_codepoint == target_codepoint:
            return
        edge = {
            "source_name": source_name,
            "relation_scope": relation_scope,
            "relation": relation,
            "source_codepoint": source_codepoint,
            "source_character": codepoint_to_char(source_codepoint),
            "target_codepoint": target_codepoint,
            "target_character": codepoint_to_char(target_codepoint),
        }
        all_edges.append(edge)
        enriched_adjacency[source_codepoint].add(target_codepoint)
        enriched_adjacency[target_codepoint].add(source_codepoint)
        if include_in_canonical:
            canonical_edge_metadata.append(edge)
            canonical_adjacency[source_codepoint].add(target_codepoint)
            canonical_adjacency[target_codepoint].add(source_codepoint)

    for codepoint, source_map in merged_by_codepoint.items():
        for source_name in SOURCE_NAMES:
            record = source_map.get(source_name)
            if record is None:
                continue
            if source_name == "unihan":
                for relation_key, targets in record.variants.items():
                    for target in targets:
                        add_edge(
                            source_name="unihan",
                            relation_scope="canonical",
                            relation=relation_key,
                            source_codepoint=codepoint,
                            target_codepoint=target,
                            include_in_canonical=True,
                        )
            else:
                for relation_key, targets in record.supplementary_variants.items():
                    for target in targets:
                        add_edge(
                            source_name=source_name,
                            relation_scope="supplementary",
                            relation=relation_key,
                            source_codepoint=codepoint,
                            target_codepoint=target,
                            include_in_canonical=False,
                        )

    for codepoint in merged_by_codepoint:
        canonical_adjacency.setdefault(codepoint, set())
        enriched_adjacency.setdefault(codepoint, set())
    canonical_adjacency["__canonical_edge_metadata__"] = canonical_edge_metadata  # type: ignore[index]
    enriched_adjacency["__canonical_edge_metadata__"] = canonical_edge_metadata  # type: ignore[index]

    canonical_representative, canonical_components = build_graph_components(canonical_adjacency, merged_by_codepoint)
    enriched_representative, enriched_components = build_graph_components(enriched_adjacency, merged_by_codepoint)

    return (
        all_edges,
        canonical_representative,
        canonical_components,
        enriched_representative,
        enriched_components,
    )


def choose_first_nonempty(*candidates: list[str]) -> list[str]:
    for values in candidates:
        unique = unique_preserve(values)
        if unique:
            return unique
    return []


def fill_canonical_record(
    codepoint: str,
    source_map: dict[str, SourceRecord],
    canonical_representative: dict[str, str],
    canonical_components: dict[str, list[str]],
    enriched_representative: dict[str, str],
    enriched_components: dict[str, list[str]],
) -> dict[str, Any]:
    unihan = source_map.get("unihan")
    ehanja = source_map.get("ehanja")
    kanjidic2 = source_map.get("kanjidic2")
    makemeahanzi = source_map.get("makemeahanzi")

    base_record = unihan or ehanja or kanjidic2 or makemeahanzi
    if base_record is None:
        raise ValueError(f"No source record available for {codepoint}")

    character = base_record.character
    radical = (
        (unihan.radical if unihan else None)
        or (kanjidic2.radical if kanjidic2 else None)
        or (ehanja.radical if ehanja else None)
    )
    total_strokes = (
        (unihan.total_strokes if unihan else None)
        or (kanjidic2.total_strokes if kanjidic2 else None)
        or (ehanja.total_strokes if ehanja else None)
        or (makemeahanzi.total_strokes if makemeahanzi else None)
    )

    core = {
        "radical": radical,
        "total_strokes": total_strokes,
        "definitions": {
            "english": choose_first_nonempty(
                unihan.definitions["english"] if unihan else [],
                kanjidic2.definitions["english"] if kanjidic2 else [],
                makemeahanzi.definitions["english"] if makemeahanzi else [],
            ),
            "korean_explanation": choose_first_nonempty(
                ehanja.definitions["korean_explanation"] if ehanja else []
            ),
            "korean_hun": choose_first_nonempty(
                ehanja.definitions["korean_hun"] if ehanja else []
            ),
        },
        "readings": {
            "mandarin": choose_first_nonempty(
                unihan.readings["mandarin"] if unihan else [],
                kanjidic2.readings["mandarin"] if kanjidic2 else [],
                makemeahanzi.readings["mandarin"] if makemeahanzi else [],
            ),
            "cantonese": choose_first_nonempty(
                unihan.readings["cantonese"] if unihan else []
            ),
            "korean_hangul": choose_first_nonempty(
                ehanja.readings["korean_hangul"] if ehanja else [],
                kanjidic2.readings["korean_hangul"] if kanjidic2 else [],
            ),
            "korean_romanized": choose_first_nonempty(
                kanjidic2.readings["korean_romanized"] if kanjidic2 else []
            ),
            "japanese_on": choose_first_nonempty(
                kanjidic2.readings["japanese_on"] if kanjidic2 else [],
                unihan.readings["japanese_on"] if unihan else [],
            ),
            "japanese_kun": choose_first_nonempty(
                kanjidic2.readings["japanese_kun"] if kanjidic2 else [],
                unihan.readings["japanese_kun"] if unihan else [],
            ),
            "vietnamese": choose_first_nonempty(
                unihan.readings["vietnamese"] if unihan else [],
                kanjidic2.readings["vietnamese"] if kanjidic2 else [],
            ),
        },
    }

    variants = make_empty_variants()
    if unihan:
        for key in VARIANT_KEYS:
            variants[key] = unique_preserve(unihan.variants.get(key, []))
    variants["representative_form"] = canonical_representative.get(codepoint, codepoint)
    variants["family_members"] = canonical_components.get(codepoint, [codepoint])

    supplementary_variants = make_empty_supplementary_variants()
    for record in source_map.values():
        for key, targets in record.supplementary_variants.items():
            supplementary_variants[key].extend(targets)
    supplementary_variants = {
        key: unique_preserve(values)
        for key, values in supplementary_variants.items()
    }

    structure = {
        "decomposition": makemeahanzi.structure.get("decomposition") if makemeahanzi else None,
        "etymology_type": makemeahanzi.structure.get("etymology_type") if makemeahanzi else None,
        "etymology_hint": makemeahanzi.structure.get("etymology_hint") if makemeahanzi else None,
        "phonetic_component": makemeahanzi.structure.get("phonetic_component") if makemeahanzi else None,
        "semantic_component": makemeahanzi.structure.get("semantic_component") if makemeahanzi else None,
    }

    media = {
        "stroke_svg_paths": (makemeahanzi.media.get("stroke_svg_paths") if makemeahanzi else []) or [],
        "stroke_medians": (makemeahanzi.media.get("stroke_medians") if makemeahanzi else []) or [],
    }

    references = {
        "unihan": unihan.references if unihan else {},
        "kanjidic2": kanjidic2.references if kanjidic2 else {},
        "ehanja": ehanja.references if ehanja else {},
    }

    return {
        "character": character,
        "codepoint": codepoint,
        "source_flags": source_presence_flags(source_map),
        "core": core,
        "variants": variants,
        "supplementary_variants": supplementary_variants,
        "variant_graph": {
            "canonical_family_members": canonical_components.get(codepoint, [codepoint]),
            "canonical_representative_form": canonical_representative.get(codepoint, codepoint),
            "enriched_family_members": enriched_components.get(codepoint, [codepoint]),
            "enriched_representative_form": enriched_representative.get(codepoint, codepoint),
        },
        "structure": structure,
        "media": media,
        "references": references,
        "source_payloads": {
            "unihan_raw": unihan.source_payload if unihan else None,
            "ehanja_raw": ehanja.source_payload if ehanja else None,
            "kanjidic2_raw": kanjidic2.source_payload if kanjidic2 else None,
            "makemeahanzi_raw": makemeahanzi.source_payload if makemeahanzi else None,
        },
    }


def build_canonical_records(
    source_records: dict[str, list[SourceRecord]]
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    merged = merge_source_records(source_records)
    (
        variant_edges,
        canonical_representative_map,
        canonical_components,
        enriched_representative_map,
        enriched_components,
    ) = build_variant_views(merged)

    canonical_records: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for codepoint in sorted(merged):
        canonical = fill_canonical_record(
            codepoint,
            merged[codepoint],
            canonical_representative_map,
            canonical_components,
            enriched_representative_map,
            enriched_components,
        )
        canonical_records.append(canonical)
        component_rows.append(
            {
                "codepoint": codepoint,
                "character": canonical["character"],
                "representative_form": canonical["variants"]["representative_form"],
                "family_members": canonical["variants"]["family_members"],
                "component_size": len(canonical["variants"]["family_members"]),
                "canonical_representative_form": canonical["variant_graph"]["canonical_representative_form"],
                "canonical_family_members": canonical["variant_graph"]["canonical_family_members"],
                "enriched_representative_form": canonical["variant_graph"]["enriched_representative_form"],
                "enriched_family_members": canonical["variant_graph"]["enriched_family_members"],
            }
        )

    return canonical_records, variant_edges, component_rows


def export_sqlite(
    sqlite_path: Path,
    canonical_records: list[dict[str, Any]],
    variant_edges: list[dict[str, str]],
    component_rows: list[dict[str, Any]],
) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if sqlite_path.exists():
        sqlite_path.unlink()

    conn = sqlite3.connect(sqlite_path)
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            CREATE TABLE characters (
              codepoint TEXT PRIMARY KEY,
              character TEXT NOT NULL,
              radical TEXT,
              total_strokes INTEGER,
              representative_form TEXT,
              data_json TEXT NOT NULL
            );

            CREATE TABLE character_readings (
              codepoint TEXT NOT NULL,
              character TEXT NOT NULL,
              language TEXT NOT NULL,
              reading_type TEXT NOT NULL,
              value TEXT NOT NULL
            );

            CREATE TABLE character_meanings (
              codepoint TEXT NOT NULL,
              character TEXT NOT NULL,
              language TEXT NOT NULL,
              meaning_type TEXT NOT NULL,
              value TEXT NOT NULL
            );

            CREATE TABLE variant_edges (
              source_name TEXT NOT NULL,
              relation_scope TEXT NOT NULL,
              source_codepoint TEXT NOT NULL,
              source_character TEXT NOT NULL,
              relation TEXT NOT NULL,
              target_codepoint TEXT NOT NULL,
              target_character TEXT NOT NULL
            );

            CREATE TABLE variant_components (
              codepoint TEXT NOT NULL,
              character TEXT NOT NULL,
              representative_form TEXT NOT NULL,
              component_size INTEGER NOT NULL,
              family_members_json TEXT NOT NULL,
              canonical_representative_form TEXT NOT NULL,
              canonical_family_members_json TEXT NOT NULL,
              enriched_representative_form TEXT NOT NULL,
              enriched_family_members_json TEXT NOT NULL
            );

            CREATE TABLE source_presence (
              codepoint TEXT NOT NULL,
              character TEXT NOT NULL,
              unihan INTEGER NOT NULL,
              ehanja INTEGER NOT NULL,
              kanjidic2 INTEGER NOT NULL,
              makemeahanzi INTEGER NOT NULL
            );

            CREATE TABLE source_payloads (
              codepoint TEXT NOT NULL,
              character TEXT NOT NULL,
              source_name TEXT NOT NULL,
              payload_json TEXT NOT NULL
            );

            CREATE TABLE character_media (
              codepoint TEXT NOT NULL,
              character TEXT NOT NULL,
              stroke_index INTEGER NOT NULL,
              stroke_svg_path TEXT,
              median_json TEXT
            );
            """
        )

        for record in canonical_records:
            codepoint = record["codepoint"]
            character = record["character"]
            cur.execute(
                """
                INSERT INTO characters(codepoint, character, radical, total_strokes, representative_form, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    codepoint,
                    character,
                    record["core"]["radical"],
                    record["core"]["total_strokes"],
                    record["variants"]["representative_form"],
                    json.dumps(record, ensure_ascii=False, sort_keys=True),
                ),
            )

            for reading_key, values in record["core"]["readings"].items():
                for value in values:
                    cur.execute(
                        """
                        INSERT INTO character_readings(codepoint, character, language, reading_type, value)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            codepoint,
                            character,
                            reading_key,
                            reading_key,
                            value,
                        ),
                    )

            for meaning_key, values in record["core"]["definitions"].items():
                for value in values:
                    cur.execute(
                        """
                        INSERT INTO character_meanings(codepoint, character, language, meaning_type, value)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            codepoint,
                            character,
                            "ko" if meaning_key.startswith("korean") else "en",
                            meaning_key,
                            value,
                        ),
                    )

            cur.execute(
                """
                INSERT INTO source_presence(codepoint, character, unihan, ehanja, kanjidic2, makemeahanzi)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    codepoint,
                    character,
                    int(record["source_flags"]["unihan"]),
                    int(record["source_flags"]["ehanja"]),
                    int(record["source_flags"]["kanjidic2"]),
                    int(record["source_flags"]["makemeahanzi"]),
                ),
            )

            for source_name, payload in record["source_payloads"].items():
                if payload is None:
                    continue
                cur.execute(
                    """
                    INSERT INTO source_payloads(codepoint, character, source_name, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (codepoint, character, source_name, json.dumps(payload, ensure_ascii=False, sort_keys=True)),
                )

            stroke_paths = record["media"]["stroke_svg_paths"]
            stroke_medians = record["media"]["stroke_medians"]
            max_len = max(len(stroke_paths), len(stroke_medians))
            for index in range(max_len):
                cur.execute(
                    """
                    INSERT INTO character_media(codepoint, character, stroke_index, stroke_svg_path, median_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        codepoint,
                        character,
                        index,
                        stroke_paths[index] if index < len(stroke_paths) else None,
                        json.dumps(stroke_medians[index], ensure_ascii=False) if index < len(stroke_medians) else None,
                    ),
                )

        for edge in variant_edges:
            cur.execute(
                """
                INSERT INTO variant_edges(source_name, relation_scope, source_codepoint, source_character, relation, target_codepoint, target_character)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge["source_name"],
                    edge["relation_scope"],
                    edge["source_codepoint"],
                    edge["source_character"],
                    edge["relation"],
                    edge["target_codepoint"],
                    edge["target_character"],
                ),
            )

        for row in component_rows:
            cur.execute(
                """
                INSERT INTO variant_components(
                    codepoint, character, representative_form, component_size, family_members_json,
                    canonical_representative_form, canonical_family_members_json,
                    enriched_representative_form, enriched_family_members_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["codepoint"],
                    row["character"],
                    row["representative_form"],
                    row["component_size"],
                    json.dumps(row["family_members"], ensure_ascii=False),
                    row["canonical_representative_form"],
                    json.dumps(row["canonical_family_members"], ensure_ascii=False),
                    row["enriched_representative_form"],
                    json.dumps(row["enriched_family_members"], ensure_ascii=False),
                ),
            )

        cur.executescript(
            """
            CREATE INDEX idx_character_readings_codepoint ON character_readings(codepoint);
            CREATE INDEX idx_character_meanings_codepoint ON character_meanings(codepoint);
            CREATE INDEX idx_variant_edges_source ON variant_edges(source_codepoint);
            CREATE INDEX idx_variant_edges_target ON variant_edges(target_codepoint);
            CREATE INDEX idx_variant_components_rep ON variant_components(representative_form);
            """
        )
        conn.commit()
    finally:
        conn.close()


def build_summary(
    source_records: dict[str, list[SourceRecord]],
    canonical_records: list[dict[str, Any]],
    variant_edges: list[dict[str, str]],
    component_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    canonical_component_counter = Counter(row["canonical_representative_form"] for row in component_rows)
    enriched_component_counter = Counter(row["enriched_representative_form"] for row in component_rows)
    edge_scope_counter = Counter(edge["relation_scope"] for edge in variant_edges)
    edge_source_counter = Counter(edge["source_name"] for edge in variant_edges if edge["relation_scope"] == "supplementary")
    gained_members = sum(
        1
        for row in component_rows
        if len(row["enriched_family_members"]) > len(row["canonical_family_members"])
    )
    return {
        "source_record_counts": {
            source: len(records)
            for source, records in source_records.items()
        },
        "canonical_character_count": len(canonical_records),
        "variant_edge_count": len(variant_edges),
        "variant_edge_scope_counts": dict(edge_scope_counter),
        "supplementary_edge_source_counts": dict(edge_source_counter),
        "canonical_variant_component_count": len(canonical_component_counter),
        "enriched_variant_component_count": len(enriched_component_counter),
        "max_canonical_component_size": max((len(row["canonical_family_members"]) for row in component_rows), default=0),
        "max_enriched_component_size": max((len(row["enriched_family_members"]) for row in component_rows), default=0),
        "characters_with_enriched_growth": gained_members,
        "sample_codepoints": [record["codepoint"] for record in canonical_records[:10]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Sinograph Canonical DB v1 staging JSONL, canonical JSONL, and SQLite artifacts."
    )
    parser.add_argument(
        "--skip-sqlite",
        action="store_true",
        help="Build JSONL artifacts only and skip SQLite export.",
    )
    args = parser.parse_args()

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    source_records = build_source_adapters()
    for source_name, records in source_records.items():
        write_jsonl(STAGING_DIR / f"{source_name}.normalized.jsonl", [record.to_dict() for record in records])

    canonical_records, variant_edges, component_rows = build_canonical_records(source_records)
    write_jsonl(OUT_DIR / "canonical_characters.jsonl", canonical_records)
    write_jsonl(OUT_DIR / "canonical_variants.jsonl", variant_edges)
    write_jsonl(OUT_DIR / "variant_components.jsonl", component_rows)

    summary = build_summary(source_records, canonical_records, variant_edges, component_rows)
    with (OUT_DIR / "build_summary.json").open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    if not args.skip_sqlite:
        export_sqlite(OUT_DIR / "sinograph_canonical_v1.sqlite", canonical_records, variant_edges, component_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
