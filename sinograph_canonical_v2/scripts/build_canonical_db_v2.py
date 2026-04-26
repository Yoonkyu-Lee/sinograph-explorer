"""Sinograph Canonical DB v2 — build pipeline.

ETL stages:
  A. Source adapters → staging/<source>.normalized.jsonl
  B. Identity merge (codepoint → source_map)
  C. Variant graph — edges merged by (src, tgt, scope) with sources[] + support_count
     + canonical/enriched connected-component computation
  D. Canonical projection — core + provenance + core_alternatives + source_exclusive
  E. SQLite export

Run:
    python scripts/build_canonical_db_v2.py

Inputs:
    ../db_src/Unihan/Unihan_txt/*.txt
    ../db_src/e-hanja_online/tree.jsonl + detail.jsonl + strokes_manifest.jsonl
    ../db_src/KANJIDIC2/KANJIDIC2_xml/kanjidic2.xml
    ../db_src/MAKEMEAHANZI/dictionary.txt

Outputs (under this workspace):
    staging/<source>.normalized.jsonl
    out/canonical_characters.jsonl
    out/canonical_variants.jsonl
    out/variant_components.jsonl
    out/sinograph_canonical_v2.sqlite
    out/build_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------- paths ----------

SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BUILD_ROOT.parent
DB_SRC_ROOT = REPO_ROOT / "db_src"

UNIHAN_DIR = DB_SRC_ROOT / "Unihan" / "Unihan_txt"
EHANJA_ONLINE_DIR = DB_SRC_ROOT / "e-hanja_online"
EHANJA_ONLINE_TREE = EHANJA_ONLINE_DIR / "tree.jsonl"
EHANJA_ONLINE_DETAIL = EHANJA_ONLINE_DIR / "detail.jsonl"
EHANJA_ONLINE_MANIFEST = EHANJA_ONLINE_DIR / "strokes_manifest.jsonl"
KANJIDIC2_XML = DB_SRC_ROOT / "KANJIDIC2" / "KANJIDIC2_xml" / "kanjidic2.xml"
MAKEMEAHANZI_DICT = DB_SRC_ROOT / "MAKEMEAHANZI" / "dictionary.txt"

STAGING_DIR = BUILD_ROOT / "staging"
OUT_DIR = BUILD_ROOT / "out"


# ---------- constants ----------

SOURCE_NAMES = ("unihan", "ehanja_online", "kanjidic2", "makemeahanzi")

# Core readings that ≥2 sources potentially cover.
READING_KEYS = (
    "mandarin",
    "cantonese",
    "korean_hangul",
    "japanese_on",
    "japanese_kun",
    "vietnamese",
)

# Unihan variant backbone — relation buckets in `variants.*`.
UNIHAN_VARIANT_FIELD_MAP = {
    "kTraditionalVariant": "traditional",
    "kSimplifiedVariant": "simplified",
    "kSemanticVariant": "semantic",
    "kSpecializedSemanticVariant": "specialized_semantic",
    "kZVariant": "z_variants",
    "kSpoofingVariant": "spoofing",
}
VARIANT_KEYS = tuple(UNIHAN_VARIANT_FIELD_MAP.values())

UNIHAN_READING_FIELD_MAP = {
    "kMandarin": "mandarin",
    "kCantonese": "cantonese",
    "kKorean": "korean_hangul",
    "kJapaneseOn": "japanese_on",
    "kJapaneseKun": "japanese_kun",
    "kVietnamese": "vietnamese",
}

# supplementary variant keys — source-explicit. Empty list when that source
# doesn't speak about the char.
SUPPLEMENTARY_VARIANT_KEYS = (
    # from e-hanja_online schoolCom (10 relations)
    "ehanja_yakja", "ehanja_bonja", "ehanja_simple", "ehanja_kanji",
    "ehanja_dongja", "ehanja_tongja", "ehanja_waja", "ehanja_goja",
    "ehanja_sokja", "ehanja_hDup",
    # from e-hanja_online detail dropdown-hover (3 detail-only relations)
    "ehanja_synonyms", "ehanja_opposites", "ehanja_alt_forms",
    # from KANJIDIC2 resolvable variant refs
    "kanjidic2_resolved",
)

VARIANT_CODEPOINT_RE = re.compile(r"U\+[0-9A-F]{4,6}")
KANJIDIC2_RESOLVABLE_VARIANT_TYPES = {"jis208", "jis212", "jis213", "ucs"}

# Relations that are stored on each record but do NOT participate in the
# variant-family graph. These relations are semantic-adjacent (유의 / 상대)
# or detail-page-only (별자) and are not drawn on e-hanja's official
# 이체-관계도 (comsTree). Edges still exist in canonical_variants.jsonl /
# SQLite variant_edges, but canonical_/enriched_family_members are computed
# ignoring them.
FAMILY_GRAPH_EXCLUDED_RELATIONS = {
    "ehanja_synonyms",
    "ehanja_opposites",
    "ehanja_alt_forms",
}


# ---------- utilities ----------


def codepoint_from_char(character: str) -> str:
    return f"U+{ord(character):04X}"


def codepoint_to_char(codepoint: str) -> str:
    return chr(int(codepoint[2:], 16))


def ensure_single_character(text: str | None) -> str | None:
    s = (text or "").strip()
    return s if len(s) == 1 else None


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


def unique_preserve(values: list[Any]) -> list[Any]:
    """Drop empties, preserve order, dedup. Values are str unless marked."""
    seen: set[Any] = set()
    out: list[Any] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, str):
            t = v.strip()
            if not t:
                continue
            key = t
            val = t
        else:
            key = json.dumps(v, ensure_ascii=False, sort_keys=True)
            val = v
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    return out


def parse_character_targets(raw: str) -> list[str]:
    """'鍳,鑒,鑬,𨰲' → ['U+9373', 'U+9452', 'U+946C', 'U+28C32'] (single-char only)."""
    out: list[str] = []
    for piece in (raw or "").split(","):
        ch = ensure_single_character(piece)
        if ch is None:
            continue
        out.append(codepoint_from_char(ch))
    return unique_preserve(out)


def extract_unihan_variant_codepoints(values: list[str]) -> list[str]:
    out: list[str] = []
    for v in values:
        out.extend(VARIANT_CODEPOINT_RE.findall(v))
    return unique_preserve(out)


def parse_unihan_radical(rs_unicode_values: list[str]) -> str | None:
    for v in rs_unicode_values:
        text = v.strip()
        if not text:
            continue
        first = text.split()[0]
        radical = first.split(".")[0].split("'")[0]
        if radical:
            return radical
    return None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def empty_readings() -> dict[str, list[str]]:
    return {k: [] for k in READING_KEYS}


def empty_variants() -> dict[str, list[str]]:
    return {k: [] for k in VARIANT_KEYS}


def empty_supp_variants() -> dict[str, list[str]]:
    return {k: [] for k in SUPPLEMENTARY_VARIANT_KEYS}


# ---------- staging record ----------


@dataclass
class NormalizedSourceRecord:
    source_name: str
    character: str
    codepoint: str
    radical: str | None
    total_strokes: int | None
    readings: dict[str, list[str]]
    definitions: dict[str, list[str]]     # only "english" key used
    variants: dict[str, list[str]]        # Unihan backbone only
    supplementary_variants: dict[str, list[str]]
    exclusive: dict[str, Any]             # source-exclusive fields (goes to final source_exclusive.<source>)
    source_payload: dict[str, Any]        # near-raw (goes to final source_payloads.<source>_raw)

    def to_dict(self) -> dict:
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
            "exclusive": self.exclusive,
            "source_payload": self.source_payload,
        }


# ---------- Adapter: Unihan ----------


def build_unihan_records() -> list[NormalizedSourceRecord]:
    per_codepoint: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(UNIHAN_DIR.glob("Unihan_*.txt")):
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    cp, field_name, value = line.split("\t", 2)
                except ValueError:
                    continue
                per_codepoint[cp][field_name].append(value)

    records: list[NormalizedSourceRecord] = []
    for cp in sorted(per_codepoint):
        try:
            ch = chr(int(cp[2:], 16))
        except ValueError:
            continue
        fm = per_codepoint[cp]
        readings = empty_readings()
        for raw_name, canonical_key in UNIHAN_READING_FIELD_MAP.items():
            readings[canonical_key] = unique_preserve(fm.get(raw_name, []))
        definitions = {"english": unique_preserve(fm.get("kDefinition", []))}
        variants = empty_variants()
        for raw_name, canonical_key in UNIHAN_VARIANT_FIELD_MAP.items():
            variants[canonical_key] = extract_unihan_variant_codepoints(fm.get(raw_name, []))
        strokes = [parse_int(v) for v in fm.get("kTotalStrokes", [])]
        total_strokes = next((v for v in strokes if v is not None), None)

        exclusive = {
            "rs_unicode": unique_preserve(fm.get("kRSUnicode", [])),
            "kangxi": unique_preserve(fm.get("kKangXi", [])),
            "irg_kangxi": unique_preserve(fm.get("kIRGKangXi", [])),
            "hanyu": unique_preserve(fm.get("kHanYu", [])),
            "unihan_core_2020": unique_preserve(fm.get("kUnihanCore2020", [])),
        }
        # drop empties from exclusive
        exclusive = {k: v for k, v in exclusive.items() if v}

        records.append(NormalizedSourceRecord(
            source_name="unihan",
            character=ch,
            codepoint=cp,
            radical=parse_unihan_radical(fm.get("kRSUnicode", [])),
            total_strokes=total_strokes,
            readings=readings,
            definitions=definitions,
            variants=variants,
            supplementary_variants=empty_supp_variants(),
            exclusive=exclusive,
            source_payload={k: unique_preserve(v) for k, v in fm.items()},
        ))
    return records


# ---------- Adapter: e-hanja_online ----------

_EHANJA_SCHOOLCOM_MAP = {
    "yakja":  "ehanja_yakja",
    "bonja":  "ehanja_bonja",
    "simple": "ehanja_simple",
    "kanji":  "ehanja_kanji",
    "dongja": "ehanja_dongja",
    "tongja": "ehanja_tongja",
    "waja":   "ehanja_waja",
    "goja":   "ehanja_goja",
    "sokja":  "ehanja_sokja",
    "hDup":   "ehanja_hDup",
}

_EHANJA_DETAIL_RELATION_MAP = {
    "synonyms":  "ehanja_synonyms",
    "opposites": "ehanja_opposites",
    "alt_forms": "ehanja_alt_forms",
}


def _parse_hread(hread: str) -> tuple[list[str], str]:
    """Split '거울 감' into (hun='거울', eum=['감']). The e-hanja hRead field is
    space-separated 'meaning reading' (both may be multi-syllable). We take
    everything before the last space as hun, last token as reading."""
    text = (hread or "").strip()
    if not text:
        return [], ""
    parts = text.rsplit(" ", 1)
    if len(parts) == 1:
        return [], parts[0]
    return [parts[0].strip()], parts[1].strip()


def build_ehanja_online_records() -> list[NormalizedSourceRecord]:
    tree_rows = read_jsonl(EHANJA_ONLINE_TREE)
    detail_rows = read_jsonl(EHANJA_ONLINE_DETAIL)
    manifest_rows = read_jsonl(EHANJA_ONLINE_MANIFEST)

    detail_by_cp = {r["cp"]: r for r in detail_rows if "cp" in r}
    manifest_by_cp = {r["cp"]: r for r in manifest_rows if "cp" in r}

    records: list[NormalizedSourceRecord] = []
    for tree in tree_rows:
        cp_int = tree.get("cp")
        if cp_int is None:
            continue
        ch = tree.get("char") or chr(cp_int)
        if ensure_single_character(ch) is None:
            continue
        cp = f"U+{cp_int:04X}"
        det = detail_by_cp.get(cp_int) or {}
        man = manifest_by_cp.get(cp_int) or {}

        # readings.korean_hangul — from getHunum hRead reading token
        hunum_list = tree.get("getHunum") or []
        korean_hangul: list[str] = []
        korean_hun_all: list[str] = []
        for item in hunum_list:
            hh = item.get("hRead", "")
            hun_parts, eum = _parse_hread(hh)
            if eum:
                korean_hangul.append(eum)
            if hun_parts:
                korean_hun_all.extend(hun_parts)
        readings = empty_readings()
        readings["korean_hangul"] = unique_preserve(korean_hangul)

        # detail.pinyin → readings.mandarin (strip variant in parens etc.)
        if "pinyin" in det:
            # Input shape "jiàn (jiàn)" or just "jiàn"
            main = re.split(r"[\(\s]", det["pinyin"].strip(), maxsplit=1)[0].strip()
            if main:
                readings["mandarin"] = [main]

        # definitions.english from detail.english
        definitions = {"english": []}
        if det.get("english"):
            definitions["english"] = [det["english"].rstrip(". ").strip()]

        # total_strokes: prefer manifest.stroke_count (animated), fallback detail.total_strokes
        ts: int | None = None
        if man.get("stroke_count"):
            ts = parse_int(man.get("stroke_count"))
        if ts is None and det.get("total_strokes") is not None:
            ts = parse_int(det.get("total_strokes"))

        # radical (from detail.radical.char)
        radical_ch = None
        if det.get("radical") and det["radical"].get("char"):
            radical_ch = det["radical"]["char"]

        # supplementary_variants from schoolcom
        supp = empty_supp_variants()
        sc_list = tree.get("getSchoolCom") or []
        sc = sc_list[0] if sc_list else {}
        for src_key, dst_key in _EHANJA_SCHOOLCOM_MAP.items():
            supp[dst_key] = parse_character_targets(sc.get(src_key, ""))
        # supplementary_variants from detail.related_characters
        rc = det.get("related_characters") or {}
        for src_key, dst_key in _EHANJA_DETAIL_RELATION_MAP.items():
            items = rc.get(src_key) or []
            cps = []
            for item in items:
                ch2 = ensure_single_character(item.get("char"))
                if ch2:
                    cps.append(codepoint_from_char(ch2))
            supp[dst_key] = unique_preserve(cps)

        # exclusive — source-exclusive ehanja_online block
        exclusive: dict[str, Any] = {}
        if korean_hun_all:
            exclusive["korean_hun"] = unique_preserve(korean_hun_all)
        if tree.get("getJahae"):
            # keep structure but strip redundant 'hanja' field
            cleaned = []
            for item in tree["getJahae"]:
                cleaned.append({k: v for k, v in item.items() if k != "hanja"})
            exclusive["korean_explanation"] = cleaned
        if det.get("classification"):
            exclusive["classification"] = det["classification"]
        if det.get("shape"):
            exclusive["shape"] = det["shape"]
        if det.get("etymology"):
            exclusive["etymology"] = det["etymology"]
        if det.get("word_usage"):
            exclusive["word_usage"] = det["word_usage"]
        if det.get("related_words"):
            exclusive["related_words"] = det["related_words"]
        if det.get("radical") and (det["radical"].get("etymology")
                                    or det["radical"].get("variant")
                                    or det["radical"].get("name")):
            # keep radical sub-info that's not in core.radical
            exclusive["radical_detail"] = {
                k: v for k, v in det["radical"].items() if k != "char" and v
            }
        if man:
            exclusive["svg_type"] = man.get("type")
            if man.get("stroke_count") is not None:
                exclusive["stroke_count_animated"] = man.get("stroke_count")

        source_payload: dict[str, Any] = {"tree_raw": tree}
        if det:
            source_payload["detail_raw"] = det
        if man:
            source_payload["manifest_raw"] = man

        records.append(NormalizedSourceRecord(
            source_name="ehanja_online",
            character=ch,
            codepoint=cp,
            radical=radical_ch,
            total_strokes=ts,
            readings=readings,
            definitions=definitions,
            variants=empty_variants(),
            supplementary_variants=supp,
            exclusive=exclusive,
            source_payload=source_payload,
        ))
    return records


# ---------- Adapter: KANJIDIC2 ----------


def build_kanjidic2_records() -> list[NormalizedSourceRecord]:
    parsed: list[dict] = []
    cp_value_maps: dict[str, dict[str, str]] = defaultdict(dict)

    context = ET.iterparse(KANJIDIC2_XML, events=("end",))
    for _, elem in context:
        if elem.tag != "character":
            continue
        ch = ensure_single_character(elem.findtext("literal", ""))
        if not ch:
            elem.clear()
            continue
        readings = empty_readings()
        meanings: list[str] = []
        nanori: list[str] = []
        # Korean romanized kept source-exclusive
        korean_romanized: list[str] = []

        rm = elem.find("reading_meaning")
        rmgroup = rm.find("rmgroup") if rm is not None else None
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
                    korean_romanized.append(text)
                elif r_type == "vietnam":
                    readings["vietnamese"].append(text)
            for meaning in rmgroup.findall("meaning"):
                lang = meaning.get("m_lang")
                t = (meaning.text or "").strip()
                if not t:
                    continue
                if lang in (None, "", "en"):
                    meanings.append(t)
        if rm is not None:
            nanori = unique_preserve([(n.text or "").strip() for n in rm.findall("nanori")])

        radical = None
        for rad in elem.findall("radical/rad_value"):
            if rad.get("rad_type") == "classical":
                radical = (rad.text or "").strip() or None
                if radical:
                    break

        stroke_counts = [parse_int((n.text or "").strip()) for n in elem.findall("misc/stroke_count")]
        total_strokes = next((v for v in stroke_counts if v is not None), None)

        codepoint_refs: list[dict] = []
        for n in elem.findall("codepoint/cp_value"):
            cp_type = n.get("cp_type", "")
            value = (n.text or "").strip()
            if cp_type and value:
                cp_value_maps[cp_type][value] = ch
                codepoint_refs.append({"cp_type": cp_type, "value": value})

        variant_refs: list[dict] = []
        misc = elem.find("misc")
        if misc is not None:
            for n in misc.findall("variant"):
                v = (n.text or "").strip()
                if not v:
                    continue
                variant_refs.append({"var_type": n.get("var_type", ""), "value": v})

        dic_number: dict[str, list[str]] = {}
        query_code: list[dict] = []
        dnum = elem.find("dic_number")
        if dnum is not None:
            for n in dnum.findall("dic_ref"):
                dic_number.setdefault(n.get("dr_type", ""), []).append((n.text or "").strip())
        qc = elem.find("query_code")
        if qc is not None:
            for n in qc.findall("q_code"):
                query_code.append({
                    "qc_type": n.get("qc_type", ""),
                    "skip_misclass": n.get("skip_misclass", ""),
                    "value": (n.text or "").strip(),
                })

        parsed.append({
            "character": ch, "codepoint": codepoint_from_char(ch),
            "radical": radical, "total_strokes": total_strokes,
            "readings": {k: unique_preserve(v) for k, v in readings.items()},
            "english_meanings": unique_preserve(meanings),
            "nanori": nanori,
            "korean_romanized": unique_preserve(korean_romanized),
            "variant_refs": variant_refs,
            "codepoint_refs": codepoint_refs,
            "dic_number": dic_number,
            "query_code": query_code,
        })
        elem.clear()

    records: list[NormalizedSourceRecord] = []
    for row in parsed:
        resolved: list[str] = []
        unresolved: list[dict] = []
        for ref in row["variant_refs"]:
            vt = ref["var_type"]; val = ref["value"]
            if vt not in KANJIDIC2_RESOLVABLE_VARIANT_TYPES:
                unresolved.append(ref)
                continue
            target_char = cp_value_maps.get(vt, {}).get(val)
            if target_char is None:
                unresolved.append(ref)
                continue
            resolved.append(codepoint_from_char(target_char))

        supp = empty_supp_variants()
        supp["kanjidic2_resolved"] = unique_preserve(resolved)

        exclusive: dict[str, Any] = {}
        if row["korean_romanized"]:
            exclusive["korean_romanized"] = row["korean_romanized"]
        if row["dic_number"]:
            exclusive["dictionary_refs"] = {
                k: unique_preserve(v) for k, v in row["dic_number"].items()
            }
        if row["query_code"]:
            exclusive["query_codes"] = row["query_code"]
        if row["codepoint_refs"]:
            exclusive["codepoint_refs"] = row["codepoint_refs"]
        if row["nanori"]:
            exclusive["nanori"] = row["nanori"]
        if unresolved:
            exclusive["unresolved_variant_refs"] = unresolved

        records.append(NormalizedSourceRecord(
            source_name="kanjidic2",
            character=row["character"],
            codepoint=row["codepoint"],
            radical=row["radical"],
            total_strokes=row["total_strokes"],
            readings=row["readings"],
            definitions={"english": row["english_meanings"]},
            variants=empty_variants(),
            supplementary_variants=supp,
            exclusive=exclusive,
            source_payload={
                "variant_refs": row["variant_refs"],
                "unresolved_variant_refs": unresolved,
                "dic_number": row["dic_number"],
                "query_code": row["query_code"],
                "codepoint_refs": row["codepoint_refs"],
                "nanori": row["nanori"],
            },
        ))
    return records


# ---------- Adapter: MakeMeAHanzi (reduced: structure only) ----------


def build_makemeahanzi_records() -> list[NormalizedSourceRecord]:
    """MMH in v2: decomposition + etymology only. Stroke media lives in db_src/
    and is consumed by the v3 synth engine directly — canonical layer ignores it."""
    dictionary_rows: dict[str, dict] = {}
    if MAKEMEAHANZI_DICT.exists():
        for row in read_jsonl(MAKEMEAHANZI_DICT):
            ch = ensure_single_character(row.get("character", ""))
            if ch is None:
                continue
            dictionary_rows[ch] = row

    records: list[NormalizedSourceRecord] = []
    for ch in sorted(dictionary_rows):
        drow = dictionary_rows[ch]
        etymology = drow.get("etymology") or {}
        readings = empty_readings()
        readings["mandarin"] = unique_preserve(drow.get("pinyin", []) or [])
        definitions = {"english": unique_preserve([drow.get("definition", "")])}

        exclusive: dict[str, Any] = {}
        if drow.get("decomposition"):
            exclusive["decomposition"] = drow["decomposition"]
        if etymology.get("type"):
            exclusive["etymology_type"] = etymology["type"]
        if etymology.get("hint"):
            exclusive["etymology_hint"] = etymology["hint"]
        if etymology.get("phonetic"):
            exclusive["phonetic_component"] = etymology["phonetic"]
        if etymology.get("semantic"):
            exclusive["semantic_component"] = etymology["semantic"]

        records.append(NormalizedSourceRecord(
            source_name="makemeahanzi",
            character=ch,
            codepoint=codepoint_from_char(ch),
            radical=(drow.get("radical") or "").strip() or None,
            total_strokes=None,   # MMH doesn't give count directly without graphics.txt
            readings=readings,
            definitions=definitions,
            variants=empty_variants(),
            supplementary_variants=empty_supp_variants(),
            exclusive=exclusive,
            source_payload={"dictionary": drow},
        ))
    return records


# ---------- Stage B: identity merge ----------


def merge_source_records(
    all_records: dict[str, list[NormalizedSourceRecord]]
) -> dict[str, dict[str, NormalizedSourceRecord]]:
    """codepoint → {source_name: record}"""
    merged: dict[str, dict[str, NormalizedSourceRecord]] = defaultdict(dict)
    for source_name, rows in all_records.items():
        for rec in rows:
            key = rec.codepoint or codepoint_from_char(rec.character)
            merged[key][source_name] = rec
    return dict(merged)


# ---------- Stage C: variant graph ----------


@dataclass
class MergedEdge:
    source_codepoint: str
    target_codepoint: str
    relation_scope: str   # "canonical" | "supplementary"
    relation: str         # primary (first-seen) relation label
    sources: list[str] = field(default_factory=list)
    source_relations: dict[str, str] = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "source_character": codepoint_to_char(self.source_codepoint),
            "source_codepoint": self.source_codepoint,
            "target_character": codepoint_to_char(self.target_codepoint),
            "target_codepoint": self.target_codepoint,
            "relation_scope": self.relation_scope,
            "relation": self.relation,
            "sources": list(self.sources),
            "source_relations": dict(self.source_relations),
            "support_count": len(self.sources),
        }


def build_variant_graph(
    merged_by_cp: dict[str, dict[str, NormalizedSourceRecord]]
) -> tuple[list[MergedEdge], dict[str, str], dict[str, list[str]],
           dict[str, str], dict[str, list[str]]]:
    """Returns (merged_edges, canonical_rep, canonical_components, enriched_rep, enriched_components)."""
    edges_by_key: dict[tuple[str, str, str], MergedEdge] = {}

    def add_edge(*, src_cp: str, tgt_cp: str, scope: str, relation: str, source_name: str) -> None:
        if not tgt_cp or src_cp == tgt_cp:
            return
        key = (src_cp, tgt_cp, scope)
        e = edges_by_key.get(key)
        if e is None:
            e = MergedEdge(
                source_codepoint=src_cp, target_codepoint=tgt_cp,
                relation_scope=scope, relation=relation,
            )
            edges_by_key[key] = e
        if source_name not in e.source_relations:
            e.sources.append(source_name)
            e.source_relations[source_name] = relation

    for cp, source_map in merged_by_cp.items():
        for source_name in SOURCE_NAMES:
            rec = source_map.get(source_name)
            if rec is None:
                continue
            if source_name == "unihan":
                for rel_key, targets in rec.variants.items():
                    for t in targets:
                        add_edge(src_cp=cp, tgt_cp=t, scope="canonical",
                                  relation=rel_key, source_name="unihan")
            else:
                for rel_key, targets in rec.supplementary_variants.items():
                    if not targets:
                        continue
                    for t in targets:
                        add_edge(src_cp=cp, tgt_cp=t, scope="supplementary",
                                  relation=rel_key, source_name=source_name)

    merged_edges = list(edges_by_key.values())

    canonical_adj: dict[str, set[str]] = defaultdict(set)
    enriched_adj: dict[str, set[str]] = defaultdict(set)
    for e in merged_edges:
        # Detail-only / semantic-adjacent relations are persisted as edges but
        # are not used to expand the family graph.
        if e.relation in FAMILY_GRAPH_EXCLUDED_RELATIONS:
            continue
        enriched_adj[e.source_codepoint].add(e.target_codepoint)
        enriched_adj[e.target_codepoint].add(e.source_codepoint)
        if e.relation_scope == "canonical":
            canonical_adj[e.source_codepoint].add(e.target_codepoint)
            canonical_adj[e.target_codepoint].add(e.source_codepoint)
    for cp in merged_by_cp:
        canonical_adj.setdefault(cp, set())
        enriched_adj.setdefault(cp, set())

    canonical_edge_metadata = [e for e in merged_edges if e.relation_scope == "canonical"]
    canonical_rep, canonical_components = _components(canonical_adj, merged_by_cp, canonical_edge_metadata)
    enriched_rep, enriched_components = _components(enriched_adj, merged_by_cp, canonical_edge_metadata)
    return merged_edges, canonical_rep, canonical_components, enriched_rep, enriched_components


def _components(
    adjacency: dict[str, set[str]],
    merged_by_cp: dict[str, dict[str, NormalizedSourceRecord]],
    canonical_edges: list[MergedEdge],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    incoming_traditional: Counter = Counter()
    outgoing_simplified: Counter = Counter()
    for e in canonical_edges:
        if e.relation == "traditional":
            incoming_traditional[e.target_codepoint] += 1
        if e.relation == "simplified":
            outgoing_simplified[e.source_codepoint] += 1

    def metadata_score(cp: str) -> int:
        source_map = merged_by_cp.get(cp, {})
        score = len(source_map)
        for rec in source_map.values():
            score += sum(1 for v in rec.readings.values() if v)
            score += sum(1 for v in rec.definitions.values() if v)
            if rec.radical:
                score += 1
            if rec.total_strokes is not None:
                score += 1
            if "decomposition" in rec.exclusive:
                score += 1
        return score

    def rep_key(cp: str) -> tuple[int, int, str]:
        traditionality = incoming_traditional[cp] + outgoing_simplified[cp]
        return (-traditionality, -metadata_score(cp), cp)

    visited: set[str] = set()
    rep: dict[str, str] = {}
    comp: dict[str, list[str]] = {}
    for start in sorted(adjacency):
        if start in visited:
            continue
        q = deque([start])
        part: list[str] = []
        visited.add(start)
        while q:
            cur = q.popleft()
            part.append(cur)
            for nb in sorted(adjacency.get(cur, set())):
                if nb in visited:
                    continue
                visited.add(nb)
                q.append(nb)
        part_sorted = sorted(part)
        representative = sorted(part_sorted, key=rep_key)[0]
        for cp in part_sorted:
            rep[cp] = representative
            comp[cp] = part_sorted
    return rep, comp


# ---------- Stage D: canonical projection ----------


def _pick_first_nonempty(
    candidates: list[tuple[str, list[str]]]
) -> tuple[list[str], str | None]:
    """Return (picked_values, picked_source). candidates = [(source_name, values), ...]"""
    for source_name, values in candidates:
        cleaned = unique_preserve(values)
        if cleaned:
            return cleaned, source_name
    return [], None


def _pick_first_scalar(
    candidates: list[tuple[str, Any]]
) -> tuple[Any, str | None]:
    for source_name, value in candidates:
        if value is not None and value != "" and value != []:
            return value, source_name
    return None, None


def _register_alternative(
    alternatives: dict[str, dict[str, Any]],
    field_path: str,
    picked_source: str | None,
    candidates: list[tuple[str, Any]],
) -> None:
    """For each non-picked candidate that HAS data, record it under
    alternatives[field_path][source_name]."""
    non_empty = [(s, v) for s, v in candidates
                 if s != picked_source and v is not None and v != "" and v != []]
    if not non_empty:
        return
    alternatives.setdefault(field_path, {})
    for s, v in non_empty:
        # normalize lists to unique_preserve for readings / english
        if isinstance(v, list):
            v2 = unique_preserve(v)
            if not v2:
                continue
            alternatives[field_path][s] = v2
        else:
            alternatives[field_path][s] = v


def fill_canonical(
    cp: str,
    source_map: dict[str, NormalizedSourceRecord],
    canonical_rep: dict[str, str],
    canonical_comp: dict[str, list[str]],
    enriched_rep: dict[str, str],
    enriched_comp: dict[str, list[str]],
) -> dict:
    """Build one canonical_characters record."""
    unihan = source_map.get("unihan")
    ehanja = source_map.get("ehanja_online")
    kd2 = source_map.get("kanjidic2")
    mmh = source_map.get("makemeahanzi")
    base = unihan or ehanja or kd2 or mmh
    if base is None:
        raise ValueError(f"no source for {cp}")
    character = base.character

    provenance: dict[str, str] = {}
    alternatives: dict[str, dict[str, Any]] = {}

    # --- core.radical ---
    rad_candidates: list[tuple[str, Any]] = []
    if unihan: rad_candidates.append(("unihan", unihan.radical))
    if kd2:    rad_candidates.append(("kanjidic2", kd2.radical))
    if ehanja: rad_candidates.append(("ehanja_online", ehanja.radical))
    if mmh:    rad_candidates.append(("makemeahanzi", mmh.radical))
    radical, rad_source = _pick_first_scalar(rad_candidates)
    if rad_source:
        provenance["radical"] = rad_source
        _register_alternative(alternatives, "radical", rad_source, rad_candidates)

    # --- core.total_strokes ---
    ts_candidates: list[tuple[str, Any]] = []
    if unihan: ts_candidates.append(("unihan", unihan.total_strokes))
    if kd2:    ts_candidates.append(("kanjidic2", kd2.total_strokes))
    if ehanja: ts_candidates.append(("ehanja_online", ehanja.total_strokes))
    if mmh:    ts_candidates.append(("makemeahanzi", mmh.total_strokes))
    total_strokes, ts_source = _pick_first_scalar(ts_candidates)
    if ts_source:
        provenance["total_strokes"] = ts_source
        _register_alternative(alternatives, "total_strokes", ts_source, ts_candidates)

    # --- core.definitions.english ---
    en_candidates: list[tuple[str, list[str]]] = []
    if unihan: en_candidates.append(("unihan", unihan.definitions.get("english", [])))
    if kd2:    en_candidates.append(("kanjidic2", kd2.definitions.get("english", [])))
    if mmh:    en_candidates.append(("makemeahanzi", mmh.definitions.get("english", [])))
    if ehanja: en_candidates.append(("ehanja_online", ehanja.definitions.get("english", [])))
    english, en_source = _pick_first_nonempty(en_candidates)
    if en_source:
        provenance["definitions.english"] = en_source
        _register_alternative(alternatives, "definitions.english", en_source, en_candidates)

    # --- core.readings.* ---
    def reading_candidates(order: list[str], key: str) -> list[tuple[str, list[str]]]:
        c: list[tuple[str, list[str]]] = []
        lookup = {"unihan": unihan, "ehanja_online": ehanja, "kanjidic2": kd2, "makemeahanzi": mmh}
        for s in order:
            rec = lookup.get(s)
            if rec is not None:
                c.append((s, rec.readings.get(key, [])))
        return c

    reading_authority = {
        "mandarin":     ["unihan", "kanjidic2", "ehanja_online", "makemeahanzi"],
        "cantonese":    ["unihan"],
        "korean_hangul":["ehanja_online", "kanjidic2"],
        "japanese_on":  ["kanjidic2", "unihan"],
        "japanese_kun": ["kanjidic2", "unihan"],
        "vietnamese":   ["unihan", "kanjidic2"],
    }
    core_readings: dict[str, list[str]] = {}
    for key, order in reading_authority.items():
        cands = reading_candidates(order, key)
        picked, src = _pick_first_nonempty(cands)
        core_readings[key] = picked
        if src:
            provenance[f"readings.{key}"] = src
            _register_alternative(alternatives, f"readings.{key}", src, cands)

    core = {
        "radical": radical,
        "total_strokes": total_strokes,
        "definitions": {"english": english},
        "readings": core_readings,
    }

    # --- variants (Unihan backbone only) ---
    variants: dict[str, Any] = empty_variants()
    if unihan:
        variants = {k: list(v) for k, v in unihan.variants.items()}
    variants["representative_form"] = canonical_rep.get(cp)
    variants["family_members"] = list(canonical_comp.get(cp, [cp]))

    # --- supplementary_variants (source-explicit bucket; merged across sources) ---
    supplementary: dict[str, list[str]] = empty_supp_variants()
    for source_name in SOURCE_NAMES:
        rec = source_map.get(source_name)
        if rec is None:
            continue
        for k, v in rec.supplementary_variants.items():
            if v:
                merged_vals = unique_preserve(supplementary[k] + v)
                supplementary[k] = merged_vals

    # --- variant_graph (canonical + enriched) ---
    variant_graph = {
        "canonical_family_members": list(canonical_comp.get(cp, [cp])),
        "canonical_representative_form": canonical_rep.get(cp),
        "enriched_family_members": list(enriched_comp.get(cp, [cp])),
        "enriched_representative_form": enriched_rep.get(cp),
    }

    # --- source_exclusive ---
    source_exclusive: dict[str, dict] = {}
    for source_name in SOURCE_NAMES:
        rec = source_map.get(source_name)
        if rec is None or not rec.exclusive:
            continue
        source_exclusive[source_name] = dict(rec.exclusive)

    # --- source_payloads ---
    source_payloads: dict[str, dict] = {}
    for source_name in SOURCE_NAMES:
        rec = source_map.get(source_name)
        if rec is None:
            continue
        source_payloads[f"{source_name}_raw"] = rec.source_payload

    record = {
        "character": character,
        "codepoint": cp,
        "source_flags": {s: (s in source_map) for s in SOURCE_NAMES},
        "core": core,
        "provenance": provenance,
        "core_alternatives": alternatives,
        "variants": variants,
        "supplementary_variants": supplementary,
        "variant_graph": variant_graph,
        "source_exclusive": source_exclusive,
        "source_payloads": source_payloads,
    }
    return record


# ---------- Stage E: SQLite export ----------


def build_sqlite(
    sqlite_path: Path,
    canonical_rows: list[dict],
    edge_rows: list[dict],
    component_rows: list[dict],
) -> None:
    if sqlite_path.exists():
        sqlite_path.unlink()
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE characters (
        codepoint TEXT PRIMARY KEY,
        character TEXT NOT NULL,
        radical TEXT,
        total_strokes INTEGER,
        canonical_representative TEXT,
        enriched_representative TEXT,
        data_json TEXT NOT NULL
    );
    CREATE TABLE character_readings (
        codepoint TEXT,
        character TEXT,
        reading_type TEXT,
        value TEXT
    );
    CREATE INDEX idx_readings_cp ON character_readings(codepoint);
    CREATE TABLE character_meanings (
        codepoint TEXT,
        character TEXT,
        language TEXT,
        value TEXT
    );
    CREATE INDEX idx_meanings_cp ON character_meanings(codepoint);
    CREATE TABLE variant_edges (
        source_codepoint TEXT,
        source_character TEXT,
        target_codepoint TEXT,
        target_character TEXT,
        relation_scope TEXT,
        relation TEXT,
        sources_json TEXT,
        source_relations_json TEXT,
        support_count INTEGER
    );
    CREATE INDEX idx_edges_src ON variant_edges(source_codepoint);
    CREATE INDEX idx_edges_tgt ON variant_edges(target_codepoint);
    CREATE TABLE variant_components (
        codepoint TEXT PRIMARY KEY,
        character TEXT,
        representative_form TEXT,
        component_size INTEGER,
        family_members_json TEXT,
        canonical_representative_form TEXT,
        canonical_family_members_json TEXT,
        enriched_representative_form TEXT,
        enriched_family_members_json TEXT
    );
    CREATE TABLE source_presence (
        codepoint TEXT PRIMARY KEY,
        character TEXT,
        unihan INTEGER,
        ehanja_online INTEGER,
        kanjidic2 INTEGER,
        makemeahanzi INTEGER
    );
    CREATE TABLE core_provenance (
        codepoint TEXT,
        character TEXT,
        field_path TEXT,
        source_name TEXT
    );
    CREATE INDEX idx_prov_cp ON core_provenance(codepoint);
    CREATE TABLE core_alternatives (
        codepoint TEXT,
        character TEXT,
        field_path TEXT,
        source_name TEXT,
        value_json TEXT
    );
    CREATE INDEX idx_alt_cp ON core_alternatives(codepoint);
    CREATE TABLE source_exclusive (
        codepoint TEXT,
        character TEXT,
        source_name TEXT,
        field_path TEXT,
        value_json TEXT
    );
    CREATE INDEX idx_excl_cp ON source_exclusive(codepoint);
    CREATE TABLE source_payloads (
        codepoint TEXT,
        character TEXT,
        source_name TEXT,
        payload_json TEXT
    );
    CREATE INDEX idx_payloads_cp ON source_payloads(codepoint);
    """)

    for r in canonical_rows:
        cp = r["codepoint"]; ch = r["character"]
        cur.execute("INSERT INTO characters VALUES (?,?,?,?,?,?,?)", (
            cp, ch,
            r["core"]["radical"], r["core"]["total_strokes"],
            r["variant_graph"].get("canonical_representative_form"),
            r["variant_graph"].get("enriched_representative_form"),
            json.dumps(r, ensure_ascii=False),
        ))
        for rt, vals in r["core"]["readings"].items():
            for v in vals:
                cur.execute("INSERT INTO character_readings VALUES (?,?,?,?)",
                            (cp, ch, rt, v))
        for v in r["core"]["definitions"]["english"]:
            cur.execute("INSERT INTO character_meanings VALUES (?,?,?,?)",
                        (cp, ch, "en", v))
        # source-exclusive korean def if present
        eh_excl = r.get("source_exclusive", {}).get("ehanja_online", {})
        for item in eh_excl.get("korean_explanation", []) or []:
            m = item.get("meaning") if isinstance(item, dict) else None
            if m:
                cur.execute("INSERT INTO character_meanings VALUES (?,?,?,?)",
                            (cp, ch, "ko", m))

        sf = r["source_flags"]
        cur.execute("INSERT INTO source_presence VALUES (?,?,?,?,?,?)", (
            cp, ch,
            int(bool(sf.get("unihan"))),
            int(bool(sf.get("ehanja_online"))),
            int(bool(sf.get("kanjidic2"))),
            int(bool(sf.get("makemeahanzi"))),
        ))

        for field_path, src_name in r.get("provenance", {}).items():
            cur.execute("INSERT INTO core_provenance VALUES (?,?,?,?)",
                        (cp, ch, field_path, src_name))

        for field_path, alt_map in r.get("core_alternatives", {}).items():
            for src_name, val in alt_map.items():
                cur.execute("INSERT INTO core_alternatives VALUES (?,?,?,?,?)",
                            (cp, ch, field_path, src_name,
                             json.dumps(val, ensure_ascii=False)))

        for src_name, excl in r.get("source_exclusive", {}).items():
            for field_path, val in excl.items():
                cur.execute("INSERT INTO source_exclusive VALUES (?,?,?,?,?)",
                            (cp, ch, src_name, field_path,
                             json.dumps(val, ensure_ascii=False)))

        for src_key, payload in r.get("source_payloads", {}).items():
            src_name = src_key.removesuffix("_raw")
            cur.execute("INSERT INTO source_payloads VALUES (?,?,?,?)",
                        (cp, ch, src_name,
                         json.dumps(payload, ensure_ascii=False)))

    for e in edge_rows:
        cur.execute("INSERT INTO variant_edges VALUES (?,?,?,?,?,?,?,?,?)", (
            e["source_codepoint"], e["source_character"],
            e["target_codepoint"], e["target_character"],
            e["relation_scope"], e["relation"],
            json.dumps(e["sources"], ensure_ascii=False),
            json.dumps(e["source_relations"], ensure_ascii=False),
            e["support_count"],
        ))

    for c in component_rows:
        cur.execute("INSERT INTO variant_components VALUES (?,?,?,?,?,?,?,?,?)", (
            c["codepoint"], c["character"],
            c["representative_form"], c["component_size"],
            json.dumps(c["family_members"], ensure_ascii=False),
            c["canonical_representative_form"],
            json.dumps(c["canonical_family_members"], ensure_ascii=False),
            c["enriched_representative_form"],
            json.dumps(c["enriched_family_members"], ensure_ascii=False),
        ))

    con.commit()
    con.close()


# ---------- main ----------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-staging", action="store_true",
                     help="reuse existing staging JSONL instead of rebuilding")
    args = ap.parse_args()

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Stage A: adapters ---
    import time
    def _log(msg: str) -> None:
        print(f"[v2] {msg}", flush=True)

    all_records: dict[str, list[NormalizedSourceRecord]] = {}
    for src_name, build_fn in [
        ("unihan", build_unihan_records),
        ("ehanja_online", build_ehanja_online_records),
        ("kanjidic2", build_kanjidic2_records),
        ("makemeahanzi", build_makemeahanzi_records),
    ]:
        staging_path = STAGING_DIR / f"{src_name}.normalized.jsonl"
        if args.skip_staging and staging_path.exists():
            _log(f"reusing staging {src_name}")
            rows = [NormalizedSourceRecord(**{**r, "readings": r["readings"],
                                               "definitions": r["definitions"],
                                               "variants": r["variants"],
                                               "supplementary_variants": r["supplementary_variants"],
                                               "exclusive": r["exclusive"],
                                               "source_payload": r["source_payload"]})
                    for r in read_jsonl(staging_path)]
            all_records[src_name] = rows
            continue
        t0 = time.perf_counter()
        recs = build_fn()
        _log(f"{src_name} adapter: {len(recs):,} records   ({time.perf_counter()-t0:.1f}s)")
        write_jsonl(staging_path, [r.to_dict() for r in recs])
        all_records[src_name] = recs

    # --- Stage B: identity merge ---
    merged_by_cp = merge_source_records(all_records)
    _log(f"merged: {len(merged_by_cp):,} codepoints")

    # --- Stage C: variant graph ---
    merged_edges, can_rep, can_comp, enr_rep, enr_comp = build_variant_graph(merged_by_cp)
    canonical_edge_count = sum(1 for e in merged_edges if e.relation_scope == "canonical")
    supplementary_edge_count = sum(1 for e in merged_edges if e.relation_scope == "supplementary")
    multi_source_edge_count = sum(1 for e in merged_edges if len(e.sources) > 1)
    _log(f"variant edges: {len(merged_edges):,} "
         f"(canonical {canonical_edge_count:,}, supplementary {supplementary_edge_count:,}, "
         f"multi-source {multi_source_edge_count:,})")

    # --- Stage D: canonical projection ---
    canonical_rows: list[dict] = []
    for cp in sorted(merged_by_cp):
        canonical_rows.append(fill_canonical(cp, merged_by_cp[cp],
                                              can_rep, can_comp, enr_rep, enr_comp))
    _log(f"canonical rows: {len(canonical_rows):,}")
    write_jsonl(OUT_DIR / "canonical_characters.jsonl", canonical_rows)

    edge_rows = [e.to_row() for e in sorted(merged_edges,
                                              key=lambda e: (e.source_codepoint, e.target_codepoint,
                                                              e.relation_scope))]
    write_jsonl(OUT_DIR / "canonical_variants.jsonl", edge_rows)

    component_rows: list[dict] = []
    for cp in sorted(merged_by_cp):
        can = can_comp.get(cp, [cp])
        enr = enr_comp.get(cp, [cp])
        component_rows.append({
            "codepoint": cp, "character": codepoint_to_char(cp),
            "representative_form": can_rep.get(cp),
            "component_size": len(can),
            "family_members": can,
            "canonical_representative_form": can_rep.get(cp),
            "canonical_family_members": can,
            "enriched_representative_form": enr_rep.get(cp),
            "enriched_family_members": enr,
        })
    write_jsonl(OUT_DIR / "variant_components.jsonl", component_rows)

    # --- Stage E: SQLite ---
    sqlite_path = OUT_DIR / "sinograph_canonical_v2.sqlite"
    _log(f"writing SQLite: {sqlite_path}")
    build_sqlite(sqlite_path, canonical_rows, edge_rows, component_rows)

    # --- build_summary.json ---
    source_counts = {s: len(all_records.get(s, [])) for s in SOURCE_NAMES}
    largest_canonical = max((len(c) for c in can_comp.values()), default=0)
    largest_enriched = max((len(c) for c in enr_comp.values()), default=0)
    canonical_comp_count = len({tuple(v) for v in can_comp.values()})
    enriched_comp_count = len({tuple(v) for v in enr_comp.values()})
    smp_rare_count = sum(1 for cp in merged_by_cp
                         if int(cp[2:], 16) >= 0x20000
                         and any(e.source_codepoint == cp or e.target_codepoint == cp
                                  for e in merged_edges))
    summary = {
        "canonical_character_count": len(canonical_rows),
        "source_record_counts": source_counts,
        "variant_edge_count": len(merged_edges),
        "variant_edge_scope_counts": {
            "canonical": canonical_edge_count,
            "supplementary": supplementary_edge_count,
        },
        "multi_source_edge_count": multi_source_edge_count,
        "canonical_variant_component_count": canonical_comp_count,
        "enriched_variant_component_count": enriched_comp_count,
        "max_canonical_component_size": largest_canonical,
        "max_enriched_component_size": largest_enriched,
        "smp_covered_count": smp_rare_count,
        "sample_codepoints": sorted(merged_by_cp)[:10],
    }
    with (OUT_DIR / "build_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    _log(f"build_summary: {summary}")
    _log("done")


if __name__ == "__main__":
    main()
