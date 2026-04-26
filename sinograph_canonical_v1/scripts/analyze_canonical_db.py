from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BUILD_ROOT = SCRIPT_DIR.parent
DEFAULT_JSONL = BUILD_ROOT / "out" / "canonical_characters.jsonl"
DEFAULT_VARIANTS_JSONL = BUILD_ROOT / "out" / "canonical_variants.jsonl"

SOURCE_ORDER = ("unihan", "ehanja", "kanjidic2", "makemeahanzi")
SOURCE_ABBREV = {
    "unihan": "U",
    "ehanja": "E",
    "kanjidic2": "K",
    "makemeahanzi": "M",
}

READING_KEYS = (
    "mandarin",
    "cantonese",
    "korean_hangul",
    "korean_romanized",
    "japanese_on",
    "japanese_kun",
    "vietnamese",
)

DEFINITION_KEYS = (
    "english",
    "korean_explanation",
    "korean_hun",
)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count * 100.0 / total, 2)


def combo_key(flags: dict[str, bool]) -> str:
    key = "".join(SOURCE_ABBREV[name] for name in SOURCE_ORDER if flags.get(name))
    return key or "(none)"


def compute_stats(records: list[dict[str, Any]], variant_edges: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    total = len(records)
    combo_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    field_presence: Counter[str] = Counter()
    enriched_growth = 0
    canonical_component_reps: Counter[str] = Counter()
    enriched_component_reps: Counter[str] = Counter()

    for record in records:
        flags = record["source_flags"]
        core = record["core"]
        variants = record["variants"]
        supplementary = record.get("supplementary_variants", {})
        variant_graph = record.get("variant_graph", {})
        structure = record["structure"]
        media = record["media"]

        combo_counts[combo_key(flags)] += 1
        for source_name in SOURCE_ORDER:
            if flags.get(source_name):
                source_counts[source_name] += 1

        for left_index, left_name in enumerate(SOURCE_ORDER):
            for right_name in SOURCE_ORDER[left_index + 1 :]:
                if flags.get(left_name) and flags.get(right_name):
                    pair_counts[f"{left_name}_{right_name}"] += 1

        if core["radical"] is not None:
            field_presence["radical"] += 1
        if core["total_strokes"] is not None:
            field_presence["total_strokes"] += 1

        for field_name in DEFINITION_KEYS:
            if core["definitions"].get(field_name):
                field_presence[f"definition_{field_name}"] += 1

        for field_name in READING_KEYS:
            if core["readings"].get(field_name):
                field_presence[f"reading_{field_name}"] += 1

        if structure.get("decomposition"):
            field_presence["structure_decomposition"] += 1
        if structure.get("etymology_type"):
            field_presence["structure_etymology_type"] += 1
        if structure.get("etymology_hint"):
            field_presence["structure_etymology_hint"] += 1
        if structure.get("phonetic_component"):
            field_presence["structure_phonetic_component"] += 1
        if structure.get("semantic_component"):
            field_presence["structure_semantic_component"] += 1

        if media.get("stroke_svg_paths"):
            field_presence["media_stroke_svg_paths"] += 1
        if media.get("stroke_medians"):
            field_presence["media_stroke_medians"] += 1

        family_members = variants.get("family_members", [])
        if family_members:
            field_presence["variants_has_family"] += 1
        if len(family_members) > 1:
            field_presence["variants_family_gt1"] += 1
        canonical_component_reps[variant_graph.get("canonical_representative_form", record["codepoint"])] += 1
        enriched_component_reps[variant_graph.get("enriched_representative_form", record["codepoint"])] += 1
        if len(variant_graph.get("enriched_family_members", [])) > len(variant_graph.get("canonical_family_members", [])):
            enriched_growth += 1
        for variant_key in (
            "traditional",
            "simplified",
            "semantic",
            "specialized_semantic",
            "z_variants",
            "spoofing",
        ):
            if variants.get(variant_key):
                field_presence[f"variant_{variant_key}"] += 1
        for variant_key, values in supplementary.items():
            if values:
                field_presence[f"supplementary_{variant_key}"] += 1

    supplementary_edge_source_counts: dict[str, dict[str, float | int]] = {}
    edge_scope_counts: dict[str, dict[str, float | int]] = {}
    if variant_edges is not None:
        scope_counter = Counter(edge["relation_scope"] for edge in variant_edges)
        supplementary_source_counter = Counter(
            edge["source_name"] for edge in variant_edges if edge["relation_scope"] == "supplementary"
        )
        edge_scope_counts = {
            key: {"count": count, "pct": pct(count, len(variant_edges))}
            for key, count in sorted(scope_counter.items())
        }
        supplementary_edge_source_counts = {
            key: {"count": count, "pct": pct(count, sum(supplementary_source_counter.values()))}
            for key, count in sorted(supplementary_source_counter.items())
        }

    combo_stats = [
        {"combo": key, "count": count, "pct": pct(count, total)}
        for key, count in combo_counts.most_common()
    ]
    pair_stats = {
        key: {"count": count, "pct": pct(count, total)}
        for key, count in sorted(pair_counts.items())
    }
    source_stats = {
        key: {"count": count, "pct": pct(count, total)}
        for key, count in source_counts.items()
    }
    field_stats = {
        key: {"count": count, "pct": pct(count, total)}
        for key, count in sorted(field_presence.items())
    }

    return {
        "total_records": total,
        "source_presence": source_stats,
        "source_combinations": combo_stats,
        "pair_overlaps": pair_stats,
        "field_presence": field_stats,
        "variant_graph": {
            "canonical_component_count": len(canonical_component_reps),
            "enriched_component_count": len(enriched_component_reps),
            "characters_with_enriched_growth": {
                "count": enriched_growth,
                "pct": pct(enriched_growth, total),
            },
        },
        "variant_edge_scopes": edge_scope_counts,
        "supplementary_edge_sources": supplementary_edge_source_counts,
        "highlights": {
            "all_four_sources": combo_counts["UEKM"],
            "unihan_only": combo_counts["U"],
            "unihan_ehanja_kanjidic2": combo_counts["UEK"],
            "unihan_kanjidic2_makemeahanzi": combo_counts["UKM"],
        },
    }


def print_human_summary(stats: dict[str, Any]) -> None:
    total = stats["total_records"]
    print("=== Sinograph Canonical DB v1 Coverage Audit ===")
    print(f"Total canonical records: {total:,}")
    print()

    print("[Source presence]")
    for source_name in SOURCE_ORDER:
        item = stats["source_presence"].get(source_name, {"count": 0, "pct": 0.0})
        print(f"- {source_name:<13} {item['count']:>7,} / {total:,} ({item['pct']:>6.2f}%)")
    print()

    print("[Source combinations]")
    for item in stats["source_combinations"]:
        print(f"- {item['combo']:<6} {item['count']:>7,} / {total:,} ({item['pct']:>6.2f}%)")
    print()

    print("[Pair overlaps]")
    for pair_name, item in stats["pair_overlaps"].items():
        print(f"- {pair_name:<24} {item['count']:>7,} / {total:,} ({item['pct']:>6.2f}%)")
    print()

    print("[Field presence]")
    for field_name, item in stats["field_presence"].items():
        print(f"- {field_name:<30} {item['count']:>7,} / {total:,} ({item['pct']:>6.2f}%)")
    print()

    print("[Variant graph]")
    print(f"- canonical_component_count       {stats['variant_graph']['canonical_component_count']:>7,}")
    print(f"- enriched_component_count        {stats['variant_graph']['enriched_component_count']:>7,}")
    growth = stats["variant_graph"]["characters_with_enriched_growth"]
    print(f"- characters_with_enriched_growth {growth['count']:>7,} / {total:,} ({growth['pct']:>6.2f}%)")
    print()

    if stats["variant_edge_scopes"]:
        print("[Variant edge scopes]")
        for scope_name, item in stats["variant_edge_scopes"].items():
            print(f"- {scope_name:<30} {item['count']:>7,} / {sum(v['count'] for v in stats['variant_edge_scopes'].values()):,} ({item['pct']:>6.2f}%)")
        print()

    if stats["supplementary_edge_sources"]:
        print("[Supplementary edge sources]")
        total_supp = sum(v["count"] for v in stats["supplementary_edge_sources"].values())
        for source_name, item in stats["supplementary_edge_sources"].items():
            print(f"- {source_name:<30} {item['count']:>7,} / {total_supp:,} ({item['pct']:>6.2f}%)")
        print()

    print("[Highlights]")
    for key, count in stats["highlights"].items():
        print(f"- {key:<30} {count:>7,} / {total:,} ({pct(count, total):>6.2f}%)")


def main() -> int:
    configure_stdout()

    parser = argparse.ArgumentParser(
        description="Analyze coverage/intersection statistics for the Sinograph Canonical DB v1 JSONL artifact."
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help=f"Path to canonical_characters.jsonl (default: {DEFAULT_JSONL})",
    )
    parser.add_argument(
        "--variants-jsonl",
        type=Path,
        default=DEFAULT_VARIANTS_JSONL,
        help=f"Path to canonical_variants.jsonl (default: {DEFAULT_VARIANTS_JSONL})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON stats instead of the human-readable summary.",
    )
    args = parser.parse_args()

    if not args.jsonl.exists():
        print(f"Canonical JSONL not found: {args.jsonl}")
        print("Run build_canonical_db.py first.")
        return 1

    records = load_records(args.jsonl)
    variant_edges = load_records(args.variants_jsonl) if args.variants_jsonl.exists() else None
    stats = compute_stats(records, variant_edges)

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_human_summary(stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
