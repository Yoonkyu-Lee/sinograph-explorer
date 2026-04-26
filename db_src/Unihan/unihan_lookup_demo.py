import argparse
from collections import deque
import re
import sys
from pathlib import Path


UNIHAN_DIR = Path(__file__).resolve().parent / "Unihan_txt"

# Fields that are most useful for the current project demo.
WANTED_FIELDS = {
    "kDefinition",
    "kMandarin",
    "kCantonese",
    "kJapanese",
    "kJapaneseOn",
    "kJapaneseKun",
    "kKorean",
    "kTotalStrokes",
    "kRSUnicode",
    "kTraditionalVariant",
    "kSimplifiedVariant",
    "kSemanticVariant",
    "kSpecializedSemanticVariant",
    "kSpoofingVariant",
    "kZVariant",
    "kIRGKangXi",
    "kKangXi",
    "kHanYu",
    "kUnihanCore2020",
}

# Unihan_Variants.txt 안에서 현재 프로젝트에 의미 있는 variant 관계들.
VARIANT_FIELDS = (
    "kTraditionalVariant",
    "kSimplifiedVariant",
    "kSemanticVariant",
    "kSpecializedSemanticVariant",
    "kSpoofingVariant",
    "kZVariant",
)


def parse_unihan_subset(unihan_dir: Path) -> dict[str, dict[str, str]]:
    db: dict[str, dict[str, str]] = {}

    # Unihan의 여러 txt 파일을 한 번에 읽어서
    # "코드포인트 -> 필요한 속성들" 형태의 작은 DB로 합친다.
    for path in sorted(unihan_dir.glob("Unihan_*.txt")):
        with path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                codepoint, field, value = line.split("\t", 2)
                if field not in WANTED_FIELDS:
                    continue

                entry = db.setdefault(codepoint, {})
                entry["char"] = chr(int(codepoint[2:], 16))
                entry[field] = value

    return db


def variant_codepoints(value: str) -> list[str]:
    # 예: "U+5B78<kLau,kMatthews>" 같은 값에서
    # 실제 코드포인트(U+XXXX)만 뽑아낸다.
    # Values can look like: "U+5B78<kLau,kMatthews,kMeyerWempe>"
    return re.findall(r"U\+[0-9A-F]{4,6}", value)


def cp_to_char(cp: str) -> str:
    return chr(int(cp[2:], 16))


def build_variant_graph(db: dict[str, dict[str, str]]) -> dict[str, dict[str, set[str]]]:
    graph: dict[str, dict[str, set[str]]] = {}

    # Unihan의 variant 관계를 "문자 간 연결 그래프"로 바꾼다.
    # 여기서는 관계를 무방향처럼 다뤄서, 역방향으로만 적혀 있는 문자도 함께 찾을 수 있게 한다.
    for cp, entry in db.items():
        for field in VARIANT_FIELDS:
            raw_value = entry.get(field)
            if not raw_value:
                continue

            for target_cp in variant_codepoints(raw_value):
                graph.setdefault(cp, {}).setdefault(field, set()).add(target_cp)
                graph.setdefault(target_cp, {}).setdefault(field, set()).add(cp)

    return graph


def traverse_variant_component(
    start_cp: str,
    graph: dict[str, dict[str, set[str]]],
) -> tuple[list[str], list[tuple[str, str, str]]]:
    if start_cp not in graph:
        return [start_cp], []

    # 재귀 대신 BFS를 사용해 연결된 variant 문자군 전체를 찾는다.
    # visited가 있기 때문에 순환 참조가 있어도 무한 루프에 빠지지 않는다.
    queue = deque([start_cp])
    visited = {start_cp}
    order = [start_cp]
    edges: set[tuple[str, str, str]] = set()

    while queue:
        current = queue.popleft()
        for field in sorted(graph.get(current, {})):
            for neighbor in sorted(graph[current][field]):
                edge_key = tuple(sorted((current, neighbor)) + [field])
                edges.add(edge_key)

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    order.append(neighbor)

    return order, sorted(edges)


def describe_lookup(ch: str, db: dict[str, dict[str, str]]) -> str:
    cp = f"U+{ord(ch):04X}"
    entry = db.get(cp)
    variant_graph = build_variant_graph(db)

    # 최종적으로 터미널에 보여줄 설명 텍스트를 줄 단위로 쌓는다.
    lines: list[str] = []
    lines.append(f"Input character : {ch}")
    lines.append(f"Unicode codepoint: {cp}")

    if not entry:
        lines.append("Status          : Not found in loaded Unihan subset")
        return "\n".join(lines)

    lines.append("Status          : Found in Unihan")
    lines.append("")
    lines.append("Basic info")
    lines.append(f"  Definition    : {entry.get('kDefinition', '(none)')}")
    lines.append(f"  Mandarin      : {entry.get('kMandarin', '(none)')}")
    lines.append(f"  Cantonese     : {entry.get('kCantonese', '(none)')}")
    lines.append(f"  Japanese      : {entry.get('kJapanese', '(none)')}")
    lines.append(f"  Japanese On   : {entry.get('kJapaneseOn', '(none)')}")
    lines.append(f"  Japanese Kun  : {entry.get('kJapaneseKun', '(none)')}")
    lines.append(f"  Korean        : {entry.get('kKorean', '(none)')}")
    lines.append(f"  Total strokes : {entry.get('kTotalStrokes', '(none)')}")
    lines.append(f"  Radical/Stroke: {entry.get('kRSUnicode', '(none)')}")
    lines.append(f"  Unihan Core   : {entry.get('kUnihanCore2020', '(none)')}")

    lines.append("")
    lines.append("Variant relations")
    # 현재는 Unihan_Variants.txt에 있는 대표적인 variant 필드들을 모두 보여준다.
    for field in VARIANT_FIELDS:
        raw_value = entry.get(field)
        if not raw_value:
            lines.append(f"  {field:<18}: (none)")
            continue

        cps = variant_codepoints(raw_value)
        pretty = ", ".join(f"{cp_to_char(v)} ({v})" for v in cps) if cps else raw_value
        lines.append(f"  {field:<18}: {pretty}")

    lines.append("")
    lines.append("Dictionary references")
    lines.append(f"  KangXi        : {entry.get('kKangXi', '(none)')}")
    lines.append(f"  IRG KangXi    : {entry.get('kIRGKangXi', '(none)')}")
    lines.append(f"  HanYu         : {entry.get('kHanYu', '(none)')}")

    lines.append("")
    lines.append("Variant component search (BFS over variant graph)")
    lines.append("  Algorithm     : Treat each variant relation as an undirected edge and")
    lines.append("                  run BFS from the input character to collect all linked variants.")
    component, edges = traverse_variant_component(cp, variant_graph)
    pretty_component = " -> ".join(f"{cp_to_char(v)} ({v})" for v in component)
    lines.append(f"  Visit order   : {pretty_component}")

    if edges:
        lines.append("  Discovered edges:")
        for left, right, field in edges:
            lines.append(f"    {cp_to_char(left)} ({left}) --{field}-- {cp_to_char(right)} ({right})")

    if len(component) > 1:
        lines.append("")
        lines.append("Linked character summaries")
        for linked_cp in component[1:]:
            linked_entry = db.get(linked_cp, {})
            lines.append(
                f"  {cp_to_char(linked_cp)} ({linked_cp})"
                f" | Definition: {linked_entry.get('kDefinition', '(none)')}"
                f" | Mandarin: {linked_entry.get('kMandarin', '(none)')}"
            )

    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        # Windows 콘솔에서 한자가 깨지지 않도록 UTF-8 출력으로 고정.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Demo lookup of a recognized character against local Unihan text files."
    )
    parser.add_argument(
        "character",
        nargs="?",
        default="斈",
        help="Single character to look up. Default: 斈",
    )
    args = parser.parse_args()

    if len(args.character) != 1:
        raise SystemExit("Please pass exactly one character, e.g. python unihan_lookup_demo.py 斈")

    db = parse_unihan_subset(UNIHAN_DIR)
    print(describe_lookup(args.character, db))


if __name__ == "__main__":
    main()
