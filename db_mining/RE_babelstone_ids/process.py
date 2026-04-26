"""Parse BabelStone IDS.TXT → canonical JSONL in db_src/babelstone_ids/.

BabelStone format (after header comment):
  U+XXXX<TAB>char<TAB>^<IDS>$(region_flags)[<TAB>^<IDS>$(...)...]

Region flags: G=PRC, H=HK, T=Taiwan, J=Japan, K=Korea, P=PRK/North-Korea,
V=Vietnam, X=extended/minor. Multiple columns mean region-specific alternates.

Output (db_src/babelstone_ids/):
  ids.jsonl  — one line per codepoint:
    {"codepoint":"U+XXXX","char":"…","ids":["…"],"regions":[["G","H","T","J","K","P"], …]}
  ids_primary.jsonl — "primary" IDS (first alternate) only, simpler format:
    {"codepoint":"U+XXXX","char":"…","ids":"…"}
  stats.json — {total_entries, codepoints_unique, by_block, …}
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(r"d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/db_mining/RE_babelstone_ids/data/IDS.TXT")
OUT_DIR = Path(r"d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/db_src/babelstone_ids")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# BabelStone IDS column pattern: ^<IDS>$(<flags>)
IDS_COL_RE = re.compile(r"\^(.*?)\$\(([A-Z]+)\)")


def unicode_block(cp: int) -> str:
    """Crude block classifier."""
    if 0x4E00 <= cp <= 0x9FFF:    return "CJK_Unified"
    if 0x3400 <= cp <= 0x4DBF:    return "CJK_Ext_A"
    if 0x20000 <= cp <= 0x2A6DF:  return "CJK_Ext_B"
    if 0x2A700 <= cp <= 0x2B73F:  return "CJK_Ext_C"
    if 0x2B740 <= cp <= 0x2B81F:  return "CJK_Ext_D"
    if 0x2B820 <= cp <= 0x2CEAF:  return "CJK_Ext_E"
    if 0x2CEB0 <= cp <= 0x2EBEF:  return "CJK_Ext_F"
    if 0x30000 <= cp <= 0x3134F:  return "CJK_Ext_G"
    if 0x31350 <= cp <= 0x323AF:  return "CJK_Ext_H"
    if 0x2EBF0 <= cp <= 0x2EE5F:  return "CJK_Ext_I"
    if 0x323B0 <= cp <= 0x33479:  return "CJK_Ext_J"
    if 0xF900 <= cp <= 0xFAFF:    return "CJK_Compat"
    if 0x2F800 <= cp <= 0x2FA1F:  return "CJK_Compat_Supp"
    if 0x2E80 <= cp <= 0x2EFF:    return "CJK_Radicals_Supp"
    if 0x2F00 <= cp <= 0x2FDF:    return "Kangxi_Radicals"
    return "Other"


def main() -> None:
    assert SRC.exists(), SRC
    print(f"[src] {SRC}  size={SRC.stat().st_size/1024/1024:.2f} MB")

    out_full = OUT_DIR / "ids.jsonl"
    out_primary = OUT_DIR / "ids_primary.jsonl"
    n_total = 0
    n_multi = 0
    by_block: Counter = Counter()
    cps_seen: set[str] = set()

    with open(SRC, encoding="utf-8-sig") as f, \
         open(out_full, "w", encoding="utf-8") as fw_full, \
         open(out_primary, "w", encoding="utf-8") as fw_prim:
        for line in f:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            cp_str, ch = parts[0], parts[1]
            if not cp_str.startswith("U+"):
                continue
            try:
                cp = int(cp_str[2:], 16)
            except ValueError:
                continue

            ids_alts: list[str] = []
            regions: list[list[str]] = []
            for col in parts[2:]:
                for m in IDS_COL_RE.finditer(col):
                    ids_str, flags = m.group(1), m.group(2)
                    ids_alts.append(ids_str)
                    regions.append(sorted(set(flags)))

            if not ids_alts:
                continue

            cps_seen.add(cp_str)
            n_total += 1
            if len(ids_alts) > 1:
                n_multi += 1
            by_block[unicode_block(cp)] += 1

            fw_full.write(json.dumps({
                "codepoint": cp_str, "char": ch,
                "ids": ids_alts, "regions": regions,
            }, ensure_ascii=False) + "\n")
            fw_prim.write(json.dumps({
                "codepoint": cp_str, "char": ch, "ids": ids_alts[0],
            }, ensure_ascii=False) + "\n")

    stats = {
        "source": "BabelStone IDS",
        "source_file": str(SRC),
        "total_entries": n_total,
        "unique_codepoints": len(cps_seen),
        "multi_alternate_entries": n_multi,
        "by_block": dict(sorted(by_block.items(), key=lambda x: -x[1])),
    }
    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[out] {out_full}")
    print(f"[out] {out_primary}")
    print(f"[out] {OUT_DIR / 'stats.json'}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
