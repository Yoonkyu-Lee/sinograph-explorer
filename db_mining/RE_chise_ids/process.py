"""Parse CHISE IDS-*.txt → canonical JSONL in db_src/chise_ids/.

CHISE format (per block file):
  ;; header comment
  U+XXXX<TAB>char<TAB>IDS[<TAB>additional_ids...]
  ;@apparent-ancient = …    (metadata lines, skipped)

We concat all IDS-UCS-*.txt files (UCS codepoint sources).
Optional: also include IDS-JIS-*, IDS-Daikanwa-*, IDS-CNS-*, IDS-CBETA.txt.
For now: UCS files only (primary Unicode coverage).

Output:
  ids.jsonl          — full (all columns if multiple IDS listed)
  ids_primary.jsonl  — primary IDS only (first column)
  stats.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR = Path(r"d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/db_mining/RE_chise_ids/data")
OUT_DIR = Path(r"d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/db_src/chise_ids")
OUT_DIR.mkdir(parents=True, exist_ok=True)

UCS_GLOB = "IDS-UCS-*.txt"


def unicode_block(cp: int) -> str:
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
    files = sorted(SRC_DIR.glob(UCS_GLOB))
    if not files:
        raise SystemExit(f"no {UCS_GLOB} in {SRC_DIR}")
    print(f"[src] {len(files)} files:")
    for f in files:
        print(f"  {f.name}")

    out_full = OUT_DIR / "ids.jsonl"
    out_primary = OUT_DIR / "ids_primary.jsonl"
    n_total = 0
    n_multi = 0
    by_block: Counter = Counter()
    cps_seen: set[str] = set()

    with open(out_full, "w", encoding="utf-8") as fw_full, \
         open(out_primary, "w", encoding="utf-8") as fw_prim:
        for src in files:
            with open(src, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if not line or line.startswith(";"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    cp_raw, ch = parts[0], parts[1]
                    # CHISE uses "U+XXXX" for BMP and "U-XXXXXXXX" for supplementary
                    if cp_raw.startswith("U+") or cp_raw.startswith("U-"):
                        try:
                            cp = int(cp_raw[2:], 16)
                        except ValueError:
                            continue
                    else:
                        continue
                    cp_str = f"U+{cp:04X}"
                    # Remaining columns are IDS alternates (may include
                    # source-qualified variants like "@apparent=...").
                    ids_alts: list[str] = []
                    for col in parts[2:]:
                        col = col.strip()
                        if not col or col.startswith("@"):
                            continue
                        # Some columns carry trailing region/source annotations
                        # in brackets [...] — keep as-is for now.
                        ids_alts.append(col)
                    if not ids_alts:
                        continue
                    if cp_str in cps_seen:
                        # dup across files — extend existing with additional
                        # alternates? Simplest: skip dup.
                        continue
                    cps_seen.add(cp_str)
                    n_total += 1
                    if len(ids_alts) > 1:
                        n_multi += 1
                    by_block[unicode_block(cp)] += 1

                    fw_full.write(json.dumps({
                        "codepoint": cp_str, "char": ch, "ids": ids_alts,
                    }, ensure_ascii=False) + "\n")
                    fw_prim.write(json.dumps({
                        "codepoint": cp_str, "char": ch, "ids": ids_alts[0],
                    }, ensure_ascii=False) + "\n")

    stats = {
        "source": "CHISE IDS (UCS files)",
        "source_files": [f.name for f in files],
        "total_entries": n_total,
        "unique_codepoints": len(cps_seen),
        "multi_alternate_entries": n_multi,
        "by_block": dict(sorted(by_block.items(), key=lambda x: -x[1])),
    }
    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[out] {out_full}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
