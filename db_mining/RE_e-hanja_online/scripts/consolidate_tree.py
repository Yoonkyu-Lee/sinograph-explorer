"""Consolidate 76k per-char tree JSONs into a single JSONL under db_src/.

Input:
  db_mining/RE_e-hanja_online/data/tree/{HEX}.json    success (76,013)
  db_mining/RE_e-hanja_online/data/tree/{HEX}.404     404 markers (83)

Each success file is already the raw getHunum + getJahae + getSchoolCom dict.
We add `cp`, `hex`, `char` at the top level for joinability but leave the
original fields (`unicode`, `getHunum`, `getJahae`, `getSchoolCom`) untouched
so the v2 adapter can consume them as-is.

Output:
  db_src/e-hanja_online/tree.jsonl        success records
  db_src/e-hanja_online/tree_404.jsonl    404 codepoints (audit)
"""
from __future__ import annotations

import json
from pathlib import Path

IN = Path("d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/db_mining/RE_e-hanja_online/data/tree")
OUT_DIR = Path("d:/Library/01 Admissions/01 UIUC/4-2 SP26/ECE 479/lab3/db_src/e-hanja_online")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "tree.jsonl"
OUT_404 = OUT_DIR / "tree_404.jsonl"


def main() -> None:
    success = 0
    missed = 0
    bad = 0
    hex_set: set[str] = set()

    with open(OUT_JSONL, "w", encoding="utf-8") as fout, \
         open(OUT_404, "w", encoding="utf-8") as fout404:
        for entry in sorted(IN.iterdir()):
            name = entry.name
            if name.endswith(".json"):
                hex_part = name[:-5]
                try:
                    cp = int(hex_part, 16)
                except ValueError:
                    bad += 1
                    continue
                try:
                    with open(entry, "r", encoding="utf-8") as f:
                        rec = json.load(f)
                except Exception as e:
                    bad += 1
                    print(f"[skip] {name}: {e}")
                    continue
                rec_out = {"cp": cp, "hex": hex_part.upper(), "char": chr(cp)}
                # preserve original keys
                for k, v in rec.items():
                    rec_out[k] = v
                fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
                hex_set.add(hex_part.upper())
                success += 1
            elif name.endswith(".404"):
                hex_part = name[:-4]
                try:
                    cp = int(hex_part, 16)
                except ValueError:
                    bad += 1
                    continue
                fout404.write(json.dumps({"cp": cp, "hex": hex_part.upper(), "char": chr(cp)},
                                          ensure_ascii=False) + "\n")
                missed += 1

    print(f"success={success:,}  missed(404)={missed}  malformed={bad}")
    print(f"wrote: {OUT_JSONL}")
    print(f"wrote: {OUT_404}")


if __name__ == "__main__":
    main()
