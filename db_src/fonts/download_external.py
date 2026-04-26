"""Download external CJK fonts (OFL / SIL) into db_src/fonts/external/.

Reads DOWNLOAD_MANIFEST.json, fetches each URL once (skips if file already
exists and has nonzero size), writes to target_dir. Verifies each downloaded
file is a valid TTF / OTF by peeking the magic bytes.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
MANIFEST = HERE / "DOWNLOAD_MANIFEST.json"

MAGIC_TTF = b"\x00\x01\x00\x00"
MAGIC_OTF = b"OTTO"
MAGIC_TTC = b"ttcf"


def is_font_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
        return magic in (MAGIC_TTF, MAGIC_OTF, MAGIC_TTC)
    except Exception:
        return False


def main() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    target_dir = HERE.parent.parent / data["target_dir"]
    target_dir.mkdir(parents=True, exist_ok=True)

    ok, skip, fail = 0, 0, 0
    for entry in data["fonts"]:
        out = target_dir / entry["filename"]
        if out.exists() and out.stat().st_size > 0 and is_font_file(out):
            print(f"  [skip] {entry['filename']}  ({out.stat().st_size:,} bytes, already ok)")
            skip += 1
            continue

        url = entry["url"]
        print(f"  [get ] {entry['filename']}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "synth_engine_v3/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                blob = resp.read()
            if not blob:
                raise RuntimeError("empty response")
            out.write_bytes(blob)
            if not is_font_file(out):
                raise RuntimeError(f"not a valid font file (magic={blob[:4]!r})")
            print(f"         {len(blob):,} bytes  ok")
            ok += 1
        except (urllib.error.URLError, RuntimeError, TimeoutError) as e:
            print(f"         FAILED: {e}", file=sys.stderr)
            fail += 1

    summary = {"ok": ok, "skip": skip, "fail": fail, "total": len(data["fonts"])}
    (target_dir.parent / "download_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSummary: ok={ok}  skip={skip}  fail={fail}  of {len(data['fonts'])}")


if __name__ == "__main__":
    main()
