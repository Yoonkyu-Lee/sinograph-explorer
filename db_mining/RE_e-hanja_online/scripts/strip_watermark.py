"""
Copy every SVG from the crawl output to `db_src/e-hanja_online/svg/` with
the "(c)2020.(e-hanja)" text watermark stripped.

Watermark format (exact, trailing newline irrelevant):
    <text x="512" y="1108" text-anchor="middle" fill="#aaa"
          font-family="Segoe UI" font-size="88px" pointer-events="none"
    >&#169;2020.(e-hanja)</text>

A single `<text ...>...e-hanja...</text>` line appears right before `</svg>`
in every file seen so far. We drop it with a regex — no XML parsing needed.
Bytes outside the watermark are preserved verbatim.

The original crawl directory is left untouched. Only `.404` markers and
errors are skipped; actual SVGs are written under the clean output root.

Idempotent: already-existing destination files are skipped. Re-run is safe.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "svg"
DST_DIR = ROOT.parent.parent / "db_src" / "e-hanja_online" / "svg"
DST_DIR.mkdir(parents=True, exist_ok=True)

# Matches the watermark element. We deliberately tolerate slight whitespace
# variation but require the literal "e-hanja)" fingerprint to avoid eating
# unrelated text nodes.
WATERMARK = re.compile(rb'<text\b[^>]*>&#169;2020\.\(e-hanja\)</text>\s*')


def strip(svg_bytes: bytes) -> tuple[bytes, bool]:
    """Return (clean_bytes, had_watermark)."""
    new, n = WATERMARK.subn(b"", svg_bytes, count=1)
    return new, n > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process first N (smoke test)")
    ap.add_argument("--progress-every", type=int, default=2000)
    ap.add_argument("--overwrite", action="store_true",
                    help="rewrite destination even if it exists")
    args = ap.parse_args()

    all_svgs = sorted(SRC_DIR.glob("*.svg"))
    if args.limit:
        all_svgs = all_svgs[:args.limit]
    total = len(all_svgs)
    print(f"stripping watermarks from {total:,} SVGs")
    print(f"  src: {SRC_DIR}")
    print(f"  dst: {DST_DIR}")
    print()

    t0 = time.time()
    copied = 0
    skipped = 0
    no_watermark = 0

    for i, src in enumerate(all_svgs, 1):
        dst = DST_DIR / src.name
        if dst.exists() and not args.overwrite:
            skipped += 1
        else:
            raw = src.read_bytes()
            clean, had = strip(raw)
            if not had:
                no_watermark += 1
            dst.write_bytes(clean)
            copied += 1
        if i % args.progress_every == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / max(elapsed, 0.01)
            eta = (total - i) / max(rate, 0.01)
            print(f"  {i:,}/{total:,}  copied={copied:,}  skipped={skipped:,}  "
                  f"no_watermark={no_watermark:,}  rate={rate:.0f}/s  eta={eta:.0f}s")

    print()
    print(f"done  elapsed={(time.time()-t0):.1f}s")
    print(f"  copied:        {copied:,}")
    print(f"  skipped:       {skipped:,}  (existed already; pass --overwrite to rewrite)")
    print(f"  no_watermark:  {no_watermark:,}  (no text element matching the fingerprint)")


if __name__ == "__main__":
    main()
