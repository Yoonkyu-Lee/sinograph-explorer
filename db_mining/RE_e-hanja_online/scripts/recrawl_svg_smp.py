"""
Re-crawl SVG axis for codepoints that 404'd due to the URL folder-mask bug.

Bug
---
Original `fetch_svg` used:
    folder = f"{cp & 0xFF00:X}"
The `0xFF00` mask keeps only bits 8–15, so for SMP codepoints (cp ≥ 0x10000)
bits 16+ are zeroed. Example: U+2B503 → folder "B500" (wrong, should be
"2B500") → every SMP request went to a non-existent path and 404'd.

BMP codepoints were unaffected because they fit in 16 bits — `cp & 0xFF00`
equals `cp & ~0xFF` when cp < 0x10000.

What this script does
---------------------
1. Scan `data/svg/*.404` and select the ones with ≥5 hex-digit codepoints.
2. Re-fetch with the corrected URL (`folder = cp & ~0xFF`).
3. On 200: atomic-write `.svg`, remove the `.404` marker.
4. On real 404: leave `.404` in place (true absence from server).
5. `.svg` files are never touched — already-successful entries skipped.

Safe to run alongside the main detail crawler — different subdomain
(img.e-hanja.kr vs www.e-hanja.kr), no session, no collision.

Usage
    python recrawl_svg_smp.py                  # all SMP 404s
    python recrawl_svg_smp.py --limit 50       # smoke test
    python recrawl_svg_smp.py --concurrency 3  # default 3
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. run: pip install httpx", file=sys.stderr)
    sys.exit(1)


SVG_BASE = "http://img.e-hanja.kr/hanjaSvg/aniSVG"
USER_AGENT = "ECE479-Lab3-Research/1.0 (academic; contact: yoonguri21@gmail.com)"
TIMEOUT = 20.0
MAX_RETRIES = 3

ROOT = Path(__file__).resolve().parent.parent
SVG_DIR = ROOT / "data" / "svg"


def find_smp_404s(min_hex_len: int = 5) -> list[int]:
    """Collect codepoints whose .404 marker exists AND have ≥ min_hex_len hex
    digits (i.e. are in SMP range, where the folder bug struck)."""
    out: list[int] = []
    for p in SVG_DIR.glob("*.404"):
        stem = p.stem
        if len(stem) < min_hex_len:
            continue
        try:
            cp = int(stem, 16)
        except ValueError:
            continue
        out.append(cp)
    out.sort()
    return out


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)


async def fetch_one(client: httpx.AsyncClient, cp: int,
                    sem: asyncio.Semaphore) -> tuple[int, str]:
    """Return (cp, outcome) where outcome ∈ {'fixed', 'still_404', 'error'}."""
    folder = f"{cp & ~0xFF:X}"          # corrected mask
    hex_cp = f"{cp:X}"
    url = f"{SVG_BASE}/{folder}/{hex_cp}.svg"
    svg_path = SVG_DIR / f"{hex_cp}.svg"
    marker_404 = SVG_DIR / f"{hex_cp}.404"

    last_err = ""
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                r = await client.get(url, timeout=TIMEOUT)
        except Exception as e:
            last_err = str(e)
            await asyncio.sleep(0.8 * (attempt + 1))
            continue
        if r.status_code == 200 and r.content:
            _atomic_write_bytes(svg_path, r.content)
            if marker_404.exists():
                marker_404.unlink()
            return cp, "fixed"
        if r.status_code == 404:
            return cp, "still_404"
        if r.status_code in (429, 503):
            last_err = f"HTTP {r.status_code} (throttled)"
            await asyncio.sleep(2.0 * (attempt + 1))
            continue
        last_err = f"HTTP {r.status_code}"
        await asyncio.sleep(0.8 * (attempt + 1))
    return cp, f"error:{last_err}"


async def main_async(args):
    cps = find_smp_404s()
    if args.limit:
        cps = cps[:args.limit]
    total = len(cps)
    if total == 0:
        print("[svg-fix] no SMP .404 markers found — nothing to do")
        return

    print(f"[svg-fix] targets={total}  range=U+{cps[0]:X}..U+{cps[-1]:X}  "
          f"concurrency={args.concurrency}  "
          f"url pattern: folder = cp & ~0xFF (bug-fixed)")

    sem = asyncio.Semaphore(args.concurrency)
    fixed = still_404 = errored = 0
    t0 = time.time()

    # Connection pool sized to concurrency so httpx doesn't bottleneck
    limits = httpx.Limits(max_connections=args.concurrency * 2,
                          max_keepalive_connections=args.concurrency * 2)
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT},
                                  limits=limits) as client:
        tasks = [fetch_one(client, cp, sem) for cp in cps]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            _, outcome = await coro
            if outcome == "fixed":
                fixed += 1
            elif outcome == "still_404":
                still_404 += 1
            else:
                errored += 1
            if i % args.progress_every == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 0.01)
                eta_h = (total - i) / max(rate, 0.01) / 3600
                print(f"[svg-fix] processed={i}/{total}  fixed={fixed}  "
                      f"still_404={still_404}  error={errored}  "
                      f"rate={rate:.2f}/s  eta={eta_h:.2f}h")

    print()
    print(f"[svg-fix] done  fixed={fixed}  still_404={still_404}  "
          f"error={errored}  elapsed={(time.time()-t0)/60:.1f}m")
    if errored:
        print(f"[svg-fix] {errored} still errored — re-run same command to retry "
              f"(.404 markers for those were not modified)")


def main():
    p = argparse.ArgumentParser(
        description="Re-crawl SVG axis for SMP codepoints affected by the "
                    "folder-mask bug. Leaves .svg files untouched.")
    p.add_argument("--concurrency", type=int, default=12,
                   help="parallel in-flight requests (default 12 — "
                        "static CDN tolerates higher than detail axis)")
    p.add_argument("--limit", type=int, default=0,
                   help="process only the first N targets (smoke test)")
    p.add_argument("--progress-every", type=int, default=500,
                   help="log every N processed (default 500)")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
