"""
Download N random images from picsum.photos to samples/backgrounds/.

Picsum is a free random-image service backed by public stock photos
(Unsplash origin), no API key required. Images come as JPG redirects.

Usage:
    python fetch_picsum_backgrounds.py                 # default 50 images, 800x800
    python fetch_picsum_backgrounds.py --count 100     # more images
    python fetch_picsum_backgrounds.py --size 1024     # larger resolution
    python fetch_picsum_backgrounds.py --prefix photo  # filename prefix

Existing files with same name are skipped (resumable).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import httpx
except ImportError:
    print("ERROR: install httpx first:  pip install httpx", file=sys.stderr)
    sys.exit(1)


OUT = Path(__file__).resolve().parent.parent / "samples" / "backgrounds"
OUT.mkdir(parents=True, exist_ok=True)

USER_AGENT = "ECE479-Lab3-Research/1.0 (academic; contact: yoonguri21@gmail.com)"
URL_TMPL = "https://picsum.photos/seed/{seed}/{w}/{h}"


async def fetch_one(client, seed: str, w: int, h: int, out_path: Path,
                    max_retries: int = 3) -> tuple[bool, int | str]:
    if out_path.exists():
        return True, -1  # already have it
    url = URL_TMPL.format(seed=seed, w=w, h=h)
    for attempt in range(max_retries):
        try:
            r = await client.get(url, timeout=45.0, follow_redirects=True)
            if r.status_code == 200 and len(r.content) > 1000:
                out_path.write_bytes(r.content)
                return True, len(r.content)
            await asyncio.sleep(1.0 * (attempt + 1))
        except Exception:
            await asyncio.sleep(1.0 * (attempt + 1))
    return False, "failed after retries"


async def main_async(args):
    sem = asyncio.Semaphore(args.concurrency)
    start = time.time()

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        async def bounded(i):
            seed = f"{args.prefix}{i:04d}"
            out_path = OUT / f"{args.prefix}_{i:03d}.jpg"
            async with sem:
                return i, await fetch_one(client, seed, args.size, args.size, out_path)

        tasks = [bounded(i) for i in range(args.count)]
        ok = 0
        skipped = 0
        failed: list[int] = []
        for coro in asyncio.as_completed(tasks):
            i, (success, info) = await coro
            if success:
                if info == -1:
                    skipped += 1
                else:
                    ok += 1
                    if (ok + skipped) % 10 == 0:
                        elapsed = time.time() - start
                        print(f"  {ok+skipped}/{args.count}  ({ok} new, {skipped} skipped)  elapsed={elapsed:.1f}s")
            else:
                failed.append(i)

    print()
    print(f"done: {ok} new, {skipped} skipped, {len(failed)} failed")
    if failed:
        print(f"  failed indices: {failed[:10]}{'...' if len(failed)>10 else ''}")
    total = sum(1 for p in OUT.glob("*.jpg"))
    print(f"total in {OUT}: {total} images")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--size", type=int, default=800)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--prefix", default="picsum")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
