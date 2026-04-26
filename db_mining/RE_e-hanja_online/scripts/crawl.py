"""
e-hanja online crawler — resumable, rate-limited, per-axis.

Phases (see COLLECTION_PLAN.md):
  - svg    : GET  img.e-hanja.kr/hanjaSvg/aniSVG/{folder}/{cp}.svg
  - tree   : POST tool.img.e-hanja.kr/hanjaSvg/asp/dbHandle.comsTree.asp  (3 types combined)
  - detail : POST www.e-hanja.kr/dic/contents/jajun_contentA.asp  (session required)

Usage
    python crawl.py --axis svg
    python crawl.py --axis tree
    python crawl.py --axis detail
    python crawl.py --axis all
    python crawl.py --axis svg --limit 50        # smoke test
    python crawl.py --axis all --concurrency 3 --delay 0.3

Resume semantics
    Each codepoint × axis produces either:
      data/{axis}/{cp:X}.{ext}       on 200 success
      data/{axis}/{cp:X}.404         on 404 (no content)
      data/{axis}/{cp:X}.err         on persistent failure (with error message)
    Re-running skips any codepoint that already has one of these three markers.

Safety
    Ctrl+C is clean — the crawler stops after the current in-flight requests.
    Next run resumes where it left off.

Dependencies
    pip install httpx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import signal
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. run: pip install httpx", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from codepoints import enumerate_codepoints, total_count


# -------- configuration ------------------------------------------------------

USER_AGENT = "ECE479-Lab3-Research/1.0 (academic; contact: yoonguri21@gmail.com)"
TIMEOUT = 20.0
MAX_RETRIES = 3

SVG_BASE = "http://img.e-hanja.kr/hanjaSvg/aniSVG"
TREE_URL = "http://tool.img.e-hanja.kr/hanjaSvg/asp/dbHandle.comsTree.asp"
DETAIL_SHELL = "http://www.e-hanja.kr/dic/contents/jajun_content.asp"
DETAIL_AJAX = "http://www.e-hanja.kr/dic/contents/jajun_contentA.asp"
WWW_SEED = "http://www.e-hanja.kr/dic/dictionary.asp"

AXIS_EXT = {"svg": "svg", "tree": "json", "detail": "html"}


# -------- paths --------------------------------------------------------------


def out_dir(root: Path, axis: str) -> Path:
    d = root / "data" / axis
    d.mkdir(parents=True, exist_ok=True)
    return d


def status_of(axis_dir: Path, cp: int) -> str:
    """Return one of: 'done', 'notfound', 'error', 'pending'."""
    hex_cp = f"{cp:X}"
    ext = AXIS_EXT[axis_dir.name]
    if (axis_dir / f"{hex_cp}.{ext}").exists():
        return "done"
    if (axis_dir / f"{hex_cp}.404").exists():
        return "notfound"
    if (axis_dir / f"{hex_cp}.err").exists():
        return "error"
    return "pending"


def _atomic_write(path: Path, content: bytes | str) -> None:
    """Write to {path}.tmp then rename to {path}.
    Guarantees {path} never exists in partial state — on any crash mid-write,
    only the .tmp file may be partial (ignored on resume)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    if isinstance(content, bytes):
        tmp.write_bytes(content)
    else:
        tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)  # atomic rename on both POSIX and Windows


def write_result(axis_dir: Path, cp: int, kind: str, content: bytes | str) -> None:
    """kind: 'done' | 'notfound' | 'error'"""
    hex_cp = f"{cp:X}"
    ext = AXIS_EXT[axis_dir.name]
    if kind == "done":
        _atomic_write(axis_dir / f"{hex_cp}.{ext}", content)
    elif kind == "notfound":
        _atomic_write(axis_dir / f"{hex_cp}.404", "")
    else:  # error
        _atomic_write(axis_dir / f"{hex_cp}.err", str(content))


# -------- logging ------------------------------------------------------------


class JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a", encoding="utf-8")

    def log(self, **kw):
        kw["ts"] = time.time()
        self.f.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


# -------- per-axis fetchers --------------------------------------------------


async def fetch_svg(client: httpx.AsyncClient, cp: int) -> tuple[str, Any]:
    folder = f"{cp & 0xFF00:X}"
    hex_cp = f"{cp:X}"
    url = f"{SVG_BASE}/{folder}/{hex_cp}.svg"
    try:
        r = await client.get(url, timeout=TIMEOUT)
    except Exception as e:
        return "error", str(e)
    if r.status_code == 200:
        return "done", r.content
    if r.status_code == 404:
        return "notfound", None
    if r.status_code in (429, 503):
        # Server throttling us — extra long wait built into error
        return "error", f"HTTP {r.status_code} (rate-limited); long-back-off"
    return "error", f"HTTP {r.status_code}"


TREE_REQUESTS = ("getHunum", "getJahae", "getSchoolCom")


async def fetch_tree(client: httpx.AsyncClient, cp: int) -> tuple[str, Any]:
    """Combine 3 JSON calls into a single object."""
    unicode_str = f"U+{cp:X}"
    combined: dict[str, Any] = {"unicode": unicode_str}
    any_success = False
    for req_type in TREE_REQUESTS:
        body = {"request": req_type, "unicode": unicode_str, "db": "server"}
        try:
            r = await client.post(TREE_URL, json=body, timeout=TIMEOUT)
        except Exception as e:
            combined[req_type] = {"_error": str(e)}
            continue
        if r.status_code != 200:
            combined[req_type] = {"_error": f"HTTP {r.status_code}"}
            continue
        try:
            data = r.json()
        except Exception as e:
            combined[req_type] = {"_error": f"json parse: {e}", "_raw": r.text[:200]}
            continue
        rs = data.get("recordSet")
        if isinstance(rs, list) and len(rs) == 0:
            combined[req_type] = None  # empty = no data for this cp
        else:
            combined[req_type] = rs
            any_success = True
    if not any_success:
        return "notfound", None
    return "done", json.dumps(combined, ensure_ascii=False)


async def fetch_detail(client: httpx.AsyncClient, cp: int) -> tuple[str, Any]:
    hex_cp = f"{cp:X}"
    # URL-encoded UTF-8 of the character (httpx will handle it in params/data)
    ch = chr(cp)
    try:
        r = await client.post(
            DETAIL_AJAX,
            data={
                "qry": "", "snd": "", "hanja": ch, "pageNo": "1",
                "keyfield": "", "keyword": ch, "hanjaGrade": "", "backUrl": "",
            },
            headers={
                "Referer": f"{DETAIL_SHELL}?hanja={quote(ch)}",
                "X-Requested-With": "XMLHttpRequest",
            },
            timeout=TIMEOUT,
        )
    except Exception as e:
        return "error", str(e)
    if r.status_code == 429 or r.status_code == 503:
        return "error", f"HTTP {r.status_code} (rate-limited)"
    if r.status_code != 200:
        return "error", f"HTTP {r.status_code}"
    text = r.text
    if "해당 정보가 없습니다" in text or len(text) < 500:
        return "notfound", None
    return "done", text


FETCHERS = {
    "svg": fetch_svg,
    "tree": fetch_tree,
    "detail": fetch_detail,
}


# -------- client factories ---------------------------------------------------


async def make_client_img() -> httpx.AsyncClient:
    return httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, follow_redirects=True)


async def make_client_tool() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT, "Content-Type": "application/json"},
        follow_redirects=True,
    )


async def make_client_www() -> httpx.AsyncClient:
    """For detail axis — seed session via GET first."""
    client = httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=TIMEOUT,
    )
    try:
        await client.get(WWW_SEED, timeout=TIMEOUT)
    except Exception as e:
        print(f"[warn] session seed failed: {e}", file=sys.stderr)
    return client


CLIENT_FACTORY = {
    "svg": make_client_img,
    "tree": make_client_tool,
    "detail": make_client_www,
}


# -------- worker -------------------------------------------------------------


@dataclass
class Progress:
    done: int = 0
    notfound: int = 0
    error: int = 0
    start_ts: float = 0.0
    last_success_cp: int | None = None
    last_success_ts: float = 0.0

    def summary(self, total: int) -> str:
        processed = self.done + self.notfound + self.error
        elapsed = time.time() - self.start_ts
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (total - processed) / rate if rate > 0 else float("inf")
        return (
            f"processed={processed}/{total}  "
            f"done={self.done}  notfound={self.notfound}  error={self.error}  "
            f"rate={rate:.2f}/s  eta={remaining/3600:.1f}h"
        )


@dataclass
class CircuitBreaker:
    """Trip when too many recent requests fail.

    Only ERROR is counted (timeout/500/conn-reset/…).
    404 NOT counted — it's a normal response meaning 'cp not covered'.
    """
    window: int = 50
    threshold: int = 30
    recent: deque = field(default_factory=lambda: deque(maxlen=50))
    error_kinds: Counter = field(default_factory=Counter)
    tripped: bool = False
    trip_reason: str = ""

    def record(self, is_error: bool, err_text: str = "") -> None:
        self.recent.append(is_error)
        if is_error:
            short = err_text[:50].replace("\n", " ")
            self.error_kinds[short] += 1
        if len(self.recent) >= self.window and self.recent.count(True) >= self.threshold:
            if not self.tripped:
                self.tripped = True
                fails = self.recent.count(True)
                self.trip_reason = f"{fails}/{len(self.recent)} consecutive-window failures"


def iso_ts(t: float | None) -> str:
    if not t:
        return "n/a"
    return datetime.fromtimestamp(t, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


async def worker(
    name: int, axis: str, cps: asyncio.Queue[int],
    axis_dir: Path, client_holder: list, client_factory,
    delay: float, logger: JsonlLogger, progress: Progress,
    breaker: CircuitBreaker, stop_event: asyncio.Event,
) -> None:
    """`client_holder` is a 1-element list so we can swap the client object
    on session re-establishment without breaking the reference."""
    fn = FETCHERS[axis]
    consecutive_errs = 0
    while not stop_event.is_set() and not breaker.tripped:
        try:
            cp = cps.get_nowait()
        except asyncio.QueueEmpty:
            return

        # attempt with retries
        last_err = None
        final_kind = None
        for attempt in range(1, MAX_RETRIES + 1):
            t0 = time.time()
            kind, payload = await fn(client_holder[0], cp)
            elapsed_ms = int((time.time() - t0) * 1000)
            if kind == "done":
                write_result(axis_dir, cp, "done", payload)
                progress.done += 1
                progress.last_success_cp = cp
                progress.last_success_ts = time.time()
                logger.log(axis=axis, cp=cp, status="done", bytes=len(payload) if payload else 0, ms=elapsed_ms, worker=name)
                final_kind = "done"
                consecutive_errs = 0
                break
            if kind == "notfound":
                write_result(axis_dir, cp, "notfound", "")
                progress.notfound += 1
                logger.log(axis=axis, cp=cp, status="notfound", ms=elapsed_ms, worker=name)
                final_kind = "notfound"
                consecutive_errs = 0
                break
            # error
            last_err = payload
            err_text = str(last_err)
            # 429/503 → long backoff (server is throttling)
            if "rate-limited" in err_text or "HTTP 429" in err_text or "HTTP 503" in err_text:
                wait = 30.0 * attempt  # 30s, 60s, 90s
                print(f"  [{axis}] worker {name}: rate-limit hit, backing off {wait:.0f}s")
                await asyncio.sleep(wait)
                continue
            if attempt < MAX_RETRIES:
                await asyncio.sleep(0.5 * 2 ** (attempt - 1))  # 0.5, 1.0, 2.0
                continue
            write_result(axis_dir, cp, "error", last_err or "unknown")
            progress.error += 1
            logger.log(axis=axis, cp=cp, status="error", err=err_text[:200], ms=elapsed_ms, worker=name)
            final_kind = "error"

        # track consecutive errors in this worker only (for session re-establishment)
        if final_kind == "error":
            consecutive_errs += 1
        else:
            consecutive_errs = 0

        # Axis 'detail' session re-establishment after 5 consecutive worker errors
        if axis == "detail" and consecutive_errs >= 5:
            print(f"  [{axis}] worker {name}: 5 consecutive errors — re-establishing session")
            try:
                old = client_holder[0]
                new_client = await client_factory()
                client_holder[0] = new_client
                await old.aclose()
                consecutive_errs = 0
            except Exception as e:
                print(f"  [{axis}] worker {name}: session re-establishment failed: {e}")

        # record in circuit breaker (only error matters; 404 is normal)
        breaker.record(final_kind == "error", str(last_err) if last_err else "")
        if breaker.tripped and not stop_event.is_set():
            print(f"\n⚠️  [{axis}] CIRCUIT BREAKER TRIPPED — {breaker.trip_reason}")
            stop_event.set()
            return

        # polite delay
        await asyncio.sleep(delay + random.uniform(0, delay * 0.5))


# -------- axis runner --------------------------------------------------------


async def run_axis(axis: str, root: Path, concurrency: int, delay: float,
                    limit: int | None, stop_event: asyncio.Event,
                    retry_errors: bool = False,
                    breaker_window: int = 50, breaker_threshold: int = 30) -> tuple[Progress, CircuitBreaker]:
    axis_dir = out_dir(root, axis)

    # clean up any stale .tmp files from a previous crashed run (harmless if none)
    stale = list(axis_dir.glob("*.tmp"))
    if stale:
        print(f"[{axis}] cleaning {len(stale)} stale .tmp files from previous interrupted run")
        for p in stale:
            try:
                p.unlink()
            except Exception:
                pass

    # optionally remove .err markers to retry previously-failed codepoints
    if retry_errors:
        errs = list(axis_dir.glob("*.err"))
        if errs:
            print(f"[{axis}] removing {len(errs)} .err markers — will retry those codepoints")
            for p in errs:
                try:
                    p.unlink()
                except Exception:
                    pass

    # enumerate remaining codepoints
    all_cps = list(enumerate_codepoints())
    remaining = [cp for cp in all_cps if status_of(axis_dir, cp) == "pending"]
    if limit is not None:
        remaining = remaining[:limit]

    print(f"[{axis}] total={len(all_cps)}  remaining={len(remaining)}  concurrency={concurrency}  delay={delay}s  "
          f"breaker={breaker_threshold}/{breaker_window}")

    breaker = CircuitBreaker(window=breaker_window, threshold=breaker_threshold)
    breaker.recent = deque(maxlen=breaker_window)  # re-init with correct maxlen

    if not remaining:
        return Progress(start_ts=time.time()), breaker

    # queue
    cps: asyncio.Queue[int] = asyncio.Queue()
    for cp in remaining:
        cps.put_nowait(cp)

    # logger
    logger = JsonlLogger(root / "logs" / "crawl.log.jsonl")
    progress = Progress(start_ts=time.time())

    # client (shared across workers) — wrapped in list so it can be replaced
    client_factory = CLIENT_FACTORY[axis]
    client_holder = [await client_factory()]

    try:
        tasks = [
            asyncio.create_task(
                worker(i, axis, cps, axis_dir, client_holder, client_factory,
                       delay, logger, progress, breaker, stop_event)
            )
            for i in range(concurrency)
        ]
        # periodic summary
        async def periodic():
            while not stop_event.is_set() and not all(t.done() for t in tasks):
                await asyncio.sleep(30)
                print(f"  [{axis}] {progress.summary(len(remaining))}")
        summary_task = asyncio.create_task(periodic())

        await asyncio.gather(*tasks, return_exceptions=True)
        summary_task.cancel()
    finally:
        await client_holder[0].aclose()
        logger.close()

    return progress, breaker


# -------- main ---------------------------------------------------------------


async def main_async(args) -> None:
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)

    stop_event = asyncio.Event()

    def sigint_handler():
        print("\n[ctrl-c] stopping gracefully... finish in-flight and exit.")
        stop_event.set()

    loop = asyncio.get_event_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, sigint_handler)
    except NotImplementedError:
        pass  # Windows may not support this

    axes = ["svg", "tree", "detail"] if args.axis == "all" else [args.axis]

    if args.parallel and len(axes) > 1:
        print(f"[main] running {len(axes)} axes in parallel")
        results = await asyncio.gather(
            *(run_axis(a, root, args.concurrency, args.delay, args.limit, stop_event,
                        args.retry_errors, args.breaker_window, args.breaker_threshold)
              for a in axes),
            return_exceptions=True,
        )
    else:
        results = []
        for a in axes:
            if stop_event.is_set():
                break
            results.append(await run_axis(
                a, root, args.concurrency, args.delay, args.limit, stop_event,
                args.retry_errors, args.breaker_window, args.breaker_threshold,
            ))

    print_and_save_final_report(root, axes, results, args)


def print_and_save_final_report(root: Path, axes: list[str], results, args) -> None:
    """Print a termination report and save a JSON artifact for later review."""
    tripped_axes: list[tuple[str, CircuitBreaker, Progress]] = []
    report: dict[str, Any] = {
        "timestamp": iso_ts(time.time()),
        "axes": {},
    }
    print()
    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    for axis, result in zip(axes, results):
        if isinstance(result, Exception):
            print(f"[{axis}] unexpected exception: {result}")
            report["axes"][axis] = {"exception": str(result)}
            continue
        progress, breaker = result
        # scan disk to report authoritative current state
        axis_dir = root / "data" / axis
        all_cps = list(enumerate_codepoints())
        done = sum(1 for cp in all_cps if (axis_dir / f"{cp:X}.{AXIS_EXT[axis]}").exists())
        notfound = sum(1 for cp in all_cps if (axis_dir / f"{cp:X}.404").exists())
        err = sum(1 for cp in all_cps if (axis_dir / f"{cp:X}.err").exists())
        pending = len(all_cps) - done - notfound - err

        state = "TRIPPED" if breaker.tripped else "OK"
        print(f"\n[{axis}]  state={state}")
        print(f"  disk: done={done:,}  notfound={notfound:,}  err={err:,}  pending={pending:,}  total={len(all_cps):,}")
        print(f"  session: done={progress.done} notfound={progress.notfound} error={progress.error}")
        if progress.last_success_cp is not None:
            print(f"  last success: U+{progress.last_success_cp:X}  at  {iso_ts(progress.last_success_ts)}")
        if breaker.tripped:
            print(f"  ⚠️  BREAKER TRIPPED: {breaker.trip_reason}")
            if breaker.error_kinds:
                top = breaker.error_kinds.most_common(5)
                print(f"     top error patterns in window:")
                for txt, n in top:
                    print(f"       [{n:2d}]  {txt}")
            tripped_axes.append((axis, breaker, progress))

        report["axes"][axis] = {
            "state": state,
            "disk": {"done": done, "notfound": notfound, "err": err, "pending": pending, "total": len(all_cps)},
            "session": {"done": progress.done, "notfound": progress.notfound, "error": progress.error},
            "last_success_cp": f"U+{progress.last_success_cp:X}" if progress.last_success_cp else None,
            "last_success_ts": iso_ts(progress.last_success_ts) if progress.last_success_ts else None,
            "breaker_tripped": breaker.tripped,
            "breaker_reason": breaker.trip_reason,
            "top_errors": breaker.error_kinds.most_common(10),
        }

    # resume instructions
    print("\n" + "-" * 60)
    if any(r["disk"]["pending"] > 0 or r["disk"]["err"] > 0
           for r in report["axes"].values() if isinstance(r, dict) and "disk" in r):
        print("재개 방법:")
        axis_arg = ",".join(axes) if len(axes) > 1 else axes[0]
        axis_arg = "all" if sorted(axes) == ["detail", "svg", "tree"] else axis_arg
        par_flag = " --parallel" if args.parallel and len(axes) > 1 else ""
        print(f"  (A) pending만 이어받기 (.err는 스킵):")
        print(f"      python crawl.py --axis {axis_arg}{par_flag}")
        print(f"  (B) .err도 재시도 (transient 실패 다시 도전):")
        print(f"      python crawl.py --axis {axis_arg}{par_flag} --retry-errors")
    else:
        print("모든 축이 완료되었습니다. 🎉")

    # save json
    report_path = root / "logs" / "abort_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n상세 보고서 저장: {report_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axis", choices=["svg", "tree", "detail", "all"], default="all")
    p.add_argument("--concurrency", type=int, default=3,
                   help="parallel workers within a single axis")
    p.add_argument("--delay", type=float, default=0.3,
                   help="per-worker base delay between requests in seconds")
    p.add_argument("--limit", type=int, default=None,
                   help="limit codepoint count for smoke test")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent.parent),
                   help="output root (will write to ./data/ and ./logs/)")
    p.add_argument("--parallel", action="store_true",
                   help="run different axes concurrently (default: sequential)")
    p.add_argument("--retry-errors", action="store_true", dest="retry_errors",
                   help="delete all .err markers before starting — retry previously-failed codepoints")
    p.add_argument("--breaker-window", type=int, default=50,
                   help="how many recent requests the circuit breaker considers (default 50)")
    p.add_argument("--breaker-threshold", type=int, default=30,
                   help="trip the breaker if this many of the last N are errors (default 30)")
    args = p.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
