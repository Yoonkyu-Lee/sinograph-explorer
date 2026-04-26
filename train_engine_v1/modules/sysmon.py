"""System resource monitoring: VRAM, GPU util%, CPU RSS, system RAM.

Designed to be called cheaply inside the training loop without touching
the hot path (samples each cost O(μs) via cached NVML handle).

Usage:
    mon = SysMon()
    mon.reset_peaks()
    ...
    for step in ...:
        ...
        if step % log_every == 0:
            snap = mon.snapshot()  # dict of scalars
            print(format_snapshot(snap))
    peaks = mon.peaks()  # dict of max values seen since reset_peaks()
"""
from __future__ import annotations

import atexit
import os
import threading

import psutil
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_NVML = True
    _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _HAS_NVML = False
    _HANDLE = None


class _GpuUtilSampler:
    """Background thread that samples GPU utilization% every `interval` seconds.

    Call `.flush()` to drain the buffer and get min/mean/max since last flush.
    Continuous sampling closes the blind spots between log_every ticks,
    so the training loop's snapshot reflects interval statistics, not
    whatever happened to be true at the sampling instant.
    """

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self._samples: list[int] = []
        self._vram_samples: list[tuple[int, int]] = []  # (alloc_bytes, reserved_bytes)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread is not None or not _HAS_NVML:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="sysmon-gpu")
        self._thread.start()
        atexit.register(self.stop)

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(_HANDLE).gpu
                with self._lock:
                    self._samples.append(int(u))
            except Exception:
                pass
            # also snapshot torch VRAM so peak catches inter-step spikes
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                try:
                    a = torch.cuda.memory_allocated()
                    r = torch.cuda.memory_reserved()
                    with self._lock:
                        self._vram_samples.append((a, r))
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def flush(self):
        with self._lock:
            s = self._samples[:]
            v = self._vram_samples[:]
            self._samples.clear()
            self._vram_samples.clear()
        out = {}
        if s:
            out["gpu_util_n"] = len(s)
            out["gpu_util_mean"] = sum(s) / len(s)
            out["gpu_util_min"] = min(s)
            out["gpu_util_max"] = max(s)
        if v:
            out["vram_alloc_max_gb"] = max(x[0] for x in v) / 1e9
            out["vram_reserved_max_gb"] = max(x[1] for x in v) / 1e9
        return out


class SysMon:
    def __init__(self, device_index: int = 0, gpu_sample_interval: float = 0.2):
        self.device_index = device_index
        self.proc = psutil.Process(os.getpid())
        self._peak_util = 0
        self._peak_vram_alloc = 0
        self._peak_vram_reserved = 0
        self._peak_rss = 0
        self._sampler = _GpuUtilSampler(interval=gpu_sample_interval)
        self._sampler.start()

    def reset_peaks(self):
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            try:
                torch.cuda.reset_peak_memory_stats(self.device_index)
            except RuntimeError:
                pass
        self._sampler.flush()  # drop old interval data
        self._peak_util = 0
        self._peak_vram_alloc = 0
        self._peak_vram_reserved = 0
        self._peak_rss = 0

    def snapshot(self) -> dict:
        """Flush interval stats (mean/min/max since last snapshot) and
        sample instantaneous VRAM + CPU. Updates peak counters."""
        s = {}
        interval = self._sampler.flush()
        s.update(interval)
        if torch.cuda.is_available():
            s["vram_alloc_gb"] = torch.cuda.memory_allocated(self.device_index) / 1e9
            s["vram_reserved_gb"] = torch.cuda.memory_reserved(self.device_index) / 1e9
            s["vram_peak_alloc_gb"] = torch.cuda.max_memory_allocated(self.device_index) / 1e9
            s["vram_peak_reserved_gb"] = torch.cuda.max_memory_reserved(self.device_index) / 1e9
            self._peak_vram_alloc = max(self._peak_vram_alloc, s["vram_peak_alloc_gb"])
            self._peak_vram_reserved = max(self._peak_vram_reserved, s["vram_peak_reserved_gb"])
            # Device-wide VRAM (matches Task Manager 전용 GPU 메모리): includes
            # CUDA context, cuDNN workspace, other processes. torch's reserved
            # is a strict subset.
            try:
                free, total = torch.cuda.mem_get_info(self.device_index)
                s["gpu_used_gb"] = (total - free) / 1e9
                s["gpu_total_gb"] = total / 1e9
            except Exception:
                pass
        if "gpu_util_max" in s:
            self._peak_util = max(self._peak_util, s["gpu_util_max"])
        try:
            rss_gb = self.proc.memory_info().rss / 1e9
            s["cpu_rss_gb"] = rss_gb
            self._peak_rss = max(self._peak_rss, rss_gb)
            vm = psutil.virtual_memory()
            s["sys_ram_used_gb"] = (vm.total - vm.available) / 1e9
            s["sys_ram_total_gb"] = vm.total / 1e9
        except Exception:
            pass
        return s

    def peaks(self) -> dict:
        return {
            "peak_gpu_util_pct": self._peak_util,
            "peak_vram_alloc_gb": self._peak_vram_alloc,
            "peak_vram_reserved_gb": self._peak_vram_reserved,
            "peak_cpu_rss_gb": self._peak_rss,
        }


def format_snapshot(s: dict) -> str:
    parts = []
    if "gpu_util_mean" in s:
        parts.append(
            f"gpu={s['gpu_util_mean']:.0f}% "
            f"(min {s['gpu_util_min']}, max {s['gpu_util_max']}, n={s['gpu_util_n']})"
        )
    if "vram_alloc_gb" in s:
        parts.append(f"vram_torch={s['vram_alloc_gb']:.2f}/{s['vram_reserved_gb']:.2f}GB")
    if "gpu_used_gb" in s:
        parts.append(f"vram_dev={s['gpu_used_gb']:.2f}/{s['gpu_total_gb']:.1f}GB")
    if "cpu_rss_gb" in s:
        parts.append(f"rss={s['cpu_rss_gb']:.2f}GB")
    if "sys_ram_used_gb" in s:
        parts.append(f"sys={s['sys_ram_used_gb']:.1f}/{s['sys_ram_total_gb']:.0f}GB")
    return " ".join(parts)
