#!/usr/bin/env python3
"""Benchmark single-shot async vs sync to measure async overhead.

Quantifies the overhead of tokio + spawn_blocking + asyncio event loop
for a single read.

Usage:
    python benchmark_async_vs_sync.py [path/to/file.edf]
"""

import asyncio
import statistics
import sys
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def pick_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return max(FIXTURES.glob("*.edf"), key=lambda p: p.stat().st_size)


def format_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def median_sync(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


async def median_async(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        await fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def report(label: str, t_sync: float, t_async: float) -> None:
    overhead = (t_async - t_sync) / t_sync * 100.0 if t_sync > 0 else 0.0
    sign = "+" if overhead >= 0 else ""
    print(f"  {label:<24} sync={format_time(t_sync):>10}  "
          f"async={format_time(t_async):>10}  "
          f"overhead={sign}{overhead:5.1f}%")


async def main():
    import edfarray
    import edfarray.aio as aio

    path = pick_path()
    size_mb = path.stat().st_size / (1024 * 1024)

    print("=" * 72)
    print("Async vs Sync Single-Shot Benchmark")
    print("=" * 72)
    print(f"File: {path.name} ({size_mb:.1f} MB)")
    print()

    repeats = 10

    t_sync_open = median_sync(lambda: edfarray.EdfFile(str(path)), repeats)

    async def _aopen():
        await aio.open(str(path))
    t_async_open = await median_async(_aopen, repeats)
    report("open", t_sync_open, t_async_open)

    f_sync = edfarray.EdfFile(str(path))
    sig_sync = f_sync.signal(0)
    f_async = await aio.open(str(path))
    sig_async = f_async.signal(0)

    t_sync_full = median_sync(lambda: sig_sync.to_numpy(), repeats)

    async def _afull():
        await sig_async.to_numpy()
    t_async_full = await median_async(_afull, repeats)
    report("full read (physical)", t_sync_full, t_async_full)

    t_sync_dig = median_sync(lambda: sig_sync.to_digital(), repeats)

    async def _adig():
        await sig_async.to_digital()
    t_async_dig = await median_async(_adig, repeats)
    report("full read (digital)", t_sync_dig, t_async_dig)

    sr = int(sig_sync.sample_rate)
    n_samples = len(sig_sync)
    mid = n_samples // 2

    t_sync_slice = median_sync(lambda: sig_sync[mid:mid + sr], repeats)

    async def _aslice():
        await sig_async.read_physical(mid, mid + sr)
    t_async_slice = await median_async(_aslice, repeats)
    report("1-second slice", t_sync_slice, t_async_slice)

    print()
    print("Async overhead is per-call cost of tokio dispatch + event loop hop.")


if __name__ == "__main__":
    asyncio.run(main())
