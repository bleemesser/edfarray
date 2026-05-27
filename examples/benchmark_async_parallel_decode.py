#!/usr/bin/env python3
"""Benchmark parallel decode through the async API.

Compares sequential awaits vs asyncio.gather of N concurrent reads.
The async runtime dispatches each decode to spawn_blocking on the tokio
thread pool with the GIL released.

Usage:
    python benchmark_async_parallel_decode.py [path/to/file.edf]
"""

import asyncio
import statistics
import sys
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def format_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def pick_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return max(FIXTURES.glob("*.edf"), key=lambda p: p.stat().st_size)


async def time_one(coro_factory, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        await coro_factory()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


async def time_gather(coro_factory, n: int, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        await asyncio.gather(*(coro_factory() for _ in range(n)))
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


async def time_sequential(coro_factory, n: int, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            await coro_factory()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


async def main():
    import edfarray.aio as aio

    path = pick_path()
    size_mb = path.stat().st_size / (1024 * 1024)

    f = await aio.open(str(path))
    sig = f.signal(0)
    n_samples = len(sig)

    print("=" * 72)
    print("Async Parallel Decode Benchmark (edfarray.aio)")
    print("=" * 72)
    print(f"File:     {path.name} ({size_mb:.1f} MB)")
    print(f"Signal 0: {n_samples} samples @ {sig.sample_rate} Hz")
    print()

    repeats = 5
    concurrencies = [1, 2, 4, 8]

    async def read_full():
        await sig.to_numpy()

    t_single = await time_one(read_full, repeats)
    print(f"Single full read (baseline): {format_time(t_single)}")
    print()

    print(f"{'N':>3}  {'sequential':>14}  {'gather':>14}  "
          f"{'speedup':>9}  {'efficiency':>11}")
    print(f"{'-' * 3}  {'-' * 14}  {'-' * 14}  {'-' * 9}  {'-' * 11}")
    for n in concurrencies:
        t_seq = await time_sequential(read_full, n, repeats)
        t_par = await time_gather(read_full, n, repeats)
        speedup = t_seq / t_par if t_par > 0 else float("inf")
        eff = speedup / n * 100.0
        print(f"{n:>3}  {format_time(t_seq):>14}  {format_time(t_par):>14}  "
              f"{speedup:>8.2f}x  {eff:>10.1f}%")

    print()
    print("Speedup near N indicates true parallel decode on the tokio pool.")

    await f.close() if asyncio.iscoroutinefunction(f.close) else f.close()


if __name__ == "__main__":
    asyncio.run(main())
