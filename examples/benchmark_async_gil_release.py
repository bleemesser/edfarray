#!/usr/bin/env python3
"""Benchmark async vs sync decode under concurrent Python load.

A busy Python thread spins while decodes run. Both paths release the GIL
during Rust decode. The async path dispatches to a tokio worker thread,
avoiding contention with the busy thread on GIL re-acquisition.

Usage:
    python benchmark_async_gil_release.py [path/to/file.edf]
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def pick_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    return max(FIXTURES.glob("*.edf"), key=lambda p: p.stat().st_size)


class BusyCounter:
    """Daemon thread that increments a counter as fast as possible."""

    def __init__(self):
        self.count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.count += 1

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()


def measure_freerun(duration: float) -> float:
    """Ticks per second with no competing work."""
    bc = BusyCounter()
    bc.start()
    t0 = time.perf_counter()
    time.sleep(duration)
    elapsed = time.perf_counter() - t0
    bc.stop()
    return bc.count / elapsed


def measure_sync(path: Path, repeats: int) -> tuple[float, float]:
    """Sync full-signal decode with busy thread."""
    import edfarray

    f = edfarray.EdfFile(str(path))
    sig = f.signal(0)

    bc = BusyCounter()
    bc.start()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = sig.to_physical()
    elapsed = time.perf_counter() - t0
    bc.stop()
    return bc.count / elapsed, elapsed


async def measure_async(path: Path, repeats: int) -> tuple[float, float]:
    """Async full-signal decode with busy thread."""
    import edfarray.aio as aio

    f = await aio.open(str(path))
    sig = f.signal(0)

    bc = BusyCounter()
    bc.start()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = await sig.to_physical()
    elapsed = time.perf_counter() - t0
    bc.stop()
    return bc.count / elapsed, elapsed


async def main():
    path = pick_path()
    size_mb = path.stat().st_size / (1024 * 1024)

    print("=" * 72)
    print("Async GIL Release Benchmark (edfarray.aio vs sync)")
    print("=" * 72)
    print(f"File: {path.name} ({size_mb:.1f} MB)")
    print()

    print("Calibrating free-running counter (1 s with no decode)...")
    freerun = measure_freerun(1.0)
    print(f"  freerun: {freerun:,.0f} ticks/sec")
    print()

    repeats = 10

    print(f"Running {repeats} full-signal decodes per mode")
    print("(busy Python thread spinning the whole time)")
    print()

    sync_rate, sync_elapsed = measure_sync(path, repeats)
    print(f"Sync API:")
    print(f"  wall:  {sync_elapsed * 1000:.1f} ms")
    print(f"  ticks: {sync_rate:,.0f}/sec  "
          f"(ratio to freerun: {sync_rate / freerun:.2%})")
    print()

    async_rate, async_elapsed = await measure_async(path, repeats)
    print(f"Async API:")
    print(f"  wall:  {async_elapsed * 1000:.1f} ms")
    print(f"  ticks: {async_rate:,.0f}/sec  "
          f"(ratio to freerun: {async_rate / freerun:.2%})")
    print()

    speedup = sync_elapsed / async_elapsed if async_elapsed > 0 else float("inf")
    print(f"Wall-clock speedup under busy-thread load: {speedup:.1f}x")
    print()
    print("Both paths drop the GIL during Rust decode. The wall-clock gap")
    print("comes from where decode runs: sync contends with the busy thread;")
    print("async dispatches to a tokio worker on a separate OS thread.")


if __name__ == "__main__":
    asyncio.run(main())
