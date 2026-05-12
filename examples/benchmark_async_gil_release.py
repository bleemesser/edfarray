#!/usr/bin/env python3
"""Benchmark async behavior under concurrent Python load.

A background Python thread spins on a counter while decodes are in flight.
Both the sync and async paths release the GIL during the Rust decode, so the
busy thread should advance in both cases. The interesting signal is
*wall-clock time for the decodes themselves*: the sync path runs decode on
the calling Python thread (which contends with the busy thread on GIL
re-acquisition and CPU), while the async path dispatches each decode to a
tokio worker on a separate OS thread, so the calling thread only runs
event-loop bookkeeping and Python work proceeds in parallel.

Reports:
- counter ticks/second observed during each run (both should be close to
  free-running, confirming GIL is dropped during decode)
- wall-clock decode time under busy-thread load — the real headline

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
    """A Python thread that increments a counter as fast as it can."""

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
    """Ticks per second with no other Python work competing."""
    bc = BusyCounter()
    bc.start()
    t0 = time.perf_counter()
    time.sleep(duration)
    elapsed = time.perf_counter() - t0
    bc.stop()
    return bc.count / elapsed


def measure_sync(path: Path, repeats: int) -> tuple[float, float]:
    """Run a sync full-signal decode while the busy thread runs."""
    import edfarray

    f = edfarray.EdfFile(str(path))
    sig = f.signal(0)

    bc = BusyCounter()
    bc.start()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = sig.to_numpy()
    elapsed = time.perf_counter() - t0
    bc.stop()
    return bc.count / elapsed, elapsed


async def measure_async(path: Path, repeats: int) -> tuple[float, float]:
    """Run async full-signal decodes while the busy thread runs."""
    import edfarray.aio as aio

    f = await aio.open(str(path))
    sig = f.signal(0)

    bc = BusyCounter()
    bc.start()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = await sig.to_numpy()
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

    # Calibrate free-running tick rate.
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
    print("Both paths drop the GIL during the Rust decode (high tick ratio in")
    print("both confirms this). The wall-clock gap comes from where the decode")
    print("runs: sync uses the calling thread (contending with the busy thread")
    print("for GIL/CPU); async dispatches to a tokio worker on a separate OS")
    print("thread so Python work proceeds in parallel without contention.")


if __name__ == "__main__":
    asyncio.run(main())
