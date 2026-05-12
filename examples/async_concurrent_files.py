#!/usr/bin/env python3
"""Concurrent multi-file scan.

Opens several EDF files in parallel, gathers summary statistics for one
ordinary signal from each, and prints a small report. Demonstrates
`aio.inspect` for cheap header-only metadata, `aio.open` + async context
managers, and `asyncio.gather` for true parallel decode across files.
"""

import asyncio
from pathlib import Path

import numpy as np

import edfarray.aio as aio

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
FILES = [
    "test_generator.edf",
    "test_generator_2.edf",
    "edfPlusC.edf",
    "edfPlusD.edf",
    "S001R01.edf",
]


async def quick_metadata(path: Path) -> dict:
    meta = await aio.inspect(str(path))
    return {"file": path.name, **{k: meta[k] for k in ("variant", "num_signals", "duration")}}


async def signal_stats(path: Path) -> dict:
    async with await aio.open(str(path)) as f:
        ordinary = f.ordinary_signal_indices()
        sig = f.signal(ordinary[0])
        data = await sig.to_numpy()
        return {
            "file": path.name,
            "label": sig.label,
            "rate": sig.sample_rate,
            "n": data.size,
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }


async def main() -> None:
    paths = [FIXTURES / name for name in FILES]

    metas = await asyncio.gather(*(quick_metadata(p) for p in paths))
    print(f"Inspected {len(metas)} headers concurrently:")
    for m in metas:
        print(f"  {m['file']:25s} {m['variant']:6s}  "
              f"{m['num_signals']:>3d} signals  {m['duration']:>7.1f} s")
    print()

    stats = await asyncio.gather(*(signal_stats(p) for p in paths))
    print("Per-file stats (first ordinary signal):")
    for s in stats:
        print(f"  {s['file']:25s} {s['label']:14s} {s['rate']:>6.1f} Hz  "
              f"n={s['n']:<8d} mean={s['mean']:+.3f} std={s['std']:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
