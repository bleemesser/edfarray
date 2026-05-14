#!/usr/bin/env python3
"""Streaming write driven by an async producer-consumer pipeline."""

import asyncio
import math
import tempfile
from pathlib import Path

import numpy as np

import edfarray
import edfarray.aio as aio

SAMPLE_RATE = 256
RECORD_DURATION = 1.0
SAMPLES_PER_RECORD = int(SAMPLE_RATE * RECORD_DURATION)
NUM_RECORDS = 10
NUM_CHANNELS = 4

PHYS_MIN, PHYS_MAX = -3200.0, 3200.0
DIG_MIN, DIG_MAX = -32768, 32767


def channel_definitions() -> list[aio.WriterSignal]:
    return [
        aio.WriterSignal(
            label=f"EEG ch{i}",
            physical_dimension="uV",
            physical_min=PHYS_MIN,
            physical_max=PHYS_MAX,
            digital_min=DIG_MIN,
            digital_max=DIG_MAX,
            samples_per_record=SAMPLES_PER_RECORD,
        )
        for i in range(NUM_CHANNELS)
    ]


def synth_record(t0: float, channel: int) -> np.ndarray:
    t = t0 + np.arange(SAMPLES_PER_RECORD) / SAMPLE_RATE
    base = 50.0 * np.sin(2 * math.pi * (10.0 + channel) * t)
    noise = np.random.default_rng(channel).normal(0.0, 5.0, t.shape)
    return (base + noise).astype(np.float64)


async def producer(queue: asyncio.Queue) -> None:
    for r in range(NUM_RECORDS):
        await asyncio.sleep(0.01)
        t0 = r * RECORD_DURATION
        channels = [synth_record(t0, c) for c in range(NUM_CHANNELS)]
        annotation = None
        if r in (3, 7):
            channels[0] += 600.0
            annotation = edfarray.Annotation(
                onset=t0 + 0.5, duration=None, text=f"artifact-{r}"
            )
        await queue.put((channels, annotation))
    await queue.put(None)


async def consumer(queue: asyncio.Queue, out_path: Path) -> int:
    written = 0
    async with await aio.EdfWriter.create(
        str(out_path),
        variant="EDF+C",
        record_duration=RECORD_DURATION,
        signals=channel_definitions(),
    ) as w:
        while True:
            item = await queue.get()
            if item is None:
                break
            channels, annotation = item
            if annotation is not None:
                # Embedded in the next write_record call
                w.add_annotation(annotation)
            await w.write_record(channels)
            written += 1
    return written


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "live.edf"
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)

        prod = asyncio.create_task(producer(queue))
        cons = asyncio.create_task(consumer(queue, out_path))
        _, written = await asyncio.gather(prod, cons)
        print(f"Wrote {written} records to {out_path.name} "
              f"({out_path.stat().st_size} bytes)")
        print()

        async with await aio.open(str(out_path)) as f:
            print(f"Re-opened: {f.variant}, {f.num_signals} signals, "
                  f"{f.duration:.1f} s")
            await f.wait_for_annotations()
            print(f"Annotations: {len(f.annotations)}")
            for a in f.annotations:
                print(f"  t={a.onset:>5.2f}s  {a.text!r}")
            print()

            pages = await f.read_page(0.0, f.duration)
            block = np.stack(pages)
            rms = np.sqrt(np.mean(block * block, axis=1))
            print("Per-channel RMS over full file:")
            for i, r in enumerate(rms):
                print(f"  ch{i}: {r:7.2f} uV")


if __name__ == "__main__":
    asyncio.run(main())
