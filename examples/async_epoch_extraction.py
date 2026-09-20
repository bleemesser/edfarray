#!/usr/bin/env python3
"""Parallel epoch feature extraction with the async extract_epochs API."""

import asyncio
from pathlib import Path

import numpy as np

import edfarray.aio as aio

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
PATH = FIXTURES / "S001R01.edf"

EPOCH_SEC = 2.0
HOP_SEC = 2.0


def features(block: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (rms, alpha-power) per channel for a (epochs, channels, samples) block."""
    n = block.shape[-1]
    rms = np.sqrt(np.mean(block * block, axis=-1))
    spec = np.fft.rfft(block, axis=-1)
    power = (spec.real ** 2 + spec.imag ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / rate)
    alpha = (freqs >= 8.0) & (freqs < 13.0)
    return rms, power[:, :, alpha].mean(axis=-1)


async def main() -> None:
    async with await aio.open(str(PATH)) as f:
        await f.wait_for_annotations()
        group = max(f.signal_groups(), key=len)
        rate = group.sample_rate
        labels = [f.signal(i).label for i in group.indices]

        print(f"File:     {PATH.name}")
        print(f"Variant:  {f.variant}    duration: {f.duration:.1f} s")
        print(f"Group:    {len(labels)} channels @ {rate} Hz")
        print()

        # Grid epochs: hand extract_epochs the window centers; decoding is
        # parallelized inside a single blocking task, not one task per epoch.
        centers = np.arange(EPOCH_SEC / 2,
                            f.duration - EPOCH_SEC / 2 + 1e-9, HOP_SEC)
        epochs = await f.extract_epochs(list(centers),
                                        pre=EPOCH_SEC / 2, post=EPOCH_SEC / 2,
                                        group=group)
        rms, alpha = features(epochs.data, rate)
        print(f"Grid epochs: {len(epochs)} x {EPOCH_SEC}s "
              f"(hop {HOP_SEC}s) -> data shape {epochs.data.shape}")
        loudest = int(np.argmax(rms.mean(axis=0)))
        strongest = int(np.argmax(alpha.mean(axis=0)))
        print(f"  loudest channel:          {labels[loudest]} "
              f"(mean RMS {rms[:, loudest].mean():.2f})")
        print(f"  strongest alpha (8-13Hz): {labels[strongest]} "
              f"(mean power {alpha[:, strongest].mean():.3f})")
        print()

        # Event-locked epochs straight from annotation text. The T0 marker sits
        # at t=0, so its pre-window runs off the file; zero-fill keeps it.
        events = f.events("T0") or f.annotations
        if not events:
            print("No annotations for event-locked epoching.")
            return
        ev = await f.extract_epochs(events, pre=0.5, post=1.5, group=group, pad="zero")
        rms, _ = features(ev.data, rate)
        print(f"Event-locked epochs: {len(ev)} "
              f"(0.5s pre, 1.5s post), valid={ev.valid.tolist()}")
        for onset, ok, row in zip(ev.onsets, ev.valid, rms):
            top = int(np.argmax(row))
            pad = "" if ok else "  (zero-padded pre-window)"
            print(f"  t={onset:>7.2f}s  loudest={labels[top]} "
                  f"rms={row[top]:.2f}{pad}")


if __name__ == "__main__":
    asyncio.run(main())
