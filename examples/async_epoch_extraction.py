#!/usr/bin/env python3
"""Parallel epoch feature extraction on a multi-channel EEG file."""

import asyncio
from pathlib import Path

import numpy as np

import edfarray.aio as aio

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
PATH = FIXTURES / "S001R01.edf"

EPOCH_SEC = 2.0
HOP_SEC = 2.0
EVENT_PRE = 0.5
EVENT_POST = 1.5


async def epoch_features(f, idxs, rate: float, start: float, dur: float) -> dict:
    n = int(dur * rate)
    pages = await f.read_page(start, start + dur, signal_indices=idxs)
    block = np.stack(pages)[:, :n]  # (n_chan, n_samples)
    rms = np.sqrt(np.mean(block * block, axis=1))
    spec = np.fft.rfft(block, axis=1)
    power = (spec.real ** 2 + spec.imag ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / rate)
    alpha = (freqs >= 8.0) & (freqs < 13.0)
    alpha_power = power[:, alpha].mean(axis=1)
    return {"start": start, "rms": rms, "alpha": alpha_power}


async def main() -> None:
    async with await aio.open(str(PATH)) as f:
        ordinary = f.ordinary_signal_indices()
        groups = f.signal_groups()
        group = max(groups, key=len)
        idxs = group.indices
        rate = group.sample_rate if group else None
        if not rate:
            print("No valid sample rate found.")
            return
        labels = [f.signal(i).label for i in idxs]

        print(f"File:      {PATH.name}")
        print(f"Variant:   {f.variant}    duration: {f.duration:.1f} s")
        print(f"Channels:  {len(idxs)} @ {rate} Hz   "
              f"(out of {len(ordinary)} ordinary signals)")
        print()

        starts = np.arange(0.0, f.duration - EPOCH_SEC + 1e-9, HOP_SEC)
        print(f"Sliding window: {len(starts)} epochs of {EPOCH_SEC}s "
              f"(hop {HOP_SEC}s) dispatched via asyncio.gather")

        epochs = await asyncio.gather(
            *(epoch_features(f, idxs, rate, float(s), EPOCH_SEC) for s in starts)
        )
        print()

        rms_matrix = np.stack([e["rms"] for e in epochs])
        alpha_matrix = np.stack([e["alpha"] for e in epochs])
        loudest_ch = int(np.argmax(rms_matrix.mean(axis=0)))
        most_alpha_ch = int(np.argmax(alpha_matrix.mean(axis=0)))
        print(f"Loudest channel:        {labels[loudest_ch]:8s} "
              f"(mean RMS {rms_matrix[:, loudest_ch].mean():.2f})")
        print(f"Strongest alpha (8-13Hz): {labels[most_alpha_ch]:8s} "
              f"(mean power {alpha_matrix[:, most_alpha_ch].mean():.3f})")
        print()

        await f.wait_for_annotations()
        events = [a for a in f.annotations
                  if a.onset >= EVENT_PRE
                  and a.onset + EVENT_POST <= f.duration]
        if not events:
            print("No annotations suitable for event-locked epoching.")
            return

        print(f"Event-locked epochs: {len(events)} events "
              f"({EVENT_PRE}s pre, {EVENT_POST}s post)")
        windows = await asyncio.gather(*(
            epoch_features(f, idxs, rate, a.onset - EVENT_PRE, EVENT_PRE + EVENT_POST)
            for a in events
        ))
        for ev, w in zip(events, windows):
            top = int(np.argmax(w["rms"]))
            print(f"  t={ev.onset:>7.2f}s  {ev.text!r:>10s}  "
                  f"loudest={labels[top]:6s} rms={w['rms'][top]:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
