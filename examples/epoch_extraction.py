#!/usr/bin/env python3
"""Event-locked epoch extraction: windows, gap handling, and pad policies."""

from pathlib import Path

import numpy as np

import edfarray

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
PATH = FIXTURES / "edfPlusD.edf"  # discontinuous: has time gaps between records

PRE, POST = 0.5, 0.5


def rms(block: np.ndarray) -> np.ndarray:
    """Root-mean-square over the sample axis, ignoring padded NaN samples.

    An epoch with no real samples left yields NaN (no spurious warnings)."""
    squared = block * block
    mask = ~np.isnan(block)
    count = mask.sum(axis=-1)
    total = np.where(mask, squared, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore"):
        return np.sqrt(total / count)


def main() -> None:
    with edfarray.EdfFile(str(PATH)) as f:
        group = max(f.signal_groups(), key=len)
        rate = group.sample_rate
        print(f"File:     {PATH.name}")
        print(f"Variant:  {f.variant}    duration: {f.duration:.1f} s")
        print(f"Group:    {len(group.indices)} channels @ {rate} Hz")
        print()

        # A handful of events, some of which land next to a recording gap.
        events = [1.0, 2.0, 3.5, 5.0, 8.0, 10.0]

        # Plan first: (onset, s_start, s_end) plus a validity flag, no reads yet.
        windows, valid = f.epoch_windows(events, pre=PRE, post=POST)
        print(f"Planned {len(events)} windows of {PRE + POST}s "
              f"({int((PRE + POST) * rate)} samples):")
        for (onset, s0, s1), ok in zip(windows, valid):
            flag = "ok" if ok else "gap/off-file"
            print(f"  t={onset:>6.2f}s  samples [{s0}:{s1})  {flag}")
        print()

        # Drop policy: epochs that straddle a gap or run off the file are removed.
        dropped = f.extract_epochs(events, pre=PRE, post=POST, group=group)
        print(f"pad=drop: kept {len(dropped)} of {len(events)}  "
              f"(dropped event indices {dropped.dropped})")
        print(f"          shape {dropped.data.shape}  "
              f"(epochs, channels, samples)")
        print()

        # Fill policy: keep every epoch, pad the offending ones, flag them invalid.
        filled = f.extract_epochs(events, pre=PRE, post=POST, group=group, pad="nan")
        print(f"pad=nan:  shape {filled.data.shape}  "
              f"valid={filled.valid.tolist()}")

        # Compute a feature and skip padded samples automatically via np.nan*.
        per_epoch = rms(filled.data).mean(axis=1)  # mean RMS across channels
        for onset, ok, val in zip(filled.onsets, filled.valid, per_epoch):
            if not np.isfinite(val):
                print(f"  t={onset:>6.2f}s  window entirely off the file")
            elif ok:
                print(f"  t={onset:>6.2f}s  mean RMS {val:8.3f}")
            else:
                print(f"  t={onset:>6.2f}s  mean RMS {val:8.3f}"
                      f"   <- padded, mean over real samples only")


if __name__ == "__main__":
    main()
