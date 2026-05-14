#!/usr/bin/env python3
"""Basic API usage: opening files, reading metadata, and accessing signal data."""

from pathlib import Path

import edfarray

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main():
    # Open a plain EDF file using the context manager
    with edfarray.EdfFile(str(FIXTURES / "test_generator.edf")) as f:
        print(f"=== {f.variant} file ===")
        print(f"Signals: {f.num_signals}")
        print(f"Records: {f.num_records}")
        print(f"Record duration: {f.record_duration}s")
        print(f"Total duration: {f.duration}s")
        print(f"Start: {f.start_datetime}")
        print()

        # Patient metadata (parsed from header subfields)
        print(f"Patient name: {f.patient_name}")
        print(f"Patient sex: {f.patient_sex}")
        print(f"Patient birthdate: {f.patient_birthdate}")
        print()

        # Signal labels
        print(f"Labels: {f.signal_labels()}")
        print()

        # Signal metadata
        for i in range(min(f.num_signals, 5)):
            sig = f.signal(i)
            print(
                f"  [{i}] {sig.label:20s}  "
                f"{sig.sample_rate:>8.1f} Hz  "
                f"{sig.physical_dimension:>6s}  "
                f"[{sig.physical_min}, {sig.physical_max}]  "
                f"{len(sig)} samples"
            )
        print()

        # Access by label
        sig = f.signal("F4")
        print(f"F4: {len(sig)} samples at {sig.sample_rate} Hz")

        # Single sample
        print(f"  First sample: {sig[0]:.4f} {sig.physical_dimension}")

        # Slice
        chunk = sig[0:5]
        print(f"  First 5 samples: {chunk}")

        # Strided access (every 10th sample = 10x downsample)
        downsampled = sig[::10]
        print(f"  Downsampled 10x: {len(downsampled)} samples")

        # Full signal as numpy array
        all_data = sig.to_numpy()
        print(f"  Full signal: shape={all_data.shape}, dtype={all_data.dtype}")

        # Raw digital values
        digital = sig.to_digital()
        print(f"  Digital: shape={digital.shape}, dtype={digital.dtype}, "
              f"range=[{digital.min()}, {digital.max()}]")

        # Timestamps
        times = sig.times()
        print(f"  Time range: {times[0]:.3f}s to {times[-1]:.3f}s")

    # EDF+ files with annotations
    print()
    with edfarray.EdfFile(str(FIXTURES / "test_generator_2.edf")) as f:
        print(f"=== {f.variant} file with annotations ===")
        print(f"Annotations: {len(f.annotations)}")
        for ann in f.annotations:
            dur = f" (duration={ann.duration}s)" if ann.duration else ""
            print(f"  {ann.onset:>8.3f}s: {ann.text}{dur}")

    # EDF+D (discontinuous) files
    print()
    with edfarray.EdfFile(str(FIXTURES / "edfPlusD.edf")) as f:
        print(f"=== {f.variant} (discontinuous) ===")
        print(f"Signals: {f.num_signals}")
        print(f"Annotations: {len(f.annotations)}")
        print(f"Warnings: {len(f.warnings)}")

        sig = f.signal(0)
        times = sig.times()
        gaps = []
        expected_dt = 1.0 / sig.sample_rate
        for i in range(1, min(len(times), 100_000)):
            dt = times[i] - times[i - 1]
            if dt > expected_dt * 1.5:
                gaps.append((times[i - 1], times[i], dt))
        print(f"Time gaps found in first 100k samples: {len(gaps)}")
        for t_before, t_after, gap in gaps[:5]:
            print(f"  Gap at {t_before:.3f}s -> {t_after:.3f}s ({gap:.3f}s)")

    # Multi-channel EEG (64 channels)
    print()
    with edfarray.EdfFile(str(FIXTURES / "S001R01.edf")) as f:
        print(f"=== Multi-channel EEG ({f.variant}) ===")
        ordinary = f.ordinary_signal_indices()
        print(f"Ordinary signals: {len(ordinary)}")
        print(f"Total signals: {f.num_signals}")

        # Bulk read all channels for 1 second
        pages = f.read_page(0.0, 1.0)
        print(f"read_page(0, 1): {len(pages)} arrays, "
              f"first has {len(pages[0])} samples")
        print()

        # Discover same-rate signal groups for numpy-style multi-channel access.
        groups = f.signal_groups()
        print(f"Signal groups: {len(groups)}")
        for g in groups:
            print(f"  {g.kind:12s} n={len(g):>3d}  rate={g.sample_rate}Hz  "
                  f"covers_all={g.covers_all_ordinary}")
        print()

        # 2D proxy — numpy-like indexing across signals and samples.
        group = max(groups, key=len)
        p2 = f.proxy_2d(group)
        rate = p2.sample_rate
        assert rate is not None  # rectangular group => has a sample rate
        print(f"Proxy2D: shape={p2.shape}, rate={rate}Hz")
        first_second = p2[:, : int(rate)]
        print(f"  p2[:, :rate] -> {first_second.shape}, dtype={first_second.dtype}")
        print(f"  p2[0, 0]     -> {p2[0, 0]:.3f}")
        print(f"  p2[[0,1,2], 0:5] -> shape {p2[[0, 1, 2], 0:5].shape}")
        print()

        # 3D proxy — natural record-by-channel layout for epoch-based ML.
        # Rectangular groups (one shared sample rate) are required.
        p3 = f.proxy_3d(group)
        print(f"Proxy3D: shape={p3.shape}  "
              f"(num_records, num_channels, samples_per_record)")
        epoch = p3[0:5, :, :]  # first 5 records, all channels
        print(f"  p3[0:5, :, :] -> {epoch.shape}, dtype={epoch.dtype}")
        print(f"  p3[0, 10, 0]  -> {p3[0, 10, 0]:.3f}")

    # Mixed sample rates — Open group + PadMode
    print()
    with edfarray.EdfFile(str(FIXTURES / "test_generator.edf")) as f:
        groups = f.signal_groups()
        print(f"Signal groups: {len(groups)}")
        for g in groups:
            print(f"  {g.kind:12s} n={len(g):>3d}  rate={g.sample_rate}Hz  "
                  f"covers_all={g.covers_all_ordinary}")
        all_ch = f.signal_group(f.ordinary_signal_indices())
        print(f"=== Mixed-rate file: {all_ch.kind} group ===")
        if all_ch.kind == "open":
            # Proxy3D rejects Open groups; use Proxy2D with a pad mode.
            p2 = f.proxy_2d(all_ch, pad_mode="nan")
            print(f"Proxy2D(pad_mode='nan'): shape={p2.shape}, "
                  f"rate={p2.sample_rate}")
            print(f"  valid_lengths: {p2.valid_lengths}")
            # Reading past a short channel's end fills with NaN.
            tail = p2[:, p2.shape[1] - 4 : p2.shape[1]]
            import numpy as np
            print(f"  NaNs in tail: {int(np.isnan(tail).sum())} / {tail.size}")


if __name__ == "__main__":
    main()
