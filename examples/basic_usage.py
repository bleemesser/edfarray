#!/usr/bin/env python3
"""Basic API usage: opening files, reading metadata, and accessing signal data."""

from pathlib import Path

import edfarray

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main():
    # Open a file using the context manager
    with edfarray.EdfFile(str(FIXTURES / "short_psg.edf")) as f:
        print(f"=== {f.variant} file ===")
        print(f"Signals: {f.num_signals}")
        print(f"Records: {f.num_records}")
        print(f"Record duration: {f.record_duration}s")
        print(f"Total duration: {f.duration}s")
        print(f"Start: {f.start_datetime}")
        print()

        # Patient metadata (parsed from EDF+ header subfields)
        print(f"Patient name: {f.patient_name}")
        print(f"Patient sex: {f.patient_sex}")
        print(f"Patient birthdate: {f.patient_birthdate}")
        print()

        # Recording metadata
        print(f"Admin code: {f.admin_code}")
        print(f"Technician: {f.technician}")
        print(f"Equipment: {f.equipment}")
        print()

        # Signal metadata
        for i in range(f.num_signals):
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
        eeg = f.signal("EEG Fpz-Cz")
        print(f"EEG Fpz-Cz: {len(eeg)} samples at {eeg.sample_rate} Hz")

        # Single sample
        print(f"  First sample: {eeg[0]:.4f} {eeg.physical_dimension}")

        # Slice
        chunk = eeg[0:5]
        print(f"  First 5 samples: {chunk}")

        # Strided access (every 10th sample = 10x downsample)
        downsampled = eeg[::10]
        print(f"  Downsampled 10x: {len(downsampled)} samples")

        # Full signal as numpy array
        all_data = eeg.to_numpy()
        print(f"  Full signal: shape={all_data.shape}, dtype={all_data.dtype}")

        # Raw digital values
        digital = eeg.to_digital()
        print(f"  Digital: shape={digital.shape}, dtype={digital.dtype}, "
              f"range=[{digital.min()}, {digital.max()}]")

        # Timestamps
        times = eeg.times()
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
    with edfarray.EdfFile(str(FIXTURES / "edf+D_sample.edf")) as f:
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
            print(f"  Gap at {t_before:.3f}s → {t_after:.3f}s ({gap:.3f}s)")

    # Anonymized dates
    print()
    with edfarray.EdfFile(str(FIXTURES / "edf+c_sample_short_future.edf")) as f:
        print(f"=== Anonymized date handling ===")
        print(f"start_datetime type: {type(f.start_datetime).__name__}")
        print(f"start_datetime value: {f.start_datetime}")


if __name__ == "__main__":
    main()
