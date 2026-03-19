#!/usr/bin/env python3
"""Benchmark the async annotation scan.

Measures:
- Time to open (header parse only, returns immediately)
- Time to first signal read (should be instant, no annotation dependency)
- Time to access annotations (blocks until scan completes)
- annotations_ready status at each point
"""

import time
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def format_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def bench_file(name, sig_idx=0):
    import edfarray

    path = FIXTURES / name
    if not path.exists():
        return
    size_mb = path.stat().st_size / (1024 * 1024)

    print(f"\n{'─' * 70}")
    print(f"File: {name} ({size_mb:.1f} MB)")
    print(f"{'─' * 70}")

    # Phase 1: Open (header + layout only, scan starts in background)
    t0 = time.perf_counter()
    f = edfarray.EdfFile(str(path))
    t_open = time.perf_counter() - t0
    ready_after_open = f.annotations_ready
    progress_after_open = f.scan_progress

    print(f"  Open file:              {format_time(t_open):>12s}   "
          f"annotations_ready={ready_after_open}  "
          f"scan_progress={progress_after_open[0]}/{progress_after_open[1]}")

    # Phase 2: Read first page of signal data (no annotation dependency)
    t0 = time.perf_counter()
    sig = f.signal(sig_idx)
    n = min(1000, len(sig))
    _ = sig[0:n]
    t_first_read = time.perf_counter() - t0
    ready_after_read = f.annotations_ready
    progress_after_read = f.scan_progress

    print(f"  First signal read:      {format_time(t_first_read):>12s}   "
          f"annotations_ready={ready_after_read}  "
          f"scan_progress={progress_after_read[0]}/{progress_after_read[1]}")

    # Phase 3: Access annotations (blocks until scan completes)
    t0 = time.perf_counter()
    anns = f.annotations
    t_annotations = time.perf_counter() - t0
    ready_after_anns = f.annotations_ready

    print(f"  Access annotations:     {format_time(t_annotations):>12s}   "
          f"annotations_ready={ready_after_anns}  "
          f"n_annotations={len(anns)}")

    # Phase 4: Second access (should be instant, scan already done)
    t0 = time.perf_counter()
    _ = f.annotations
    t_second = time.perf_counter() - t0

    print(f"  Second annotation read: {format_time(t_second):>12s}   (cached)")

    # Phase 5: Warnings (also depends on scan)
    t0 = time.perf_counter()
    w = f.warnings
    t_warnings = time.perf_counter() - t0

    print(f"  Access warnings:        {format_time(t_warnings):>12s}   "
          f"n_warnings={len(w)}")

    return {
        "open": t_open,
        "first_read": t_first_read,
        "annotations": t_annotations,
        "second_annotations": t_second,
    }


def main():
    print("=" * 70)
    print("Async Annotation Scan Benchmark")
    print("=" * 70)
    print()
    print("The annotation scan now runs in a background thread.")
    print("Signal reads work immediately without waiting for it.")

    test_files = [
        ("test_generator.edf", 0),
        ("test_generator_2.edf", 0),
        ("edfPlusC.edf", 0),
        ("S001R01.edf", 0),
        ("edfPlusD.edf", 0),
    ]

    for name, sig_idx in test_files:
        bench_file(name, sig_idx)


if __name__ == "__main__":
    main()
