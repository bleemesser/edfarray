#!/usr/bin/env python3
"""Simulate an EEG viewer paging through a recording.

Models the access pattern of a typical EEG review application:
- Display all channels simultaneously for a time window (the "page")
- User pages forward/backward through the recording
- Measures initial load, sequential paging, and random seek times
"""

import time
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

PAGE_DURATION_SEC = 10.0
NUM_SEQUENTIAL_PAGES = 50
NUM_RANDOM_SEEKS = 100


def bench_edfarray_paging(path, page_duration=PAGE_DURATION_SEC):
    """Benchmark using the bulk read_page() API -- single call for all channels."""
    import edfarray

    results = {}

    f = edfarray.EdfFile(str(path))
    total_duration = f.duration
    signal_indices = f.ordinary_signal_indices()

    results["num_channels"] = len(signal_indices)
    results["total_duration"] = total_duration
    results["page_duration"] = page_duration

    num_pages = int(total_duration / page_duration)
    if num_pages < 2:
        print(f"  File too short for paging benchmark ({total_duration}s)")
        return None

    def load_page(page_idx):
        t_start = page_idx * page_duration
        t_end = t_start + page_duration
        return f.read_page(t_start, t_end)

    # Initial page load (cold)
    t0 = time.perf_counter()
    load_page(0)
    results["initial_load"] = time.perf_counter() - t0

    # Sequential forward paging
    n_pages = min(NUM_SEQUENTIAL_PAGES, num_pages)
    times = []
    for i in range(n_pages):
        t0 = time.perf_counter()
        load_page(i)
        times.append(time.perf_counter() - t0)
    results["sequential_forward_median"] = np.median(times)
    results["sequential_forward_p95"] = np.percentile(times, 95)
    results["sequential_forward_max"] = np.max(times)

    # Sequential backward paging
    times = []
    for i in range(n_pages - 1, -1, -1):
        t0 = time.perf_counter()
        load_page(i)
        times.append(time.perf_counter() - t0)
    results["sequential_backward_median"] = np.median(times)

    # Random seeks
    rng = np.random.default_rng(42)
    random_pages = rng.integers(0, num_pages, size=NUM_RANDOM_SEEKS)
    times = []
    for page_idx in random_pages:
        t0 = time.perf_counter()
        load_page(int(page_idx))
        times.append(time.perf_counter() - t0)
    results["random_seek_median"] = np.median(times)
    results["random_seek_p95"] = np.percentile(times, 95)
    results["random_seek_max"] = np.max(times)

    return results


def bench_pyedflib_paging(path, page_duration=PAGE_DURATION_SEC):
    import pyedflib

    results = {}

    f = pyedflib.EdfReader(str(path))
    num_signals = f.signals_in_file
    total_duration = f.file_duration

    signal_indices = []
    for i in range(num_signals):
        if f.getLabel(i) == "EDF Annotations":
            continue
        signal_indices.append(i)

    num_channels = len(signal_indices)
    results["num_channels"] = num_channels
    results["total_duration"] = total_duration
    results["page_duration"] = page_duration

    num_pages = int(total_duration / page_duration)
    if num_pages < 2:
        return None

    sample_rates = [f.getSampleFrequency(i) for i in signal_indices]
    n_samples = f.getNSamples()

    def load_page(page_idx):
        t_start = page_idx * page_duration
        buffers = []
        for sig_i, sr in zip(signal_indices, sample_rates):
            s_start = int(t_start * sr)
            n = int(min(int(page_duration * sr), n_samples[sig_i] - s_start))
            if s_start >= n_samples[sig_i] or n <= 0:
                break
            buffers.append(f.readSignal(sig_i, start=s_start, n=n))
        return buffers

    # Initial page load
    t0 = time.perf_counter()
    load_page(0)
    results["initial_load"] = time.perf_counter() - t0

    # Sequential forward paging
    n_pages = min(NUM_SEQUENTIAL_PAGES, num_pages)
    times = []
    for i in range(n_pages):
        t0 = time.perf_counter()
        load_page(i)
        times.append(time.perf_counter() - t0)
    results["sequential_forward_median"] = np.median(times)
    results["sequential_forward_p95"] = np.percentile(times, 95)
    results["sequential_forward_max"] = np.max(times)

    # Sequential backward paging
    times = []
    for i in range(n_pages - 1, -1, -1):
        t0 = time.perf_counter()
        load_page(i)
        times.append(time.perf_counter() - t0)
    results["sequential_backward_median"] = np.median(times)

    # Random seeks
    rng = np.random.default_rng(42)
    random_pages = rng.integers(0, num_pages, size=NUM_RANDOM_SEEKS)
    times = []
    for page_idx in random_pages:
        t0 = time.perf_counter()
        load_page(int(page_idx))
        times.append(time.perf_counter() - t0)
    results["random_seek_median"] = np.median(times)
    results["random_seek_p95"] = np.percentile(times, 95)
    results["random_seek_max"] = np.max(times)

    f.close()
    return results


def format_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def print_results(ours, theirs=None):
    metrics = [
        ("initial_load", "Initial page load"),
        ("sequential_forward_median", "Page forward (median)"),
        ("sequential_forward_p95", "Page forward (p95)"),
        ("sequential_forward_max", "Page forward (max)"),
        ("sequential_backward_median", "Page backward (median)"),
        ("random_seek_median", "Random seek (median)"),
        ("random_seek_p95", "Random seek (p95)"),
        ("random_seek_max", "Random seek (max)"),
    ]

    header = f"  {'Operation':<28s} {'edfarray':>12s}"
    if theirs:
        header += f" {'pyedflib':>12s} {'speedup':>10s}"
    print(header)

    for key, name in metrics:
        ours_t = format_time(ours[key])
        line = f"  {name:<28s} {ours_t:>12s}"
        if theirs and key in theirs:
            theirs_t = format_time(theirs[key])
            speedup = theirs[key] / ours[key] if ours[key] > 0 else float("inf")
            line += f" {theirs_t:>12s} {speedup:>9.1f}x"
        print(line)


def main():
    test_files = [
        "test_generator.edf",
        "test_generator_2.edf",
        "S001R01.edf",
    ]

    print("=" * 78)
    print(f"EEG Viewer Paging Benchmark  (page={PAGE_DURATION_SEC}s, "
          f"seq={NUM_SEQUENTIAL_PAGES} pages, rand={NUM_RANDOM_SEEKS} seeks)")
    print("=" * 78)

    for name in test_files:
        path = FIXTURES / name
        if not path.exists():
            continue

        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"\n{'-' * 78}")
        print(f"File: {name} ({size_mb:.1f} MB)")

        ours = bench_edfarray_paging(path)
        if ours is None:
            continue

        print(f"Channels: {ours['num_channels']}, "
              f"Duration: {ours['total_duration']:.1f}s, "
              f"Page: {ours['page_duration']}s")
        print(f"{'-' * 78}")

        try:
            theirs = bench_pyedflib_paging(path)
        except Exception as e:
            print(f"  pyedflib: {e}")
            theirs = None

        print_results(ours, theirs)


if __name__ == "__main__":
    main()
