#!/usr/bin/env python3
"""Performance benchmarks comparing edfarray against pyedflib."""

import time
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def find_largest_fixture():
    """Find the largest EDF file in fixtures for benchmarking."""
    edfs = list(FIXTURES.glob("*.edf"))
    if not edfs:
        raise FileNotFoundError("No EDF fixtures found")
    return max(edfs, key=lambda p: p.stat().st_size)


def bench_edfarray(path, signal_idx=0, n_iterations=5):
    import edfarray

    results = {}

    # File open time
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        f = edfarray.EdfFile(str(path))
        times.append(time.perf_counter() - t0)
    results["open"] = np.median(times)

    f = edfarray.EdfFile(str(path))
    sig = f.signal(signal_idx)
    n_samples = len(sig)

    # Read full signal
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _data = sig.to_numpy()
        times.append(time.perf_counter() - t0)
    results["read_full"] = np.median(times)
    results["n_samples"] = n_samples
    results["throughput_MSps"] = n_samples / results["read_full"] / 1e6

    # Read 1-second chunk from the middle
    sr = int(sig.sample_rate)
    mid = n_samples // 2
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = sig[mid:mid + sr]
        times.append(time.perf_counter() - t0)
    results["read_1s_chunk"] = np.median(times)

    # Random access (1000 single samples)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, n_samples, size=1000)
    t0 = time.perf_counter()
    for idx in indices:
        _ = sig[int(idx)]
    results["random_1000"] = time.perf_counter() - t0

    # Digital read
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = sig.to_digital()
        times.append(time.perf_counter() - t0)
    results["read_digital"] = np.median(times)

    return results


def bench_pyedflib(path, signal_idx=0, n_iterations=5):
    import pyedflib

    results = {}

    # File open time
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        f = pyedflib.EdfReader(str(path))
        f.close()
        times.append(time.perf_counter() - t0)
    results["open"] = np.median(times)

    f = pyedflib.EdfReader(str(path))
    n_samples = f.getNSamples()[signal_idx]

    # Read full signal
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _data = f.readSignal(signal_idx)
        times.append(time.perf_counter() - t0)
    results["read_full"] = np.median(times)
    results["n_samples"] = n_samples
    results["throughput_MSps"] = n_samples / results["read_full"] / 1e6

    # Read 1-second chunk from the middle
    sr = int(f.getSampleFrequency(signal_idx))
    mid = n_samples // 2
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = f.readSignal(signal_idx, start=mid, n=sr)
        times.append(time.perf_counter() - t0)
    results["read_1s_chunk"] = np.median(times)

    # Random access (1000 single samples)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, n_samples, size=1000)
    t0 = time.perf_counter()
    for idx in indices:
        _ = f.readSignal(signal_idx, start=int(idx), n=1)
    results["random_1000"] = time.perf_counter() - t0

    # Digital read
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = f.readSignal(signal_idx, digital=True)
        times.append(time.perf_counter() - t0)
    results["read_digital"] = np.median(times)

    f.close()
    return results


def format_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def main():
    # Use a few different file sizes
    test_files = [
        ("test_generator.edf", 0),
        ("test_generator_2.edf", 0),
        ("edfPlusC.edf", 0),
        ("S001R01.edf", 0),
    ]

    available = []
    for name, sig_idx in test_files:
        path = FIXTURES / name
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            available.append((name, sig_idx, path, size_mb))

    if not available:
        print("No fixture files found!")
        return

    print("=" * 80)
    print("edfarray vs pyedflib benchmark")
    print("=" * 80)

    for name, sig_idx, path, size_mb in available:
        print(f"\n{'-' * 80}")
        print(f"File: {name} ({size_mb:.1f} MB)")
        print(f"{'-' * 80}")

        ours = bench_edfarray(path, sig_idx)

        theirs = None
        try:
            theirs = bench_pyedflib(path, sig_idx)
        except Exception as e:
            print(f"  pyedflib could not open this file: {e}")

        print(f"  Samples: {ours['n_samples']:,}")
        print()
        print(f"  {'Operation':<25s} {'edfarray':>12s}", end="")
        if theirs:
            print(f" {'pyedflib':>12s} {'speedup':>10s}")
        else:
            print()

        for key, label in [
            ("open", "File open"),
            ("read_full", "Read full signal"),
            ("read_digital", "Read digital"),
            ("read_1s_chunk", "Read 1s chunk"),
            ("random_1000", "1000 random samples"),
        ]:
            ours_t = format_time(ours[key])
            print(f"  {label:<25s} {ours_t:>12s}", end="")
            if theirs:
                theirs_t = format_time(theirs[key])
                speedup = theirs[key] / ours[key] if ours[key] > 0 else float("inf")
                print(f" {theirs_t:>12s} {speedup:>9.1f}x")
            else:
                print()

        print(f"\n  Throughput: {ours['throughput_MSps']:.1f} M samples/s", end="")
        if theirs:
            print(f" (vs {theirs['throughput_MSps']:.1f} M samples/s)")
        else:
            print()


if __name__ == "__main__":
    main()
