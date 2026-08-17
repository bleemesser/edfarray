"""Large-file benchmark for edfarray vs pyedflib, warm and cold.

Every other benchmark in this repo measures a warm page cache on files of a few MB, which is
the regime where mmap looks best. This one targets the opposite case: a file too large to stay
resident, where reading a single channel must touch every record.

    python scripts/bench_large.py --gb 4          # generate, then warm + cold
    python scripts/bench_large.py --file BIG.edf --cold-only

Cold runs evict only the file under test, and assert the eviction worked before reporting.
"""

import argparse
import os
import resource
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coldcache import evict, residency  # noqa: E402
from gen_large_edf import generate  # noqa: E402


def major_faults():
    return resource.getrusage(resource.RUSAGE_SELF).ru_majflt


def time_call(fn):
    before = major_faults()
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return elapsed, major_faults() - before, result


def edfarray_full_channel(path, strategy=None):
    import edfarray

    def run():
        f = edfarray.EdfFile(path, scan_annotations=False)
        return len(f.signal(0, strategy=strategy).to_physical())

    return run


def edfarray_page(path):
    import edfarray

    def run():
        f = edfarray.EdfFile(path, scan_annotations=False)
        return sum(len(a) for a in f.read_page(0.0, 30.0))

    return run


def pyedflib_full_channel(path):
    import pyedflib

    def run():
        r = pyedflib.EdfReader(path)
        try:
            return len(r.readSignal(0))
        finally:
            r._close()

    return run


def pyedflib_page(path):
    import pyedflib

    def run():
        r = pyedflib.EdfReader(path)
        try:
            n = int(r.getSampleFrequency(0) * 30)
            return sum(len(r.readSignal(i, 0, n)) for i in range(r.signals_in_file))
        finally:
            r._close()

    return run


def bench(name, cases, *, cold, path, repeats):
    print(f"\n{name}")
    print(f"  {'case':<28}{'time':>12}{'majflt':>12}")
    results = {}
    for label, factory in cases.items():
        times, faults = [], []
        for _ in range(repeats):
            if cold:
                evict(path)
                assert residency(path) < 0.1, "eviction failed; cold numbers would be bogus"
            elapsed, majflt, _ = time_call(factory)
            times.append(elapsed)
            faults.append(majflt)
        best = min(times)
        results[label] = best
        print(f"  {label:<28}{best * 1000:>10.1f} ms{statistics.median(faults):>12.0f}")
    if "edfarray" in results and "pyedflib" in results and results["pyedflib"] > 0:
        ratio = results["pyedflib"] / results["edfarray"]
        print(f"  {'speedup vs pyedflib':<28}{ratio:>10.2f}x")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="existing EDF to benchmark; generated if omitted")
    ap.add_argument("--gb", type=float, default=4.0)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--rate", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--cold-only", action="store_true")
    ap.add_argument("--warm-only", action="store_true")
    args = ap.parse_args(argv)

    path = args.file or "/tmp/edfarray_large.edf"
    if not os.path.exists(path):
        generate(path, gb=args.gb, channels=args.channels, rate=args.rate)
    size_gib = os.path.getsize(path) / (1 << 30)
    print(f"file: {path} ({size_gib:.2f} GiB)")

    full = {
        "edfarray": edfarray_full_channel(path),
        "edfarray (force mmap)": edfarray_full_channel(path, "mmap"),
        "edfarray (force stream)": edfarray_full_channel(path, "stream"),
        "pyedflib": pyedflib_full_channel(path),
    }
    page = {"edfarray": edfarray_page(path), "pyedflib": pyedflib_page(path)}

    if not args.cold_only:
        for run in full.values():
            run()  # warm the cache
        bench("WARM -- full single-channel read", full, cold=False, path=path,
              repeats=args.repeats)
        bench("WARM -- 30s page, all channels", page, cold=False, path=path,
              repeats=args.repeats)

    if not args.warm_only:
        method = evict(path)
        print(f"\ncold-cache eviction: {method} (residency now {residency(path):.3f})")
        bench("COLD -- full single-channel read", full, cold=True, path=path,
              repeats=args.repeats)
        bench("COLD -- 30s page, all channels", page, cold=True, path=path,
              repeats=args.repeats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
