"""The sync API must release the GIL around blocking reads.

Without this, a single large read stalls every other Python thread for its whole duration,
which makes the sync API unusable from a thread pool.
"""

import os
import threading
import time

import edfarray
import pytest
from conftest import FIXTURES

PATH = str(FIXTURES / "test_generator.edf")

needs_cores = pytest.mark.skipif(
    (os.cpu_count() or 1) < 4, reason="needs >=4 cores to observe parallel speedup"
)


def _time_burst(work, total, threads):
    per_thread = total // threads
    workers = [threading.Thread(target=lambda: [work() for _ in range(per_thread)])
               for _ in range(threads)]
    start = time.perf_counter()
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    return time.perf_counter() - start


@needs_cores
def test_signal_read_scales_across_threads():
    # If the GIL were held for the duration of each read, threads would serialize and the
    # speedup would be ~1x regardless of core count.
    f = edfarray.EdfFile(PATH)
    sig = f.signal(0)
    work = sig.to_numpy
    work()  # warm the page cache so the comparison measures decode, not first-touch I/O

    sequential = _time_burst(work, 200, 1)
    parallel = _time_burst(work, 200, 4)
    assert sequential / parallel > 1.5, (
        f"no parallel speedup: {sequential:.3f}s sequential vs {parallel:.3f}s on 4 threads"
    )


@needs_cores
def test_read_page_scales_across_threads():
    f = edfarray.EdfFile(PATH)
    work = lambda: f.read_page(0.0, 10.0)  # noqa: E731
    work()

    sequential = _time_burst(work, 200, 1)
    parallel = _time_burst(work, 200, 4)
    assert sequential / parallel > 1.5, (
        f"no parallel speedup: {sequential:.3f}s sequential vs {parallel:.3f}s on 4 threads"
    )


def test_concurrent_reads_from_threads_agree():
    f = edfarray.EdfFile(PATH)
    sig = f.signal(0)
    expected = sig.to_numpy()
    results = []
    errors = []

    def worker():
        try:
            results.append(sig.to_numpy())
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 8
    for got in results:
        assert (got == expected).all()
