"""The sync API must be safe to call concurrently from multiple threads.

Blocking reads release the GIL and must not corrupt shared state, so concurrent reads from
a thread pool have to return identical results without raising.
"""

import threading

import edfarray
from conftest import FIXTURES

PATH = str(FIXTURES / "test_generator.edf")


def test_concurrent_reads_from_threads_agree():
    f = edfarray.EdfFile(PATH)
    sig = f.signal(0)
    expected = sig.to_physical()
    results = []
    errors = []

    def worker():
        try:
            results.append(sig.to_physical())
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
