"""Annotation scan modes.

The eager background scan reads every data record at open. For very large files that competes
with the caller's own reads for page cache, so it can be deferred.
"""

import threading

import edfarray
from conftest import FIXTURES

ANNOTATED = str(FIXTURES / "edfPlusC.edf")


def test_lazy_scan_defers_until_annotations_are_touched():
    f = edfarray.EdfFile(ANNOTATED, scan_annotations=False)
    assert f.annotations_ready is False
    assert [a.text for a in f.annotations] == ["RECORD START", "REC STOP"]
    assert f.annotations_ready is True


def test_lazy_and_eager_agree():
    lazy = edfarray.EdfFile(ANNOTATED, scan_annotations=False)
    eager = edfarray.EdfFile(ANNOTATED)
    # `annotations` blocks until the index is ready in both modes.
    assert [(a.onset, a.duration, a.text) for a in lazy.annotations] == [
        (a.onset, a.duration, a.text) for a in eager.annotations
    ]
    assert lazy.warnings == eager.warnings


def test_lazy_scan_builds_once_under_concurrent_access():
    f = edfarray.EdfFile(ANNOTATED, scan_annotations=False)
    results = []
    errors = []

    def worker():
        try:
            results.append([a.text for a in f.annotations])
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert all(r == results[0] for r in results)


def test_signal_reads_work_without_scanning():
    # Reading sample data must not require the annotation index.
    f = edfarray.EdfFile(ANNOTATED, scan_annotations=False)
    data = f.signal(0).to_numpy()
    assert len(data) > 0
    assert f.annotations_ready is False
