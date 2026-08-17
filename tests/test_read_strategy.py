"""Read strategies.

`auto` streams large reads that are not already cached and uses the memory map otherwise. All
three strategies must return identical data; only their I/O behavior differs.
"""

import edfarray
import numpy as np
import pytest
from conftest import FIXTURES

PATH = str(FIXTURES / "test_generator.edf")
STRATEGIES = ["auto", "mmap", "stream"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_full_read_identical_across_strategies(strategy):
    f = edfarray.EdfFile(PATH)
    expected = f.signal(0, strategy="mmap").to_numpy()
    got = f.signal(0, strategy=strategy).to_numpy()
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_digital_read_identical_across_strategies(strategy):
    f = edfarray.EdfFile(PATH)
    expected = f.signal(0, strategy="mmap").to_digital()
    got = f.signal(0, strategy=strategy).to_digital()
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("start,stop", [(0, 1), (0, 999), (1, 1000), (513, 2049), (0, None)])
def test_slices_identical_across_strategies(start, stop):
    f = edfarray.EdfFile(PATH)
    mm = f.signal(0, strategy="mmap")
    st = f.signal(0, strategy="stream")
    stop = len(mm) if stop is None else stop
    np.testing.assert_array_equal(st[start:stop], mm[start:stop])


def test_streaming_crosses_record_boundaries():
    # Ranges that start and end mid-record are where a chunked reader is most likely to slip.
    f = edfarray.EdfFile(PATH)
    mm = f.signal(0, strategy="mmap")
    st = f.signal(0, strategy="stream")
    spr = mm.samples_per_record
    for start in (spr - 1, spr, spr + 1, 2 * spr + 3):
        for length in (1, spr, spr + 1, 3 * spr - 2):
            if start + length > len(mm):
                continue
            np.testing.assert_array_equal(
                st[start : start + length], mm[start : start + length]
            )


def test_read_at_identical_across_strategies():
    f = edfarray.EdfFile(PATH)
    mm = f.signal(0, strategy="mmap").read_at(1.0, 3.5)
    st = f.signal(0, strategy="stream").read_at(1.0, 3.5)
    np.testing.assert_array_equal(st, mm)


def test_unknown_strategy_rejected():
    f = edfarray.EdfFile(PATH)
    with pytest.raises(ValueError):
        f.signal(0, strategy="fastest")


def test_strategy_works_with_cache():
    f = edfarray.EdfFile(PATH)
    expected = f.signal(0, strategy="mmap").to_numpy()
    got = f.signal(0, cache_capacity=4, strategy="stream").to_numpy()
    np.testing.assert_array_equal(got, expected)
