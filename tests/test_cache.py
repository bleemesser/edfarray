"""The optional per-signal record cache.

`cache_capacity` counts decoded records. The cache must never change what a read returns, only
how often it re-decodes, and it must stay bounded.
"""

import edfarray
import numpy as np
import pytest
from conftest import FIXTURES

PATH = str(FIXTURES / "test_generator.edf")


@pytest.fixture
def uncached():
    return edfarray.EdfFile(PATH).signal(0)


@pytest.mark.parametrize("capacity", [1, 2, 8, 1000])
def test_cache_never_changes_results(uncached, capacity):
    cached = edfarray.EdfFile(PATH).signal(0, cache_capacity=capacity)
    np.testing.assert_array_equal(cached.to_physical(), uncached.to_physical())
    np.testing.assert_array_equal(cached[100:5000], uncached[100:5000])
    assert cached[42] == uncached[42]


def test_capacity_zero_leaves_the_cache_disabled(uncached):
    cached = edfarray.EdfFile(PATH).signal(0, cache_capacity=0)
    np.testing.assert_array_equal(cached.to_physical(), uncached.to_physical())


def test_repeated_overlapping_reads_are_consistent(uncached):
    # The case the cache exists for: re-reading the same records many times.
    sig = edfarray.EdfFile(PATH).signal(0, cache_capacity=4)
    spr = sig.samples_per_record
    expected = uncached[0 : spr * 6]
    for _ in range(5):
        np.testing.assert_array_equal(sig[0 : spr * 6], expected)


def test_cache_smaller_than_the_window_still_correct(uncached):
    # Capacity 1 with a multi-record window forces eviction on every record.
    sig = edfarray.EdfFile(PATH).signal(0, cache_capacity=1)
    spr = sig.samples_per_record
    np.testing.assert_array_equal(sig[0 : spr * 10], uncached[0 : spr * 10])


def test_backwards_and_random_access_is_correct(uncached):
    sig = edfarray.EdfFile(PATH).signal(0, cache_capacity=3)
    spr = sig.samples_per_record
    for start in (spr * 5, 0, spr * 3, spr, spr * 5):
        np.testing.assert_array_equal(sig[start : start + spr], uncached[start : start + spr])


def test_digital_reads_bypass_the_cache(uncached):
    # Documented: the cache holds decoded physical values only.
    sig = edfarray.EdfFile(PATH).signal(0, cache_capacity=4)
    sig.to_physical()
    np.testing.assert_array_equal(sig.to_digital(), uncached.to_digital())


def test_cache_is_per_signal_instance():
    f = edfarray.EdfFile(PATH)
    first = f.signal(0, cache_capacity=4)
    second = f.signal(0, cache_capacity=4)
    np.testing.assert_array_equal(first.to_physical(), second.to_physical())


def test_cache_combines_with_read_strategies(uncached):
    for strategy in ("auto", "mmap", "stream"):
        sig = edfarray.EdfFile(PATH).signal(0, cache_capacity=4, strategy=strategy)
        np.testing.assert_array_equal(sig.to_physical(), uncached.to_physical())
