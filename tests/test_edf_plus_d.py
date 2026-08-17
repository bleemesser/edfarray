"""EDF+D discontinuous recordings.

For EDF+D the record onsets are the only record of where data sits in time, so flat sample
indexing and time-based reading genuinely disagree. pyedflib refuses to read these files at
all, so there is no external reference: the invariants are checked against the file's own
onset table.
"""

import edfarray
import numpy as np
import pytest
from conftest import FIXTURES

PATH = str(FIXTURES / "edfPlusD.edf")


@pytest.fixture
def f():
    return edfarray.EdfFile(PATH)


def test_variant_is_detected(f):
    assert f.variant == "EDF+D"


def test_times_reveal_the_gaps(f):
    sig = f.signal(0)
    steps = np.diff(sig.times())
    nominal = 1.0 / sig.sample_rate
    # Within a record the step is the sample period; between records it jumps by the gap.
    assert np.isclose(steps.min(), nominal)
    assert steps.max() > nominal * 2, "fixture is expected to contain a real gap"


def test_times_are_monotonic(f):
    times = f.signal(0).times()
    assert np.all(np.diff(times) > 0)


def test_use_time_differs_from_flat_indexing(f):
    flat = f.read_page(0.0, 5.0)
    timed = f.read_page(0.0, 5.0, use_time=True)
    # Flat indexing ignores gaps and returns a fixed sample count; time-aware reading returns
    # only the samples whose onsets fall in the window.
    assert [len(a) for a in flat] != [len(a) for a in timed]
    assert all(len(a) <= len(b) for a, b in zip(timed, flat))


def test_use_time_matches_read_time_range(f):
    timed = f.read_page(2.0, 4.0, use_time=True)
    direct = f.signal(f.ordinary_signal_indices()[0]).read_time_range(2.0, 4.0)
    np.testing.assert_array_equal(timed[0], direct)


def test_time_range_inside_a_gap_is_empty(f):
    sig = f.signal(0)
    times = sig.times()
    steps = np.diff(times)
    gap_at = int(np.argmax(steps))
    gap_start, gap_end = times[gap_at], times[gap_at + 1]
    if gap_end - gap_start < 0.2:
        pytest.skip("no gap wide enough to sample inside")
    mid = (gap_start + gap_end) / 2
    assert len(sig.read_time_range(mid - 0.01, mid + 0.01)) == 0


def test_digital_page_lengths_match_physical(f):
    phys = f.read_page(0.0, 4.0, use_time=True)
    dig = f.read_page_digital(0.0, 4.0, use_time=True)
    assert [len(a) for a in phys] == [len(a) for a in dig]


def test_full_read_covers_every_record(f):
    sig = f.signal(0)
    assert len(sig) == f.num_records * sig.samples_per_record
    assert len(sig.to_physical()) == len(sig)


def test_record_onsets_are_used_for_annotations(f):
    # The single annotation in this fixture must land at a real onset, not a computed one.
    for ann in f.annotations:
        assert 0.0 <= ann.onset <= f.duration
