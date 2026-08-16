"""Annotation onset normalization and time-keeping TAL selection.

Ground truth for the compliant cases is pyedflib, which rejects a first record onset >= 1s
as non-compliant.
"""

import datetime

import edfarray
import numpy as np
import pytest
from edfbuilder import build_edf_plus


@pytest.mark.parametrize("subsecond", [0, 0.25, 0.9])
def test_subsecond_start_is_normalized_away(tmp_path, subsecond):
    # The first record's time-keeping onset carries the subsecond start offset, so annotation
    # onsets are reported relative to file start.
    onsets = [subsecond + i for i in range(4)]
    path = build_edf_plus(
        tmp_path / "sub.edf",
        record_onsets=onsets,
        annotations=[(2, onsets[2], "EVENT", 0)],
    )
    f = edfarray.EdfFile(path)
    assert [a.onset for a in f.annotations] == [2.0]
    assert not any("subsecond" in w for w in f.warnings)


@pytest.mark.parametrize("first", [1.0, 5.0])
def test_non_subsecond_first_onset_warns_and_keeps_stored_onsets(tmp_path, first):
    # pyedflib rejects these files outright. Rather than silently shifting every annotation by
    # a bogus offset, report the onsets as stored and warn.
    onsets = [first + i for i in range(4)]
    path = build_edf_plus(
        tmp_path / "notsub.edf",
        record_onsets=onsets,
        annotations=[(2, onsets[2], "EVENT", 0)],
    )
    f = edfarray.EdfFile(path)
    assert [a.onset for a in f.annotations] == [onsets[2]]
    assert any("subsecond start offset" in w for w in f.warnings)


def test_timekeeping_tal_only_taken_from_first_annotation_channel(tmp_path):
    # An empty-text TAL leading a *second* annotation channel is not a time-keeping TAL.
    path = build_edf_plus(
        tmp_path / "twochan.edf",
        record_onsets=[0, 1, 2, 3],
        annotation_channels=2,
        annotations=[(1, 1.0, "REAL", 1), (2, 2.0, "OTHER", 1)],
    )
    f = edfarray.EdfFile(path)
    assert [(a.onset, a.text) for a in f.annotations] == [(1.0, "REAL"), (2.0, "OTHER")]


def test_empty_tal_on_second_channel_is_not_mistaken_for_timekeeping(tmp_path):
    # Channel 0 carries no time-keeping TAL. A leading empty-text TAL on channel 1 must not be
    # adopted as the record onset, which would shift every annotation in the file.
    path = build_edf_plus(
        tmp_path / "notk.edf",
        record_onsets=[0, 1],
        annotation_channels=2,
        timekeeping=False,
        annotations=[(0, 9.0, "", 1), (0, 0.5, "EVENT", 1)],
    )
    f = edfarray.EdfFile(path)
    assert [(a.onset, a.text) for a in f.annotations] == [(0.5, "EVENT")]
    assert any("missing time-keeping" in w for w in f.warnings)


def test_annotations_on_second_channel_are_not_consumed(tmp_path):
    path = build_edf_plus(
        tmp_path / "chan2.edf",
        record_onsets=[0, 1],
        annotation_channels=2,
        annotations=[(0, 0.5, "A", 0), (0, 0.75, "B", 1)],
    )
    f = edfarray.EdfFile(path)
    assert {a.text for a in f.annotations} == {"A", "B"}


@pytest.mark.parametrize("microsecond", [0, 250000, 900000])
def test_subsecond_write_read_roundtrip(tmp_path, microsecond):
    # The writer folds the subsecond start into every TAL onset; the reader removes it again.
    sig = edfarray.WriterSignal(
        label="S1",
        physical_dimension="uV",
        physical_min=-100.0,
        physical_max=100.0,
        digital_min=-32768,
        digital_max=32767,
        samples_per_record=10,
    )
    path = str(tmp_path / "rt.edf")
    edfarray.write_edf(
        path,
        variant="EDF+C",
        record_duration=1.0,
        signals=[sig],
        data=[np.zeros(40)],
        annotations=[edfarray.Annotation(onset=2.0, text="EVENT")],
        start_datetime=datetime.datetime(2020, 1, 1, 0, 0, 0, microsecond),
    )
    f = edfarray.EdfFile(path)
    assert [(a.onset, a.text) for a in f.annotations] == [(2.0, "EVENT")]


@pytest.mark.parametrize("year,ok", [(1984, False), (1985, True), (2084, True), (2085, False)])
def test_unrepresentable_start_year_is_rejected(tmp_path, year, ok):
    # The EDF startdate field holds two year digits read back with an 85-pivot, so a year
    # outside 1985..=2084 would silently read back a century off.
    sig = edfarray.WriterSignal(
        label="S1",
        physical_dimension="uV",
        physical_min=-100.0,
        physical_max=100.0,
        digital_min=-32768,
        digital_max=32767,
        samples_per_record=10,
    )
    kwargs = dict(
        variant="EDF",
        record_duration=1.0,
        signals=[sig],
        data=[np.zeros(20)],
        start_datetime=datetime.datetime(year, 6, 1, 12, 0, 0),
    )
    path = str(tmp_path / f"y{year}.edf")
    if ok:
        edfarray.write_edf(path, **kwargs)
        assert edfarray.EdfFile(path).start_datetime.year == year
    else:
        with pytest.raises(Exception):
            edfarray.write_edf(path, **kwargs)


def test_annotation_text_with_reserved_tal_bytes_is_rejected(tmp_path):
    sig = edfarray.WriterSignal(
        label="S1",
        physical_dimension="uV",
        physical_min=-100.0,
        physical_max=100.0,
        digital_min=-32768,
        digital_max=32767,
        samples_per_record=10,
    )
    for bad in ["a\x14b", "a\x15b", "a\x00b"]:
        with pytest.raises(Exception):
            edfarray.write_edf(
                str(tmp_path / "bad.edf"),
                variant="EDF+C",
                record_duration=1.0,
                signals=[sig],
                data=[np.zeros(20)],
                annotations=[edfarray.Annotation(onset=1.0, text=bad)],
            )
