"""Epoch extraction: inputs, windows, data, padding."""

import edfarray
import edfarray.aio as aio
import numpy as np
import pytest
from conftest import FIXTURES

GEN = str(FIXTURES / "test_generator.edf")
GEN2 = str(FIXTURES / "test_generator_2.edf")  # EDF+C, has annotations


def test_single_float_event():
    with edfarray.EdfFile(GEN) as f:
        ep = f.extract_epochs(10.0, pre=1.0, post=1.0)
        assert len(ep) == 1


def test_annotation_events_and_alias():
    with edfarray.EdfFile(GEN2) as f:
        anns = f.annotations
        assert anns, "fixture expected to carry annotations"
        ep = f.extract_epochs(anns, pre=0.5, post=0.5)
        # Both fixture annotations sit at file edges (0 and 600 of 600), so with
        # pre=post=0.5 both are edge-affected and dropped. The bookkeeping is what
        # the assertion pins: kept + dropped == all events, order preserved.
        assert ep.dropped == [0, 1]
        assert len(ep) + len(ep.dropped) == len(anns)
        assert f.events("record") == f.filter_annotations("record", regex=False)


def test_query_sugar_matches_events_path():
    with edfarray.EdfFile(GEN2) as f:
        anns = f.filter_annotations("record", regex=False)
        ep_q = f.extract_epochs(None, pre=0.5, post=0.5, query="record")
        ep_e = f.extract_epochs(anns, pre=0.5, post=0.5)
        np.testing.assert_array_equal(ep_q.data, ep_e.data)


def test_events_and_query_together_rejected():
    with edfarray.EdfFile(GEN) as f:
        with pytest.raises(edfarray.InvalidArgumentError):
            f.extract_epochs([1.0], pre=0.5, post=0.5, query="record")


def test_bad_pre_post_rejected():
    with edfarray.EdfFile(GEN) as f:
        for pre, post in [(-1.0, 1.0), (1.0, -0.5), (0.0, 0.0), (float("nan"), 1.0)]:
            with pytest.raises(edfarray.InvalidArgumentError):
                f.extract_epochs([10.0], pre=pre, post=post)


def test_unsupported_event_types_rejected():
    with edfarray.EdfFile(GEN) as f:
        with pytest.raises(TypeError):
            f.extract_epochs([None], pre=1.0, post=1.0)
        with pytest.raises(TypeError):
            f.extract_epochs(object(), pre=1.0, post=1.0)
        with pytest.raises(ValueError):
            f.extract_epochs([float("nan")], pre=1.0, post=1.0)


def test_ndarray_events():
    with edfarray.EdfFile(GEN) as f:
        ep = f.extract_epochs(np.array([10.0, 20.0]), pre=1.0, post=1.0)
        assert ep.onsets.tolist() == [10.0, 20.0]


def test_epochs_contract():
    with edfarray.EdfFile(GEN) as f:
        ep = f.extract_epochs([10.0, 20.0], pre=1.0, post=1.0)
        arr = np.asarray(ep)
        np.testing.assert_array_equal(arr, ep.data)
        assert ep.data.dtype == np.float64
        assert ep.data.ndim == 3
        assert ep.data.shape[0] == 2
        assert ep.data.shape[2] == int(2.0 * max(g.sample_rate for g in f.signal_groups()))
        assert len(ep) == 2
        assert ep.valid.tolist() == [True, True]
        assert ep.dropped == []
        group = max(f.signal_groups(), key=len)
        assert ep.labels == [f.signal(i).label for i in group.indices]
        with pytest.raises(ValueError):
            np.asarray(ep, copy=False)
        assert "Epochs" in repr(ep)


def test_windows_table_matches_extraction():
    with edfarray.EdfFile(GEN) as f:
        events = [10.0, 20.0]
        windows, valid = f.epoch_windows(events, pre=1.0, post=1.0)
        assert valid.tolist() == [True, True]
        assert len(windows) == 2
        rate = max(f.signal_groups(), key=len).sample_rate
        for (onset, s_start, s_end), t in zip(windows, events):
            assert onset == t
            assert s_end - s_start == int(2.0 * rate)


def test_rows_match_single_signal_reads():
    with edfarray.EdfFile(GEN) as f:
        group = max(f.signal_groups(), key=len)
        events = [10.0, 25.0]
        ep = f.extract_epochs(events, pre=1.0, post=1.0, group=group)
        rate = group.sample_rate
        for e, t in enumerate(events):
            sig = f.signal(group.indices[0])
            want = sig[int((t - 1.0) * rate) : int(np.ceil((t + 1.0) * rate))]
            np.testing.assert_allclose(ep.data[e, 0, : len(want)], want)


def test_pad_modes():
    with edfarray.EdfFile(GEN) as f:
        rate = max(f.signal_groups(), key=len).sample_rate
        pre_n = int(2.0 * rate)  # t0=0 with pre=2.0 pads exactly the first pre_n columns
        events = [0.0, 30.0]  # 0.0 is an edge event; 30.0 is well inside
        ep = f.extract_epochs(events, pre=2.0, post=2.0, pad="drop")
        assert ep.dropped == [0]
        assert len(ep) == 1
        ep = f.extract_epochs(events, pre=2.0, post=2.0, pad="nan")
        assert ep.valid.tolist() == [False, True]
        assert np.isnan(ep.data[0, 0, :pre_n]).all()
        assert not np.isnan(ep.data[0, 0, pre_n:]).any()
        ep = f.extract_epochs(events, pre=2.0, post=2.0, pad="zero")
        assert ep.valid.tolist() == [False, True]
        assert (ep.data[0, 0, :pre_n] == 0.0).all()
        ep = f.extract_epochs(events, pre=2.0, post=2.0, pad=7.0)
        assert (ep.data[0, 0, :pre_n] == 7.0).all()
        ep = f.extract_epochs(events, pre=2.0, post=2.0, pad="edge")
        assert ep.valid.tolist() == [False, True]
        assert (ep.data[0, 0, :pre_n] == ep.data[0, 0, pre_n]).all()


def test_pad_raise():
    with edfarray.EdfFile(GEN) as f:
        with pytest.raises(edfarray.OutOfRangeError):
            f.extract_epochs([0.0, 30.0], pre=2.0, post=2.0, pad="raise")


def test_empty_events():
    with edfarray.EdfFile(GEN) as f:
        group = max(f.signal_groups(), key=len)
        ep = f.extract_epochs([], pre=1.0, post=1.0, group=group)
        assert ep.data.shape == (0, len(group.indices), int(2.0 * group.sample_rate))
        assert len(ep) == 0


def test_open_group_rejected():
    with edfarray.EdfFile(GEN) as f:
        groups = sorted(f.signal_groups(), key=len, reverse=True)
        indices = list(groups[0].indices[:1]) + list(groups[-1].indices[:1])
        mixed = f.signal_group(indices)
        with pytest.raises(edfarray.InvalidArgumentError):
            f.extract_epochs([10.0], pre=1.0, post=1.0, group=mixed)


async def test_async_extract_matches_sync():
    async with await aio.open(GEN) as af:
        ep = await af.extract_epochs([10.0, 20.0], pre=1.0, post=1.0)
        windows, valid = await af.epoch_windows([10.0, 20.0], pre=1.0, post=1.0)
    with edfarray.EdfFile(GEN) as f:
        sync_ep = f.extract_epochs([10.0, 20.0], pre=1.0, post=1.0)
    np.testing.assert_array_equal(ep.data, sync_ep.data)
    assert ep.labels == sync_ep.labels
    assert len(windows) == 2
    assert valid.tolist() == [True, True]


async def test_async_extract_pad_and_drop():
    async with await aio.open(GEN) as af:
        ep = await af.extract_epochs([0.0, 30.0], pre=2.0, post=2.0, pad="drop")
        assert ep.dropped == [0]
        ep2 = await af.extract_epochs([0.0, 30.0], pre=2.0, post=2.0, pad="nan")
        assert ep2.valid.tolist() == [False, True]


async def test_async_events_alias_and_query():
    async with await aio.open(GEN2) as af:
        assert af.events("record") == af.filter_annotations("record", regex=False)
        anns = af.filter_annotations("record", regex=False)
        ep = await af.extract_epochs(None, pre=0.5, post=0.5, query="record")
        assert len(ep) + len(ep.dropped) == len(anns)
