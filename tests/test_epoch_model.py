"""Randomized epoch extraction checked against a slow model built from `Signal.times()`.

The model never touches the epoch planner. It maps each sample's time to a grid index and
fills every row column by column, so a rounding or placement defect in the planner shows up
as a mismatch. Seeds are fixed, so a failure reproduces.
"""

import math

import edfarray
import edfarray.aio as aio
import numpy as np
import pytest
from edfbuilder import build_edf_plus

GRID_EPS = 1e-6


def _first_at_or_after(x):
    k = math.floor(x) - 2
    while k < x - GRID_EPS:
        k += 1
    return k


def _build(tmp_path, rng, gapped):
    rate = int(rng.integers(1, 33))
    onsets = [0.0]
    for _ in range(int(rng.integers(0, 6))):
        gap = int(rng.integers(0, 4)) if gapped else 0
        onsets.append(onsets[-1] + 1.0 + gap)
    path = tmp_path / f"model_{rate}_{len(onsets)}_{int(gapped)}.edf"
    build_edf_plus(
        str(path),
        record_onsets=onsets,
        variant="EDF+D" if gapped else "EDF+C",
        signals=[
            {"label": "A", "rate": rate, "values": lambda r, s, n=rate: r * n + s},
            {"label": "B", "rate": rate, "values": lambda r, s, n=rate: -(r * n + s)},
        ],
    )
    return str(path), rate, onsets[-1] + 1.0


def _events(rng, span):
    free = rng.uniform(-3.0, span + 3.0, size=6)
    lattice = rng.integers(-30, int(span + 3) * 10, size=6) * 0.1
    noise = rng.choice([-1e-12, 0.0, 1e-12], size=6)
    return np.concatenate([free, lattice + noise]).tolist()


def _model(f, rate, events, pre, post):
    n = max(_first_at_or_after((pre + post) * rate), 1)
    channels = []
    for label in ("A", "B"):
        sig = f.signal(label)
        by_grid = {round(t * rate): v for t, v in zip(sig.times(), sig[:])}
        rows = np.full((len(events), n), np.nan)
        for e, t in enumerate(events):
            first = _first_at_or_after((t - pre) * rate)
            for j in range(n):
                rows[e, j] = by_grid.get(first + j, np.nan)
        channels.append(rows)
    return np.stack(channels, axis=1)


@pytest.mark.parametrize("gapped", [False, True])
@pytest.mark.parametrize("seed", range(40))
def test_extraction_matches_model(tmp_path, seed, gapped):
    rng = np.random.default_rng(seed)
    path, rate, span = _build(tmp_path, rng, gapped)
    events = _events(rng, span)
    pre, post = (float(v) for v in rng.uniform(0.0, 3.0, size=2))
    if rng.random() < 0.3:
        pre = round(pre, 1)
        post = round(post, 1)
    if pre + post == 0.0:
        post = 0.5

    with edfarray.EdfFile(path) as f:
        want = _model(f, rate, events, pre, post)
        want_valid = ~np.isnan(want).any(axis=(1, 2))

        filled = f.extract_epochs(events, pre=pre, post=post, pad="nan")
        np.testing.assert_array_equal(filled.data, want)
        np.testing.assert_array_equal(filled.valid, want_valid)

        kept = f.extract_epochs(events, pre=pre, post=post)
        np.testing.assert_array_equal(kept.data, want[want_valid])
        assert kept.dropped == np.flatnonzero(~want_valid).tolist()
        np.testing.assert_array_equal(kept.onsets, np.asarray(events)[want_valid])
        assert kept.valid.all()

        _, planned = f.epoch_windows(events, pre=pre, post=post)
        np.testing.assert_array_equal(planned, want_valid)

        held = f.extract_epochs(events, pre=pre, post=post, pad="edge")
        assert np.isfinite(held.data).all()
        real = ~np.isnan(want)
        np.testing.assert_array_equal(held.data[real], want[real])

        sentinel = f.extract_epochs(events, pre=pre, post=post, pad=-12345.0)
        np.testing.assert_array_equal(sentinel.data[~real], -12345.0)

        if want_valid.all():
            f.extract_epochs(events, pre=pre, post=post, pad="raise")
        else:
            with pytest.raises(edfarray.OutOfRangeError):
                f.extract_epochs(events, pre=pre, post=post, pad="raise")


@pytest.mark.parametrize("seed", range(10))
def test_time_range_reads_match_model(tmp_path, seed):
    """`read_time_range` and `read_page(use_time=True)` select samples by the same rule."""
    rng = np.random.default_rng(1000 + seed)
    path, rate, span = _build(tmp_path, rng, True)
    with edfarray.EdfFile(path) as f:
        sig = f.signal("A")
        grid = np.round(sig.times() * rate).astype(int)
        values = sig[:]
        for start, end in zip(_events(rng, span), _events(rng, span)):
            lo = _first_at_or_after(max(start, 0.0) * rate)
            hi = _first_at_or_after(max(end, 0.0) * rate)
            inside = np.flatnonzero((grid >= lo) & (grid < hi))
            got = sig.read_time_range(start, end)
            page = f.read_page(start, end, [0], use_time=True)[0]
            np.testing.assert_array_equal(got, page)
            if inside.size == 0:
                assert got.size == 0
            else:
                np.testing.assert_array_equal(got, values[inside[0] : inside[-1] + 1])


@pytest.mark.parametrize("seed", range(5))
async def test_async_matches_sync(tmp_path, seed):
    rng = np.random.default_rng(2000 + seed)
    path, _, span = _build(tmp_path, rng, True)
    events = _events(rng, span)
    with edfarray.EdfFile(path) as f:
        want = f.extract_epochs(events, pre=0.7, post=1.3, pad="nan")
    async with await aio.open(path) as f:
        await f.wait_for_annotations()
        got = await f.extract_epochs(events, pre=0.7, post=1.3, pad="nan")
    np.testing.assert_array_equal(got.data, want.data)
    np.testing.assert_array_equal(got.valid, want.valid)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pre": -0.1, "post": 1.0},
        {"pre": 0.0, "post": 0.0},
        {"pre": float("nan"), "post": 1.0},
        {"pre": float("inf"), "post": 1.0},
    ],
)
def test_bad_windows_raise(tmp_path, kwargs):
    path, _, _ = _build(tmp_path, np.random.default_rng(7), False)
    with edfarray.EdfFile(path) as f, pytest.raises(ValueError):
        f.extract_epochs([0.5], **kwargs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_events_raise(tmp_path, bad):
    path, _, _ = _build(tmp_path, np.random.default_rng(7), False)
    with edfarray.EdfFile(path) as f, pytest.raises(ValueError):
        f.extract_epochs([0.5, bad], pre=0.1, post=0.1, pad="nan")


def test_extreme_events_do_not_overflow(tmp_path):
    path, _, _ = _build(tmp_path, np.random.default_rng(7), True)
    with edfarray.EdfFile(path) as f:
        ep = f.extract_epochs([-1e15, 1e15, 1e300], pre=0.1, post=0.1, pad="nan")
        assert not ep.valid.any()
        assert np.isnan(ep.data).all()
        held = f.extract_epochs([-1e15, 1e15], pre=0.1, post=0.1, pad="edge")
        assert np.isfinite(held.data).all()


def test_empty_events_keep_shape(tmp_path):
    path, rate, _ = _build(tmp_path, np.random.default_rng(7), False)
    with edfarray.EdfFile(path) as f:
        ep = f.extract_epochs([], pre=0.1, post=0.2)
        assert ep.data.shape == (0, 2, _first_at_or_after(0.3 * rate))
