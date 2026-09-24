"""Channel subset selection in `EdfFile.write_to` (sync and async).

`signals` picks which ordinary channels land in the destination file, in the
given order. Annotation handling, variant transcoding, and EDF+D gaps behave
exactly as for a full copy.
"""

from pathlib import Path

import numpy as np
import pytest

import edfarray
import edfarray.aio as aio
from edfbuilder import build_edf_plus

FLAT_ONSETS = [0.0, 1.0, 2.0, 3.0]
GAP_ONSETS = [0.0, 1.0, 3.0, 6.0]


def _multi(path, variant="EDF+C", onsets=None, rates=(10, 20, 30)):
    """A 3-channel file where each sample value encodes (record, channel, sample).

    Values stay inside the builder's physical range, so the write_to
    physical->digital re-encode round-trips exactly.
    """
    onsets = onsets or FLAT_ONSETS
    signals = [
        {"label": f"Ch{i}", "rate": rate, "values": lambda r, s, i=i: r * 1000 + i * 100 + s}
        for i, rate in enumerate(rates)
    ]
    build_edf_plus(
        str(path),
        record_onsets=onsets,
        annotations=[(1, 1.5, "mid", 0), (1, 1.75, "late", 0)],
        variant=variant,
        signals=signals,
    )
    return str(path)


def test_subset_by_index_preserves_values_and_annotations(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    dst = tmp_path / "dst.edf"

    f = edfarray.EdfFile(src)
    f.write_to(str(dst), signals=[1])

    g = edfarray.EdfFile(str(dst))
    assert g.variant == "EDF+C"
    assert g.num_records == 4
    assert g.patient_id == f.patient_id
    ordinary = [l for l in g.signal_labels() if l != "EDF Annotations"]
    assert ordinary == ["Ch1"]
    assert [a.text for a in g.annotations] == ["mid", "late"]
    np.testing.assert_array_equal(
        g.signal("Ch1").to_digital(), f.signal(1).to_digital()
    )


def test_subset_mixed_selection_preserves_order(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    dst = tmp_path / "dst.edf"

    f = edfarray.EdfFile(src)
    f.write_to(str(dst), signals=["Ch2", 0])

    g = edfarray.EdfFile(str(dst))
    ordinary = [l for l in g.signal_labels() if l != "EDF Annotations"]
    assert ordinary == ["Ch2", "Ch0"]
    np.testing.assert_array_equal(
        g.signal(0).to_digital(), f.signal(2).to_digital()
    )
    np.testing.assert_array_equal(
        g.signal(1).to_digital(), f.signal(0).to_digital()
    )


def test_subset_scalar_index_and_label(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")

    f = edfarray.EdfFile(src)
    f.write_to(str(tmp_path / "by_index.edf"), signals=1)
    f.write_to(str(tmp_path / "by_label.edf"), signals="Ch1")

    for name in ("by_index.edf", "by_label.edf"):
        g = edfarray.EdfFile(str(tmp_path / name))
        ordinary = [l for l in g.signal_labels() if l != "EDF Annotations"]
        assert ordinary == ["Ch1"]
        np.testing.assert_array_equal(
            g.signal(0).to_digital(), f.signal(1).to_digital()
        )


def test_subset_accepts_signal_group(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    dst = tmp_path / "dst.edf"

    f = edfarray.EdfFile(src)
    group = f.signal_group([2, 0])
    f.write_to(str(dst), signals=group)

    g = edfarray.EdfFile(str(dst))
    ordinary = [l for l in g.signal_labels() if l != "EDF Annotations"]
    assert ordinary == ["Ch2", "Ch0"]
    np.testing.assert_array_equal(
        g.signal(0).to_digital(), f.signal(2).to_digital()
    )


def test_subset_transcode_to_plain_drops_annotations(tmp_path: Path):
    src = _multi(tmp_path / "src.edf", variant="EDF+C")
    dst = tmp_path / "dst.edf"

    f = edfarray.EdfFile(src)
    f.write_to(str(dst), variant="EDF", signals=[0])

    g = edfarray.EdfFile(str(dst))
    assert g.variant == "EDF"
    assert g.signal_labels() == ["Ch0"]
    assert list(g.annotations) == []
    np.testing.assert_array_equal(
        g.signal(0).to_digital(), f.signal(0).to_digital()
    )


def test_subset_bdf_source_to_bdf(tmp_path: Path):
    src = tmp_path / "src.bdf"
    dst = tmp_path / "dst.bdf"

    def _bdf_signal(label, spr):
        return edfarray.WriterSignal(
            label=label,
            physical_dimension="uV",
            physical_min=-1000.0,
            physical_max=1000.0,
            digital_min=-8388607,
            digital_max=8388607,
            samples_per_record=spr,
        )

    n = 100 * 4
    data_a = 500.0 * np.sin(np.arange(n) * 2 * np.pi * 7 / 100)
    data_b = -300.0 * np.sin(np.arange(n) * 2 * np.pi * 13 / 100)
    edfarray.write_edf(
        str(src),
        variant="BDF",
        record_duration=1.0,
        signals=[_bdf_signal("A", 100), _bdf_signal("B", 100)],
        data=[data_a, data_b],
    )

    f = edfarray.EdfFile(str(src))
    f.write_to(str(dst), variant="BDF", signals=[1])

    g = edfarray.EdfFile(str(dst))
    assert g.variant == "BDF"
    assert g.signal_labels() == ["B"]
    # Same family, same digital range: 24-bit values round-trip exactly.
    np.testing.assert_array_equal(g.signal(0).to_digital(), f.signal(1).to_digital())


def test_subset_preserves_edf_plus_d_gaps(tmp_path: Path):
    src = _multi(tmp_path / "src.edf", variant="EDF+D", onsets=GAP_ONSETS)
    dst = tmp_path / "dst.edf"

    f = edfarray.EdfFile(src)
    src_times = f.signal(1).times()

    f.write_to(str(dst), signals=[1])

    g = edfarray.EdfFile(str(dst))
    assert g.variant == "EDF+D"
    np.testing.assert_allclose(g.signal(0).times(), src_times, atol=1e-9)
    assert [a.text for a in g.annotations] == ["mid", "late"]


def test_subset_none_explicit_copies_every_ordinary_signal(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    dst = tmp_path / "dst.edf"

    f = edfarray.EdfFile(src)
    f.write_to(str(dst), signals=None)

    g = edfarray.EdfFile(str(dst))
    assert g.signal_labels() == f.signal_labels()


def test_subset_empty_selection_rejected(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    dst = tmp_path / "x.edf"
    f = edfarray.EdfFile(src)
    with pytest.raises(ValueError):
        f.write_to(str(dst), signals=[])
    assert not dst.exists(), "a rejected write must not leave a destination file"


def test_subset_out_of_range_rejected(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    dst = tmp_path / "x.edf"
    f = edfarray.EdfFile(src)
    with pytest.raises(IndexError):
        f.write_to(str(dst), signals=[99])
    assert not dst.exists(), "a rejected write must not leave a destination file"


def test_subset_annotation_channel_rejected(tmp_path: Path):
    src = _multi(tmp_path / "src.edf", variant="EDF+C")
    f = edfarray.EdfFile(src)
    ann_idx = f.num_signals - 1
    assert f.signal_labels()[ann_idx] == "EDF Annotations"
    with pytest.raises(ValueError):
        f.write_to(str(tmp_path / "x.edf"), signals=[ann_idx])
    with pytest.raises(ValueError):
        f.write_to(str(tmp_path / "x.edf"), signals=[0, ann_idx])


def test_subset_duplicate_rejected(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    f = edfarray.EdfFile(src)
    with pytest.raises(ValueError):
        f.write_to(str(tmp_path / "x.edf"), signals=[0, 1, 0])


def test_subset_unknown_label_rejected(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    dst = tmp_path / "x.edf"
    f = edfarray.EdfFile(src)
    with pytest.raises(KeyError):
        f.write_to(str(dst), signals=["Nope"])
    assert not dst.exists(), "a rejected write must not leave a destination file"
    with pytest.raises(KeyError):
        f.write_to(str(tmp_path / "y.edf"), signals=["ch1"])  # exact match only


def test_subset_bad_element_rejected(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    f = edfarray.EdfFile(src)
    with pytest.raises(TypeError):
        f.write_to(str(tmp_path / "x.edf"), signals=[None])
    with pytest.raises(TypeError):
        f.write_to(str(tmp_path / "x.edf"), signals=[3.5])
    # Bytes are not channel identifiers: bytes and bytearrays would otherwise be
    # iterated as small ints and silently select the wrong channels.
    with pytest.raises(TypeError):
        f.write_to(str(tmp_path / "x.edf"), signals=b"AB")
    with pytest.raises(TypeError):
        f.write_to(str(tmp_path / "x.edf"), signals=[b"A"])


def test_subset_non_sequencable_rejected(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    f = edfarray.EdfFile(src)
    with pytest.raises(TypeError):
        f.write_to(str(tmp_path / "x.edf"), signals=object())


async def test_write_to_async_subset(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    dst = tmp_path / "dst.edf"

    f = await aio.open(src)
    try:
        await f.write_to(str(dst), signals=[2, "Ch0"])
    finally:
        f.close()

    g = edfarray.EdfFile(str(dst))
    ordinary = [l for l in g.signal_labels() if l != "EDF Annotations"]
    assert ordinary == ["Ch2", "Ch0"]
    assert [a.text for a in g.annotations] == ["mid", "late"]

    f = edfarray.EdfFile(src)
    np.testing.assert_array_equal(
        g.signal(0).to_digital(), f.signal(2).to_digital()
    )


async def test_write_to_async_subset_rejects_errors(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    f = await aio.open(src)
    try:
        with pytest.raises(IndexError):
            await f.write_to(str(tmp_path / "x.edf"), signals=[99])
    finally:
        f.close()
    assert not (tmp_path / "x.edf").exists()


def test_subset_negative_index_rejected(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    f = edfarray.EdfFile(src)
    with pytest.raises(IndexError):
        f.write_to(str(tmp_path / "x.edf"), signals=-1)
    with pytest.raises(IndexError):
        f.write_to(str(tmp_path / "x.edf"), signals=[0, -1])


def test_subset_bool_and_unordered_rejected(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    f = edfarray.EdfFile(src)
    for bad in (True, [True], {0, 1}, frozenset({0}), {0: "a"}):
        with pytest.raises(TypeError):
            f.write_to(str(tmp_path / "x.edf"), signals=bad)


def test_subset_accepts_numpy_indices(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    f = edfarray.EdfFile(src)
    f.write_to(str(tmp_path / "a.edf"), signals=np.array([2, 0]))
    f.write_to(str(tmp_path / "b.edf"), signals=np.int64(1))
    ordinary = lambda p: [
        l for l in edfarray.EdfFile(str(p)).signal_labels() if l != "EDF Annotations"
    ]
    assert ordinary(tmp_path / "a.edf") == ["Ch2", "Ch0"]
    assert ordinary(tmp_path / "b.edf") == ["Ch1"]


def test_write_over_own_path_is_rejected_and_source_survives(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    before = Path(src).read_bytes()
    f = edfarray.EdfFile(src)
    with pytest.raises(edfarray.EdfFileError, match="another edfarray handle"):
        f.write_to(src, signals=[0])
    with pytest.raises(OSError):
        f.write_to(src)
    f.close()
    assert Path(src).read_bytes() == before


def test_write_over_mapped_path_waits_for_derived_objects(tmp_path: Path):
    src = edfarray.EdfFile(_multi(tmp_path / "src.edf"))
    dst = _multi(tmp_path / "dst.edf")

    held = edfarray.EdfFile(dst)
    sig = held.signal(0)
    held.close()
    with pytest.raises(edfarray.EdfFileError):
        src.write_to(dst, signals=[1])

    del sig
    src.write_to(dst, signals=[1])
    assert "Ch1" in edfarray.EdfFile(dst).signal_labels()


def test_rewrite_after_drop_mid_scan(tmp_path: Path):
    src = edfarray.EdfFile(_multi(tmp_path / "src.edf"))
    dst = str(tmp_path / "dst.edf")
    for i in range(20):
        src.write_to(dst, signals=[i % 3])
        edfarray.EdfFile(dst).signal_labels()


async def test_write_to_async_own_path_rejected(tmp_path: Path):
    src = _multi(tmp_path / "src.edf")
    before = Path(src).read_bytes()
    f = await aio.open(src)
    try:
        with pytest.raises(edfarray.EdfFileError):
            await f.write_to(src, signals=[0])
    finally:
        f.close()
    assert Path(src).read_bytes() == before


def test_negative_signal_index_is_out_of_range(tmp_path: Path):
    f = edfarray.EdfFile(_multi(tmp_path / "src.edf"))
    with pytest.raises(edfarray.OutOfRangeError):
        f.signal(-1)


async def test_negative_signal_index_is_out_of_range_async(tmp_path: Path):
    f = await aio.open(_multi(tmp_path / "src.edf"))
    try:
        with pytest.raises(edfarray.OutOfRangeError):
            f.signal(-1)
    finally:
        f.close()
