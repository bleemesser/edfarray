import asyncio
from pathlib import Path

import numpy as np
import pytest

import edfarray
import edfarray.aio as aio

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _pick_fixture() -> Path:
    edfs = sorted(FIXTURES.glob("*.edf"))
    assert edfs, "no EDF fixtures found"
    return edfs[0]


async def test_open_returns_async_edffile():
    path = _pick_fixture()
    f = await aio.open(str(path))
    try:
        assert f.num_signals > 0
        assert f.num_records > 0
        assert f.duration > 0
        assert isinstance(f.variant, str)
        assert isinstance(f.signal_labels(), list)
        assert isinstance(f.header(), dict)
        assert not f.closed
    finally:
        f.close()
    assert f.closed


async def test_metadata_matches_sync():
    path = _pick_fixture()
    sync_f = edfarray.EdfFile(str(path))
    async_f = await aio.open(str(path))
    try:
        assert async_f.num_signals == sync_f.num_signals
        assert async_f.num_records == sync_f.num_records
        assert async_f.duration == sync_f.duration
        assert async_f.variant == sync_f.variant
        assert async_f.signal_labels() == sync_f.signal_labels()
        assert async_f.ordinary_signal_indices() == sync_f.ordinary_signal_indices()
    finally:
        async_f.close()


async def test_inspect_async():
    path = _pick_fixture()
    meta = await aio.inspect(str(path))
    assert isinstance(meta, dict)
    assert meta["num_signals"] > 0
    assert "variant" in meta
    assert "signal_labels" in meta


async def test_wait_for_annotations():
    path = _pick_fixture()
    f = await aio.open(str(path))
    try:
        await f.wait_for_annotations()
        assert f.annotations_ready
        await f.wait_for_annotations()
    finally:
        f.close()


async def test_async_context_manager():
    path = _pick_fixture()
    async with await aio.open(str(path)) as f:
        assert f.num_signals > 0
    assert f.closed


async def test_closed_file_raises():
    path = _pick_fixture()
    f = await aio.open(str(path))
    f.close()
    with pytest.raises(RuntimeError, match="closed"):
        await f.wait_for_annotations()


async def test_read_page_matches_sync():
    path = _pick_fixture()
    sync_f = edfarray.EdfFile(str(path))
    async_f = await aio.open(str(path))
    try:
        sync_arrays = sync_f.read_page(0.0, 1.0)
        async_arrays = await async_f.read_page(0.0, 1.0)
        assert len(sync_arrays) == len(async_arrays)
        for s, a in zip(sync_arrays, async_arrays):
            np.testing.assert_array_equal(s, a)
            assert a.dtype == np.float64
    finally:
        async_f.close()


async def test_read_page_digital_matches_sync():
    path = _pick_fixture()
    sync_f = edfarray.EdfFile(str(path))
    async_f = await aio.open(str(path))
    try:
        sync_arrays = sync_f.read_page_digital(0.0, 1.0)
        async_arrays = await async_f.read_page_digital(0.0, 1.0)
        assert len(sync_arrays) == len(async_arrays)
        for s, a in zip(sync_arrays, async_arrays):
            np.testing.assert_array_equal(s, a)
            assert a.dtype == np.int32
    finally:
        async_f.close()


async def test_read_page_signal_indices_subset():
    path = _pick_fixture()
    f = await aio.open(str(path))
    try:
        ordinary = f.ordinary_signal_indices()
        if len(ordinary) < 2:
            pytest.skip("need at least 2 ordinary signals")
        subset = ordinary[:2]
        arrays = await f.read_page(0.0, 1.0, signal_indices=subset)
        assert len(arrays) == 2
    finally:
        f.close()


async def test_read_page_concurrent_gather_works():
    path = _pick_fixture()
    f = await aio.open(str(path))
    try:
        results = await asyncio.gather(*[f.read_page(0.0, 1.0) for _ in range(4)])
        assert len(results) == 4
        for arrays in results:
            assert len(arrays) > 0
            assert all(a.dtype == np.float64 for a in arrays)
        for arrays in results[1:]:
            for a, b in zip(results[0], arrays):
                np.testing.assert_array_equal(a, b)
    finally:
        f.close()


async def test_read_page_releases_gil():
    import threading

    path = _pick_fixture()
    f = await aio.open(str(path))
    try:
        counter = [0]
        stop = threading.Event()

        def busy():
            while not stop.is_set():
                counter[0] += 1

        await f.read_page(0.0, f.duration)

        t = threading.Thread(target=busy)
        t.start()
        try:
            start = counter[0]
            await asyncio.gather(*[f.read_page(0.0, f.duration) for _ in range(8)])
            ticks = counter[0] - start
        finally:
            stop.set()
            t.join()

        assert ticks > 10_000, (
            f"python thread only advanced {ticks} ticks during 8 concurrent "
            f"reads — GIL likely held throughout decode"
        )
    finally:
        f.close()


async def test_read_page_closed_file_raises():
    path = _pick_fixture()
    f = await aio.open(str(path))
    f.close()
    with pytest.raises(RuntimeError, match="closed"):
        await f.read_page(0.0, 1.0)


async def test_signal_metadata_matches_sync():
    path = _pick_fixture()
    sync_f = edfarray.EdfFile(str(path))
    async_f = await aio.open(str(path))
    try:
        idx = async_f.ordinary_signal_indices()[0]
        sync_sig = sync_f.signal(idx)
        async_sig = async_f.signal(idx)
        assert async_sig.label == sync_sig.label
        assert async_sig.sample_rate == sync_sig.sample_rate
        assert async_sig.num_samples == sync_sig.num_samples
        assert async_sig.physical_min == sync_sig.physical_min
        assert async_sig.physical_max == sync_sig.physical_max
        assert len(async_sig) == len(sync_sig)
    finally:
        async_f.close()


async def test_signal_read_physical_matches_sync():
    path = _pick_fixture()
    sync_f = edfarray.EdfFile(str(path))
    async_f = await aio.open(str(path))
    try:
        idx = async_f.ordinary_signal_indices()[0]
        n = min(async_f.signal(idx).num_samples, 1000)
        sync_data = sync_f.signal(idx)[0:n]
        async_data = await async_f.signal(idx).read_physical(0, n)
        np.testing.assert_array_equal(sync_data, async_data)
        assert async_data.dtype == np.float64
    finally:
        async_f.close()


async def test_signal_to_numpy_and_to_digital():
    path = _pick_fixture()
    async_f = await aio.open(str(path))
    try:
        idx = async_f.ordinary_signal_indices()[0]
        sig = async_f.signal(idx)
        phys = await sig.to_numpy()
        dig = await sig.to_digital()
        assert phys.dtype == np.float64
        assert dig.dtype == np.int32
        assert phys.shape == dig.shape == (sig.num_samples,)
    finally:
        async_f.close()


async def test_signal_times():
    path = _pick_fixture()
    async_f = await aio.open(str(path))
    try:
        idx = async_f.ordinary_signal_indices()[0]
        sig = async_f.signal(idx)
        t = await sig.times()
        assert t.dtype == np.float64
        assert t.shape == (sig.num_samples,)
        assert (np.diff(t) >= 0).all()
    finally:
        async_f.close()


async def test_signal_by_label():
    path = _pick_fixture()
    async_f = await aio.open(str(path))
    try:
        labels = async_f.signal_labels()
        first_label = labels[0]
        sig = async_f.signal(first_label)
        assert sig.label == first_label
    finally:
        async_f.close()


async def test_signal_with_cache():
    path = _pick_fixture()
    async_f = await aio.open(str(path))
    try:
        idx = async_f.ordinary_signal_indices()[0]
        cached = async_f.signal(idx, cache_capacity=8)
        uncached = async_f.signal(idx)
        n = min(cached.num_samples, 500)
        a = await cached.read_physical(0, n)
        b = await uncached.read_physical(0, n)
        np.testing.assert_array_equal(a, b)
    finally:
        async_f.close()


async def test_signal_concurrent_reads_release_gil():
    import threading

    path = _pick_fixture()
    async_f = await aio.open(str(path))
    try:
        idx = async_f.ordinary_signal_indices()[0]
        sig = async_f.signal(idx)

        counter = [0]
        stop = threading.Event()
        def busy():
            while not stop.is_set():
                counter[0] += 1

        await sig.to_numpy()

        t = threading.Thread(target=busy)
        t.start()
        try:
            start = counter[0]
            await asyncio.gather(*[sig.to_numpy() for _ in range(8)])
            ticks = counter[0] - start
        finally:
            stop.set()
            t.join()

        assert ticks > 10_000, f"python thread only advanced {ticks} ticks"
    finally:
        async_f.close()


async def test_write_edf_async_roundtrip(tmp_path):
    out = tmp_path / "out.edf"
    sig = aio.WriterSignal(
        label="EEG Fz",
        physical_dimension="uV",
        physical_min=-100.0,
        physical_max=100.0,
        digital_min=-32768,
        digital_max=32767,
        samples_per_record=256,
    )
    data = np.linspace(-50.0, 50.0, 256 * 4).astype(np.float64)
    await aio.write_edf(
        str(out),
        variant="EDF",
        record_duration=1.0,
        signals=[sig],
        data=[data],
    )
    f = await aio.open(str(out))
    try:
        assert f.num_signals == 1
        assert f.num_records == 4
        s = f.signal(0)
        assert s.label.startswith("EEG Fz")
        read = await s.to_numpy()
        np.testing.assert_allclose(read, data, atol=0.01)
    finally:
        f.close()


async def test_streaming_async_writer(tmp_path):
    out = tmp_path / "stream.edf"
    sig = aio.WriterSignal(
        label="ch1",
        physical_dimension="uV",
        physical_min=-1.0,
        physical_max=1.0,
        digital_min=-32768,
        digital_max=32767,
        samples_per_record=10,
    )
    w = await aio.EdfWriter.create(
        str(out),
        variant="EDF",
        record_duration=1.0,
        signals=[sig],
    )
    for i in range(3):
        rec = np.full(10, float(i) * 0.1, dtype=np.float64)
        await w.write_record([rec])
    await w.finish()

    f = await aio.open(str(out))
    try:
        assert f.num_records == 3
        data = await f.signal(0).to_numpy()
        assert len(data) == 30
    finally:
        f.close()


async def test_async_writer_context_manager(tmp_path):
    out = tmp_path / "ctx.edf"
    sig = aio.WriterSignal(
        label="ch1",
        physical_dimension="uV",
        physical_min=-1.0,
        physical_max=1.0,
        digital_min=-32768,
        digital_max=32767,
        samples_per_record=10,
    )
    async with await aio.EdfWriter.create(
        str(out),
        variant="EDF",
        record_duration=1.0,
        signals=[sig],
    ) as w:
        rec = np.zeros(10, dtype=np.float64)
        await w.write_record([rec])
    f = await aio.open(str(out))
    try:
        assert f.num_records == 1
    finally:
        f.close()


async def test_async_writer_finish_idempotent_error(tmp_path):
    out = tmp_path / "double.edf"
    sig = aio.WriterSignal(
        label="ch1",
        physical_dimension="uV",
        physical_min=-1.0,
        physical_max=1.0,
        digital_min=-32768,
        digital_max=32767,
        samples_per_record=10,
    )
    w = await aio.EdfWriter.create(
        str(out),
        variant="EDF",
        record_duration=1.0,
        signals=[sig],
    )
    await w.write_record([np.zeros(10, dtype=np.float64)])
    await w.finish()
    with pytest.raises(ValueError, match="finished"):
        await w.finish()


async def test_write_to_async_roundtrip(tmp_path):
    src_path = _pick_fixture()
    dst_path = tmp_path / "copy.edf"
    f = await aio.open(str(src_path))
    try:
        await f.write_to(str(dst_path))
    finally:
        f.close()

    f2 = await aio.open(str(dst_path))
    try:
        assert f2.num_signals > 0
        assert f2.num_records > 0
    finally:
        f2.close()


async def test_write_to_async_transcode(tmp_path):
    src_path = _pick_fixture()
    dst_path = tmp_path / "transcoded.edf"
    f = await aio.open(str(src_path))
    src_variant = f.variant
    try:
        target = "EDF+C" if src_variant != "EDF+C" else "EDF"
        await f.write_to(str(dst_path), variant=target)
    finally:
        f.close()

    f2 = await aio.open(str(dst_path))
    try:
        assert f2.variant == ("EDF+C" if src_variant != "EDF+C" else "EDF")
    finally:
        f2.close()
