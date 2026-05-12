import asyncio
from pathlib import Path

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
