"""Tests for explicit EdfFile.close()."""

import pytest

from conftest import FIXTURES


def test_close_releases_and_blocks_access():
    f = __import__("edfarray").EdfFile(str(FIXTURES / "test_generator.edf"))
    assert f.closed is False
    f.close()
    assert f.closed is True
    with pytest.raises(BaseException):
        _ = f.num_signals


def test_close_is_idempotent():
    import edfarray

    f = edfarray.EdfFile(str(FIXTURES / "test_generator.edf"))
    f.close()
    f.close()
    assert f.closed is True


def test_signal_outlives_close():
    import edfarray

    f = edfarray.EdfFile(str(FIXTURES / "test_generator.edf"))
    sig = f.signal(0)
    expected_len = sig.num_samples
    f.close()
    assert len(sig.to_numpy()) == expected_len


def test_context_manager_closes():
    import edfarray

    with edfarray.EdfFile(str(FIXTURES / "test_generator.edf")) as f:
        assert f.closed is False
    assert f.closed is True


def test_repr_when_closed():
    import edfarray

    f = edfarray.EdfFile(str(FIXTURES / "test_generator.edf"))
    f.close()
    assert "closed" in repr(f)
