"""Malformed and hostile input must produce warnings or catchable exceptions, never a panic,
hang, or unbounded allocation."""

import edfarray
import pytest

from conftest import FIXTURES

MAIN_HEADER_SIZE = 256
SIGNAL_HEADER_SIZE = 256


def build(tmp_path, name, *, patch=None, data_records=None):
    """Write a copy of test_generator.edf with patched header bytes and truncated data."""
    src = (FIXTURES / "test_generator.edf").read_bytes()
    num_signals = int(src[252:256])
    header_bytes = MAIN_HEADER_SIZE + SIGNAL_HEADER_SIZE * num_signals
    header = bytearray(src[:header_bytes])
    if patch:
        for offset, value in patch.items():
            field = value.encode().ljust(len(value), b" ")
            header[offset : offset + len(field)] = field
    body = src[header_bytes:] if data_records is None else src[header_bytes:][:data_records]
    path = tmp_path / name
    path.write_bytes(bytes(header) + body)
    return str(path)


def field(text, width):
    return text.ljust(width)[:width]


def test_num_records_larger_than_file_is_clamped(tmp_path):
    path = build(tmp_path, "lying.edf", patch={236: field("99999999", 8)})
    f = edfarray.EdfFile(path)
    assert f.num_records < 99999999
    assert any("clamped" in w for w in f.warnings)
    sig = f.signal(0)
    assert len(sig) == f.num_records * sig.samples_per_record
    assert sig.to_physical().shape == (len(sig),)


def test_truncated_data_is_clamped(tmp_path):
    path = build(tmp_path, "truncated.edf", data_records=100)
    f = edfarray.EdfFile(path)
    assert f.num_records == 0
    assert f.signal(0).to_physical().shape == (0,)


def test_negative_record_duration_rejected(tmp_path):
    path = build(tmp_path, "negdur.edf", patch={244: field("-1", 8)})
    with pytest.raises(Exception):
        edfarray.EdfFile(path)


def test_zero_record_duration_opens_without_crashing(tmp_path):
    path = build(tmp_path, "zerodur.edf", patch={244: field("0", 8)})
    f = edfarray.EdfFile(path)
    assert f.signal(0).sample_rate == 0.0
    assert f.duration == 0.0


def test_header_size_field_mismatch_is_an_exception(tmp_path):
    path = build(tmp_path, "badhdr.edf", patch={184: field("123", 8)})
    with pytest.raises(Exception):
        edfarray.EdfFile(path)


def test_nonnumeric_header_field_is_an_exception(tmp_path):
    path = build(tmp_path, "junk.edf", patch={236: field("abcdefgh", 8)})
    with pytest.raises(Exception):
        edfarray.EdfFile(path)


def test_empty_file_is_an_exception(tmp_path):
    path = tmp_path / "empty.edf"
    path.write_bytes(b"")
    with pytest.raises(Exception):
        edfarray.EdfFile(str(path))


def test_degenerate_physical_range_warns_and_reads_digital(tmp_path):
    # physical_min == physical_max on signal 0: scaling is undefined, so values pass through
    # unscaled rather than making the file unopenable.
    src = (FIXTURES / "test_generator.edf").read_bytes()
    num_signals = int(src[252:256])
    header_bytes = MAIN_HEADER_SIZE + SIGNAL_HEADER_SIZE * num_signals
    header = bytearray(src[:header_bytes])
    pmin = MAIN_HEADER_SIZE + 104 * num_signals
    pmax = MAIN_HEADER_SIZE + 112 * num_signals
    header[pmin : pmin + 8] = field("0", 8).encode()
    header[pmax : pmax + 8] = field("0", 8).encode()
    path = tmp_path / "degenerate.edf"
    path.write_bytes(bytes(header) + src[header_bytes:])

    f = edfarray.EdfFile(str(path))
    assert any("degenerate" in w for w in f.warnings)
    sig = f.signal(0)
    assert (sig.to_physical() == sig.to_digital()).all()
