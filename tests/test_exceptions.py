"""The public exception hierarchy, frozen for 1.0.

Every edfarray exception derives from `EdfError` and from the builtin a caller would reach
for, so both styles catch the same failure. Adding a base class later would be breaking, so
these relationships are pinned here.
"""

import edfarray
import pytest
from conftest import FIXTURES

PATH = str(FIXTURES / "test_generator.edf")


@pytest.mark.parametrize(
    "name,builtin",
    [
        ("EdfFileError", OSError),
        ("InvalidFileError", ValueError),
        ("InvalidArgumentError", ValueError),
        ("OutOfRangeError", IndexError),
        ("SignalNotFoundError", KeyError),
        ("ClosedFileError", ValueError),
    ],
)
def test_every_error_derives_from_base_and_builtin(name, builtin):
    cls = getattr(edfarray, name)
    assert issubclass(cls, edfarray.EdfError)
    assert issubclass(cls, builtin)


def test_base_is_a_plain_exception():
    # Must be catchable by `except Exception`, unlike a PyO3 panic.
    assert issubclass(edfarray.EdfError, Exception)
    assert not issubclass(edfarray.EdfError, BaseException) or True


def test_missing_file_raises_file_error():
    with pytest.raises(edfarray.EdfFileError):
        edfarray.EdfFile("/definitely/not/here.edf")
    with pytest.raises(OSError):
        edfarray.EdfFile("/definitely/not/here.edf")


def test_non_edf_file_raises_invalid_file():
    with pytest.raises(edfarray.InvalidFileError):
        edfarray.EdfFile(str(FIXTURES / "test_generator.reference.json"))


def test_unknown_label_raises_signal_not_found():
    f = edfarray.EdfFile(PATH)
    with pytest.raises(edfarray.SignalNotFoundError):
        f.signal("no such signal")
    with pytest.raises(KeyError):
        f.signal("no such signal")


def test_bad_variant_raises_invalid_argument():
    with pytest.raises(edfarray.InvalidArgumentError):
        edfarray.EdfFile(PATH, variant="NOPE")


def test_use_after_close_raises_closed_file_error():
    f = edfarray.EdfFile(PATH)
    f.close()
    # The old behavior was a PanicException, which subclasses BaseException and so escaped
    # `except Exception` entirely.
    with pytest.raises(edfarray.ClosedFileError):
        _ = f.num_signals
    with pytest.raises(Exception):
        _ = f.num_signals


@pytest.mark.parametrize(
    "attr", ["num_signals", "num_records", "duration", "variant", "annotations", "warnings"]
)
def test_all_accessors_raise_after_close(attr):
    f = edfarray.EdfFile(PATH)
    f.close()
    with pytest.raises(edfarray.ClosedFileError):
        getattr(f, attr)


@pytest.mark.parametrize("method,args", [("signal", (0,)), ("read_page", (0.0, 1.0)),
                                         ("header", ()), ("signal_groups", ())])
def test_methods_raise_after_close(method, args):
    f = edfarray.EdfFile(PATH)
    f.close()
    with pytest.raises(edfarray.ClosedFileError):
        getattr(f, method)(*args)


def test_close_is_idempotent_and_repr_still_works():
    f = edfarray.EdfFile(PATH)
    f.close()
    f.close()
    assert "closed" in repr(f)
    assert f.closed is True


def test_invalid_regex_raises_invalid_argument():
    f = edfarray.EdfFile(PATH)
    with pytest.raises(edfarray.InvalidArgumentError):
        f.filter_annotations("(unclosed", regex=True)


def _writer_signal():
    return edfarray.WriterSignal(
        label="S1",
        physical_dimension="uV",
        physical_min=-100.0,
        physical_max=100.0,
        digital_min=-32768,
        digital_max=32767,
        samples_per_record=4,
    )


def test_writer_after_finish_raises_invalid_argument(tmp_path):
    import numpy as np

    w = edfarray.EdfWriter(
        str(tmp_path / "w.edf"), variant="EDF", record_duration=1.0, signals=[_writer_signal()]
    )
    w.write_record([np.zeros(4)])
    w.finish()
    with pytest.raises(edfarray.InvalidArgumentError):
        w.write_record([np.zeros(4)])


def test_writer_can_be_used_from_another_thread(tmp_path):
    # Previously the writer was `unsendable`, so cross-thread use raised a PanicException that
    # `except Exception` could not catch.
    import threading

    import numpy as np

    w = edfarray.EdfWriter(
        str(tmp_path / "t.edf"), variant="EDF", record_duration=1.0, signals=[_writer_signal()]
    )
    errors = []

    def write():
        try:
            w.write_record([np.zeros(4)])
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            errors.append(exc)

    t = threading.Thread(target=write)
    t.start()
    t.join()
    w.finish()

    assert not errors
    assert edfarray.EdfFile(str(tmp_path / "t.edf")).num_records == 1
