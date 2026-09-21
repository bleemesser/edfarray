"""Guards on the public API surface.

These exist because the 1.0 surface is frozen: an accidental rename, a dropped export, or a
stub that drifts from the runtime is a breaking change for every downstream user.
"""

import ast
import inspect as _inspect
from pathlib import Path

import edfarray
import edfarray.aio as aio
import pytest

CORE_STUB = Path(edfarray.__file__).parent / "_core" / "__init__.pyi"
AIO_STUB = Path(edfarray.__file__).parent / "aio" / "__init__.pyi"

EXPECTED_EXPORTS = {
    "EdfFile",
    "Epochs",
    "Signal",
    "Annotation",
    "Proxy2D",
    "Proxy3D",
    "SignalGroup",
    "EdfWriter",
    "WriterSignal",
    "inspect",
    "write_edf",
    "edit_header",
    "anonymize",
    "audit",
    "aio",
    "EdfError",
    "EdfFileError",
    "InvalidFileError",
    "InvalidArgumentError",
    "OutOfRangeError",
    "SignalNotFoundError",
    "ClosedFileError",
}


def test_exports_are_exactly_the_frozen_set():
    assert set(edfarray.__all__) == EXPECTED_EXPORTS


def test_py_typed_marker_is_present():
    # Without this file, both stubs are invisible to type checkers.
    assert (Path(edfarray.__file__).parent / "py.typed").is_file()


def test_stub_files_ship_and_parse():
    for stub in (CORE_STUB, AIO_STUB):
        assert stub.is_file(), f"missing stub: {stub}"
        ast.parse(stub.read_text())


def _stub_class_members(stub_path, class_name):
    tree = ast.parse(stub_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            names = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    names.add(item.target.id)
            return names
    return None


@pytest.mark.parametrize(
    "class_name", ["EdfFile", "Signal", "Annotation", "Proxy2D", "Proxy3D", "SignalGroup"]
)
def test_core_stub_covers_runtime_members(class_name):
    stub_members = _stub_class_members(CORE_STUB, class_name)
    assert stub_members is not None, f"{class_name} missing from the generated stub"
    runtime = {
        name
        for name in dir(getattr(edfarray, class_name))
        if not name.startswith("_") or name in {"__getitem__", "__len__", "__array__"}
    }
    missing = runtime - stub_members
    assert not missing, f"{class_name} runtime members absent from stub: {sorted(missing)}"


def test_async_file_mirrors_sync_metadata():
    # Metadata accessors must not drift between the two APIs.
    shared = {
        "num_signals",
        "num_records",
        "record_duration",
        "duration",
        "variant",
        "patient_id",
        "recording_id",
        "start_datetime",
        "patient_name",
        "patient_code",
        "patient_sex",
        "annotations",
        "warnings",
        "signal_labels",
        "ordinary_signal_indices",
        "find_all_signals",
        "signal_group",
        "signal_groups",
        "proxy_2d",
        "proxy_3d",
        "header",
    }
    missing_async = shared - set(dir(aio.EdfFile))
    missing_sync = shared - set(dir(edfarray.EdfFile))
    assert not missing_async, f"missing on aio.EdfFile: {sorted(missing_async)}"
    assert not missing_sync, f"missing on EdfFile: {sorted(missing_sync)}"


def test_epoch_methods_present_in_both_apis():
    # Epoch extraction is async-offloaded on the aio side but must exist on both.
    for name in ("extract_epochs", "epoch_windows", "events"):
        assert hasattr(edfarray.EdfFile, name), f"missing on EdfFile: {name}"
        assert hasattr(aio.EdfFile, name), f"missing on aio.EdfFile: {name}"


def test_signal_read_methods_match_between_apis():
    shared = {"to_physical", "to_digital", "times", "read_time_range"}
    assert shared <= set(dir(edfarray.Signal))
    assert shared <= set(dir(aio.Signal))


def test_writer_signal_is_one_class_across_apis():
    # A spec built for the sync writer must be accepted by the async writer.
    assert edfarray.WriterSignal is aio.WriterSignal


def test_header_is_a_method_on_both_apis():
    assert callable(getattr(edfarray.EdfFile, "header"))
    assert not isinstance(_inspect.getattr_static(edfarray.EdfFile, "header"), property)
    assert callable(getattr(aio.EdfFile, "header"))
