use std::sync::OnceLock;

use edfarray_core::error::EdfError;
use pyo3::PyErr;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

/// Exception classes exported by the extension module.
///
/// Every one derives from `EdfError` *and* from the builtin a caller would reach for, so both
/// `except edfarray.SignalNotFoundError` and `except KeyError` catch the same failure. Adding
/// a common base later would be a breaking change, so the hierarchy is fixed at 1.0.
struct ErrorClasses {
    base: Py<PyType>,
    file: Py<PyType>,
    invalid_file: Py<PyType>,
    invalid_argument: Py<PyType>,
    out_of_range: Py<PyType>,
    signal_not_found: Py<PyType>,
    closed: Py<PyType>,
}

static CLASSES: OnceLock<ErrorClasses> = OnceLock::new();

/// Build a new exception class deriving from `bases`.
fn new_exception<'py>(
    py: Python<'py>,
    name: &str,
    bases: &[&Bound<'py, PyType>],
    doc: &str,
) -> PyResult<Py<PyType>> {
    let builtins = py.import("builtins")?;
    let dict = PyDict::new(py);
    dict.set_item("__doc__", doc)?;
    dict.set_item("__module__", "edfarray")?;
    let bases = PyTuple::new(py, bases.iter().map(|b| b.as_any()))?;
    let cls = builtins
        .getattr("type")?
        .call1((name, bases, dict))?
        .cast_into::<PyType>()?;
    Ok(cls.unbind())
}

/// Create the exception hierarchy and add it to the module.
pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let exception = py.get_type::<pyo3::exceptions::PyException>();
    let os_error = py.get_type::<pyo3::exceptions::PyOSError>();
    let value_error = py.get_type::<pyo3::exceptions::PyValueError>();
    let index_error = py.get_type::<pyo3::exceptions::PyIndexError>();
    let key_error = py.get_type::<pyo3::exceptions::PyKeyError>();

    let base = new_exception(
        py,
        "EdfError",
        &[&exception],
        "Base class for edfarray errors.",
    )?;
    let base_bound = base.bind(py).clone();

    let classes = ErrorClasses {
        file: new_exception(
            py,
            "EdfFileError",
            &[&base_bound, &os_error],
            "The file could not be opened or mapped.",
        )?,
        invalid_file: new_exception(
            py,
            "InvalidFileError",
            &[&base_bound, &value_error],
            "The file is not valid EDF/BDF, or its header is inconsistent.",
        )?,
        invalid_argument: new_exception(
            py,
            "InvalidArgumentError",
            &[&base_bound, &value_error],
            "An argument was outside the range the format or API allows.",
        )?,
        out_of_range: new_exception(
            py,
            "OutOfRangeError",
            &[&base_bound, &index_error],
            "A record, signal, or sample index was out of range.",
        )?,
        signal_not_found: new_exception(
            py,
            "SignalNotFoundError",
            &[&base_bound, &key_error],
            "No signal matched the requested label.",
        )?,
        closed: new_exception(
            py,
            "ClosedFileError",
            &[&base_bound, &value_error],
            "The file was used after close().",
        )?,
        base,
    };

    for (name, cls) in [
        ("EdfError", &classes.base),
        ("EdfFileError", &classes.file),
        ("InvalidFileError", &classes.invalid_file),
        ("InvalidArgumentError", &classes.invalid_argument),
        ("OutOfRangeError", &classes.out_of_range),
        ("SignalNotFoundError", &classes.signal_not_found),
        ("ClosedFileError", &classes.closed),
    ] {
        module.add(name, cls.bind(py))?;
    }

    let _ = CLASSES.set(classes);
    Ok(())
}

fn classes() -> &'static ErrorClasses {
    CLASSES
        .get()
        .expect("error classes are registered during module initialization")
}

/// Raise `InvalidArgumentError` for an argument the bindings validate themselves.
pub fn invalid_argument_err(message: impl Into<String>) -> PyErr {
    let message = message.into();
    Python::attach(|py| PyErr::from_type(classes().invalid_argument.bind(py).clone(), message))
}

/// Raise `OutOfRangeError` for an index the bindings check themselves.
pub fn out_of_range_err(message: impl Into<String>) -> PyErr {
    let message = message.into();
    Python::attach(|py| PyErr::from_type(classes().out_of_range.bind(py).clone(), message))
}

/// Raise `ClosedFileError`, used by every accessor on a closed handle.
pub fn closed_file_err() -> PyErr {
    Python::attach(|py| {
        PyErr::from_type(
            classes().closed.bind(py).clone(),
            "operation on closed EdfFile",
        )
    })
}

/// Convert an `EdfError` into the matching Python exception.
pub fn to_py_err(e: EdfError) -> PyErr {
    let message = e.to_string();
    Python::attach(|py| {
        let c = classes();
        let cls = match &e {
            EdfError::FileOpen { .. } | EdfError::MmapFailed { .. } | EdfError::Io { .. } => {
                &c.file
            }

            EdfError::FileTooSmall { .. }
            | EdfError::InvalidHeaderField { .. }
            | EdfError::HeaderSizeMismatch { .. }
            | EdfError::NoSignals
            | EdfError::InvalidSignalField { .. }
            | EdfError::InvalidDigitalRange { .. }
            | EdfError::InvalidPhysicalRange { .. } => &c.invalid_file,

            EdfError::InvalidArgument { .. } | EdfError::BufferSizeMismatch { .. } => {
                &c.invalid_argument
            }

            EdfError::RecordOutOfRange { .. }
            | EdfError::SignalOutOfRange { .. }
            | EdfError::SampleOutOfRange { .. } => &c.out_of_range,

            EdfError::SignalNotFound { .. } => &c.signal_not_found,
        };
        PyErr::from_type(cls.bind(py).clone(), message)
    })
}
