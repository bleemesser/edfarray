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

/// Exception classes as `(name, builtin base, docstring)`. Every class except the first also
/// derives from `EdfError`. `register` and the stub generator both read this table.
pub const EXCEPTIONS: [(&str, &str, &str); 7] = [
    ("EdfError", "Exception", "Base class for edfarray errors."),
    (
        "EdfFileError",
        "OSError",
        "The file could not be opened, mapped, locked, or written.",
    ),
    (
        "InvalidFileError",
        "ValueError",
        "The file is not valid EDF/BDF, or its header is inconsistent.",
    ),
    (
        "InvalidArgumentError",
        "ValueError",
        "An argument was outside the range the format or API allows.",
    ),
    (
        "OutOfRangeError",
        "IndexError",
        "A record, signal, or sample index was out of range.",
    ),
    (
        "SignalNotFoundError",
        "KeyError",
        "No signal matched the requested label.",
    ),
    (
        "ClosedFileError",
        "ValueError",
        "The file was used after close().",
    ),
];

/// Create the exception hierarchy and add it to the module.
pub fn register(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let builtins = py.import("builtins")?;
    let mut created: Vec<Py<PyType>> = Vec::with_capacity(EXCEPTIONS.len());
    for (name, builtin, doc) in EXCEPTIONS {
        let builtin = builtins.getattr(builtin)?.cast_into::<PyType>()?;
        let cls = match created.first() {
            None => new_exception(py, name, &[&builtin], doc)?,
            Some(base) => new_exception(py, name, &[base.bind(py), &builtin], doc)?,
        };
        module.add(name, cls.bind(py))?;
        created.push(cls);
    }

    let [
        _base,
        file,
        invalid_file,
        invalid_argument,
        out_of_range,
        signal_not_found,
        closed,
    ] = <[Py<PyType>; 7]>::try_from(created).expect("one class per EXCEPTIONS entry");
    let _ = CLASSES.set(ErrorClasses {
        file,
        invalid_file,
        invalid_argument,
        out_of_range,
        signal_not_found,
        closed,
    });
    Ok(())
}

/// Stub declarations for the exception classes, for appending to the generated `.pyi`.
pub fn stub_declarations() -> String {
    let mut out = String::new();
    for (i, (name, builtin, doc)) in EXCEPTIONS.iter().enumerate() {
        let bases = if i == 0 {
            format!("builtins.{builtin}")
        } else {
            format!("{}, builtins.{builtin}", EXCEPTIONS[0].0)
        };
        out.push_str(&format!(
            "\nclass {name}({bases}):\n    r\"\"\"{doc}\"\"\"\n"
        ));
    }
    out
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
