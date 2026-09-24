use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyBool;

use crate::errors::out_of_range_err;

/// Extract an integer index, rejecting `bool`.
///
/// `bool` subclasses `int` in Python, so a plain `extract::<isize>()` accepts `True` and reads
/// it as index 1. numpy reads a bool as a mask. If this function returned element 1, the caller
/// gets a wrong answer and not an error for an unsupported index.
pub fn extract_index(key: &Bound<'_, PyAny>) -> Option<isize> {
    if key.is_instance_of::<PyBool>() {
        return None;
    }
    key.extract::<isize>().ok()
}

/// True if `key` is a bool, used to give a specific error message.
pub fn is_bool(key: &Bound<'_, PyAny>) -> bool {
    key.is_instance_of::<PyBool>()
}

/// Error for an index whose type is not supported.
pub fn unsupported_index_err(key: &Bound<'_, PyAny>, accepted: &str) -> PyErr {
    if is_bool(key) {
        return PyTypeError::new_err(
            "boolean indexing is not supported; edfarray accepts integers, slices, \
             and lists of integers",
        );
    }
    if key.is_none() {
        return PyTypeError::new_err(
            "None/np.newaxis is not supported; reshape the returned array instead",
        );
    }
    if key.is_instance_of::<pyo3::types::PyEllipsis>() {
        return PyTypeError::new_err(
            "Ellipsis is not supported; index every axis explicitly, or use a full slice",
        );
    }
    let type_name = key
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "object".to_string());
    PyTypeError::new_err(format!(
        "invalid index type {type_name}; expected {accepted}"
    ))
}

/// Convert a possibly negative index into a bounds-checked offset.
pub fn normalize_index(idx: isize, len: usize, what: &str) -> PyResult<usize> {
    let adjusted = if idx < 0 { idx + len as isize } else { idx };
    if adjusted < 0 || adjusted as usize >= len {
        return Err(out_of_range_err(format!(
            "{what} index {idx} out of range for length {len}"
        )));
    }
    Ok(adjusted as usize)
}
