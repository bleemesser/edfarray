use edfarray_core::error::EdfError;
use pyo3::PyErr;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyOSError, PyValueError};

/// Convert an `EdfError` into the appropriate Python exception.
pub fn to_py_err(e: EdfError) -> PyErr {
    match &e {
        EdfError::FileOpen { .. } | EdfError::MmapFailed { .. } => {
            PyOSError::new_err(e.to_string())
        }

        EdfError::FileTooSmall { .. }
        | EdfError::InvalidHeaderField { .. }
        | EdfError::HeaderSizeMismatch { .. }
        | EdfError::NoSignals
        | EdfError::InvalidSignalField { .. }
        | EdfError::InvalidDigitalRange { .. }
        | EdfError::InvalidPhysicalRange { .. }
        | EdfError::InvalidTal { .. }
        | EdfError::InvalidAnnotationEncoding { .. }
        | EdfError::MissingTimekeepingAnnotation { .. } => PyValueError::new_err(e.to_string()),

        EdfError::RecordOutOfRange { .. }
        | EdfError::SignalOutOfRange { .. }
        | EdfError::SampleOutOfRange { .. } => PyIndexError::new_err(e.to_string()),

        EdfError::SignalNotFound { .. } => PyKeyError::new_err(e.to_string()),
    }
}
