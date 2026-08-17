use pyo3::prelude::*;

/// Look up a numpy dtype object by name.
pub fn numpy_dtype<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.call_method1("dtype", (name,))
}
