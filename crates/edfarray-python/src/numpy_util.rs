use pyo3::prelude::*;

/// Look up a numpy dtype object by name.
pub fn numpy_dtype<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.call_method1("dtype", (name,))
}

/// Copy `src` into a new `Vec`, raising `MemoryError` if the allocation is refused.
pub fn try_copy<T: Copy>(src: &[T]) -> PyResult<Vec<T>> {
    let mut out = Vec::new();
    out.try_reserve_exact(src.len())
        .map_err(|_| memory_error::<T>(src.len() as u128))?;
    out.extend_from_slice(src);
    Ok(out)
}

/// Concatenate equal-length `rows` into one row-major `Vec`, raising `MemoryError` if the
/// allocation is refused.
pub fn try_flatten<T: Copy>(rows: Vec<Vec<T>>, row_len: usize) -> PyResult<Vec<T>> {
    let mut rows = rows;
    if rows.len() == 1 {
        return Ok(rows.pop().unwrap_or_default());
    }
    let total = rows.len() as u128 * row_len as u128;
    let mut out = Vec::new();
    usize::try_from(total)
        .ok()
        .and_then(|n| out.try_reserve_exact(n).ok())
        .ok_or_else(|| memory_error::<T>(total))?;
    for row in rows {
        out.extend_from_slice(&row);
    }
    Ok(out)
}

fn memory_error<T>(elements: u128) -> PyErr {
    let bytes = elements * std::mem::size_of::<T>() as u128;
    pyo3::exceptions::PyMemoryError::new_err(format!(
        "cannot allocate {bytes} bytes for the output array ({elements} elements)"
    ))
}
