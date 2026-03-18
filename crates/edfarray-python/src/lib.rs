use pyo3::prelude::*;

#[pymodule]
fn edfarray(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
