mod annotations;
mod errors;
mod file;
mod signal;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<file::PyEdfFile>()?;
    m.add_class::<annotations::PyAnnotation>()?;
    Ok(())
}
