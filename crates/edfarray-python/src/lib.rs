mod annotations;
mod array_proxy;
mod errors;
mod file;
mod signal;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<file::PyEdfFile>()?;
    m.add_class::<signal::PySignal>()?;
    m.add_class::<annotations::PyAnnotation>()?;
    m.add_class::<array_proxy::PyArrayProxy>()?;
    Ok(())
}

/// Gather type information from annotated PyO3 classes/functions for `.pyi` stub generation.
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    let manifest_dir: &std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    let pyproject_path = manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("pyproject.toml");
    pyo3_stub_gen::StubInfo::from_pyproject_toml(pyproject_path)
}
