mod aio;
mod annotations;
mod errors;
mod file;
mod group;
mod indexing;
mod numpy_util;
mod proxy_2d;
mod proxy_3d;
mod signal;
mod writer;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Before anything that can raise: to_py_err looks these up.
    errors::register(m.py(), m)?;
    m.add_class::<file::PyEdfFile>()?;
    m.add_class::<signal::PySignal>()?;
    m.add_class::<annotations::PyAnnotation>()?;
    m.add_class::<proxy_2d::PyProxy2D>()?;
    m.add_class::<proxy_3d::PyProxy3D>()?;
    m.add_class::<group::PySignalGroup>()?;
    m.add_class::<writer::PyEdfWriter>()?;
    m.add_class::<writer::PyWriterSignal>()?;
    m.add_function(wrap_pyfunction!(file::inspect, m)?)?;
    m.add_function(wrap_pyfunction!(writer::write_edf_py, m)?)?;
    aio::register(m)?;
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
