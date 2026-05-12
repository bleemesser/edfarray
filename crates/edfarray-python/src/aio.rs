use pyo3::prelude::*;

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let aio = PyModule::new(py, "aio")?;
    // Async classes and functions will be registered here in later chunks.
    parent.add_submodule(&aio)?;

    // Make `from edfarray._core.aio import ...` resolvable by registering
    // the submodule in sys.modules under its dotted name.
    py.import("sys")?
        .getattr("modules")?
        .set_item("edfarray._core.aio", &aio)?;

    Ok(())
}
