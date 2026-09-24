use pyo3_stub_gen::Result;
use std::fs;

fn main() -> Result<()> {
    let stub = edfarray::stub_info()?;
    stub.generate()?;

    let manifest_dir: &std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    let root = manifest_dir.parent().unwrap().parent().unwrap();

    // pyo3_stub_gen does not see the exception classes, which are built at runtime.
    let core_pyi = root.join("edfarray/_core/__init__.pyi");
    let mut text = fs::read_to_string(&core_pyi)?;
    let names: String = edfarray::errors::EXCEPTIONS
        .iter()
        .map(|(name, _, _)| format!("    \"{name}\",\n"))
        .collect();
    text = text.replacen("__all__ = [\n", &format!("__all__ = [\n{names}"), 1);
    text.push_str(&edfarray::errors::stub_declarations());
    fs::write(&core_pyi, text)?;
    let init_pyi = root.join("edfarray/__init__.pyi");
    if init_pyi.exists() {
        fs::remove_file(&init_pyi)?;
    }

    Ok(())
}
