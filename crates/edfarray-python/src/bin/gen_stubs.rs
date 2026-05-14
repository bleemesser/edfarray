use pyo3_stub_gen::Result;
use std::fs;

fn main() -> Result<()> {
    let stub = edfarray::stub_info()?;
    stub.generate()?;

    let manifest_dir: &std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let init_pyi = root.join("edfarray/__init__.pyi");
    if init_pyi.exists() {
        fs::remove_file(&init_pyi)?;
    }

    Ok(())
}
