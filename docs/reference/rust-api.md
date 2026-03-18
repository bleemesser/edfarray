# Rust Crate

The `edfarray-core` crate provides the pure Rust implementation. The Python bindings are a thin layer on top of this crate. You can use it directly in Rust applications.

## Dependency

```toml
[dependencies]
edfarray-core = { git = "https://github.com/bleemesser/edfarray.git" }
```

## Modules

- `file` -- `EdfFile`, the top-level handle for opening and reading files.
- `header` -- `EdfHeader`, `EdfVariant`, `PatientInfo`, `RecordingInfo`, `MaybeDateTime`, `MaybeDate`, `Sex`.
- `signal` -- `SignalHeader`, per-signal metadata and gain/offset conversion.
- `proxy` -- `SignalProxy`, array-like view for reading samples from a single signal.
- `annotation` -- `Annotation`, `AnnotationIndex`, TAL parsing.
- `record` -- `RecordLayout`, data record byte layout and sample decoding.
- `mmap` -- `MappedFile`, memory-mapped file with parsed header and annotation index.
- `error` -- `EdfError`, the error type used throughout the crate.

## Usage

```rust
use edfarray_core::file::EdfFile;

fn main() -> edfarray_core::error::Result<()> {
    let edf = EdfFile::open("recording.edf")?;

    println!("variant: {}", edf.variant());
    println!("signals: {}", edf.num_signals());
    println!("duration: {}s", edf.duration());

    // Read a single signal.
    let proxy = edf.signal(0)?;
    let mut buf = vec![0.0f64; 1000];
    proxy.read_physical(0, 1000, &mut buf)?;
    println!("first sample: {}", buf[0]);

    // Bulk read all ordinary signals for a time window.
    let indices = edf.ordinary_signal_indices();
    let pages = edf.read_page(&indices, 0.0, 10.0)?;
    for (i, page) in pages.iter().enumerate() {
        println!("signal {}: {} samples", indices[i], page.len());
    }

    // Annotations.
    for ann in edf.annotations() {
        println!("{:.2}s: {}", ann.onset, ann.text);
    }

    Ok(())
}
```

## Error handling

All fallible operations return `Result<T, EdfError>`. Error variants carry context about what went wrong:

```rust
use edfarray_core::error::EdfError;

match edf.signal(999) {
    Err(EdfError::SignalOutOfRange { index, count }) => {
        eprintln!("signal {index} out of range (file has {count})");
    }
    Err(e) => eprintln!("error: {e}"),
    Ok(proxy) => { /* ... */ }
}
```

See the `error` module for the full list of variants.

## Key types

`EdfFile` -- Main entry point. Owns an `Arc<MappedFile>` and provides all public API methods.

`SignalProxy` -- Lightweight view of one signal. Holds an `Arc` reference to the underlying `MappedFile`. Created by `EdfFile::signal()`. Translates global sample indices to record byte offsets and decodes on the fly.

`EdfHeader` -- The complete parsed header, including signal headers, patient info, and recording info. Accessible via `EdfFile::header()`.

`MaybeDateTime` -- Either a parsed `NaiveDateTime` or a raw date/time string pair. Used for `EdfHeader::start_datetime` to handle anonymized files.

`RecordLayout` -- Byte-level layout of signals within a data record. Used internally by `SignalProxy`.
