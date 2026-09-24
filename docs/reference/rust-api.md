# Rust Crate

`edfarray-core` is the pure Rust implementation. The Python bindings are a thin layer on top of it, and the crate can also be used directly from Rust.

## Dependency

```toml
[dependencies]
edfarray-core = { git = "https://github.com/bleemesser/edfarray.git" }
```

## Modules

- `file` -- `EdfFile`, the top-level handle for opening, reading, and copying files. `EdfMetadata` and `EdfFile::inspect(path)` read the header without the annotation scan.
- `header` -- `EdfHeader`, `EdfVariant`, `PatientInfo`, `RecordingInfo`, `MaybeDateTime`, `MaybeDate`, `Sex`.
- `signal` -- `SignalHeader`, per-signal metadata and gain/offset conversion.
- `proxy` -- `SignalProxy`, array-like view for reading samples from a single signal.
- `group` -- `SignalGroup`, `GroupKind`, `PadMode`. Channel grouping by sample rate.
- `proxy_2d` -- `Proxy2D`, 2D view over a `SignalGroup`. `PadMode` sets the result of reads past the end of a shorter channel.
- `proxy_3d` -- `Proxy3D`, `StrideInfo`. 3D view `(n_records, n_channels, spr)` for rectangular groups.
- `epoch` -- `EpochWindow`, `EpochRun`, `EpochPlan`, `EpochPad`, `plan_epochs`, `extract_epochs`. Event-locked windows decoded in parallel and gap-aware for EDF+D. `plan_epochs(file, group, events, pre, post)` returns an `EpochPlan` (windows, validity mask, and the nominal row width `n_samples`) without reading data; `extract_epochs(file, group, plan, pad)` returns `(flat_row_data, valid, dropped)` and reuses `PadMode` for fill behavior.
- `annotation` -- `Annotation`, `AnnotationIndex`, TAL parsing.
- `record` -- `RecordLayout`, data record byte layout and sample decoding.
- `mmap` -- `MappedFile`, `MappedData`, `ScanMode`, `Advice`. `MappedFile` is the handle that `EdfFile` and the proxies share. `MappedData` holds the mapping, the parsed header, and the annotation index. The background annotation scan holds only the `MappedData`. When the last `MappedFile` drops, it stops the scan and waits for it. Then the mapping and the shared file lock are released. `ScanMode` selects an eager or a lazy annotation scan.
- `writer` -- `EdfWriter`, `WriterSpec`, `WriterSignal`, `write_edf`. Streaming and one-shot writers for all six variants.
- `edit` -- `edit_header`, `anonymize`, `audit`, `audit_terms`. In-place edits of the identity header fields, and a check for identity strings that remain.
- `grid` -- `first_sample_at_or_after`, `GRID_EPS`. The one rule that converts a time to a sample index.
- `error` -- `EdfError`, the error type used throughout the crate.

## Usage

```rust
use edfarray_core::file::EdfFile;

fn main() -> edfarray_core::error::Result<()> {
    let edf = EdfFile::open("recording.edf")?;

    println!("variant: {}", edf.variant());
    println!("signals: {}", edf.num_signals());
    println!("duration: {}s", edf.duration());

    // Read the first 1000 samples of one signal.
    let proxy = edf.signal(0)?;
    let mut buf = vec![0.0f64; 1000];
    proxy.read_physical(0, 1000, &mut buf)?;
    println!("first sample: {}", buf[0]);

    // Bulk read all ordinary signals for a time window.
    let indices = edf.ordinary_signal_indices();
    let pages = edf.read_page(&indices, 0.0, 10.0, false)?;
    for (i, page) in pages.iter().enumerate() {
        println!("signal {}: {} samples", indices[i], page.len());
    }

    // 2D / 3D proxies for multi-channel access. Build from a SignalGroup.
    use edfarray_core::group::PadMode;
    for group in edf.signal_groups() {
        let p2 = edf.proxy_2d(group.clone(), PadMode::Raise)?;
        println!(
            "{:?} Hz group: {} signals x {} samples",
            group.sample_rate(),
            p2.shape().0,
            p2.shape().1
        );

        // Rectangular groups can also be viewed as 3D records-by-channels.
        if group.is_rectangular() {
            let p3 = edf.proxy_3d(group)?;
            println!("  3D shape: {:?}", p3.shape());
        }
    }

    // Annotations (blocks until background scan completes).
    for ann in edf.annotations() {
        println!("{:.2}s: {}", ann.onset, ann.text);
    }

    // Check scan status without blocking.
    println!("scan ready: {}", edf.annotations_ready());

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

`EdfFile` -- Main entry point. Owns an `Arc<MappedFile>` and provides all public API methods. Opened with `EdfFile::open(path)`. Some files omit or misreport the `+C`/`+D` marker. For these files, use `EdfFile::open_with_variant(path, variant)` to force the variant. The override controls only the plain, `+C`, and `+D` distinction. An override that changes the EDF-vs-BDF sample size fails. `write_to(path, variant)` writes a copy of the file. If `variant` is `Some`, the copy uses that variant. `write_subset_to(path, variant, signals)` writes only the selected signal indices, in the given order. If another edfarray handle has the destination open, both fail with an `EdfError::Io` of kind `ResourceBusy`. The source file itself counts as open.

`SignalProxy` -- Lightweight view of one signal. Holds an `Arc` reference to the underlying `MappedFile`. Created by `EdfFile::signal()`. Translates global sample indices to record byte offsets and decodes on the fly.

`SignalGroup` -- A set of channels classified by sample rate. `GroupKind::Rectangular` (shared rate) or `GroupKind::Open` (mixed). Built by `EdfFile::signal_groups()` or `SignalGroup::from_indices(header, indices)`; its fields are private because the invariants tie them together, so read them through `indices()`, `kind()`, `sample_rate()`, `samples_per_record()`, `min_samples()`, `max_samples()`, `covers_all_ordinary()`, and `is_singleton()`. Required input to all proxy constructors.

`PadMode` -- Fill policy for reads past a channel's valid length on `Proxy2D`. Variants: `Raise` (default), `Nan`, `Zero`, `Value(f64)`, `Edge`. Interpreted in the read domain (physical f64 / digital i32). `Nan` is physical-only.

`Proxy2D` -- 2D view over a `SignalGroup`. Created by `EdfFile::proxy_2d(group, pad_mode)`. Accepts any `GroupKind`. Reads are parallelized with rayon. Key methods: `shape()`, `sample_rate() -> Option<f64>`, `valid_lengths()`, `get()`, `read_physical()`, `read_slice()`, `read_digital()`.

`Proxy3D` -- 3D view `(num_records, num_channels, samples_per_record)`. Created by `EdfFile::proxy_3d(group)`. Requires `GroupKind::Rectangular`. Key methods: `shape()`, `sample_rate()`, `get()`, `read_physical_block()`, `read_digital_block()`, `stride_info() -> Option<StrideInfo>` for zero-copy view metadata.

`EdfHeader` -- The complete parsed header, including signal headers, patient info, and recording info. Accessible via `EdfFile::header()`.

`MaybeDateTime` -- Either a parsed `NaiveDateTime` or a raw date/time string pair. Used for `EdfHeader::start_datetime` to handle anonymized files.

`RecordLayout` -- Byte-level layout of signals within a data record. Used internally by `SignalProxy`.
