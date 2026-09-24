# Rust Crate

`edfarray-core` is the Rust crate that contains the implementation. The Python bindings wrap this crate. You can also use the crate directly from Rust.

## Dependency

```toml
[dependencies]
edfarray-core = { git = "https://github.com/bleemesser/edfarray.git" }
```

## Modules

- `file`: `EdfFile`, the top-level handle to open, read, and copy files. `EdfMetadata` and `EdfFile::inspect(path)` read the header and do not scan the annotations.
- `header`: `EdfHeader`, `EdfVariant`, `PatientInfo`, `RecordingInfo`, `MaybeDateTime`, `MaybeDate`, `Sex`.
- `signal`: `SignalHeader`, the metadata of one signal, and the conversion with gain and offset.
- `proxy`: `SignalProxy`, an array-like view that reads samples from one signal. A proxy is a view that reads samples on demand.
- `group`: `SignalGroup`, `GroupKind`, `PadMode`. These types group signals by sample rate.
- `proxy_2d`: `Proxy2D`, a 2D view over a `SignalGroup`. `PadMode` sets the result of a read past the end of a shorter signal.
- `proxy_3d`: `Proxy3D`, `StrideInfo`. A 3D view `(n_records, n_channels, spr)` for rectangular groups. A record is one fixed-duration block of data in the file.
- `epoch`: `EpochWindow`, `EpochRun`, `EpochPlan`, `EpochPad`, `plan_epochs`, `extract_epochs`. This module decodes event-locked windows in parallel. It takes EDF+D gaps into account. An EDF+D gap is a time span with no recorded data.

    `plan_epochs(file, group, events, pre, post)` returns an `EpochPlan` and reads no data. The plan contains the windows, a validity mask, and the nominal row width `n_samples`. `extract_epochs(file, group, plan, pad)` returns `(flat_row_data, valid, dropped)`. It uses `PadMode` to select the fill behavior.
- `annotation`: `Annotation`, `AnnotationIndex`, and TAL parsing. A TAL is a time-stamped annotation list.
- `record`: `RecordLayout`, the byte layout of a data record, and sample decoding.
- `mmap`: `MappedFile`, `MappedData`, `ScanMode`, `Advice`. A memory map (mmap) makes the bytes of a file readable as memory. `MappedFile` is the handle that `EdfFile` and the proxies share. `MappedData` holds the mapping, the parsed header, and the annotation index. `ScanMode` selects an eager or a lazy annotation scan.

    The background annotation scan holds only the `MappedData`. When the last `MappedFile` drops, it stops the scan and waits for it. Then it releases the mapping and the shared advisory lock. An advisory lock is a file lock that only cooperating programs obey.
- `writer`: `EdfWriter`, `WriterSpec`, `WriterSignal`, `write_edf`. Streaming writers and one-shot writers for all six variants.
- `edit`: `edit_header`, `anonymize`, `audit`, `audit_terms`. These functions edit the identity fields of the header in place. They also search the file for identity strings that remain.
- `grid`: `first_sample_at_or_after`, `GRID_EPS`. This module contains the one rule that converts a time to a sample index.
- `error`: `EdfError`, the error type for all of the crate.

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

All operations that can fail return `Result<T, EdfError>`. Each error variant contains data about the failure. For example:

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

The `error` module contains the full list of variants.

## Key types

`EdfFile`: The main entry point. It owns an `Arc<MappedFile>` and provides all public API methods. You open a file with `EdfFile::open(path)`.

Some files omit or misreport the `+C`/`+D` marker. For these files, use `EdfFile::open_with_variant(path, variant)` to force the variant. The override controls only the plain, `+C`, and `+D` distinction. If the override changes the EDF-vs-BDF sample size, it fails.

`write_to(path, variant)` writes a copy of the file. If `variant` is `Some`, the copy uses that variant. `write_subset_to(path, variant, signals)` writes only the selected signal indices, in the given order. If another edfarray handle has the destination open, both methods fail with an `EdfError::Io` of kind `ResourceBusy`. The source file itself counts as open.

`SignalProxy`: A view of one signal that holds no sample data. It holds an `Arc` reference to the underlying `MappedFile`. `EdfFile::signal()` creates it. It translates global sample indices to byte offsets in the records. It decodes the samples at read time.

`SignalGroup`: A set of signals classified by sample rate. The kind is `GroupKind::Rectangular` (shared rate) or `GroupKind::Open` (mixed). `EdfFile::signal_groups()` or `SignalGroup::from_indices(header, indices)` builds a group. The invariants tie the fields together, so the fields are private. Read them through `indices()`, `kind()`, `sample_rate()`, `samples_per_record()`, `min_samples()`, `max_samples()`, `covers_all_ordinary()`, and `is_singleton()`. `Proxy2D::new` and `Proxy3D::new` take a `SignalGroup`. `SignalProxy::new` takes a signal index.

`PadMode`: The fill policy for a `Proxy2D` read past the valid length of a signal. The variants are `Raise` (default), `Nan`, `Zero`, `Value(f64)`, and `Edge`. The fill applies in the domain of the read (physical f64 / digital i32). `Nan` is valid only for physical reads.

`Proxy2D`: A 2D view over a `SignalGroup`. `EdfFile::proxy_2d(group, pad_mode)` creates it. It accepts any `GroupKind`. It runs reads in parallel on the rayon thread pool. The main methods are `shape()`, `sample_rate() -> Option<f64>`, `valid_lengths()`, `get()`, `read_physical()`, `read_slice()`, and `read_digital()`.

`Proxy3D`: A 3D view `(num_records, num_channels, samples_per_record)`. `EdfFile::proxy_3d(group)` creates it. It requires `GroupKind::Rectangular`. The main methods are `shape()`, `sample_rate()`, `get()`, `read_physical_block()`, `read_digital_block()`, and `stride_info() -> Option<StrideInfo>`. `stride_info()` gives the metadata for a zero-copy view, that is, a view that does not copy samples.

`EdfHeader`: The complete parsed header, with the signal headers, patient info, and recording info. Get it with `EdfFile::header()`.

`MaybeDateTime`: Either a parsed `NaiveDateTime` or a raw date/time string pair. `EdfHeader::start_datetime` uses this type to handle anonymized files.

`RecordLayout`: The byte layout of the signals in a data record. `SignalProxy` uses it internally.
