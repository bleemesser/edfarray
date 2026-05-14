# Python API Reference

## EdfFile

```python
edfarray.EdfFile(path: str)
```

Opens an EDF/EDF+ file at the given path. Parses the header synchronously and starts a background annotation scan for EDF+ files. Signal reads work immediately after construction.

Supports the context manager protocol (`with` statement).

### Properties

`num_signals: int` -- Total number of signals in the file, including annotation channels.

`num_records: int` -- Number of data records. For EDF-L files (header value `-1`, "unknown length") this is recovered from the file size at open time and reflects the true count; a `warnings` entry records the recovery.

`record_duration: float` -- Duration of each data record in seconds.

`duration: float` -- Total recording duration in seconds.

`variant: str` -- `"EDF"`, `"EDF+C"`, `"EDF+D"`, `"BDF"`, `"BDF+C"`, or `"BDF+D"`.

`start_datetime: datetime.datetime | str` -- Recording start date and time. Returns a `datetime.datetime` if the header date fields could be parsed, or a raw string like `"04.04.yy 12.57.02"` if the date is anonymized or non-standard. Note: the EDF header only stores integer seconds. EDF+ files encode subsecond precision in the first time-keeping annotation, which is applied to annotation onsets and sample timestamps but not to this property.

`patient_id: str` -- Raw 80-byte patient identification field from the header.

`recording_id: str` -- Raw 80-byte recording identification field from the header.

`patient_name: str | None` -- Parsed from the patient identification field. Underscores are replaced with spaces. `None` if the field is absent or set to `"X"`.

`patient_code: str | None` -- Hospital patient code from the patient identification field.

`patient_sex: str | None` -- `"M"` or `"F"`, or `None` if unknown.

`patient_birthdate: datetime.date | str | None` -- Patient birthdate. Returns a `datetime.date` if parseable, a raw string if the format is non-standard, or `None` if absent.

`patient_additional: str | None` -- Additional patient information beyond the standard subfields.

`admin_code: str | None` -- Hospital administration code from the recording identification field.

`technician: str | None` -- Technician or investigator code.

`equipment: str | None` -- Equipment code.

`recording_additional: str | None` -- Additional recording information.

`annotations: list[Annotation]` -- All non-timekeeping annotations from the file, sorted by onset. Empty for plain EDF files. Blocks until the background annotation scan is complete.

`annotations_before(t: float) -> list[Annotation]` -- Annotations with onset strictly before `t`. Uses binary search for efficiency. Blocks until the background annotation scan is complete.

`annotations_after(t: float) -> list[Annotation]` -- Annotations with onset >= `t`. Uses binary search for efficiency. Blocks until the background annotation scan is complete.

`annotations_in_range(start: float, end: float) -> list[Annotation]` -- Annotations with onset in the half-open interval `[start, end)`. Uses binary search for efficiency. Blocks until the background annotation scan is complete.

`filter_annotations(query: str, regex: bool = False) -> list[Annotation]` -- Filter annotations by text content. If `regex` is `False`, returns annotations whose text contains the query as a case-insensitive substring. If `regex` is `True`, returns annotations whose text matches the query as a case-insensitive regex pattern. Raises `ValueError` for invalid regex patterns.

`annotations_by_text(text: str) -> list[Annotation]` -- Annotations whose text exactly matches `text` (case-sensitive). Blocks until the background annotation scan is complete.

`warnings: list[str]` -- Parse warnings accumulated during file open. Empty if the file is well-formed. Blocks until the background annotation scan is complete.

`header: dict` -- Dictionary with basic header fields: `version`, `patient_id`, `recording_id`, `num_signals`, `num_records`, `record_duration`, `duration`, `variant`.

`annotations_ready: bool` -- Whether the background annotation scan has completed. Non-blocking.

`scan_progress: tuple[int, int]` -- `(records_scanned, total_records)` for the background annotation scan. Non-blocking. Can be polled to show progress for large files.

### Methods

`signal(idx_or_label: int | str) -> Signal` -- Get a signal by index or label. Raises `IndexError` for out-of-range indices, `KeyError` for unknown labels.

`find_all_signals(label: str, exact: bool = False) -> list[Signal]` -- Return all signals whose label matches `label`. If `exact` is `False` (default), performs a case-insensitive substring match. If `exact` is `True`, performs a case-sensitive exact equality match. Searches all signals including annotation signals. Skips indices that fail to construct a signal proxy.

`signal_labels() -> list[str]` -- Labels of all signals in the file.

`ordinary_signal_indices() -> list[int]` -- Indices of all non-annotation signals.

`read_page(start_sec: float, end_sec: float, signal_indices: list[int] | None = None, use_time: bool = False) -> list[numpy.ndarray]` -- Read physical (float64) data for multiple signals over a time range. Returns one array per signal. If `signal_indices` is `None`, reads all ordinary signals. Signals with different sample rates produce arrays of different lengths. When `use_time` is `False` (default), time parameters are converted to flat sample indices (`int(time * sample_rate)`). For EDF+D files with time gaps, set `use_time=True` to resolve the time range using actual record onset times. See [Annotations & Time](../guide/annotations.md#read_page-and-proxy2d-use-flat-sample-indices) for details.

`read_page_digital(start_sec: float, end_sec: float, signal_indices: list[int] | None = None, use_time: bool = False) -> list[numpy.ndarray]` -- Same as `read_page()` but returns raw int32 digital values without gain/offset conversion. When `use_time` is `True`, resolves the time range using record onset times for EDF+D files.

`signal_groups() -> list[SignalGroup]` -- Partition all ordinary signals into groups by sample rate. Each group records its classification, sample rate, sample-count range, and whether it covers every ordinary signal. Sub-Hz precision is preserved.

`signal_group(indices: list[int]) -> SignalGroup` -- Classify an arbitrary list of file-level signal indices into a `SignalGroup`. Use when you want a group that's a subset of (or crosses) the file's natural rate-based groupings.

`proxy_2d(group: SignalGroup, pad_mode: str | float | None = None) -> Proxy2D` -- Build a 2D proxy for numpy-style multi-channel indexing. Accepts any group kind; for `Open` groups (mixed sample rates), `pad_mode` decides what reads past short channels return. Values: `"raise"` (default), `"nan"`, `"zero"`, `"edge"`, or a numeric scalar (interpreted as `Value(x)`).

`proxy_3d(group: SignalGroup) -> Proxy3D` -- Build a 3D proxy `(num_records, num_channels, samples_per_record)`. Requires `group.kind == "rectangular"`. Errors otherwise.

`write_to(path: str, variant: str | None = None) -> None` -- Re-emit this file to `path`. By default uses the source variant; pass `variant` (one of `"EDF"`, `"EDF+C"`, `"EDF+D"`, `"BDF"`, `"BDF+C"`, `"BDF+D"`) to transcode. Only ordinary signals are copied; the destination's annotation channel is rebuilt from the parsed annotations.

`close() -> None` -- Explicitly release the underlying memory-mapped file. After calling `close()`, any further method or property access on the `EdfFile` raises. Existing `Signal`, `Proxy2D`, and `Proxy3D` objects keep their own references and remain usable. Idempotent. The context manager (`with` statement) calls `close()` on exit.

`closed: bool` -- Whether `close()` has been called.

---

## inspect

```python
edfarray.inspect(path: str | os.PathLike) -> dict
```

Lightweight metadata extracted from an EDF/EDF+ file header without scanning data records or building an annotation index. Does not memory-map the file, does not spawn background threads, and does not read any data records.

Returns a dict with keys:

`variant: str` -- `"EDF"`, `"EDF+C"`, `"EDF+D"`, `"BDF"`, `"BDF+C"`, or `"BDF+D"`.

`num_signals: int` -- Total number of signals in the file, including annotation channels.

`num_records: int` -- Number of data records. For EDF-L files (header value `-1`, "unknown length") this is recovered from the file size.

`record_duration: float` -- Duration of each data record in seconds.

`duration: float` -- Total recording duration in seconds.

`patient_id: str` -- Raw 80-byte patient identification field.

`recording_id: str` -- Raw 80-byte recording identification field.

`signal_labels: list[str]` -- Labels of all signals in the file.

`sample_rates: list[float]` -- Sample rates in Hz, one per signal.

---

## Signal

Returned by `EdfFile.signal()`. Proxy view of a single signal that decodes samples from the memory-mapped file on access.

### Properties

`label: str` -- Signal label, e.g. `"EEG Fpz-Cz"`.

`transducer: str` -- Transducer type, e.g. `"AgAgCl electrode"`.

`physical_dimension: str` -- Physical units, e.g. `"uV"`.

`prefiltering: str` -- Prefiltering description, e.g. `"HP:0.1Hz LP:75Hz"`.

`sample_rate: float` -- Sample frequency in Hz.

`samples_per_record: int` -- Number of samples per data record for this signal.

`physical_min: float` -- Physical minimum value.

`physical_max: float` -- Physical maximum value.

`digital_min: int` -- Digital minimum value.

`digital_max: int` -- Digital maximum value.

`num_samples: int` -- Total number of samples. Same as `len(sig)`.

### Indexing

`sig[i]` -- Returns a single physical value as a `float`. Supports negative indexing.

`sig[start:stop]` -- Returns a `numpy.ndarray` of float64 physical values.

`sig[start:stop:step]` -- Returns a strided `numpy.ndarray` of float64 physical values.

### Methods

`to_numpy() -> numpy.ndarray` -- The entire signal as a float64 numpy array.

`to_digital() -> numpy.ndarray` -- The entire signal as an int32 numpy array (raw digital values).

`times() -> numpy.ndarray` -- Timestamp in seconds from recording start for each sample. For EDF+D files, accounts for gaps between data records.

`read_at(start_sec: float, end_sec: float) -> numpy.ndarray` -- Return physical data for samples whose time falls within `[start_sec, end_sec)`. For EDF+D files, accounts for gaps between records using record onset times. For EDF and EDF+C, equivalent to indexing by flat sample number, i.e. `int(time * sample_rate)`.

`with_cache(capacity: int) -> None` -- Enable an LRU cache of decoded physical record data. `capacity` is the number of records to cache. A capacity of 0 disables the cache (default). The cache is per-Signal-instance; cloning or re-fetching from `EdfFile.signal()` starts fresh.

`__len__() -> int` -- Total number of samples.

---

## SignalGroup

A set of file-level signal indices plus classification metadata. Built by
`EdfFile.signal_groups()` or `EdfFile.signal_group(indices)`.

### Properties

`indices: list[int]` -- File-level signal indices in the group.

`kind: str` -- `"rectangular"` (all channels share a sample rate and total sample count) or `"open"` (mixed sample rates; 2D-only).

`sample_rate: float | None` -- Common rate, or `None` if `kind == "open"`.

`samples_per_record: int | None` -- Common SPR, or `None` if `kind == "open"`.

`min_samples: int` / `max_samples: int` -- Bounds on total samples across channels. Equal for `"rectangular"`.

`covers_all_ordinary: bool` -- `True` iff this group spans every ordinary signal in the file. Set only by `signal_groups()`; hand-built groups are always `False`.

`is_singleton: bool` -- `True` iff the group contains exactly one channel.

`is_rectangular: bool` -- Shortcut for `kind == "rectangular"`.

`len(group)` -- Number of channels in the group.

---

## Proxy2D

Returned by `EdfFile.proxy_2d(group, pad_mode=...)`. A 2D view over a `SignalGroup`. Reads on demand from the memory-mapped file.

### Properties

`shape: tuple[int, int]` -- `(num_signals, max_samples_across_signals)`.

`sample_rate: float | None` -- Common rate, or `None` if the underlying group is `Open`.

`valid_lengths: list[int]` -- Per-channel valid sample counts, in proxy-coordinate order. All entries equal `shape[1]` for `"rectangular"` groups; for `"open"` groups they vary.

`pad_mode: str` -- `"raise"`, `"nan"`, `"zero"`, `"value"`, or `"edge"`.

### Indexing

`proxy[int, int]` -- Returns a single physical value as a `float`. Supports negative indexing.

`proxy[int, slice]` / `proxy[slice, int]` / `proxy[slice, slice]` -- Returns the natural 1D or 2D `numpy.ndarray` of float64 physical values.

`proxy[list, slice]` -- Fancy indexing on the signal axis. The list contains proxy-coordinate signal indices.

Step values other than 1 are not supported. Reads past a channel's valid length are governed by `pad_mode`: `"raise"` (default) raises `IndexError`; other modes fill (`"nan"` is physical-only).

---

## Proxy3D

Returned by `EdfFile.proxy_3d(group)`. A 3D view over a `Rectangular` `SignalGroup`, exposing the underlying record-major layout.

### Properties

`shape: tuple[int, int, int]` -- `(num_records, num_channels, samples_per_record)`.

`sample_rate: float` -- Common sample rate.

`supports_strided_view: bool` -- `True` when the file/group support a zero-copy `as_strided` view. See `stride_info()`.

### Methods

`stride_info() -> dict | None` -- Byte-level metadata for a zero-copy strided view of the underlying int16 mmap, or `None` if ineligible. Keys: `base_offset`, `record_stride_bytes`, `channel_stride_bytes`, `sample_stride_bytes`, `shape`. Eligibility requires 2-byte samples, a contiguous channel-index range in the file, and no annotation channel interleaved within that span.

### Indexing

`proxy[rec, ch, samp]` -- Each axis accepts an int or a slice with step 1. Returns a scalar (all ints), 1D / 2D / 3D `numpy.ndarray` depending on how many axes are sliced.

---

## Writing files

See [Writing EDF/BDF files](../guide/writing.md) for a full guide.

### `edfarray.write_edf`

```python
edfarray.write_edf(
    path: str,
    *,
    variant: str,
    record_duration: float,
    signals: list[WriterSignal],
    data: list[numpy.ndarray],
    annotations: list[Annotation] | None = None,
    start_datetime: datetime.datetime | None = None,
    patient_id: str | None = None,
    recording_id: str | None = None,
    annotation_bytes_per_record: int | None = None,
) -> None
```

One-shot writer. `data[i]` must have length `num_records * signals[i].samples_per_record`. `variant` is one of `"EDF"`, `"EDF+C"`, `"EDF+D"`, `"BDF"`, `"BDF+C"`, `"BDF+D"`. The annotation channel is added automatically for `+` variants; do not include it in `signals`.

### `edfarray.EdfWriter`

```python
edfarray.EdfWriter(
    path: str,
    *,
    variant: str,
    record_duration: float,
    signals: list[WriterSignal],
    start_datetime: datetime.datetime | None = None,
    patient_id: str | None = None,
    recording_id: str | None = None,
    annotation_bytes_per_record: int | None = None,
)
```

Streaming writer. Supports the context manager protocol (`with` block calls `finish()` on exit).

`write_record(physical: list[numpy.ndarray], annotations: list[Annotation] | None = None) -> None` -- Encode and append one record. `physical[i]` must be a 1D float64 array of length `signals[i].samples_per_record`. Optional `annotations` are embedded in this record's annotation channel along with any pending ones queued via `add_annotation`.

`add_annotation(annotation: Annotation) -> None` -- Queue an annotation to be embedded with the next `write_record` call.

`finish() -> None` -- Flush, then seek back and patch `num_records` in the header. Idempotent. Called automatically by `__exit__`.

### `edfarray.WriterSignal`

```python
edfarray.WriterSignal(
    label: str,
    physical_dimension: str,
    physical_min: float,
    physical_max: float,
    digital_min: int,
    digital_max: int,
    samples_per_record: int,
    transducer: str = "",
    prefiltering: str = "",
    reserved: str = "",
)
```

Per-signal description. `samples_per_record` combined with the writer's `record_duration` gives the sample rate. For BDF/BDF+ files, `digital_min`/`digital_max` may use the full 24-bit signed range (±2²³).

---

## Annotation

Returned in `EdfFile.annotations`. Immutable.

### Properties

`onset: float` -- Time in seconds from the start of the recording.

`duration: float | None` -- Duration in seconds, or `None` if not specified.

`text: str` -- The annotation text.
