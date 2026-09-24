# Python API Reference

For the exception hierarchy, the supported-indexing rules, and threading/async guarantees, see
[API contracts](contracts.md).

## EdfFile

```python
edfarray.EdfFile(path: str, variant: str | None = None, scan_annotations: bool = True)
```

Opens an EDF/EDF+ file at the given path. Parses the header synchronously and starts a background annotation scan for EDF+ files. Signal reads work immediately after construction.

`scan_annotations=False` defers that scan until annotations are first accessed, at which point it runs on the calling thread. The scan reads every data record, so deferring it matters on very large files -- see [Performance](../guide/performance.md).

`variant` forces the file variant instead of trusting the auto-detected one, for files that omit or misreport the EDF+ `"+C"`/`"+D"` marker. It only controls the plain/`"+C"`/`"+D"` distinction; an override that changes the EDF-vs-BDF sample size (set by the version field) raises `ValueError`. A mismatch with the detected variant is recorded in `warnings`.

If an `EdfWriter` or `write_edf` call is still writing `path`, the constructor raises `EdfFileError`. The open file holds a shared lock on `path` until the `EdfFile` and every signal and proxy taken from it are dropped. While the lock is held, edfarray refuses to write to `path`. See [Memory mapping](contracts.md#memory-mapping).

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

The annotation accessors below block until the background annotation scan completes. Use `annotations_ready` to check status without blocking. Range queries use binary search.

`annotations: list[Annotation]` -- All non-timekeeping annotations from the file, sorted by onset. Empty for plain EDF files.

`annotations_before(t: float) -> list[Annotation]` -- Annotations with onset strictly before `t`.

`annotations_after(t: float) -> list[Annotation]` -- Annotations with onset >= `t`.

`annotations_in_range(start: float, end: float) -> list[Annotation]` -- Annotations with onset in the half-open interval `[start, end)`.

`filter_annotations(query: str, regex: bool = False) -> list[Annotation]` -- Filter annotations by text content. With `regex=False`, matches the query as a case-insensitive substring. With `regex=True`, matches the query as a case-insensitive regex pattern. Raises `ValueError` for invalid regex patterns.

`events(query: str, regex: bool = False) -> list[Annotation]` -- Shortcut returning the `Annotation` list that `query` matches, using the same rules as `filter_annotations`. Handy to pass straight into `extract_epochs`.

`annotations_by_text(text: str) -> list[Annotation]` -- Annotations whose text exactly matches `text` (case-sensitive).

`warnings: list[str]` -- Parse warnings accumulated during file open. Empty if the file is well-formed.

`header() -> dict` -- Dictionary with basic header fields (a method, not a property: it builds a fresh dict on each call): `version`, `patient_id`, `recording_id`, `num_signals`, `num_records`, `record_duration`, `duration`, `variant`.

`annotations_ready: bool` -- Whether the background annotation scan has completed. Non-blocking.

`scan_progress: tuple[int, int]` -- `(records_scanned, total_records)` for the background annotation scan. Non-blocking. Can be polled to show progress for large files.

`closed: bool` -- Whether `close()` has been called.

### Methods

`signal(idx_or_label: int | str, cache_capacity: int = 0, strategy: str | None = None) -> Signal` -- Get a signal by index or label. Raises `OutOfRangeError` (an `IndexError`) for out-of-range indices, `SignalNotFoundError` (a `KeyError`) for unknown labels. Negative indices are not supported and raise `OutOfRangeError`. `strategy` is `"auto"` (default), `"mmap"`, or `"stream"`; see [Performance](../guide/performance.md). `cache_capacity` enables a per-`Signal` LRU cache of decoded physical records -- see [Caching repeated reads](../guide/signals.md#caching-repeated-reads).

`find_all_signals(label: str, exact: bool = False) -> list[int]` -- Return the indices of all signals whose label matches `label`. If `exact` is `False` (default), performs a case-insensitive substring match. If `exact` is `True`, performs a case-sensitive exact equality match. Searches all signals including annotation signals. Pass an index to `signal()` to read one.

`signal_labels() -> list[str]` -- Labels of all signals in the file.

`ordinary_signal_indices() -> list[int]` -- Indices of all non-annotation signals.

`read_page(start_sec: float, end_sec: float, signal_indices: list[int] | None = None, use_time: bool = False) -> list[numpy.ndarray]` -- Read physical (float64) data for multiple signals over a time range. Returns one array per signal. If `signal_indices` is `None`, reads all ordinary signals. Signals with different sample rates produce arrays of different lengths. When `use_time` is `False` (default), each time maps to the flat index of the first sample at or after it, so `[start_sec, end_sec)` is half-open. A time within 1e-6 of a sample counts as that sample, which absorbs float noise such as `0.1 + 0.2`. Every time-based API uses this one rule. For EDF+D files with time gaps, set `use_time=True` to resolve the time range using actual record onset times. See [Annotations & Time](../guide/annotations.md#read_page-and-proxy2d-use-flat-sample-indices) for details.

`read_page_digital(start_sec: float, end_sec: float, signal_indices: list[int] | None = None, use_time: bool = False) -> list[numpy.ndarray]` -- Same as `read_page()` but returns raw int32 digital values without gain/offset conversion. When `use_time` is `True`, resolves the time range using record onset times for EDF+D files.

`extract_epochs(events, *, pre: float, post: float, group=None, pad=None, query=None, regex=False) -> Epochs` -- Decode fixed `[onset - pre, onset + post)` windows around each event in parallel into a dense `(n_epochs, n_channels, n_samples)` float64 block, where `n_samples` is always `ceil((pre + post) * sample_rate)` and does not depend on where the events fall. A product within 1e-6 of an integer counts as that integer, so `pre=0.1, post=0.2` at 200 Hz gives 60 samples, not 61. `events` accepts a float, a sequence of floats, `Annotation` objects (or a mix), or a numpy float64 array. Pass `events=None` with `query` to lock onto annotation text (mutually exclusive with a non-`None` `events`). `group` is a `SignalGroup` or list of signal indices; it must be rectangular and defaults to the largest rectangular group. `pad` governs epochs that run off the file or straddle an EDF+D gap: `"drop"` (default) omits them, `"nan"`/`"zero"`/a number/`"edge"` keep and fill them (marked `valid=False`), `"raise"` errors on the first offender. Gap-aware: samples are never spliced across a gap. See [Epoch Extraction](../guide/epochs.md).

`epoch_windows(events, *, pre: float, post: float, group=None) -> tuple[list[tuple[float, int, int]], numpy.ndarray]` -- Plan epochs without reading data. Returns `([(onset, s_start, s_end)], valid)` where the sample indices are flat offsets into the group and `valid` is a boolean array. An entry is `False` if its window runs past a file edge or crosses a gap. This is the same table `extract_epochs` decodes, so callers can inspect or filter events before paying for the decode.

`signal_groups() -> list[SignalGroup]` -- Partition all ordinary signals into groups by sample rate. Each group records its classification, sample rate, sample-count range, and whether it covers every ordinary signal. Sub-Hz precision is preserved.

`signal_group(indices: list[int]) -> SignalGroup` -- Classify an arbitrary list of file-level signal indices into a `SignalGroup`. Use when you want a group that's a subset of (or crosses) the file's natural rate-based groupings.

`proxy_2d(group: SignalGroup, pad_mode: str | float | None = None) -> Proxy2D` -- Build a 2D proxy for numpy-style multi-channel indexing. Accepts any group kind; for `Open` groups (mixed sample rates), `pad_mode` decides what reads past short channels return. Values: `"raise"` (default), `"nan"`, `"zero"`, `"edge"`, or a numeric scalar (interpreted as `Value(x)`).

`proxy_3d(group: SignalGroup) -> Proxy3D` -- Build a 3D proxy `(num_records, num_channels, samples_per_record)`. Requires `group.kind == "rectangular"`. Errors otherwise.

`write_to(path: str, variant: str | None = None, signals: SignalGroup | int | str | Sequence[int | str] | None = None) -> None` -- Re-emit this file to `path`. By default uses the source variant; pass `variant` (one of `"EDF"`, `"EDF+C"`, `"EDF+D"`, `"BDF"`, `"BDF+C"`, `"BDF+D"`) to transcode. Only ordinary signals are copied; the destination's annotation channel is rebuilt from the parsed annotations. `signals` selects which ordinary channels are written: a `SignalGroup`, a signal index, a label, or a sequence mixing both (labels match exactly, as in `signal()`); the destination's channels appear in the given order, so sets and dicts are rejected. `None` (the default) writes every ordinary signal. The annotation channel cannot be selected: it is always rebuilt automatically, and annotations are copied in full regardless of the selection. If another edfarray handle has `path` open, `write_to` raises `EdfFileError`. This includes the case where `path` is this file. Before you write to `path`, close every `EdfFile` on it, drop every signal and proxy taken from one, and finish any `EdfWriter` on it. Transcoding caveats: **EDF+D -> EDF+D** preserves the source record onsets (gaps survive), but **EDF+D -> any non-`+D` variant** flattens timing (onsets become uniform `record_idx * record_duration`); **any `+` variant -> a plain variant** drops all annotations (plain EDF/BDF has no annotation channel); and **downconverting sample size** (BDF 24-bit -> EDF 16-bit) clamps the digital range and re-encodes from physical values, losing precision.

`close() -> None` -- Explicitly release the underlying memory-mapped file. After calling `close()`, any further method or property access on the `EdfFile` raises. Existing `Signal`, `Proxy2D`, and `Proxy3D` objects keep their own references to the mapping and remain usable; the mapping is released once the file and every object derived from it are dropped. Idempotent. The context manager (`with` statement) calls `close()` on exit.

---

## inspect

```python
edfarray.inspect(path: str) -> dict
```

`path` must be a `str`. A `pathlib.Path` raises `TypeError`, so pass `str(path)`. This is the same for every function that takes a path.

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

`shape: tuple[int]` -- `(num_samples,)`.

`ndim: int` -- Always `1`.

`dtype: numpy.dtype` -- `float64`, the dtype of physical reads.

### Indexing

`sig[i]` -- Returns a single physical value as a `float`. Supports negative indexing.

`sig[start:stop]` -- Returns a `numpy.ndarray` of float64 physical values.

`sig[start:stop:step]` -- Returns a strided `numpy.ndarray` of float64 physical values.

### Methods

`to_physical() -> numpy.ndarray` -- The entire signal as a float64 numpy array.

`to_digital() -> numpy.ndarray` -- The entire signal as an int32 numpy array (raw digital values).

`times() -> numpy.ndarray` -- Timestamp in seconds from recording start for each sample. For EDF+D files, accounts for gaps between data records.

`read_range(start: int, stop: int) -> numpy.ndarray` -- Physical values for samples `[start, stop)`, indexed by sample number. Equivalent to `signal[start:stop]`.

`read_range_digital(start: int, stop: int) -> numpy.ndarray` -- Raw digital values for samples `[start, stop)`, indexed by sample number.

`read_time_range(start_sec: float, end_sec: float) -> numpy.ndarray` -- Return physical data for samples whose time falls within `[start_sec, end_sec)`. For EDF+D files, accounts for gaps between records using record onset times. For EDF and EDF+C, equivalent to indexing by flat sample number, where each time maps to the first sample at or after it.

!!! note "Caching"
    A per-`Signal` LRU cache is enabled at acquisition via `EdfFile.signal(idx, cache_capacity=N)`, not as a method on the returned `Signal`. See [Caching repeated reads](../guide/signals.md#caching-repeated-reads).

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

`ndim: int` -- Always `2`.

`dtype: numpy.dtype` -- `float64`, the dtype of physical reads.

### Methods

`read_digital(signals: list[int], start: int, stop: int) -> numpy.ndarray` -- Raw digital values for the given signals over samples `[start, stop)`, as a 2D int32 array. The counterpart to physical indexing via `[]`. `pad_mode="nan"` raises here, since NaN has no int32 representation.

### Indexing

`proxy[int, int]` -- Returns a single physical value as a `float`. Supports negative indexing.

`proxy[int, slice]` / `proxy[slice, int]` / `proxy[slice, slice]` -- Returns the natural 1D or 2D `numpy.ndarray` of float64 physical values.

`proxy[list, slice]` -- Fancy indexing on the signal axis. The list contains proxy-coordinate signal indices.

The sample (time) axis accepts a step (e.g. `proxy[:, ::4]` to downsample; negative steps supported); the signal axis requires step 1. A strided sample read still reads the full enclosing span and then subsamples, so it shrinks the result, not the I/O. Reads past a channel's valid length are governed by `pad_mode`: `"raise"` (default) raises `IndexError`; other modes fill (`"nan"` is physical-only).

---

## Proxy3D

Returned by `EdfFile.proxy_3d(group)`. A 3D view over a `Rectangular` `SignalGroup`, exposing the underlying record-major layout.

### Properties

`shape: tuple[int, int, int]` -- `(num_records, num_channels, samples_per_record)`.

`sample_rate: float` -- Common sample rate.

`supports_strided_view: bool` -- `True` when the file/group support a zero-copy `as_strided` view. See `stride_info()`.

`ndim: int` -- Always `3`.

`dtype: numpy.dtype` -- `float64`, the dtype of physical reads.

### Methods

`read_digital(record_start: int, record_stop: int, channel_start: int, channel_stop: int) -> numpy.ndarray` -- Raw digital values as a 3D int32 array shaped `(records, channels, samples_per_record)`.

`stride_info() -> dict | None` -- Byte-level metadata for a zero-copy strided view of the underlying int16 mmap, or `None` if ineligible. Keys: `base_offset`, `record_stride_bytes`, `channel_stride_bytes`, `sample_stride_bytes`, `shape`. Eligibility requires 2-byte samples, a contiguous channel-index range in the file, and no annotation channel interleaved within that span.

### Indexing

`proxy[rec, ch, samp]` -- Each axis accepts an int or a slice. The sample axis additionally accepts a step (e.g. `proxy[:, :, ::4]`; negative steps supported); the record and channel axes require step 1. Returns a scalar (all ints), 1D / 2D / 3D `numpy.ndarray` depending on how many axes are non-int. The enclosing record block is materialized regardless of sample step, so striding shrinks the result, not the work.

---

## Epochs

Returned by `EdfFile.extract_epochs(...)`. A dense epoch block plus the metadata needed to interpret it. `np.asarray(epochs)` returns `data`.

### Properties

`data: numpy.ndarray` -- Decoded float64 physical samples, shape `(n_epochs, n_channels, n_samples)`.

`onsets: numpy.ndarray` -- Event onset times in seconds, in caller order, float64.

`labels: list[str]` -- Channel labels, in group order.

`sample_rate: float` -- Common sample rate in Hz.

`valid: numpy.ndarray` -- Per-epoch boolean. `False` where any sample was padded under a fill policy. Always `True` under `pad="drop"` (offending epochs were removed instead).

`dropped: list[int]` -- Indices into the original `events` argument for epochs that `pad="drop"` removed.

`len(epochs)` returns the epoch count.

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

One-shot writer. `start_datetime` is a `datetime.datetime` or a `datetime.date`. edfarray writes its wall-clock fields as they are and ignores any `tzinfo`. `data[i]` must have length `num_records * signals[i].samples_per_record`. `variant` is one of `"EDF"`, `"EDF+C"`, `"EDF+D"`, `"BDF"`, `"BDF+C"`, `"BDF+D"`. The annotation channel is added automatically for `+` variants; do not include it in `signals`. If another edfarray handle has `path` open, `write_edf` raises `EdfFileError`.

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

Streaming writer. Supports the context manager protocol (`with` block calls `finish()` on exit). `start_datetime` follows the same rules as in `write_edf`. If another edfarray handle has `path` open, the constructor raises `EdfFileError`. The writer holds an exclusive lock on `path` until `finish()`, so opening `path` with `EdfFile` before then raises `EdfFileError`.

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

Per-signal description. `samples_per_record` combined with the writer's `record_duration` gives the sample rate. For BDF/BDF+ files, `digital_min`/`digital_max` can use the full 24-bit signed range, from -8388608 to 8388607.

Every constructor argument is also a read-only property with the same name. A `WriterSignal` compares by value and can be pickled, but it is not hashable.

---

## Annotation

```python
edfarray.Annotation(onset: float, text: str, duration: float | None = None)
```

Returned in `EdfFile.annotations`. Construct one to pass to `write_edf`, `EdfWriter`, or `extract_epochs`. Immutable. Annotations compare by value, sort by onset, are hashable, and can be pickled.

### Properties

`onset: float` -- Time in seconds from the start of the recording.

`duration: float | None` -- Duration in seconds, or `None` if not specified.

`text: str` -- The annotation text.

---

## Header editing

See [Anonymization](../guide/anonymization.md) for a full guide. These functions change the file in place. They rewrite only the fixed-width identity fields of the header and never touch data records.

### `edfarray.edit_header`

```python
edfarray.edit_header(
    path: str,
    patient_id: str | None = None,
    recording_id: str | None = None,
    start_datetime: datetime.datetime | datetime.date | None = None,
) -> dict
```

Replaces the given header fields. `None` leaves a field unchanged. edfarray checks every value before it writes anything, so a rejected edit leaves the file unchanged. `start_datetime` follows the same rules as in `write_edf`, and the header stores only whole seconds. Returns a dict that maps each changed field name to `{"before": str, "after": str}`. If another edfarray handle has `path` open, raises `EdfFileError`.

### `edfarray.anonymize`

```python
edfarray.anonymize(
    path: str,
    *,
    seed: str | None = None,
    pseudonym: str | None = None,
    date_shift_days: int | None = None,
    keep_sex: bool = True,
    keep_code: bool = False,
    keep_technician: bool = False,
    keep_equipment: bool = True,
    keep_additional: bool = False,
    dry_run: bool = False,
) -> dict
```

Replaces the patient name with a pseudonym, clears the other identity subfields unless you keep them, and shifts every date by the same number of days. With a `seed`, the pseudonym and the date shift are the same on every call. Without a seed, every call draws a new random seed. `dry_run=True` makes the same checks and returns the same report, but writes nothing. A dry run takes no lock, so it works on an open file. Returns a dict with `pseudonym`, `date_shift_days`, `patient_id_before`, `patient_id_after`, `recording_id_before`, `recording_id_after`, `start_datetime_before`, `start_datetime_after`, `scrubbed_terms`, and `dry_run`. Unless `dry_run=True`, raises `EdfFileError` if another edfarray handle has `path` open.

### `edfarray.audit`

```python
edfarray.audit(path: str, terms: Sequence[str] | None = None) -> dict
```

Searches signal labels, transducers, prefiltering text, and annotation text for identity strings. If `terms` is `None`, the terms come from the current header. After `anonymize`, pass its `scrubbed_terms`. Returns a dict with `terms`, `clean`, and `hits`. Each hit has `location`, `signal_index` (`None` for annotation text), `term`, and `excerpt`. `audit` only reads the file, so it works on an open file.

---

## edfarray.aio

An async API with the same classes and methods, in which each read returns an awaitable. `edfarray.aio.open(path, variant=None)` returns an `edfarray.aio.EdfFile`. See [Async API](../guide/async.md) for the differences from the sync API.

