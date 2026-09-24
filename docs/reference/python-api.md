# Python API Reference

For the exception hierarchy, the supported indexing rules, and the threading and async guarantees, see
[API contracts](contracts.md).

## EdfFile

```python
edfarray.EdfFile(path: str, variant: str | None = None, scan_annotations: bool = True)
```

The constructor opens an EDF/EDF+ file at the given path. It parses the header synchronously and starts a background annotation scan for EDF+ files. You can read signals immediately after construction.

If `scan_annotations=False`, edfarray does not start the scan at open. The scan then runs on the calling thread at the first access to annotations. The scan reads every data record (a fixed-duration block of samples for every signal). On very large files, the scan takes a long time. See [Performance](../guide/performance.md).

`variant` forces the file variant and ignores the detected variant. Use it for files that omit or misreport the EDF+ `"+C"`/`"+D"` marker. It controls only the plain, `"+C"`, and `"+D"` distinction. The version field sets the sample size of EDF or BDF. If an override changes that sample size, the constructor raises `ValueError`. If the override does not match the detected variant, edfarray records the mismatch in `warnings`.

If an `EdfWriter` or `write_edf` call is still writing `path`, the constructor raises `EdfFileError`. The open file holds a shared lock on `path`. A proxy is an object that reads samples only on access. The lock stays until you drop the `EdfFile` and every signal and proxy taken from it. While the lock stays, edfarray rejects writes to `path`. See [Memory mapping](contracts.md#memory-mapping).

`EdfFile` supports the context manager protocol (the `with` statement).

### Properties

`num_signals: int`: The total number of signals in the file, including annotation channels.

`num_records: int`: The number of data records. For EDF-L files, the header value is `-1` ("unknown length"). For these files, edfarray recovers the true count from the file size at open time. A `warnings` entry records the recovery.

`record_duration: float`: The duration of each data record in seconds.

`duration: float`: The total recording duration in seconds.

`variant: str`: `"EDF"`, `"EDF+C"`, `"EDF+D"`, `"BDF"`, `"BDF+C"`, or `"BDF+D"`.

`start_datetime: datetime.datetime | str`: The recording start date and time. If edfarray can parse the header date fields, the value is a `datetime.datetime`. If edfarray cannot parse the date, for example because it is anonymized or non-standard, the value is a raw string such as `"04.04.yy 12.57.02"`. The EDF header stores only whole seconds. EDF+ files encode subsecond precision in the first time-keeping annotation. edfarray applies that precision to annotation onsets and sample timestamps, but not to this property.

`patient_id: str`: The raw 80-byte patient identification field from the header.

`recording_id: str`: The raw 80-byte recording identification field from the header.

`patient_name: str | None`: Parsed from the patient identification field. edfarray replaces underscores with spaces. If the field is absent or set to `"X"`, the value is `None`.

`patient_code: str | None`: The hospital patient code from the patient identification field.

`patient_sex: str | None`: `"M"` or `"F"`. If the sex is unknown, the value is `None`.

`patient_birthdate: datetime.date | str | None`: The patient birthdate. If edfarray can parse it, the value is a `datetime.date`. If the format is non-standard, the value is a raw string. If the field is absent, the value is `None`.

`patient_additional: str | None`: Additional patient information after the standard subfields.

`admin_code: str | None`: The hospital administration code from the recording identification field.

`technician: str | None`: The technician or investigator code.

`equipment: str | None`: The equipment code.

`recording_additional: str | None`: Additional recording information.

The annotation accessors below block until the background annotation scan is complete. To get the scan status without blocking, use `annotations_ready`. Range queries use binary search.

`annotations: list[Annotation]`: All annotations from the file except time-keeping annotations, sorted by onset. For plain EDF files, the list is empty.

`annotations_before(t: float) -> list[Annotation]`: Annotations with onset strictly before `t`.

`annotations_after(t: float) -> list[Annotation]`: Annotations with onset >= `t`.

`annotations_in_range(start: float, end: float) -> list[Annotation]`: Annotations with onset in the half-open interval `[start, end)`.

`filter_annotations(query: str, regex: bool = False) -> list[Annotation]`: Filters annotations by text content. With `regex=False`, the method matches the query as a case-insensitive substring. With `regex=True`, the method matches the query as a case-insensitive regex pattern. An invalid regex pattern raises `ValueError`.

`events(query: str, regex: bool = False) -> list[Annotation]`: Returns the `Annotation` list that `query` matches, with the same rules as `filter_annotations`. You can pass the result directly to `extract_epochs`.

`annotations_by_text(text: str) -> list[Annotation]`: Annotations whose text exactly matches `text` (case-sensitive).

`warnings: list[str]`: Warnings from the header parse and from the annotation scan. It waits until the scan is complete. If the file is well-formed, the list is empty.

`header() -> dict`: Returns a dict with these basic header fields: `version`, `patient_id`, `recording_id`, `num_signals`, `num_records`, `record_duration`, `duration`, `variant`. `header()` is a method, not a property, because it builds a new dict on each call.

`annotations_ready: bool`: If the background annotation scan is complete, the value is `True`. Does not block.

`scan_progress: tuple[int, int]`: `(records_scanned, total_records)` for the background annotation scan. Does not block. For large files, you can poll it to show progress.

`closed: bool`: `True` after a call to `close()`.

### Methods

`signal(idx_or_label: int | str, cache_capacity: int = 0, strategy: str | None = None) -> Signal`: Returns a signal by index or label. An out-of-range index raises `OutOfRangeError` (an `IndexError`). An unknown label raises `SignalNotFoundError` (a `KeyError`). `signal()` does not support negative indices. A negative index raises `OutOfRangeError`. `strategy` is `"auto"` (default), `"mmap"`, or `"stream"`. See [Performance](../guide/performance.md). `cache_capacity` enables a per-`Signal` LRU cache of decoded physical records. See [Caching repeated reads](../guide/signals.md#caching-repeated-reads).

`find_all_signals(label: str, exact: bool = False) -> list[int]`: Returns the indices of all signals whose label matches `label`. If `exact` is `False` (default), the method does a case-insensitive substring match. If `exact` is `True`, the method does a case-sensitive exact match. The search includes annotation channels. To read a signal, pass its index to `signal()`.

`signal_labels() -> list[str]`: The labels of all signals in the file.

`ordinary_signal_indices() -> list[int]`: The indices of all ordinary signals. An ordinary signal is a signal that is not the annotation channel.

`read_page(start_sec: float, end_sec: float, signal_indices: list[int] | None = None, use_time: bool = False) -> list[numpy.ndarray]`: Reads physical (float64) data for multiple signals over a time range. Returns one array per signal. If `signal_indices` is `None`, the method reads all ordinary signals. Signals with different sample rates give arrays of different lengths. A flat index is a sample position counted across records, ignoring gaps. If `use_time` is `False` (default), each time maps to the flat index of the first sample at or after it. Thus `[start_sec, end_sec)` is half-open. A time within 1e-6 of a sample counts as that sample. This rule absorbs float noise such as `0.1 + 0.2`. Every time-based API uses this one rule. An EDF+D gap is a time span with no records. If an EDF+D file has time gaps, set `use_time=True`. edfarray then resolves the time range with the actual record onset times. See [Annotations & Time](../guide/annotations.md#read_page-and-proxy2d-use-flat-sample-indices) for details.

`read_page_digital(start_sec: float, end_sec: float, signal_indices: list[int] | None = None, use_time: bool = False) -> list[numpy.ndarray]`: Same as `read_page()`, but returns raw int32 digital values without gain and offset conversion. If `use_time` is `True`, the method resolves the time range with record onset times for EDF+D files.

`extract_epochs(events, *, pre: float, post: float, group=None, pad=None, query=None, regex=False) -> Epochs`: Decodes fixed `[onset - pre, onset + post)` windows around each event into a dense `(n_epochs, n_channels, n_samples)` float64 block. The decode runs in parallel. `n_samples` is always `ceil((pre + post) * sample_rate)` and does not depend on where the events fall. A product within 1e-6 of an integer counts as that integer. Thus `pre=0.1, post=0.2` at 200 Hz gives 60 samples, not 61.

`events` accepts a float, a sequence of floats, `Annotation` objects (or a mix), or a numpy float64 array. To select events by annotation text, pass `events=None` with `query`. You cannot use `query` with a non-`None` `events`. `group` is a `SignalGroup` or a list of signal indices. The group must be rectangular. The default is the largest rectangular group.

`pad` controls epochs that run off the file or cross an EDF+D gap:

- `"drop"` (default) omits these epochs.
- `"nan"`, `"zero"`, a number, or `"edge"` keeps these epochs, fills them, and marks them `valid=False`.
- `"raise"` raises an error on the first such epoch.

edfarray never joins samples across a gap. See [Epoch Extraction](../guide/epochs.md).

`epoch_windows(events, *, pre: float, post: float, group=None) -> tuple[list[tuple[float, int, int]], numpy.ndarray]`: Plans epochs without reading data. Returns `([(onset, s_start, s_end)], valid)`. The sample indices are flat offsets into the group, and `valid` is a boolean array. If a window runs past a file edge or crosses a gap, its entry is `False`. `extract_epochs` decodes this same table. Thus you can inspect or filter events before you pay for the decode.

`signal_groups() -> list[SignalGroup]`: Partitions all ordinary signals into groups by sample rate. Each group records its classification, its sample rate, its range of sample counts, and whether it covers every ordinary signal. The sample rates keep sub-Hz precision.

`signal_group(indices: list[int]) -> SignalGroup`: Classifies any list of file-level signal indices into a `SignalGroup`. Use it to get a group that is a subset of the natural rate-based groups of the file, or that crosses them.

`proxy_2d(group: SignalGroup, pad_mode: str | float | None = None) -> Proxy2D`: Builds a 2D proxy for numpy-style indexing over multiple signals. The method accepts any group kind. For `Open` groups (mixed sample rates), `pad_mode` sets what reads past the end of shorter signals return. Values: `"raise"` (default), `"nan"`, `"zero"`, `"edge"`, or a numeric scalar (interpreted as `Value(x)`).

`proxy_3d(group: SignalGroup) -> Proxy3D`: Builds a 3D proxy with shape `(num_records, num_channels, samples_per_record)`. Requires `group.kind == "rectangular"`. For any other group, the method raises an error.

`write_to(path: str, variant: str | None = None, signals: SignalGroup | int | str | Sequence[int | str] | None = None) -> None`: Writes a copy of this file to `path`. By default, the copy uses the source variant. To transcode, pass `variant` (one of `"EDF"`, `"EDF+C"`, `"EDF+D"`, `"BDF"`, `"BDF+C"`, `"BDF+D"`). edfarray copies only ordinary signals. It rebuilds the annotation channel of the destination from the parsed annotations.

`signals` selects which ordinary signals edfarray writes. It accepts a `SignalGroup`, a signal index, a label, or a sequence that mixes indices and labels. Labels must match exactly, as in `signal()`. The destination signals appear in the given order, so `write_to` rejects sets and dicts. `None` (the default) writes every ordinary signal. You cannot select the annotation channel. edfarray always rebuilds it and copies all annotations, whatever the selection.

If another edfarray handle has `path` open, `write_to` raises `EdfFileError`. This includes the case where `path` is this file. Before you write to `path`, do these steps:

- Close every `EdfFile` on `path`.
- Drop every signal and proxy taken from one of these files.
- Finish any `EdfWriter` on `path`.

Transcoding has these limits:

- From EDF+D to EDF+D, edfarray keeps the source record onsets, so the gaps stay.
- From EDF+D to any variant without `+D`, edfarray flattens the timing. The onsets become uniform `record_idx * record_duration`.
- From any `+` variant to a plain variant, edfarray drops all annotations. Plain EDF and BDF have no annotation channel.
- A smaller sample size (BDF 24-bit to EDF 16-bit) clamps the digital range and encodes again from physical values. This loses precision.

`close() -> None`: Releases the underlying memory-mapped file. After `close()`, any further method or property access on the `EdfFile` raises an error. Existing `Signal`, `Proxy2D`, and `Proxy3D` objects keep their own references to the mapping and stay usable. When you drop the file and every object taken from it, edfarray releases the mapping. A second call to `close()` does nothing. The context manager (`with` statement) calls `close()` on exit.

---

## inspect

```python
edfarray.inspect(path: str) -> dict
```

`path` must be a `str`. A `pathlib.Path` raises `TypeError`, so pass `str(path)`. This rule applies to every function that takes a path.

`inspect` reads metadata from the header of an EDF/EDF+ file. It does not memory-map the file, start background threads, read any data records, or build an annotation index.

Returns a dict with these keys:

`variant: str`: `"EDF"`, `"EDF+C"`, `"EDF+D"`, `"BDF"`, `"BDF+C"`, or `"BDF+D"`.

`num_signals: int`: The total number of signals in the file, including annotation channels.

`num_records: int`: The number of data records. For EDF-L files (header value `-1`, "unknown length"), edfarray recovers this count from the file size.

`record_duration: float`: The duration of each data record in seconds.

`duration: float`: The total recording duration in seconds.

`patient_id: str`: The raw 80-byte patient identification field.

`recording_id: str`: The raw 80-byte recording identification field.

`signal_labels: list[str]`: The labels of all signals in the file.

`sample_rates: list[float]`: The sample rates in Hz, one for each signal.

---

## Signal

`EdfFile.signal()` returns this object. It is a proxy view of one signal. It decodes samples from the memory-mapped file on access.

### Properties

`label: str`: The signal label, for example `"EEG Fpz-Cz"`.

`transducer: str`: The transducer type, for example `"AgAgCl electrode"`.

`physical_dimension: str`: The physical units, for example `"uV"`.

`prefiltering: str`: The prefiltering description, for example `"HP:0.1Hz LP:75Hz"`.

`sample_rate: float`: The sample frequency in Hz.

`samples_per_record: int`: The number of samples in each data record for this signal.

`physical_min: float`: The physical minimum value.

`physical_max: float`: The physical maximum value.

`digital_min: int`: The digital minimum value.

`digital_max: int`: The digital maximum value.

`num_samples: int`: The total number of samples. Same as `len(sig)`.

`shape: tuple[int]`: `(num_samples,)`.

`ndim: int`: Always `1`.

`dtype: numpy.dtype`: `float64`, the dtype of physical reads.

### Indexing

`sig[i]`: Returns one physical value as a `float`. Supports negative indexing.

`sig[start:stop]`: Returns a `numpy.ndarray` of float64 physical values.

`sig[start:stop:step]`: Returns a strided `numpy.ndarray` of float64 physical values.

### Methods

`to_physical() -> numpy.ndarray`: The full signal as a float64 numpy array.

`to_digital() -> numpy.ndarray`: The full signal as an int32 numpy array (raw digital values).

`times() -> numpy.ndarray`: The time in seconds from the recording start for each sample. For EDF+D files, the times include the gaps between data records.

`read_range(start: int, stop: int) -> numpy.ndarray`: Physical values for samples `[start, stop)`, indexed by sample number. Equivalent to `signal[start:stop]`.

`read_range_digital(start: int, stop: int) -> numpy.ndarray`: Raw digital values for samples `[start, stop)`, indexed by sample number.

`read_time_range(start_sec: float, end_sec: float) -> numpy.ndarray`: Returns physical data for samples whose time is in `[start_sec, end_sec)`. For EDF+D files, the method uses record onset times to include the gaps between records. For EDF and EDF+C files, the method is equivalent to indexing by flat sample number. Each time maps to the first sample at or after it.

!!! note "Caching"
    To enable a per-`Signal` LRU cache, call `EdfFile.signal(idx, cache_capacity=N)`. The returned `Signal` has no method for it. See [Caching repeated reads](../guide/signals.md#caching-repeated-reads).

`__len__() -> int`: The total number of samples.

---

## SignalGroup

A set of file-level signal indices plus classification metadata.
`EdfFile.signal_groups()` and `EdfFile.signal_group(indices)` build it.

### Properties

`indices: list[int]`: The file-level signal indices in the group.

`kind: str`: `"rectangular"` (all signals share a sample rate and a total sample count) or `"open"` (mixed sample rates, 2D only).

`sample_rate: float | None`: The common rate. If `kind == "open"`, the value is `None`.

`samples_per_record: int | None`: The common samples per record (SPR). If `kind == "open"`, the value is `None`.

`min_samples: int` / `max_samples: int`: The bounds on the total sample count of the signals in the group. For `"rectangular"` groups, they are equal.

`covers_all_ordinary: bool`: If this group spans every ordinary signal in the file, the value is `True`. Otherwise, it is `False`. Only `signal_groups()` sets this value. For groups built by hand, it is always `False`.

`is_singleton: bool`: If the group contains exactly one signal, the value is `True`. Otherwise, it is `False`.

`is_rectangular: bool`: Same as `kind == "rectangular"`.

`len(group)`: The number of signals in the group.

---

## Proxy2D

`EdfFile.proxy_2d(group, pad_mode=...)` returns this object. It is a 2D view over a `SignalGroup`. It reads from the memory-mapped file on demand.

### Properties

`shape: tuple[int, int]`: `(num_signals, max_samples_across_signals)`.

`sample_rate: float | None`: The common rate. If the underlying group is `Open`, the value is `None`.

`valid_lengths: list[int]`: The valid sample count of each signal, in proxy-coordinate order. For `"rectangular"` groups, all entries equal `shape[1]`. For `"open"` groups, the entries vary.

`pad_mode: str`: `"raise"`, `"nan"`, `"zero"`, `"value"`, or `"edge"`.

`ndim: int`: Always `2`.

`dtype: numpy.dtype`: `float64`, the dtype of physical reads.

### Methods

`read_digital(signals: list[int], start: int, stop: int) -> numpy.ndarray`: Raw digital values for the given signals over samples `[start, stop)`, as a 2D int32 array. This is the digital counterpart of physical indexing with `[]`. If `pad_mode="nan"`, this method raises an error, because int32 has no NaN value.

### Indexing

`proxy[int, int]`: Returns one physical value as a `float`. Supports negative indexing.

`proxy[int, slice]` / `proxy[slice, int]` / `proxy[slice, slice]`: Returns the natural 1D or 2D `numpy.ndarray` of float64 physical values.

`proxy[list, slice]`: Fancy indexing on the signal axis. The list contains proxy-coordinate signal indices.

The sample (time) axis accepts a step, for example `proxy[:, ::4]` to downsample. Negative steps are also supported. The signal axis requires step 1. A strided sample read still reads the full enclosing span and then takes every nth sample. Thus a step makes the result smaller, but not the I/O. `pad_mode` controls reads past the valid length of a signal. If `pad_mode` is `"raise"` (default), such a read raises `IndexError`. Other modes fill the missing samples, and `"nan"` applies only to physical reads.

---

## Proxy3D

`EdfFile.proxy_3d(group)` returns this object. It is a 3D view over a `Rectangular` `SignalGroup` and shows the record-major layout of the underlying data.

### Properties

`shape: tuple[int, int, int]`: `(num_records, num_channels, samples_per_record)`.

`sample_rate: float`: The common sample rate.

`supports_strided_view: bool`: If the file and group support a zero-copy `as_strided` view, the value is `True`. See `stride_info()`.

`ndim: int`: Always `3`.

`dtype: numpy.dtype`: `float64`, the dtype of physical reads.

### Methods

`read_digital(record_start: int, record_stop: int, channel_start: int, channel_stop: int) -> numpy.ndarray`: Raw digital values as a 3D int32 array with shape `(records, channels, samples_per_record)`.

`stride_info() -> dict | None`: Byte-level metadata for a zero-copy strided view of the underlying int16 mmap. If the view is not possible, the method returns `None`. Keys: `base_offset`, `record_stride_bytes`, `channel_stride_bytes`, `sample_stride_bytes`, `shape`. The view requires 2-byte samples, a contiguous range of signal indices in the file, and no annotation channel in that range.

### Indexing

`proxy[rec, ch, samp]`: Each axis accepts an int or a slice. The sample axis also accepts a step, for example `proxy[:, :, ::4]`. Negative steps are also supported. The record and channel axes require step 1. If all three indices are ints, the result is a scalar. Otherwise, the result is a 1D, 2D, or 3D `numpy.ndarray`, with one dimension for each axis that is not an int. edfarray decodes the full enclosing block of records, whatever the sample step. Thus a step makes the result smaller, but not the work.

---

## Epochs

`EdfFile.extract_epochs(...)` returns this object. It holds a dense block of epochs and the metadata that you need to interpret it. `np.asarray(epochs)` returns `data`.

### Properties

`data: numpy.ndarray`: The decoded float64 physical samples, with shape `(n_epochs, n_channels, n_samples)`.

`onsets: numpy.ndarray`: The event onset times in seconds, in caller order, as float64.

`labels: list[str]`: The signal labels for the channel axis, in group order.

`sample_rate: float`: The common sample rate in Hz.

`valid: numpy.ndarray`: One boolean for each epoch. The value is `False` where edfarray padded a sample under a fill policy. Under `pad="drop"`, the values are always `True`, because edfarray removes those epochs.

`dropped: list[int]`: The indices into the original `events` argument of the epochs that `pad="drop"` removed.

`len(epochs)` returns the number of epochs.

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
    start_datetime: datetime.datetime | datetime.date | None = None,
    patient_id: str | None = None,
    recording_id: str | None = None,
    annotation_bytes_per_record: int | None = None,
) -> None
```

Writes a full file in one call. `start_datetime` is a `datetime.datetime` or a `datetime.date`. edfarray writes its wall-clock fields as they are and ignores any `tzinfo`. `data[i]` must have length `num_records * signals[i].samples_per_record`. `variant` is one of `"EDF"`, `"EDF+C"`, `"EDF+D"`, `"BDF"`, `"BDF+C"`, `"BDF+D"`. For `+` variants, edfarray adds the annotation channel. Do not include it in `signals`. If another edfarray handle has `path` open, `write_edf` raises `EdfFileError`.

### `edfarray.EdfWriter`

```python
edfarray.EdfWriter(
    path: str,
    *,
    variant: str,
    record_duration: float,
    signals: list[WriterSignal],
    start_datetime: datetime.datetime | datetime.date | None = None,
    patient_id: str | None = None,
    recording_id: str | None = None,
    annotation_bytes_per_record: int | None = None,
)
```

Writes a file one record at a time. `EdfWriter` supports the context manager protocol, and the `with` block calls `finish()` on exit. `start_datetime` follows the same rules as in `write_edf`. If another edfarray handle has `path` open, the constructor raises `EdfFileError`. The writer holds an exclusive lock on `path` until `finish()`. If you open `path` with `EdfFile` before then, `EdfFile` raises `EdfFileError`.

`write_record(physical: list[numpy.ndarray], annotations: list[Annotation] | None = None) -> None`: Encodes and appends one record. `physical[i]` must be a 1D float64 array of length `signals[i].samples_per_record`. If you pass `annotations`, edfarray embeds them in the annotation channel of this record. It also embeds any pending annotations that `add_annotation` queued.

`add_annotation(annotation: Annotation) -> None`: Queues an annotation. edfarray embeds it with the next `write_record` call.

`finish() -> None`: Flushes the data, then seeks back and patches `num_records` in the header. A second call does nothing. `__exit__` calls `finish()`.

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

Describes one signal. `samples_per_record` and the `record_duration` of the writer together give the sample rate. For BDF/BDF+ files, `digital_min` and `digital_max` can use the full 24-bit signed range, from -8388608 to 8388607.

Every constructor argument is also a read-only property with the same name. A `WriterSignal` compares by value and can be pickled, but it is not hashable.

---

## Annotation

```python
edfarray.Annotation(onset: float, text: str, duration: float | None = None)
```

`EdfFile.annotations` returns these objects. Construct one to pass to `write_edf`, `EdfWriter`, or `extract_epochs`. An `Annotation` is immutable. Annotations compare by value, sort by onset, are hashable, and can be pickled.

### Properties

`onset: float`: The time in seconds from the start of the recording.

`duration: float | None`: The duration in seconds. If no duration is set, the value is `None`.

`text: str`: The annotation text.

---

## Header editing

See [Anonymization](../guide/anonymization.md) for a full guide. These functions change the file in place. They rewrite only the fixed-width identity fields of the header. They never change data records.

### `edfarray.edit_header`

```python
edfarray.edit_header(
    path: str,
    patient_id: str | None = None,
    recording_id: str | None = None,
    start_datetime: datetime.datetime | datetime.date | None = None,
) -> dict
```

Replaces the given header fields. `None` leaves a field unchanged. edfarray checks every value before it writes anything, so a rejected edit leaves the file unchanged. `start_datetime` follows the same rules as in `write_edf`, and the header stores only whole seconds. Returns a dict that maps each changed field name to `{"before": str, "after": str}`. If another edfarray handle has `path` open, `edit_header` raises `EdfFileError`.

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

Replaces the patient name with a pseudonym and shifts every date by the same number of days. The function also clears the other identity subfields, except the subfields that you keep. With a `seed`, the pseudonym and the date shift are the same on every call. Without a seed, every call uses a new random seed.

A dry run is a call that writes nothing. If `dry_run=True`, `anonymize` makes the same checks and returns the same report. A dry run takes no lock, so it works on an open file.

Returns a dict with `pseudonym`, `date_shift_days`, `patient_id_before`, `patient_id_after`, `recording_id_before`, `recording_id_after`, `start_datetime_before`, `start_datetime_after`, `scrubbed_terms`, and `dry_run`. If `dry_run` is `False` and another edfarray handle has `path` open, `anonymize` raises `EdfFileError`.

### `edfarray.audit`

```python
edfarray.audit(path: str, terms: Sequence[str] | None = None) -> dict
```

Searches signal labels, transducers, prefiltering text, and annotation text for identity strings. If `terms` is `None`, `audit` takes the terms from the current header. After `anonymize`, pass its `scrubbed_terms`. Returns a dict with `terms`, `clean`, and `hits`. Each hit has `location`, `signal_index` (`None` for annotation text), `term`, and `excerpt`. `audit` only reads the file, so it works on an open file.

---

## edfarray.aio

An async API with the same classes and methods. Each read returns an awaitable. `edfarray.aio.open(path, variant=None)` returns an `edfarray.aio.EdfFile`. See [Async API](../guide/async.md) for the differences from the sync API.
