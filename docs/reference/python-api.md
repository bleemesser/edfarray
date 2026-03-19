# Python API Reference

## EdfFile

```python
edfarray.EdfFile(path: str)
```

Opens an EDF/EDF+ file at the given path. Parses the header synchronously and starts a background annotation scan for EDF+ files. Signal reads work immediately after construction.

Supports the context manager protocol (`with` statement).

### Properties

`num_signals: int` -- Total number of signals in the file, including annotation channels.

`num_records: int` -- Number of data records.

`record_duration: float` -- Duration of each data record in seconds.

`duration: float` -- Total recording duration in seconds.

`variant: str` -- `"EDF"`, `"EDF+C"`, or `"EDF+D"`.

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

`warnings: list[str]` -- Parse warnings accumulated during file open. Empty if the file is well-formed. Blocks until the background annotation scan is complete.

`header: dict` -- Dictionary with basic header fields: `version`, `patient_id`, `recording_id`, `num_signals`, `num_records`, `record_duration`, `duration`, `variant`.

`annotations_ready: bool` -- Whether the background annotation scan has completed. Non-blocking.

`scan_progress: tuple[int, int]` -- `(records_scanned, total_records)` for the background annotation scan. Non-blocking. Can be polled to show progress for large files.

### Methods

`signal(idx_or_label: int | str) -> Signal` -- Get a signal by index or label. Raises `IndexError` for out-of-range indices, `KeyError` for unknown labels.

`signal_labels() -> list[str]` -- Labels of all signals in the file.

`ordinary_signal_indices() -> list[int]` -- Indices of all non-annotation signals.

`read_page(start_sec: float, end_sec: float, signal_indices: list[int] | None = None) -> list[numpy.ndarray]` -- Read physical (float64) data for multiple signals over a time range. Returns one array per signal. If `signal_indices` is `None`, reads all ordinary signals. Signals with different sample rates produce arrays of different lengths. **Note:** time parameters are converted to flat sample indices (`int(time * sample_rate)`). For EDF+D files with time gaps, this does not correspond to physical time. See [Annotations & Time](../guide/annotations.md#read_page-and-arrayproxy-use-flat-sample-indices) for the correct EDF+D workflow.

`read_page_digital(start_sec: float, end_sec: float, signal_indices: list[int] | None = None) -> list[numpy.ndarray]` -- Same as `read_page()` but returns raw int16 digital values without gain/offset conversion.

`array_proxy(signal_indices: list[int] | None = None) -> ArrayProxy` -- Create a 2D array proxy for numpy-style multi-channel indexing. All selected signals must have the same sample rate. If `signal_indices` is `None`, uses all ordinary signals. Raises `ValueError` if sample rates differ.

`signal_indices_by_rate() -> dict[int, list[int]]` -- Group ordinary signal indices by sample rate (Hz, rounded to integer). Useful for creating separate `ArrayProxy` instances when the file has mixed sample rates.

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

`digital_min: int` -- Digital minimum value (i16).

`digital_max: int` -- Digital maximum value (i16).

`num_samples: int` -- Total number of samples. Same as `len(sig)`.

### Indexing

`sig[i]` -- Returns a single physical value as a `float`. Supports negative indexing.

`sig[start:stop]` -- Returns a `numpy.ndarray` of float64 physical values.

`sig[start:stop:step]` -- Returns a strided `numpy.ndarray` of float64 physical values.

### Methods

`to_numpy() -> numpy.ndarray` -- The entire signal as a float64 numpy array.

`to_digital() -> numpy.ndarray` -- The entire signal as an int16 numpy array (raw digital values).

`times() -> numpy.ndarray` -- Timestamp in seconds from recording start for each sample. For EDF+D files, accounts for gaps between data records.

`__len__() -> int` -- Total number of samples.

---

## ArrayProxy

Returned by `EdfFile.array_proxy()`. A 2D view over multiple signals with the same sample rate. Reads data on demand from the memory-mapped file.

### Properties

`shape: tuple[int, int]` -- `(num_signals, total_samples_per_signal)`.

`sample_rate: float` -- Common sample rate (Hz) of all signals in the proxy.

### Indexing

`proxy[int, int]` -- Returns a single physical value as a `float`. Supports negative indexing on both axes.

`proxy[int, slice]` -- Returns a 1D `numpy.ndarray` of float64 physical values for one signal.

`proxy[slice, int]` -- Returns a 1D `numpy.ndarray` with one sample from each signal in the slice.

`proxy[slice, slice]` -- Returns a 2D `numpy.ndarray` of shape `(num_selected_signals, num_selected_samples)`.

`proxy[list, slice]` -- Fancy indexing on the signal axis. The list contains proxy-coordinate signal indices.

Step values other than 1 are not supported in slices.

---

## Annotation

Returned in `EdfFile.annotations`. Immutable.

### Properties

`onset: float` -- Time in seconds from the start of the recording.

`duration: float | None` -- Duration in seconds, or `None` if not specified.

`text: str` -- The annotation text.
