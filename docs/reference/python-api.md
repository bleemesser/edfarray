# Python API Reference

## EdfFile

```python
edfarray.EdfFile(path: str)
```

Opens an EDF/EDF+ file at the given path. Parses the header and builds the annotation index on construction.

Supports the context manager protocol (`with` statement).

### Properties

`num_signals: int` -- Total number of signals in the file, including annotation channels.

`num_records: int` -- Number of data records.

`record_duration: float` -- Duration of each data record in seconds.

`duration: float` -- Total recording duration in seconds.

`variant: str` -- `"EDF"`, `"EDF+C"`, or `"EDF+D"`.

`start_datetime: datetime.datetime | str` -- Recording start date and time. Returns a `datetime.datetime` if the header date fields could be parsed, or a raw string like `"04.04.yy 12.57.02"` if the date is anonymized or non-standard.

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

`annotations: list[Annotation]` -- All non-timekeeping annotations from the file, sorted by onset. Empty for plain EDF files.

`warnings: list[str]` -- Parse warnings accumulated during file open. Empty if the file is well-formed.

`header: dict` -- Dictionary with basic header fields: `version`, `patient_id`, `recording_id`, `num_signals`, `num_records`, `record_duration`, `duration`, `variant`.

### Methods

`signal(idx_or_label: int | str) -> Signal` -- Get a signal by index or label. Raises `IndexError` for out-of-range indices, `KeyError` for unknown labels.

`ordinary_signal_indices() -> list[int]` -- Indices of all non-annotation signals.

`read_page(start_sec: float, end_sec: float, signal_indices: list[int] | None = None) -> list[numpy.ndarray]` -- Read physical (float64) data for multiple signals over a time range. Returns one array per signal. If `signal_indices` is `None`, reads all ordinary signals. Signals with different sample rates produce arrays of different lengths.

`read_page_digital(start_sec: float, end_sec: float, signal_indices: list[int] | None = None) -> list[numpy.ndarray]` -- Same as `read_page()` but returns raw int16 digital values without gain/offset conversion.

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

## Annotation

Returned in `EdfFile.annotations`. Immutable.

### Properties

`onset: float` -- Time in seconds from the start of the recording.

`duration: float | None` -- Duration in seconds, or `None` if not specified.

`text: str` -- The annotation text.
