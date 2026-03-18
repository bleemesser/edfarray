# Getting Started

## Installation

```bash
pip install edfarray
```

From source:

```bash
git clone https://github.com/bleemesser/edfarray.git
cd edfarray
uv sync
uv run maturin develop --release
```

## Opening a file

```python
import edfarray

f = edfarray.EdfFile("recording.edf")
```

Or with a context manager:

```python
with edfarray.EdfFile("recording.edf") as f:
    print(f.num_signals)
```

## File metadata

```python
f = edfarray.EdfFile("recording.edf")

f.variant         # "EDF", "EDF+C", or "EDF+D"
f.num_signals     # total signals, including annotation channels
f.num_records     # number of data records
f.record_duration # seconds per record
f.duration        # total recording duration in seconds
f.start_datetime  # datetime.datetime, or string if anonymized
```

## Patient and recording info

EDF+ files encode structured patient and recording information in the header. edfarray parses these automatically, including plain EDF files that happen to use the EDF+ subfield format.

```python
# Patient fields. None if not present or "X" in the header.
f.patient_name       # "Smith, Casey" (underscores replaced with spaces)
f.patient_code       # "MCH-0234567"
f.patient_sex        # "M" or "F"
f.patient_birthdate  # datetime.date, string if unparseable, or None

# Recording fields.
f.admin_code         # "PSG-1234"
f.technician         # "John Doe"
f.equipment          # "Nihon Kohden"

# Raw header fields are always available.
f.patient_id         # raw 80-byte patient identification string
f.recording_id       # raw 80-byte recording identification string
```

## Listing signals

```python
for i in range(f.num_signals):
    sig = f.signal(i)
    print(f"[{i}] {sig.label}: {sig.sample_rate} Hz, {len(sig)} samples")
```

Signals can also be accessed by label:

```python
eeg = f.signal("EEG Fpz-Cz")
```

## Reading your first signal

```python
sig = f.signal(0)

# Single sample.
value = sig[0]  # returns a float

# A chunk of data.
chunk = sig[0:1000]  # returns a numpy float64 array

# The whole signal.
data = sig.to_numpy()
print(data.shape, data.dtype)  # (30000,) float64
```

## Anonymized dates

Some EDF files have anonymized date fields like `"04.04.yy"` instead of `"04.04.11"`. edfarray handles this gracefully. `start_datetime` will be a string instead of a `datetime.datetime`:

```python
dt = f.start_datetime
if isinstance(dt, str):
    print(f"Anonymized: {dt}")
else:
    print(f"Recorded on: {dt.date()}")
```

`patient_birthdate` works the same way: it may be a `datetime.date`, a raw string, or `None`.

## Parse warnings

Non-fatal issues (malformed annotations, unexpected field values) are collected in `warnings` rather than raising exceptions:

```python
if f.warnings:
    for w in f.warnings:
        print(f"Warning: {w}")
```
