# Annotations & Time

## Annotations

EDF+ files can contain annotations. An annotation is a text event with a timestamp, stored in the recording. Plain EDF files do not have annotations.

```python
f = edfarray.EdfFile("recording.edf")

for ann in f.annotations:
    print(f"{ann.onset:.2f}s: {ann.text}")
    if ann.duration is not None:
        print(f"  duration: {ann.duration}s")
```

Each annotation has:

- `onset`: Time in seconds from the start of the recording.
- `duration`: Duration in seconds. If the file does not specify a duration, the value is `None`.
- `text`: The annotation text (UTF-8).

edfarray sorts the annotations by onset time.

## EDF+ variants

A record is a fixed-duration block of samples. The `variant` property gives the format of the file:

- `"EDF"`: Plain EDF. No annotations and no subsecond precision.
- `"EDF+C"`: EDF+ contiguous. The records follow each other without gaps.
- `"EDF+D"`: EDF+ discontinuous. The records can have time gaps between them.
- `"BDF"`, `"BDF+C"`, `"BDF+D"`: The same three layouts with 24-bit samples.

```python
f.variant  # "EDF", "EDF+C", "EDF+D", "BDF", "BDF+C", or "BDF+D"
```

## Subsecond start time

The EDF header stores the recording start time with one-second resolution (`hh.mm.ss`). EDF+ files encode the subsecond precision in the first time-keeping annotation of the first record.

edfarray reads this value automatically. All annotation onsets and sample timestamps use the true start time, with subsecond accuracy.

## Discontinuous recordings (EDF+D)

EDF+D files have gaps in the recording. An EDF+D gap is a time span with no records. The timestamps from the `times()` method of a signal include these gaps:

```python
f = edfarray.EdfFile("discontinuous.edf")
sig = f.signal(0)
times = sig.times()

# Find gaps by looking for large jumps in the timestamp array.
import numpy as np
dt = np.diff(times)
expected_dt = 1.0 / sig.sample_rate
gaps = np.where(dt > expected_dt * 1.5)[0]

for idx in gaps:
    print(f"Gap at {times[idx]:.3f}s -> {times[idx+1]:.3f}s "
          f"({times[idx+1] - times[idx]:.3f}s)")
```

### `read_page` and `Proxy2D` use flat sample indices

`read_page()`, `Signal` indexing, and `Proxy2D` all address samples by flat index, not by physical time. A flat index is a sample count that ignores gaps. A proxy, such as `Proxy2D`, is an object that reads file data on request. For EDF and EDF+C, this difference has no effect, because the records are contiguous. For EDF+D, `read_page(start_sec, end_sec)` converts the time parameter to the first flat sample at or after `start_sec * sample_rate`. This conversion ignores gaps.

`Proxy3D` indexes by `(record, channel, sample)`, not by a flat sample index. For EDF+D, this layout can be a better fit. Each record has exactly one onset entry in the annotations index. The sample axis addresses only the samples in one record.

For example, a file has records at t=0s and t=1s, then a gap, then a record at t=5s:

```python
# WRONG for EDF+D: assumes 0-10s maps to the first 10s of physical time.
# Actually returns samples from records 0..N by flat index, which may
# span well beyond 10s of physical time due to gaps.
pages = f.read_page(0.0, 10.0)
```

#### Time-aware reading for EDF+D

Use `use_time=True`. It converts the time range to the actual sample indices with the record onset times:

```python
# CORRECT for EDF+D: time-aware page read.
pages = f.read_page(0.0, 10.0, use_time=True)
# pages will contain only samples from records that fall within 0-10s physical time.
```

To read one signal by time, use `Signal.read_time_range()`:

```python
sig = f.signal(0)
data = sig.read_time_range(0.0, 10.0)  # physical data within 0-10s, gaps excluded
```

Flat indexing is the same behavior as `readSignal(start, n)` in pyedflib. Flat sample indices are the standard convention. To map between sample space and time space, use `times()`. It returns the true physical timestamp of every sample.

## Time-keeping annotations

In an EDF+ file, each record starts with a time-keeping annotation. A time-keeping annotation has empty text. Its onset gives the start time of the record. edfarray uses these annotations to build an internal map from records to time.

`f.annotations` does not include the time-keeping annotations. It contains only annotations with non-empty text.

## Lenient TAL parsing

TAL (time-stamped annotation list) is the byte format of annotations. In real-world files, the TAL data is sometimes malformed. edfarray uses the same method as edflib. It parses the entries that it can, skips malformed entries, and collects warnings. One bad annotation does not stop edfarray from reading the rest of the file.

```python
if f.warnings:
    for w in f.warnings:
        print(w)
```
