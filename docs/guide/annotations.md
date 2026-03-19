# Annotations & Time

## Annotations

EDF+ files can contain annotations: timestamped text events embedded in the recording. Plain EDF files do not have annotations.

```python
f = edfarray.EdfFile("recording.edf")

for ann in f.annotations:
    print(f"{ann.onset:.2f}s: {ann.text}")
    if ann.duration is not None:
        print(f"  duration: {ann.duration}s")
```

Each annotation has:

- `onset` -- time in seconds from the start of the recording
- `duration` -- duration in seconds, or `None` if not specified
- `text` -- the annotation text (UTF-8)

Annotations are sorted by onset time.

## EDF+ variants

The `variant` property tells you which format the file uses:

- `"EDF"` -- plain EDF. No annotations, no subsecond precision.
- `"EDF+C"` -- EDF+ contiguous. Data records follow each other without gaps.
- `"EDF+D"` -- EDF+ discontinuous. Data records may have time gaps between them.

```python
f.variant  # "EDF", "EDF+C", or "EDF+D"
```

## Subsecond start time

The EDF header stores the recording start time with one-second resolution (`hh.mm.ss`). EDF+ files encode subsecond precision in the first time-keeping annotation of the first data record.

edfarray extracts this automatically. All annotation onsets and sample timestamps reflect the true subsecond-accurate start time.

## Discontinuous recordings (EDF+D)

EDF+D files have gaps in the recording. For example, a sleep study might pause during a bathroom break.

The `times()` method on a signal accounts for these gaps:

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

### `read_page` and `ArrayProxy` use flat sample indices

`read_page()`, `Signal` indexing, and `ArrayProxy` all address samples by flat index, not physical time. For EDF and EDF+C this distinction doesn't matter because records are contiguous. For EDF+D, it means the time parameter in `read_page(start_sec, end_sec)` is converted to a sample offset as `int(start_sec * sample_rate)` — it does not account for gaps.

For example, if a file has records at t=0s, t=1s, then a gap, then t=5s:

```python
# WRONG for EDF+D: assumes 0-10s maps to the first 10s of physical time.
# Actually returns samples from records 0..N by flat index, which may
# span well beyond 10s of physical time due to gaps.
pages = f.read_page(0.0, 10.0)
```

The correct way to get data within a physical time window for EDF+D is to use `times()` to identify which samples fall in your range:

```python
# CORRECT for EDF+D: use timestamps to select the right samples.
sig = f.signal(0)
all_data = sig.to_numpy()
all_times = sig.times()

t_start, t_end = 0.0, 10.0
mask = (all_times >= t_start) & (all_times < t_end)
data_in_window = all_data[mask]
times_in_window = all_times[mask]

# data_in_window will have fewer samples than (t_end - t_start) * sample_rate
# because the gaps contain no recorded data.
```

This is the same behavior as pyedflib's `readSignal(start, n)` — flat sample indices are the standard convention. The `times()` method is what makes EDF+D usable: it gives you the true physical timestamp for every sample so you can map between sample space and time space yourself.

## Time-keeping annotations

Each data record in an EDF+ file begins with a time-keeping annotation: an empty-text annotation whose onset indicates the record's start time. edfarray uses these to build a record-to-time map internally.

These time-keeping annotations are not included in `f.annotations`. Only annotations with non-empty text appear there.

## Lenient TAL parsing

The annotation byte format (TAL, Time-stamped Annotation List) is sometimes malformed in real-world files. edfarray follows edflib's approach: parse what it can, skip malformed entries, and collect warnings. A single bad annotation does not prevent the rest of the file from being read.

```python
if f.warnings:
    for w in f.warnings:
        print(w)
```
