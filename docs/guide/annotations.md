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

The `read_page()` bulk access method also works correctly with discontinuous files. The returned data corresponds to the samples that fall within the requested time window, with gaps reflected in the timestamps.

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
