# Writing EDF/BDF files

`edfarray` can write EDF, EDF+C, EDF+D, BDF, BDF+C, and BDF+D files. Two entry
points cover the common cases:

- `edfarray.write_edf(...)` — one-shot. Pass all signal data and annotations at
  once. Best for transcoding existing files or saving in-memory data.
- `edfarray.EdfWriter(...)` — streaming. Open a file, push records as they
  arrive, queue annotations between records, then `finish()` (or use a `with`
  block). Best for live recording or files too large to hold in memory.

The writer auto-creates the `EDF Annotations` channel for `+C`/`+D` variants. Do not include it in your signal list.

## One-shot: `write_edf`

```python
import datetime
import numpy as np
import edfarray

sig = edfarray.WriterSignal(
    label="EEG Fpz",
    physical_dimension="uV",
    physical_min=-3200.0,
    physical_max=3200.0,
    digital_min=-32768,
    digital_max=32767,
    samples_per_record=256,
)

# 4 records of 256 samples each
data = np.linspace(-100.0, 100.0, 256 * 4, dtype=np.float64)

edfarray.write_edf(
    "out.edf",
    variant="EDF+C",
    record_duration=1.0,
    signals=[sig],
    data=[data],
    annotations=[
        edfarray.Annotation(onset=0.5, text="start"),
        edfarray.Annotation(onset=2.0, text="event", duration=1.0),
    ],
    start_datetime=datetime.datetime(2026, 5, 9, 12, 0, 0),
)
```

`data[i]` must have length `num_records * signals[i].samples_per_record`, with
the same `num_records` across signals. Annotations are placed in the record
whose time window contains their onset.

## Streaming: `EdfWriter`

```python
with edfarray.EdfWriter(
    "live.edf",
    variant="EDF+C",
    record_duration=1.0,
    signals=[sig],
) as w:
    for record in produce_records():           # your data source
        w.write_record([record])               # one ndarray per signal
        if had_event_in_window:
            w.add_annotation(
                edfarray.Annotation(onset=now, text="event")
            )
```

The header is written with `num_records = -1` initially and patched on
`finish()` (called automatically by `__exit__`). Calling `add_annotation`
queues an annotation; it's emitted with the next `write_record` call. To embed
specific annotations in a specific record, pass them as the second arg of
`write_record`:

```python
w.write_record([record], [edfarray.Annotation(onset=t, text="cue")])
```

## Round-tripping an existing file

`EdfFile.write_to(path)` re-emits the current file's signals and annotations.
Useful for transcoding:

```python
f = edfarray.EdfFile("input.edfd")
f.write_to("output.edf", variant="EDF+C")  # collapse a discontinuous file
```

Only ordinary signals are copied; the annotation channel is rebuilt from the
parsed annotations rather than copied verbatim. If `variant` is omitted the
source variant is kept.

!!! warning "Transcoding caveats"
    Records are re-emitted contiguously, so transcoding changes more than the
    header tag:

    - **EDF+D -> EDF+D** preserves the source record onsets, so gaps survive the
      copy. **EDF+D -> any non-`+D` variant** flattens timing: the per-record
      onsets/gaps are replaced by uniform `record_idx * record_duration` timing.
    - **Any `+` variant -> a plain (non-`+`) variant** drops all annotations,
      because plain EDF/BDF has no annotation channel.
    - **Downconverting sample size** (e.g. BDF 24-bit -> EDF 16-bit) clamps the
      digital range and re-encodes from physical values, losing precision.

## Annotation channel sizing

For `+C`/`+D` variants the writer reserves a fixed byte budget per record for the annotation channel. The default is 120 bytes, enough for the time-keeping TAL plus a few short annotations. If a record's annotations don't fit, the writer returns an error rather than silently dropping data. Increase the budget via `annotation_bytes_per_record`:

```python
edfarray.write_edf(
    "verbose.edf",
    variant="EDF+C",
    record_duration=1.0,
    signals=[sig],
    data=[data],
    annotations=many_annotations,
    annotation_bytes_per_record=512,
)
```

## BDF (24-bit) writing

BDF and BDF+ use 24-bit signed samples. Set the digital range up to ±2²³:

```python
ecg = edfarray.WriterSignal(
    label="ECG",
    physical_dimension="mV",
    physical_min=-5.0,
    physical_max=5.0,
    digital_min=-(1 << 23),
    digital_max=(1 << 23) - 1,
    samples_per_record=128,
)
edfarray.write_edf(
    "out.bdf",
    variant="BDF",
    record_duration=1.0,
    signals=[ecg],
    data=[ecg_data],
)
```

## Validation

The writer rejects:

- Empty signal lists
- `digital_min >= digital_max` or `physical_min == physical_max`
- Digital ranges outside the format's signed-integer width
- User signals labelled `EDF Annotations` (the writer adds the channel)
- Annotations on plain `EDF`/`BDF` (use a `+C`/`+D` variant)
- Mismatched data lengths or `samples_per_record` not dividing the data
- Annotation channel overflow (increase `annotation_bytes_per_record`)
- Non-finite physical values: `NaN`, `+inf`, or `-inf` raise `InvalidArgumentError`,
  naming the signal and sample position. Clinical data is never silently coerced.

Finite values outside the signal's physical range are not rejected: they clamp to the
digital extremes (`digital_min` / `digital_max`) on write, exactly as the EDF format
requires. So out-of-range input is clamped, while `NaN`/`Inf` is refused -- the former
is a legal saturation, the latter carries no value to saturate to.

## Editing existing files

edfarray writes files; it does not edit them in place, with one narrow exception. The
only in-place edits are the identity header fields -- patient id, recording id, and start
datetime -- through [`edit_header`](anonymization.md) and [`anonymize`](anonymization.md),
plus the [`audit`](anonymization.md) leak checker. These rewrite only the fixed header
block and never touch data records.

Everything else -- annotations, channel labels, other header fields, and sample data --
requires a full rewrite via `write_to` or `write_edf`. There is no API to mutate a
single record or annotation in an existing file.

