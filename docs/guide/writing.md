# Writing EDF/BDF files

`edfarray` can write EDF, EDF+C, EDF+D, BDF, BDF+C, and BDF+D files. A record is
a block of samples with a fixed duration. Two functions cover the common cases:

- `edfarray.write_edf(...)`: Writes the file in one call. Pass all signal data and
  annotations at the same time. Use it to transcode existing files or to save data
  that is in memory.
- `edfarray.EdfWriter(...)`: Writes the file as a stream. Open a file, write records
  when they arrive, and queue annotations between records. Then call `finish()`, or
  use a `with` block. Use it for live recording or for files too large to keep in
  memory.

For `+C`/`+D` variants, the writer creates the annotation channel (the `EDF Annotations`
signal) automatically. Do not include it in your signal list.

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
the same `num_records` for all signals. The writer puts each annotation in the
record whose time window contains the onset of the annotation.

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

The writer first writes the header with `num_records = -1`. `finish()` updates
the header later. `__exit__` calls `finish()` automatically. `add_annotation`
puts an annotation in a queue, and the next `write_record` call writes it. To
put specific annotations in a specific record, pass them as the second argument
of `write_record`:

```python
w.write_record([record], [edfarray.Annotation(onset=t, text="cue")])
```

## Round-tripping an existing file

`EdfFile.write_to(path)` writes the signals and annotations of the current file
to a new path. You can use it to transcode a file (copy it to another variant):

```python
f = edfarray.EdfFile("input.edfd")
f.write_to("output.edf", variant="EDF+C")  # collapse a discontinuous file
```

`write_to` copies only the ordinary signals. An ordinary signal is a signal that
is not the annotation channel. `write_to` builds the annotation channel again from
the parsed annotations. It does not copy the channel byte for byte. If you omit
`variant`, the new file keeps the source variant.

### Copying a subset of channels

Pass `signals` to write only the selected signals. The value can be a
`SignalGroup`, a signal index, a label, or a sequence that mixes indices and
labels. The new file contains the signals in the given order. `write_to` always
copies all annotations:

```python
f.write_to("eeg_only.edf", signals=[0, 3])           # two channels, in order
f.write_to("eeg_only.edf", signals=["EEG Fp1", 3])   # indices and labels mix
f.write_to("eeg_only.edf", signals=f.signal_group([0, 3]))
```

You cannot select the annotation channel. `write_to` always builds it again.
`write_to` rejects sets and dicts because they have no fixed order. It also
rejects an empty selection, a duplicated signal, or an out-of-range index. It
rejects these values before it writes any data.

!!! warning "Transcoding caveats"
    Transcoding writes the records with no gaps between them, except from EDF+D
    to EDF+D. An EDF+D gap is a time gap between two records. Thus transcoding
    changes more than the variant tag in the header:

    - EDF+D to EDF+D keeps the source record onsets, so the gaps stay in the copy.
      EDF+D to any non-`+D` variant makes the timing uniform. Uniform
      `record_idx * record_duration` timing replaces the onset and gap of each
      record.
    - Any `+` variant to a plain (non-`+`) variant drops all annotations. Plain
      EDF/BDF has no annotation channel.
    - A smaller sample size (for example, BDF 24-bit to EDF 16-bit) clamps the
      digital range. The writer encodes the data again from physical values, and
      precision is lost.

### Writing over an open file

If another edfarray handle has a path open, `write_to`, `write_edf`, and
`EdfWriter` reject the write and raise `EdfFileError`. A handle is an `EdfFile`,
a signal or proxy taken from one, or an `EdfWriter`. A proxy is a `Proxy2D` or
`Proxy3D` array view. The source file also counts as a handle. Thus you cannot
remove signals from a file in place:

```python
f = edfarray.EdfFile("rec.edf")
f.write_to("rec.edf", signals=[0, 3])  # raises EdfFileError
```

A `Signal` or proxy taken from a file keeps the file open after `close()`. To
replace a file, do these steps in this order:

1. Close every `EdfFile` on the file.
2. Drop every signal and proxy taken from those `EdfFile` objects.
3. Write to the path.

If you open a file while an `EdfWriter` still writes it, edfarray also raises
`EdfFileError`.

## Annotation channel sizing

For `+C`/`+D` variants, the writer reserves a fixed number of bytes per record for the annotation channel. The default is 120 bytes. This is enough for the time-keeping TAL and a few short annotations. A TAL is a time-stamped annotation list. If the annotations of a record do not fit, the writer returns an error. It does not drop data silently. To increase the budget, set `annotation_bytes_per_record`:

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

BDF and BDF+ use 24-bit signed samples. The digital range can go from -8388608 to 8388607:

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

The writer rejects these inputs:

- Empty signal lists
- `digital_min >= digital_max` or `physical_min == physical_max`
- Digital ranges outside the format's signed-integer width
- User signals labeled `EDF Annotations` (the writer adds the annotation channel)
- Annotations on plain `EDF`/`BDF` (use a `+C`/`+D` variant)
- Data lengths that do not match, or data that `samples_per_record` does not divide
- Annotations that overflow the annotation channel (increase `annotation_bytes_per_record`)
- Non-finite physical values. `NaN`, `+inf`, or `-inf` raise `InvalidArgumentError`,
  and the error names the signal and the sample position. The writer never changes
  clinical data silently.

The writer does not reject finite values outside the physical range of the signal.
On write, it clamps them to the digital limits (`digital_min` / `digital_max`), as the
EDF format requires. This clamp is a legal saturation. `NaN`/`Inf` have no value to
saturate to, so the writer rejects them.

## Editing existing files

edfarray writes files. It does not edit files in place, with one exception: the
identity header fields. These fields are the patient id, the recording id, and the
start datetime. [`edit_header`](anonymization.md) and [`anonymize`](anonymization.md)
edit these fields in place. They rewrite only the fixed header block and never touch
data records. If another edfarray handle has the file open, they raise `EdfFileError`,
as the writers do.

A dry run is a run that writes nothing. A dry run of `anonymize` only reads the file.
The [`audit`](anonymization.md) leak checker also only reads the file.

All other changes require a full rewrite with `write_to` or `write_edf`. These changes
include annotations, signal labels, other header fields, and sample data. No API
changes a single record or annotation in an existing file.

