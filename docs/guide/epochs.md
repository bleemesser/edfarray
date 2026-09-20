# Epoch Extraction

An epoch is a short, fixed window of samples cut around an event time. Given an
onset, `pre` and `post` seconds define the window `[onset - pre, onset + post)`.
Stacking many windows gives you a dense `(n_epochs, n_channels, n_samples)` array
you can feed straight into numpy.

`extract_epochs` decodes every window in parallel, in one call. It reads the
samples it needs and nothing else, and it respects EDF+D recording gaps.

## Extract around events

```python
import edfarray

f = edfarray.EdfFile("recording.edf")

# Events are plain seconds, Annotation objects, or a mix of both.
events = [10.0, 20.0, 30.0]
epochs = f.extract_epochs(events, pre=0.5, post=1.0)

epochs.data.shape  # (3, n_channels, 240)
epochs.onsets      # array([10., 20., 30.])
```

The sample axis is always `ceil((pre + post) * sample_rate)` columns wide. That width
comes from your request alone, never from where the events landed, so shapes match
across files. Column 0 of a row is the first sample at or after `onset - pre`, and column
`j` is the sample `j` places later. An event that falls between two samples therefore
shifts its row by less than one sample period.

`epochs.data` is float64 physical samples. `np.asarray(epochs)` returns the same
array. Other metadata comes off the object:

- `labels`: channel labels, in group order.
- `sample_rate`: the common rate in Hz.
- `valid`: per-epoch boolean, `True` only when every column of the row is a real sample.
  It is `False` where the window ran off the file or touched an EDF+D gap.
- `dropped`: indices into your `events` list for epochs that `pad="drop"` removed.

### Choosing channels

Extraction needs a rectangular channel group: same rate, decoded together. Pass
`group=` a `SignalGroup` or a list of signal indices. Left unset, it uses the
largest rectangular group in the file.

```python
group = max(f.signal_groups(), key=len)
epochs = f.extract_epochs(events, pre=0.5, post=1.0, group=group)
```

### Extracting around annotation text

Pass `events=None` with `query=` to lock onto annotations by text, using the same
matching rules as `filter_annotations`. `events` and `query` are mutually exclusive.

```python
spindles = f.extract_epochs(None, query="Spindle", pre=1.0, post=1.0)
```

`f.events(query)` is a shortcut that returns the matching `Annotation` list, so you
can inspect it before extracting.

## What happens at the edges

A window can run off the start or end of the file, or straddle an EDF+D gap where
no data exists. The `pad=` argument decides what you get:

- `"drop"` (default): leave the epoch out entirely. It appears in `dropped`.
- `"nan"`: keep it, fill missing samples with NaN, mark it `valid=False`.
- `"zero"`: keep it, fill with 0.0.
- a number, for example `pad=-100.0`: fill with that value.
- `"edge"`: fill with the nearest real sample. A window with no real samples holds the
  last sample before it, or the first sample of the file when it starts before time 0.
- `"raise"`: raise `OutOfRangeError` on the first offending epoch.

```python
epochs = f.extract_epochs(events, pre=2.0, post=2.0, pad="nan")
clean = epochs.data[epochs.valid]  # keep only fully real epochs
```

NaN fill pairs well with `np.nanmean` and friends, so a partly-padded epoch still
averages over its real samples. A window that lands entirely inside a gap has no
real samples at all, and its feature comes out NaN.

## Planning without reading

`epoch_windows` runs the same planning as `extract_epochs` but reads no samples.
It returns `([(onset, s_start, s_end)], valid)`, where `s_start` and `s_end` are
flat sample indices into the group.

Use it to see what a batch of events would pull, and to drop bad epochs before you
pay for the decode.

```python
windows, valid = f.epoch_windows(events, pre=0.5, post=1.0)
good = [ev for ev, ok in zip(events, valid) if ok]
epochs = f.extract_epochs(good, pre=0.5, post=1.0)
```

## Gap awareness in EDF+D

For a discontinuous recording, `s_start` and `s_end` address the flat sample space
that `Signal` indexing uses. A window that straddles a gap is marked `valid=False`
under fill policies and dropped under `"drop"`, so you never silently treat samples
from opposite sides of a gap as contiguous.

Under a fill policy the row keeps its real samples at their true time offsets and
pads the columns that land in the gap, so the two sides never close up against each
other. A window that opens inside a gap pads its leading columns the same way.

See [Annotations & Time](annotations.md) for how record onsets and flat sample
indices relate in EDF+D.

## Async

The async API mirrors both methods. They offload the decode to a blocking task, so
your event loop stays free.

```python
import edfarray.aio as aio

async with await aio.open("recording.edf") as f:
    await f.wait_for_annotations()
    epochs = await f.extract_epochs(None, query="Stim", pre=0.5, post=1.0)
```

Run the bundled examples to see end-to-end use:

- `examples/epoch_extraction.py` shows planning, drop, and NaN fill on a gap-ridden
  EDF+D file.
- `examples/async_epoch_extraction.py` computes RMS and alpha-band power over grid
  and event-locked epochs.
