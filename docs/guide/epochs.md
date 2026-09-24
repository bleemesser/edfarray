# Epoch Extraction

An epoch is a short, fixed window of samples around an event time. For an
onset, the `pre` and `post` seconds set the window `[onset - pre, onset + post)`.
A stack of many windows gives a dense `(n_epochs, n_channels, n_samples)` array.
You can give this array directly to numpy.

`extract_epochs` decodes all windows in parallel, in one call. It reads only the
samples that it needs. It also obeys EDF+D gaps. An EDF+D gap is a time span
with no recorded data.

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

The sample axis always has `ceil((pre + post) * sample_rate)` columns. If the product
is within 1e-6 of an integer, edfarray uses that integer. Thus `pre=0.1, post=0.2` at
200 Hz gives 60 columns, not 61. The width comes only from your request, never from the
positions of the events. So the shapes are the same across files.

Column 0 of a row is the first sample at or after `onset - pre`. Column `j` is the
sample `j` places later. If an event is between two samples, its row thus moves by
less than one sample period.

`epochs.data` contains float64 physical samples. `np.asarray(epochs)` returns the same
array. The object also has this metadata:

- `labels`: The signal labels, in group order.
- `sample_rate`: The common rate in Hz.
- `valid`: A boolean for each epoch. The value is `True` only for a row in which every column is a real sample.
  The value is `False` for a window that ran off the file or touched an EDF+D gap.
- `dropped`: The indices into your `events` list of the epochs that `pad="drop"` removed.

### Choosing channels

Extraction needs a rectangular group. The signals in a rectangular group have the
same rate, and edfarray decodes them together. Set `group=` to a `SignalGroup` or
to a list of signal indices. If you do not set `group=`, edfarray uses the largest
rectangular group in the file.

```python
group = max(f.signal_groups(), key=len)
epochs = f.extract_epochs(events, pre=0.5, post=1.0, group=group)
```

### Extracting around annotation text

To select annotations by their text, pass `events=None` with `query=`. The query
uses the same matching rules as `filter_annotations`. You can set `events` or
`query`, but not both.

```python
spindles = f.extract_epochs(None, query="Spindle", pre=1.0, post=1.0)
```

`f.events(query)` is a shortcut that returns the matching `Annotation` list. You
can use it to examine the list before the extraction.

## What happens at the edges

A window can go past the start or the end of the file. A window can also cross an
EDF+D gap, where no data exists. The `pad=` argument sets the result:

- `"drop"` (default): Leave out the epoch. It appears in `dropped`.
- `"nan"`: Keep the epoch, fill the missing samples with NaN, and set `valid=False`.
- `"zero"`: Keep the epoch and fill with 0.0.
- A number, for example `pad=-100.0`: Fill with that value.
- `"edge"`: Fill with the nearest real sample. A window with no real samples holds the
  last sample before it. If that window starts before time 0, it holds the first sample of the file.
- `"raise"`: Raise `OutOfRangeError` on the first epoch that has missing samples.

```python
epochs = f.extract_epochs(events, pre=2.0, post=2.0, pad="nan")
clean = epochs.data[epochs.valid]  # keep only fully real epochs
```

NaN fill works well with `np.nanmean` and the other NaN-aware numpy functions. With
these functions, the average of a partly padded epoch uses only its real samples. A
window that is fully inside a gap has no real samples. Its feature value is NaN.

## Planning without reading

`epoch_windows` does the same planning as `extract_epochs`, but it reads no samples.
It returns `([(onset, s_start, s_end)], valid)`. `s_start` and `s_end` are flat
sample indices into the group. A flat index is a sample count that ignores gaps.

Use it to see which samples a batch of events will read. Also use it to remove bad
epochs before the decode.

```python
windows, valid = f.epoch_windows(events, pre=0.5, post=1.0)
good = [ev for ev, ok in zip(events, valid) if ok]
epochs = f.extract_epochs(good, pre=0.5, post=1.0)
```

## Gap awareness in EDF+D

For a discontinuous recording, `s_start` and `s_end` address the flat sample space
that `Signal` indexing uses. A fill policy is a `pad=` value that fills missing samples.
If a window crosses a gap, a fill policy sets `valid=False` for it, and `"drop"`
removes it. So you never treat samples from the two sides of a gap as contiguous
without an indication.

With a fill policy, the row keeps its real samples at their true time offsets. It
pads the columns that are in the gap. Thus the two sides of the gap never move
together. If a window starts inside a gap, its first columns get padding in the same way.

For the relation between record onsets and flat sample indices in EDF+D, see
[Annotations & Time](annotations.md).

## Async

The async API has the same two methods. They move the decode to a blocking task,
so your event loop stays free.

```python
import edfarray.aio as aio

async with await aio.open("recording.edf") as f:
    await f.wait_for_annotations()
    epochs = await f.extract_epochs(None, query="Stim", pre=0.5, post=1.0)
```

To see complete examples of use, run the bundled examples:

- `examples/epoch_extraction.py`: Shows planning, drop, and NaN fill on an EDF+D file
  with many gaps.
- `examples/async_epoch_extraction.py`: Calculates RMS and alpha-band power over grid
  and event-locked epochs.
