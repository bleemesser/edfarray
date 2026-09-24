# Working with Signals

## Signal objects

A `Signal` is a proxy for one signal in the file. A proxy is an object that reads file data on request. It does not hold data in memory. On each access, edfarray decodes the samples from the memory-mapped file.

```python
sig = f.signal(0)        # by index
sig = f.signal("EEG C3") # by label
```

## Signal metadata

```python
sig.label               # "EEG C3"
sig.sample_rate         # 512.0 (Hz)
sig.samples_per_record  # 256
sig.physical_dimension  # "uV"
sig.transducer          # "AgAgCl electrode"
sig.prefiltering        # "HP:0.1Hz LP:75Hz"
sig.physical_min        # -3200.0
sig.physical_max        # 3200.0
sig.digital_min         # -32768
sig.digital_max         # 32767
len(sig)                # total number of samples
```

## Indexing

Access to a single sample returns a Python float:

```python
sig[0]     # first sample
sig[-1]    # last sample
sig[1000]  # sample at index 1000
```

## Slicing

Access to a slice returns a numpy float64 array:

```python
sig[0:1000]     # first 1000 samples
sig[5000:6000]  # samples 5000-5999
sig[-1000:]     # last 1000 samples
```

To downsample, use a slice with a step:

```python
sig[::4]        # every 4th sample (4x downsample)
sig[0:10000:10] # first 10000 samples, 10x downsampled
```

## Physical vs digital values

EDF files store samples as 16-bit integers, and BDF files as 24-bit integers (digital values). edfarray calculates the physical value with a linear transform: `physical = gain * digital + offset`, where `gain = (physical_max - physical_min) / (digital_max - digital_min)`.

By default, all access returns physical values in the physical units of the signal. To get the raw digital values, use `to_digital()`:

```python
physical = sig.to_physical() # float64, in physical units (e.g. microvolts)
digital = sig.to_digital()   # int32, raw digital values from the file
```

`to_digital()` skips the gain/offset conversion. Thus it is slightly faster for applications that do their own scaling.

## Caching repeated reads

By default, a `Signal` decodes samples from the memory map on every access. The OS page cache (file data that the OS keeps in RAM) holds the raw bytes. But the gain/offset decode runs on every access. If you read the same regions many times, you can cache the decoded physical records. A record is a fixed-duration block of samples. Overlapping windows, seeks back and forth, and the same slice in a loop are examples of repeated reads:

```python
sig = f.signal("EEG Fpz-Cz", cache_capacity=8)  # cache 8 decoded records
```

- `cache_capacity` counts EDF data records, not samples or bytes. One cached record holds `samples_per_record` float64 values. Thus the cache uses approximately `cache_capacity * samples_per_record * 8` bytes of memory. The default value `0` disables the cache.
- If your reads go through the data one time, or only forward, keep the value at `0`. These reads decode each record one time, so the cache only adds overhead. The cache gives a benefit only for reads that go back to records that they read before.
- Set the capacity to a few records more than your largest repeated window: `ceil(window_samples / samples_per_record) + 2`. For example, you read 5-second windows many times from a 256 Hz signal with 256 samples per record. For this case, the capacity is `ceil(5*256 / 256) + 2 = 7`.
- The cache makes only physical reads faster (`to_physical()`, slicing). `to_digital()` always decodes again from the memory map and ignores the cache.
- Each `Signal` instance has its own cache. If you call `f.signal(...)` again, the new instance starts with an empty cache.

The `cache_capacity` argument works the same way in the [async API](async.md).

## Timestamps

`times()` returns the timestamp of every sample, in seconds from the start of the recording:

```python
times = sig.times()
print(f"Recording spans {times[0]:.3f}s to {times[-1]:.3f}s")
```

In EDF+D (discontinuous) files, the timestamps include the gaps between records. An EDF+D gap is a time span with no records. For more information, see [Annotations & Time](annotations.md).

## Mixed sample rates

EDF files can have a different sample rate for each signal. For example, a file can have EEG at 256 Hz and respiration at 1 Hz:

```python
eeg = f.signal("EEG Fpz-Cz")     # 256 Hz, 76800 samples
resp = f.signal("Resp oro-nasal")  # 1 Hz, 300 samples
```

If you use `read_page()` for bulk access, the length of the array for each signal depends on its sample rate:

```python
pages = f.read_page(0.0, 10.0)
# pages[0].shape == (2560,)  for a 256 Hz channel
# pages[5].shape == (10,)    for a 1 Hz channel
```

## Signal groups

A `SignalGroup` is a set of signals, classified by sample rate. In one EDF file,
every group is either rectangular or open. The signals in a rectangular group have
the same sample rate and the same total sample count. Every group that
`signal_groups()` returns is rectangular. Open groups contain signals with mixed rates. To
make an open group, pass indices with mixed rates to `signal_group(indices)`. No other method makes an open group.

```python
groups = f.signal_groups()
for g in groups:
    print(g.kind, g.sample_rate, len(g), "covers_all=" + str(g.covers_all_ordinary))
# Build a group from arbitrary indices:
custom = f.signal_group([0, 2, 5])
```

The fields of a group include `kind` (`"rectangular"` / `"open"`), `sample_rate`,
`samples_per_record`, `min_samples`, `max_samples`, `covers_all_ordinary`,
`is_singleton`, `is_rectangular`, `indices`, `__len__`.

## 2D proxy

`Proxy2D` gives numpy-style 2D indexing across signals and samples. It accepts
all groups, including open groups. For an open group, a `PadMode` sets the
values that a read past the end of a shorter signal returns.

```python
group = max(f.signal_groups(), key=len)
proxy = f.proxy_2d(group)              # default pad_mode="raise"
proxy.shape                             # (num_signals, max_samples)
proxy.sample_rate                       # common rate, or None for Open groups
proxy.valid_lengths                     # per-channel valid sample counts

proxy[3, 1000]                          # single float
proxy[3, 1000:2000]                     # 1D ndarray
proxy[:, 1000:2000]                     # 2D ndarray (all signals x 1000 samples)
proxy[[0, 3, 7], 0:500]                 # fancy indexing on the signal axis
proxy[:, 0:2000:4]                      # strided sample axis (downsample by 4)
proxy[:, ::-1]                          # negative step also works
```

The sample (time) axis accepts a step, so `proxy[:, ::4]` downsamples in
time. The signal axis does not accept a step. To select any set of signals, use a list.

!!! note "Striding does not reduce I/O"
    EDF stores the samples of each record contiguously. Thus a read with a
    step on the sample axis reads the full span from the memory-mapped file.
    Then it takes the subsample. So `proxy[:, ::4]` costs approximately the
    same as `proxy[:, :]`. The step makes the returned array smaller, but it
    does not decrease the work. Reads with step `1` (contiguous) use an
    unchanged fast path with no overhead.

For an open group, give a `pad_mode`:

```python
all_ch = f.signal_group(f.ordinary_signal_indices())   # may be Open
proxy = f.proxy_2d(all_ch, pad_mode="nan")
# "raise" | "nan" | "zero" | "edge" | a numeric scalar (= Value)
```

edfarray applies `pad_mode` in the domain of the read. Physical reads get the
literal `f64` fill value. Digital reads get the truncated `i32` value. edfarray
rejects `"nan"` for digital reads.

## 3D proxy

For a rectangular group, `Proxy3D` gives the record-major layout
`(num_records, num_channels, samples_per_record)`. This layout is useful for ML
pipelines that use epochs, where one record is one batch unit. An epoch is a
fixed window of samples.

```python
proxy = f.proxy_3d(group)              # group.kind must be "rectangular"
proxy.shape                             # (num_records, num_channels, spr)
proxy[0, :, :]                          # one record, shape (n_chan, spr)
proxy[0:30, :, :]                       # first 30 records, shape (30, n_chan, spr)
proxy[5, 2, 100]                        # scalar
proxy[0:30, :, ::4]                     # strided sample axis (downsample by 4)
```

As with `Proxy2D`, the sample axis accepts a step. The record axis and the
channel axis require step `1`. The read always loads the full block of records
into memory, so a step on the sample axis selects samples in memory. It never
reads more than the slice without a step.

edfarray rejects `Proxy3D` for open groups. For an open group, use `Proxy2D`
with a pad mode. Or select a group with one sample rate from `signal_groups()`.

Both proxies read from the memory-mapped file on request and hold no sample
data. Reads of more than one signal run in parallel with rayon.

## Annotation signals

EDF+ files contain one or more annotation channels. An annotation channel is an "EDF Annotations" signal that holds timing and event data. `num_signals` includes the annotation channels, and you can access them like other signals. But their "samples" are raw annotation bytes, and these bytes have no meaning as signal data.

An ordinary signal is a signal that is not the annotation channel. To get only the ordinary signals, use `ordinary_signal_indices()`:

```python
f.num_signals                    # 104 (including annotation channel)
indices = f.ordinary_signal_indices()  # [0, 1, ..., 102]

# Access annotations through the dedicated API.
f.annotations  # list[Annotation]
```
