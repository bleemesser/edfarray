# Working with Signals

## Signal objects

A `Signal` is a proxy view of a single channel in the file. It doesn't hold any data in memory. Samples are decoded from the memory-mapped file on each access.

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

Single sample access returns a Python float:

```python
sig[0]     # first sample
sig[-1]    # last sample
sig[1000]  # sample at index 1000
```

## Slicing

Slice access returns a numpy float64 array:

```python
sig[0:1000]     # first 1000 samples
sig[5000:6000]  # samples 5000-5999
sig[-1000:]     # last 1000 samples
```

Strided slicing works for quick downsampling:

```python
sig[::4]        # every 4th sample (4x downsample)
sig[0:10000:10] # first 10000 samples, 10x downsampled
```

## Physical vs digital values

EDF files store samples as 16-bit integers (digital values). The physical value is computed using a linear transform: `physical = gain * digital + offset`, where `gain = (physical_max - physical_min) / (digital_max - digital_min)`.

By default, all access returns physical values in the signal's physical units. To get the raw digital values:

```python
physical = sig.to_numpy()    # float64, in physical units (e.g. microvolts)
digital = sig.to_digital()   # int16, raw digital values from the file
```

`to_digital()` skips the gain/offset conversion, which is slightly faster for applications that do their own scaling.

## Timestamps

`times()` returns the timestamp in seconds from recording start for every sample:

```python
times = sig.times()
print(f"Recording spans {times[0]:.3f}s to {times[-1]:.3f}s")
```

For EDF+D (discontinuous) files, timestamps account for gaps between data records. See [Annotations & Time](annotations.md) for details.

## Mixed sample rates

EDF files can have different sample rates per signal. For example, EEG at 256 Hz and respiration at 1 Hz:

```python
eeg = f.signal("EEG Fpz-Cz")     # 256 Hz, 76800 samples
resp = f.signal("Resp oro-nasal")  # 1 Hz, 300 samples
```

When using `read_page()` for bulk access, each channel's array has a different length based on its sample rate:

```python
pages = f.read_page(0.0, 10.0)
# pages[0].shape == (2560,)  for a 256 Hz channel
# pages[5].shape == (10,)    for a 1 Hz channel
```

## Signal groups

A `SignalGroup` is a set of channels classified by sample rate. Within a single
EDF file every group is either *rectangular* or *open*. Rectangular groups share
a sample rate and total sample count. Every group returned by `signal_groups()`
is rectangular. Open groups have mixed rates and can only be produced by passing
mixed-rate indices to `signal_group(indices)`.

```python
groups = f.signal_groups()
for g in groups:
    print(g.kind, g.sample_rate, len(g), "covers_all=" + str(g.covers_all_ordinary))
# Build a group from arbitrary indices:
custom = f.signal_group([0, 2, 5])
```

Useful fields: `kind` (`"rectangular"` / `"open"`), `sample_rate`,
`samples_per_record`, `min_samples`, `max_samples`, `covers_all_ordinary`,
`is_singleton`, `is_rectangular`, `indices`, `__len__`.

## 2D proxy

`Proxy2D` gives numpy-style 2D indexing across signals and samples. It accepts
any group, including `Open` groups. For `Open` groups, a `PadMode` decides what
reads past a short channel's end return.

```python
group = max(f.signal_groups(), key=len)
proxy = f.proxy_2d(group)              # default pad_mode="raise"
proxy.shape                             # (num_signals, max_samples)
proxy.sample_rate                       # common rate, or None for Open groups
proxy.valid_lengths                     # per-channel valid sample counts

proxy[3, 1000]                          # single float
proxy[3, 1000:2000]                     # 1D ndarray
proxy[:, 1000:2000]                     # 2D ndarray (all signals × 1000 samples)
proxy[[0, 3, 7], 0:500]                 # fancy indexing on the signal axis
```

For `Open` groups, supply a `pad_mode`:

```python
all_ch = f.signal_group(f.ordinary_signal_indices())   # may be Open
proxy = f.proxy_2d(all_ch, pad_mode="nan")
# "raise" | "nan" | "zero" | "edge" | a numeric scalar (= Value)
```

`pad_mode` is applied in the domain of the read. Physical reads see the literal
`f64` fill. Digital reads see the truncated `i32`. `"nan"` is rejected for
digital reads.

## 3D proxy

For rectangular groups, `Proxy3D` exposes the record-major layout
`(num_records, num_channels, samples_per_record)`. This is convenient for
epoch-based ML pipelines where records align with batch units.

```python
proxy = f.proxy_3d(group)              # group.kind must be "rectangular"
proxy.shape                             # (num_records, num_channels, spr)
proxy[0, :, :]                          # one record, shape (n_chan, spr)
proxy[0:30, :, :]                       # first 30 records, shape (30, n_chan, spr)
proxy[5, 2, 100]                        # scalar
```

`Proxy3D` is rejected for `Open` groups. Use `Proxy2D` with a pad mode instead,
or pick a single-rate group from `signal_groups()`.

Both proxies read on demand from the memory-mapped file and hold no sample
data. Multi-signal reads are parallelized with rayon.

## Annotation signals

EDF+ files include one or more "EDF Annotations" signals that carry timing and event data. These are included in `num_signals` and can be accessed like any signal, but their "samples" are raw annotation bytes, not meaningful as signal data.

Use `ordinary_signal_indices()` to get only the data signals:

```python
f.num_signals                    # 104 (including annotation channel)
indices = f.ordinary_signal_indices()  # [0, 1, ..., 102]

# Access annotations through the dedicated API.
f.annotations  # list[Annotation]
```
