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

## Annotation signals

EDF+ files include one or more "EDF Annotations" signals that carry timing and event data. These are included in `num_signals` and can be accessed like any signal, but their "samples" are raw annotation bytes, not meaningful as signal data.

Use `ordinary_signal_indices()` to get only the data signals:

```python
f.num_signals                    # 104 (including annotation channel)
indices = f.ordinary_signal_indices()  # [0, 1, ..., 102]

# Access annotations through the dedicated API.
f.annotations  # list[Annotation]
```
