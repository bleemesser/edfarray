# edfarray

An EDF/EDF+ parser written in Rust with numpy-like Python bindings.

Handles EDF, EDF+C, and EDF+D files of any size without loading them into memory. Files are memory-mapped, signals are lazily decoded, and multi-channel reads are parallelized across cores.

## Install

```bash
pip install edfarray
```

## First look

```python
import edfarray

with edfarray.EdfFile("recording.edf") as f:
    sig = f.signal("EEG Fpz-Cz")
    data = sig[0:10000]  # numpy float64 array
    print(f"{sig.label}: {len(sig)} samples at {sig.sample_rate} Hz")
```

## What it does

- Memory-mapped I/O. Files are never loaded into RAM.
- Async annotation scan. Files open instantly; the annotation index builds in a background thread.
- 2D array proxy. `proxy[0:5, 1000:2000]` gives a numpy array across multiple channels.
- Multi-channel page read acceleration.
- Parallel event-locked epoch extraction into a dense numpy array, gap-aware for EDF+D.
- Full EDF+D support, including discontinuous recordings with time gaps.
- Lenient parsing. Malformed annotations and anonymized dates produce warnings, not errors.
- Subsecond precision from time-keeping annotations applied automatically.
- Full metadata access: patient info, recording info, signal properties, all as typed properties.
- In-place anonymization. Header edits cost kilobytes, not a full rewrite, and the leak audit finds identity strings hiding in labels and annotations.
- Ships with `.pyi` type stubs for IDE autocompletion.

## Guide

- [Getting Started](guide/getting-started.md) -- open a file, read signals
- [Working with Signals](guide/signals.md) -- indexing, slicing, physical vs digital
- [Annotations & Time](guide/annotations.md) -- EDF+ annotations, discontinuous recordings
- [Epoch Extraction](guide/epochs.md) -- event-locked windows into a numpy array
- [Performance](guide/performance.md) -- bulk reads, architecture
- [Writing Files](guide/writing.md) -- create EDF/BDF files from scratch
- [Anonymization](guide/anonymization.md) -- scrub identifiers, audit for leaks
- [Async API](guide/async.md) -- non-blocking reads and background annotation scan

## Reference

- [Python API](reference/python-api.md)
- [Rust Crate](reference/rust-api.md)
- [EDF Format](reference/edf-format.md)
