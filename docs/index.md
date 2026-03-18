# edfarray

An EDF/EDF+ parser written in Rust with numpy-like Python bindings.

Handles EDF, EDF+C, and EDF+D files of any size without loading them into memory. Files are memory-mapped, signals are decoded on the fly, and multi-channel reads are parallelized across cores.

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
- Multi-channel page read acceleration.
- Full EDF+D support, including discontinuous recordings with time gaps.
- Lenient parsing. Malformed annotations and anonymized dates produce warnings, not errors.
- Subsecond precision from time-keeping annotations applied automatically.
- Full metadata access: patient info, recording info, signal properties, all as typed properties.
- Ships with `.pyi` type stubs for IDE autocompletion.

## Guide

- [Getting Started](guide/getting-started.md) -- open a file, read signals
- [Working with Signals](guide/signals.md) -- indexing, slicing, physical vs digital
- [Annotations & Time](guide/annotations.md) -- EDF+ annotations, discontinuous recordings
- [Performance](guide/performance.md) -- bulk reads, architecture

## Reference

- [Python API](reference/python-api.md)
- [Rust Crate](reference/rust-api.md)
- [EDF Format](reference/edf-format.md)
