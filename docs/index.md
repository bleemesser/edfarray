# edfarray

edfarray is an EDF/EDF+ parser. It is written in Rust and has Python bindings with a numpy-like interface.

It reads EDF, EDF+C, EDF+D, BDF, BDF+C, and BDF+D files of any size and does not load them into memory. edfarray opens each file as a memory map. A memory map makes the bytes of a file readable as memory. edfarray decodes a signal only at read time. It runs reads of many signals in parallel across CPU cores.

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

- edfarray reads files through a memory map. It never loads a file into RAM.
- A file opens without a wait for the annotation scan. A background thread builds the annotation index.
- A 2D proxy returns numpy arrays. A proxy is a view that reads samples on demand. For example, `proxy[0:5, 1000:2000]` gives a numpy array across multiple channels.
- A page read returns many signals for one time window. edfarray accelerates these reads.
- edfarray extracts event-locked epochs in parallel into a dense numpy array. The extraction takes EDF+D gaps into account. An EDF+D gap is a time span with no recorded data.
- edfarray fully supports EDF+D, including discontinuous recordings with time gaps.
- The parser is lenient. Malformed annotations and anonymized dates produce warnings, not errors.
- edfarray takes the subsecond part of the start time from the time-keeping annotations. It applies this part automatically.
- You can read all metadata as typed properties: patient info, recording info, and signal properties.
- edfarray can anonymize a file in place. A header edit writes kilobytes and does not rewrite the full file. The leak audit finds identity strings in labels and annotations.
- The package includes `.pyi` type stubs for IDE autocompletion.

## Guide

- [Getting Started](guide/getting-started.md): Open a file and read signals.
- [Working with Signals](guide/signals.md): Indexing, slicing, and physical versus digital values.
- [Annotations & Time](guide/annotations.md): EDF+ annotations and discontinuous recordings.
- [Epoch Extraction](guide/epochs.md): Event-locked windows into a numpy array.
- [Performance](guide/performance.md): Bulk reads and architecture.
- [Writing Files](guide/writing.md): Create EDF/BDF files from scratch.
- [Anonymization](guide/anonymization.md): Anonymize identifiers and audit for leaks.
- [Async API](guide/async.md): Non-blocking reads and the background annotation scan.

## Reference

- [Python API](reference/python-api.md)
- [Rust Crate](reference/rust-api.md)
- [EDF Format](reference/edf-format.md)
- [API contracts](reference/contracts.md)
