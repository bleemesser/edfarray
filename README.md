# edfarray

A Rust-backed EDF/EDF+ file parser with numpy-like Python bindings. Handles EDF, EDF+C (contiguous), and EDF+D (discontinuous) recordings.

**Read the [documentation](https://bleemesser.github.io/edfarray/)**

## Install

```bash
pip install edfarray
```

## Quick example

```python
import edfarray

with edfarray.EdfFile("recording.edf") as f:
    print(f.variant)       # "EDF", "EDF+C", or "EDF+D"
    print(f.num_signals)   # number of signals (including annotation channels)
    print(f.duration)      # total duration in seconds

    # Access a signal by index or label
    sig = f.signal("EEG Fpz-Cz")
    print(sig.sample_rate) # e.g. 256.0
    print(len(sig))        # total number of samples

    # Numpy-style indexing
    first_sample = sig[0]         # single float
    chunk = sig[1000:2000]        # numpy float64 array
    downsampled = sig[::4]        # strided access

    # Bulk access -- all channels for a time window, parallelized with rayon
    pages = f.read_page(0.0, 10.0)

    # Annotations (EDF+ only)
    for ann in f.annotations:
        print(f"{ann.onset:.2f}s: {ann.text}")
```

See the [docs](https://bleemesser.github.io/edfarray/) for the full guide on signals, annotations, EDF+D time gaps, and performance.

## Examples

```bash
uv run examples/basic_usage.py
uv run examples/benchmark.py
uv run examples/benchmark_paging.py
```

## Building from source

Prerequisites: [Rust toolchain](https://rustup.rs/), Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run maturin develop
cargo run --bin gen_stubs --no-default-features --package edfarray  # regenerate .pyi stubs
```

## Running tests

```bash
uv run pytest
cargo test --package edfarray-core
```

## Releasing

Bump the version in `crates/edfarray-core/Cargo.toml` and `crates/edfarray-python/Cargo.toml`, make sure stubs are up to date, then:

```bash
git add -A && git commit -m "bump version to 0.x.y"
git tag v0.x.y
git push && git push origin v0.x.y
```

## License

[MIT](LICENSE)
