# edfarray

edfarray is a library that parses EDF/EDF+ and BDF/BDF+ files. It has Python bindings with a numpy-like interface. It reads EDF, EDF+C (contiguous), and EDF+D (discontinuous) recordings, and the matching 24-bit BDF variants.

Read the [documentation](https://bleemesser.github.io/edfarray/).

## Install

```bash
pip install edfarray
```

## Quick example

```python
import edfarray

with edfarray.EdfFile("recording.edf") as f:
    print(f.variant) # "EDF", "EDF+C", "EDF+D", "BDF", "BDF+C", or "BDF+D"
    print(f.num_signals) # number of signals (including annotation channels)
    print(f.duration) # total duration in seconds

    # Access a signal by index or label
    sig = f.signal("EEG Fpz-Cz")
    print(sig.sample_rate)
    print(len(sig))

    # Numpy-style indexing
    first_sample = sig[0] # single float
    chunk = sig[1000:2000] # numpy float64 array
    downsampled = sig[::4] # strided access

    # Bulk access to all channels for a time window, parallelized with rayon
    pages = f.read_page(0.0, 10.0)

    # 2D / 3D proxies for multi-channel numpy access. Build from a SignalGroup,
    # which is discovered via signal_groups() (partitions by sample rate).
    group = max(f.signal_groups(), key=len) # the largest same-rate group
    p2 = f.proxy_2d(group) # shape: (n_signals, n_samples)
    data = p2[:, 0:10000] # 2D numpy array

    # For uniform-rate groups, a 3D record view is natural for ML batching.
    p3 = f.proxy_3d(group) # shape: (n_records, n_chan, spr)
    epoch = p3[0:30, :, :] # first 30 records, all channels

    # Annotations (EDF+ only)
    for ann in f.annotations:
        print(f"{ann.onset:.2f}s: {ann.text}")

    # Event-locked epochs: fixed windows around onsets, decoded in parallel
    # into a dense (n_epochs, n_channels, n_samples) array. Gap-aware for EDF+D.
    epochs = f.extract_epochs([10.0, 20.0, 30.0], pre=0.5, post=1.0, group=group)
    print(epochs.data.shape) # (3, 64, 240)
```

The [docs](https://bleemesser.github.io/edfarray/) contain the full guide to signals, annotations, epoch extraction, EDF+D time gaps, and performance.

## Examples

```bash
uv run examples/basic_usage.py
uv run examples/epoch_extraction.py
uv run examples/async_epoch_extraction.py
uv run examples/benchmark.py
uv run examples/benchmark_paging.py
```

The benchmark scripts compare edfarray with pyedflib. By default, pyedflib is not installed.
To get the comparison columns, install the `bench` group:

```bash
uv sync --group bench
```

pyedflib builds from source. Thus this step needs a C compiler and the Python
development headers (`python3-devel` on Fedora, `python3-dev` on Debian).

## Benchmarks

These benchmarks compare edfarray with [pyedflib](https://github.com/holgern/pyedflib), the Python bindings for the edflib C library.

Single-signal reads:

```text
File                      Operation              edfarray     pyedflib    speedup
---------------------------------------------------------------------------------
test_generator.edf        Read full signal         183 us      8.07 ms      44x
(4.0 MB, 180k samples)    Read digital              87 us      8.05 ms      93x
                           1000 random samples     383 us      5.29 ms      14x

S001R01.edf               Read full signal          11 us       450 us      39x
(1.2 MB, 64 channels)     Read digital               7 us       450 us      67x
```

EEG viewer paging (all signals, 10s window):

```text
File                      Operation              edfarray     pyedflib    speedup
---------------------------------------------------------------------------------
test_generator.edf        Page forward (median)     57 us      1.15 ms      20x
(16 channels)              Random seek (median)     50 us      1.14 ms      23x

S001R01.edf               Page forward (median)    114 us      5.30 ms      47x
(64 channels)              Random seek (median)     81 us      5.34 ms      66x
```

Large files, where the data does not fit in the page cache. The page cache is memory where the OS keeps file data. The test reads one full signal from a 4.0 GiB recording with 64 signals:

```text
Cache state    edfarray   pyedflib    speedup
---------------------------------------------
Warm             113 ms    1021 ms        9.1x
Cold             659 ms    5624 ms        8.5x
```

EDF interleaves the signals within each record. A record is one fixed-duration block of samples. Thus a read of one signal touches every record in the file. edfarray first finds how much of the range is already in the page cache. If the range is not in the page cache, edfarray switches to a bounded sequential read. This switch keeps the cold number close to the warm number.

A memory map makes the bytes of a file readable as memory. If you force the memory map (`strategy="mmap"`), the same read takes 3959 ms and causes 131,073 major page faults.

^ `examples/benchmark.py`, `examples/benchmark_paging.py`, `scripts/bench_large.py`
(M3 Pro MacBook Pro, release build)

## Building from source

Before you build, install the [Rust toolchain](https://rustup.rs/), Python 3.12+, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`uv sync` creates `.venv`, compiles the Rust extension with maturin, and installs
`edfarray` into that environment. Run it once before anything else. After you change
Rust code, rebuild the extension in place:

```bash
uv run maturin develop
```

To regenerate the `.pyi` stubs, run this command:

```bash
cargo run --bin gen_stubs --no-default-features --package edfarray
```

`--no-default-features` turns off `pyo3/extension-module`. Thus this binary links against
libpython. If the link step fails with `unable to find library -lpython3.x`, your system
Python has no shared library to link against. In that case, install the Python development
package. As an alternative, point the build at an interpreter that has a shared library,
for example a Python that uv manages:

```bash
PYO3_PYTHON="$(uv python find 3.13)" cargo run --bin gen_stubs --no-default-features --package edfarray
```

## Running tests

Run `uv sync` first. The Python tests import the built extension. If the package is not
installed in `.venv`, pytest cannot collect the tests.

```bash
uv run pytest
cargo test --package edfarray-core
cargo clippy --workspace --all-targets -- -D warnings
```

`tests/test_differential.py` compares edfarray with pyedflib. If the `bench` group is not
installed, the test skips itself. CI installs that group. To run the full suite locally,
run `uv sync --group bench`, as the Examples section describes.

## Releasing

Increase the version in `crates/edfarray-core/Cargo.toml` and `crates/edfarray-python/Cargo.toml`. Then run these commands to regenerate the stubs, commit, tag, and push:

```bash
cargo run --bin gen_stubs --no-default-features --package edfarray
git add crates/edfarray-core/Cargo.toml crates/edfarray-python/Cargo.toml Cargo.lock edfarray/_core/__init__.pyi
git commit -m "bump version to x.y.z"
git tag vx.y.z
git push && git push origin vx.y.z
```

When you push the tag, the release job in `.github/workflows/ci.yml` starts. The job builds
wheels for CPython 3.12-3.14 on Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows.
It also builds an sdist. Then it publishes to PyPI through trusted publishing from the `pypi`
environment.

## License

[MIT](LICENSE)
