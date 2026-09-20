# edfarray

An EDF/EDF+ file parsing library with numpy-like Python bindings. Handles EDF, EDF+C (contiguous), and EDF+D (discontinuous) recordings.

**Read the [documentation](https://bleemesser.github.io/edfarray/)**

## Install

```bash
pip install edfarray
```

## Quick example

```python
import edfarray

with edfarray.EdfFile("recording.edf") as f:
    print(f.variant) # "EDF", "EDF+C", or "EDF+D"
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

See the [docs](https://bleemesser.github.io/edfarray/) for the full guide on signals, annotations, epoch extraction, EDF+D time gaps, and performance.

## Examples

```bash
uv run examples/basic_usage.py
uv run examples/epoch_extraction.py
uv run examples/async_epoch_extraction.py
uv run examples/benchmark.py
uv run examples/benchmark_paging.py
```

The benchmark scripts compare against pyedflib. pyedflib is not installed by default.
To get the comparison columns, install the `bench` group:

```bash
uv sync --group bench
```

pyedflib builds from source, so this step needs a C compiler and the Python
development headers (`python3-devel` on Fedora, `python3-dev` on Debian).

## Benchmarks

Compared against [pyedflib](https://github.com/holgern/pyedflib) (Python bindings for edflib C library).

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

EEG viewer paging (all channels, 10s window):

```text
File                      Operation              edfarray     pyedflib    speedup
---------------------------------------------------------------------------------
test_generator.edf        Page forward (median)     57 us      1.15 ms      20x
(16 channels)              Random seek (median)     50 us      1.14 ms      23x

S001R01.edf               Page forward (median)    114 us      5.30 ms      47x
(64 channels)              Random seek (median)     81 us      5.34 ms      66x
```

Large files, where the data does not fit in the page cache. One full channel read from a 4.0 GiB, 64-channel recording:

```text
Cache state    edfarray   pyedflib    speedup
---------------------------------------------
Warm             113 ms    1021 ms        9.1x
Cold             659 ms    5624 ms        8.5x
```

EDF interleaves channels within each record, so reading one channel touches every record in the file. edfarray checks how much of the range is already resident and switches to a bounded sequential read when it is not, which is what keeps the cold number close to the warm one. Forcing the memory map (`strategy="mmap"`) instead takes 3959 ms and causes 131,073 major page faults.

^ `examples/benchmark.py`, `examples/benchmark_paging.py`, `scripts/bench_large.py`
(M3 Pro MacBook Pro, release build)

## Building from source

Prerequisites: [Rust toolchain](https://rustup.rs/), Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`uv sync` creates `.venv`, compiles the Rust extension through maturin, and installs
`edfarray` into that environment. Run it once before anything else. After you change
Rust code, rebuild the extension in place:

```bash
uv run maturin develop
```

To regenerate the `.pyi` stubs:

```bash
cargo run --bin gen_stubs --no-default-features --package edfarray
```

`--no-default-features` turns off `pyo3/extension-module`, so this binary links against
libpython. If the link step fails with `unable to find library -lpython3.x`, your system
Python has no shared library to link against. Install the Python development package, or
point the build at an interpreter that ships one, such as a uv-managed Python:

```bash
PYO3_PYTHON="$(uv python find 3.13)" cargo run --bin gen_stubs --no-default-features --package edfarray
```

## Running tests

Run `uv sync` first. The Python tests import the built extension, and they fail to
collect if the package is not installed in `.venv`.

```bash
uv run pytest
cargo test --package edfarray-core
cargo clippy --workspace --all-targets -- -D warnings
```

`tests/test_differential.py` compares edfarray against pyedflib, so it skips itself
unless the `bench` group is installed. CI installs that group. To run the full suite
locally, use `uv sync --group bench` as described under Examples.

## Releasing

Bump the version in `crates/edfarray-core/Cargo.toml` and `crates/edfarray-python/Cargo.toml`, regenerate the stubs, then:

```bash
cargo run --bin gen_stubs --no-default-features --package edfarray
git add crates/edfarray-core/Cargo.toml crates/edfarray-python/Cargo.toml Cargo.lock edfarray/_core/__init__.pyi
git commit -m "bump version to x.y.z"
git tag vx.y.z
git push && git push origin vx.y.z
```

Pushing the tag runs the release job in `.github/workflows/ci.yml`, which builds wheels for
CPython 3.12-3.14 on Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows, plus an
sdist, then publishes to PyPI via trusted publishing from the `pypi` environment.

## License

[MIT](LICENSE)
