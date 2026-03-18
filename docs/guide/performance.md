# Performance

## Bulk reads with `read_page()`

The recommended way to load multi-channel data is `read_page()`. It reads all requested signals for a time window in a single call, parallelizing the decode across CPU cores using rayon.

```python
f = edfarray.EdfFile("recording.edf")

# Read all ordinary signals for a 10-second window.
pages = f.read_page(0.0, 10.0)

# Or specify which signals you want.
pages = f.read_page(0.0, 10.0, signal_indices=[0, 1, 5])
```

`read_page()` returns a list of numpy float64 arrays, one per signal. Signals with different sample rates produce arrays of different lengths.

There's also `read_page_digital()` which returns int16 arrays without the gain/offset conversion.

## `ordinary_signal_indices()`

EDF+ files include annotation channels alongside data channels. `ordinary_signal_indices()` gives you just the data channel indices:

```python
indices = f.ordinary_signal_indices()
pages = f.read_page(0.0, 10.0, signal_indices=indices)
```

When you call `read_page()` without specifying `signal_indices`, it defaults to `ordinary_signal_indices()`.

## Architecture

edfarray uses memory-mapped I/O via `memmap2`. The file is mapped into the process's address space on open, and the OS page cache handles bringing data in and out of physical RAM. This means:

- Opening large files is near-instant. The only upfront work is parsing the header and scanning for annotations.
- Sequential reads (paging forward) benefit from OS readahead.
- Random seeks (jumping to a timestamp) only fault in the pages you touch.
- Multiple signals reading from the same data records share cached pages.

On open, edfarray does one sequential scan of the file to parse the header and build the annotation index. After that, it switches to random-access mode and uses `madvise(MADV_WILLNEED)` hints before bulk reads.

## Why it's fast

Three things contribute to the performance on multi-channel page reads:

1. Rayon parallelism. Each signal's decode runs on a separate thread. With 100+ channels, this scales well across cores.

2. SIMD-friendly decode. The i16-to-f64 conversion is split into a widening pass and a multiply-add pass, which the compiler autovectorizes.

3. Zero-copy from mmap. Signal bytes are read directly from the memory-mapped file into the output buffer. No intermediate copies.

## Single-signal access

For single-signal reads, the `Signal` proxy object is already efficient. Each access resolves the global sample index to a record offset and decodes directly from the mmap:

```python
sig = f.signal(0)
chunk = sig[10000:20000]  # decoded directly from mmap
```

This path is single-threaded and doesn't benefit from rayon, but it avoids all unnecessary allocation and copying. For reading one channel at a time, there's no overhead beyond the decode itself.

## Tips for EEG viewer applications

If you're building an application that pages through a recording:

```python
f = edfarray.EdfFile("recording.edf")
page_duration = 10.0  # seconds
current_time = 0.0

# Page forward.
def next_page():
    global current_time
    pages = f.read_page(current_time, current_time + page_duration)
    current_time += page_duration
    return pages

# Page backward.
def prev_page():
    global current_time
    current_time = max(0, current_time - page_duration)
    pages = f.read_page(current_time, current_time + page_duration)
    return pages
```

Each `read_page()` call takes well under 1 ms for typical EEG recordings (30-100 channels). This is fast enough to call on every frame without any prefetching or caching layer on the Python side.
