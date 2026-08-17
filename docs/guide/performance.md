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

### Time-aware reading for EDF+D

For EDF+D files with time gaps, use `use_time=True` to resolve the time range to actual sample indices:

```python
# For EDF+D files: time-aware page read.
pages = f.read_page(0.0, 10.0, use_time=True)
# Only samples from records within 0-10s physical time are returned.

# For single-signal time-aware reading:
sig = f.signal(0)
data = sig.read_time_range(0.0, 10.0)  # physical data within 0-10s, gaps excluded
```

## `ordinary_signal_indices()`

EDF+ files include annotation channels alongside data channels. `ordinary_signal_indices()` gives you just the data channel indices:

```python
indices = f.ordinary_signal_indices()
pages = f.read_page(0.0, 10.0, signal_indices=indices)
```

When you call `read_page()` without specifying `signal_indices`, it defaults to `ordinary_signal_indices()`.

## Architecture

edfarray reads through a memory map (`memmap2`) by default. The file is mapped into the process's address space on open, and the OS page cache brings data in and out of physical RAM. This means:

- Opening large files is near-instant. The header is parsed synchronously (fixed-size, fast), and the annotation scan runs in a background thread.
- Signal reads work immediately after open, without waiting for the annotation scan.
- Sequential reads (paging forward) benefit from OS readahead.
- Random seeks (jumping to a timestamp) only fault in the pages you touch.
- Multiple signals reading from the same data records share cached pages.

On open, edfarray parses the header and record layout synchronously (microseconds), then spawns a background thread to build the annotation index by scanning the file's TAL data. `madvise` hints prime the page cache before bulk reads and mark the annotation scan as a sequential pass.

### Memmap failure case

EDF interleaves channels inside each data record, so reading one channel touches a small slice
of every record. For a 64-channel, 256 Hz file each record is 32 KB and one channel's slice is
512 B: the kernel must fault in the whole file to hand back a sixty-fourth of it. When the data
is already cached that is nearly free, which is why warm benchmarks never show it. When it is
not cached, it costs one major page fault per record.

edfarray detects this. Before a large read it samples how much of the target range is resident
(`mincore`); if the range is big and mostly not cached, it reads sequentially through a bounded
buffer instead of faulting through the mapping. It also re-checks periodically during a long
read, so a read that starts warm and loses its pages to memory pressure switches over partway
rather than paying a fault per record for the rest of the file.

Measured on a 4 GiB, 64-channel file, reading one full channel:

| Cache state | Mapping only | Automatic | pyedflib |
| --- | --- | --- | --- |
| Warm | 107 ms | 113 ms | 1021 ms |
| Cold | 3959 ms | 659 ms | 5624 ms |

Cold, the mapping-only path takes 131,073 major faults (one per record); the automatic path
takes 2. You can force either mode with `f.signal(0, strategy="mmap")` or `strategy="stream"`;
the default is `"auto"`. Streaming is slower than a warm mapping, so `"stream"` is only the
right explicit choice when you know the data will not be cached.

## Async annotation scan

For large files (e.g. a 24-hour EEG at ~12 GB), the annotation scan can take noticeable time. edfarray runs it in the background so you can start reading signal data immediately:

```python
f = edfarray.EdfFile("large_recording.edf")

# These work right away, no waiting:
sig = f.signal(0)
data = sig[0:10000]
pages = f.read_page(0.0, 10.0)

# Check scan status without blocking:
f.annotations_ready   # True/False
f.scan_progress       # (records_scanned, total_records)

# These block until the scan finishes:
f.annotations         # waits, then returns the annotation list
f.warnings            # waits, then returns warnings including annotation parse issues
```

For plain EDF files (no annotation signals), there is no scan at all -- record onsets are computed directly from the header. For EDF+C files, signal reads never block because record onsets are uniform. Only EDF+D files need the scan results for correct time mapping via `sample_time()` / `times()`.

The scan reads every data record, so on a very large file it competes with your own reads for
page cache. Pass `scan_annotations=False` to skip it at open; the index is then built on first
annotation access, on the calling thread:

```python
f = edfarray.EdfFile("large_recording.edf", scan_annotations=False)
data = f.signal(0).to_physical() # no scan happens
f.annotations # builds the index now, blocking until done
```

## Why it's fast

Three things contribute to the performance on multi-channel page reads:

1. Rayon parallelism. Each signal's decode runs on a separate thread. With 100+ channels, this scales well across cores.

2. SIMD-friendly decode. The i16-to-f64 conversion is split into a widening pass and a multiply-add pass, which the compiler autovectorizes.

3. No intermediate copies. Signal bytes are decoded from the mapping straight into the output
   buffer. (The streaming path described above copies once into its read buffer, which is the
   price of bounded memory use.)

Reads release the GIL, so decoding scales across Python threads as well as rayon workers.

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
