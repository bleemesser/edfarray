# Performance

## Bulk reads with `read_page()`

To load data for many signals, use `read_page()`. It reads all requested signals for a time window in one call. It uses rayon to decode the signals in parallel on the CPU cores.

```python
f = edfarray.EdfFile("recording.edf")

# Read all ordinary signals for a 10-second window.
pages = f.read_page(0.0, 10.0)

# Or specify which signals you want.
pages = f.read_page(0.0, 10.0, signal_indices=[0, 1, 5])
```

`read_page()` returns a list of numpy float64 arrays, one per signal. Signals with different sample rates produce arrays of different lengths.

`read_page_digital()` returns int32 arrays without the gain/offset conversion.

### Time-aware reading for EDF+D

In an EDF+D file, the records can have gaps between them. A record is a fixed-duration block of samples. An EDF+D gap is a time span with no records. For these files, use `use_time=True` to convert the time range to the actual sample indices:

```python
# For EDF+D files: time-aware page read.
pages = f.read_page(0.0, 10.0, use_time=True)
# Only samples from records within 0-10s physical time are returned.

# For single-signal time-aware reading:
sig = f.signal(0)
data = sig.read_time_range(0.0, 10.0)  # physical data within 0-10s, gaps excluded
```

## `ordinary_signal_indices()`

EDF+ files contain annotation channels in addition to the ordinary signals. An ordinary signal is a signal that is not the annotation channel. `ordinary_signal_indices()` returns only the indices of the ordinary signals:

```python
indices = f.ordinary_signal_indices()
pages = f.read_page(0.0, 10.0, signal_indices=indices)
```

If you call `read_page()` without `signal_indices`, it uses `ordinary_signal_indices()`.

## Architecture

By default, edfarray reads through a memory map (`memmap2`). When you open a file, edfarray maps the file into the address space of the process. The OS then moves pages of the file into and out of physical RAM. The page cache is the set of file pages that the OS holds in RAM. This design has these results:

- A large file opens almost immediately. edfarray parses the header synchronously, which is fast because the header has a fixed size. The annotation scan runs in a background thread.
- Signal reads work immediately after open. They do not wait for the annotation scan.
- Sequential reads (paging forward) get the benefit of OS readahead.
- Random seeks (jumps to a timestamp) fault in only the pages that they touch.
- Signals that read from the same records share the cached pages.

When you open a file, edfarray parses the header and the record layout synchronously. This takes microseconds. Then edfarray starts a background thread that scans the TAL data of the file to build the annotation index. TAL (time-stamped annotation list) is the byte format of annotations. edfarray sends `madvise` hints to prepare the page cache before bulk reads. The hints also mark the annotation scan as a sequential pass.

### Memmap failure case

EDF interleaves the signals inside each record. Thus a read of one signal touches a small slice
of every record. For example, a file has 64 signals at 256 Hz and 1-second records. Each record is 32 KB, and the
slice for one signal is 512 B. The kernel must fault in the whole file to return one sixty-fourth
of it.

If the data is already in the page cache, this cost is almost zero. For this reason, benchmarks
with a warm cache never show the problem. A warm cache already holds the data, and a cold cache
does not. If the data is not in the page cache, the cost is one major page fault per record.

edfarray detects this case. Before a large read, edfarray uses `mincore` to sample how much of the
target range is resident. Resident data is data that is in the page cache. If the range is large
and most of it is not cached, edfarray reads sequentially through a bounded buffer. It does not
fault through the mapping.

During a long read, edfarray also repeats this test at intervals. A read can start warm and
then lose its pages to memory pressure. If this occurs, the read changes to the buffered method
partway. It does not pay one fault per record for the remainder of the file.

These results are for a read of one full signal from a 4 GiB file with 64 signals:

| Cache state | Mapping only | Automatic | pyedflib |
| --- | --- | --- | --- |
| Warm | 107 ms | 113 ms | 1021 ms |
| Cold | 3959 ms | 659 ms | 5624 ms |

With a cold cache, the mapping-only path causes 131,073 major faults (one for each record). The
automatic path causes 2. To force one mode, use `f.signal(0, strategy="mmap")` or
`strategy="stream"`. The default is `"auto"`. Streaming is slower than a warm mapping. If you
know that the data will not be in the page cache, you can set `"stream"` explicitly. In other cases, do not set it.

## Async annotation scan

For large files, the annotation scan can take a noticeable time. An example is a 24-hour EEG of approximately 12 GB. edfarray runs the scan in the background, so you can read signal data immediately:

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

Plain EDF files have no annotation channel, so edfarray does not scan them. It calculates the record onsets directly from the header. For EDF+C files, signal reads never block, because the record onsets are uniform. Only EDF+D files need the scan results to map time correctly in `times()` and `read_time_range()`.

The scan reads every record. Thus, on a very large file, the scan competes with your reads for
space in the page cache. To skip the scan at open, pass `scan_annotations=False`. edfarray then
builds the index on the calling thread, at the first access to the annotations:

```python
f = edfarray.EdfFile("large_recording.edf", scan_annotations=False)
data = f.signal(0).to_physical() # no scan happens
f.annotations # builds the index now, blocking until done
```

## Why it's fast

Three design decisions set the speed of page reads for many signals:

1. edfarray decodes signals in parallel on the rayon thread pool. With 100 or more signals, the work spreads across the cores.

2. For 16-bit EDF samples, the decode splits the i16-to-f64 conversion into a widening pass and a multiply-add pass. The compiler autovectorizes these passes into SIMD instructions. SIMD means one instruction that operates on many values.

3. The decode writes the signal bytes from the mapping directly into the output buffer, with
   no intermediate copies. The streaming path above copies the data one time into its read
   buffer. This copy is the cost of bounded memory use.

Reads release the GIL (the Python global interpreter lock). Thus the decode scales across Python threads and also across rayon workers.

## Single-signal access

For reads of one signal, the `Signal` proxy is efficient. A proxy is an object that reads file data on request. Each access converts the global sample index to a record offset. Then it decodes directly from the mmap:

```python
sig = f.signal(0)
chunk = sig[10000:20000]  # decoded directly from mmap
```

This path uses one thread and does not use rayon. But it does no unnecessary allocation and no unnecessary copies. For reads of one signal at a time, the decode is the only cost.

## Tips for EEG viewer applications

This example is for an application that pages through a recording:

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

For typical EEG recordings (30-100 signals), each `read_page()` call takes much less than 1 ms. Thus you can call it on every frame, without a prefetch layer or a cache layer in Python.
