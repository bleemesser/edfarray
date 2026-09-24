# Async API

The async API is in `edfarray.aio`. It has the same classes and methods as the sync API, with two exceptions. `aio.open(path, variant=None)` has no `scan_annotations` argument, and `aio.EdfFile.signal(idx_or_label, cache_capacity=0)` has no `strategy` argument.

The async API is for applications that run an event loop and must not block on mmap decode. An event loop is the scheduler that runs asyncio coroutines. Typical applications are:

- A desktop reader that serves a browser UI.
- A local server that sends data to multiple clients.
- A pipeline where Python work must interleave with signal decoding.

Every async method releases the GIL (the Python global interpreter lock) during decode. It uses `tokio::task::spawn_blocking` on a multi-threaded tokio runtime. Concurrent reads on the same file run in parallel.

## Opening files

Open a file asynchronously. The `open` call reads the header and starts an
annotation scan in the background. Thus it returns quickly:

```python
import asyncio
import edfarray.aio as aio

async def main():
    f = await aio.open("recording.edf")
    print(f.num_signals, f.duration)
```

To get the metadata quickly without keeping the file open, use `inspect`:

```python
meta = await aio.inspect("recording.edf")
print(meta["variant"], meta["num_signals"], meta["duration"])
```

When the block exits, the async context manager closes the file:

```python
async with await aio.open("recording.edf") as f:
    data = await f.signal(0).read_range(0, 1000)
```

`aio.open` takes the same `variant` override as the sync constructor. Use it for
files whose `+C`/`+D` marker is missing or wrong:

```python
f = await aio.open("recording.edf", variant="EDF+D")
```

Metadata getters such as `num_signals`, `variant`, and `start_datetime` are sync, also on the async `EdfFile`. They read from a header that is already loaded and shared through an `Arc`. They return immediately and do no I/O.

## Reading signals

Get a `Signal` object with `f.signal()` (sync). Then read data asynchronously:

```python
f = await aio.open("recording.edf")

sig = f.signal(0) # sync: returns a Signal proxy
chunk = await sig.read_range(0, 1000) # async: decodes and returns numpy array
full  = await sig.to_physical() # async: entire signal as float64 array
raw   = await sig.to_digital() # async: entire signal as int32 array
times = await sig.times() # async: timestamp for each sample
```

Time-based reads convert seconds to sample indices internally:

```python
data = await sig.read_time_range(0.0, 10.0)  # physical values in [0, 10) seconds
```

If you read the same signal again and again, pass `cache_capacity`. This
enables an LRU (least recently used) cache of decoded physical records. A record
is a block of samples with a fixed duration. The unit of `cache_capacity` is data
records, not samples:

```python
sig = f.signal(0, cache_capacity=4)  # cache 4 decoded records
```

The cache behaves the same as in the sync API. [Caching repeated reads](signals.md#caching-repeated-reads) tells how to size
`cache_capacity` and in which cases the cache helps.

## Concurrent reads

If multiple coroutines read different regions of the same file, they run concurrently on separate OS threads:

```python
import asyncio

f = await aio.open("recording.edf")

# Read four non-overlapping pages concurrently
results = await asyncio.gather(
    f.read_page(0.0, 30.0),
    f.read_page(30.0, 60.0),
    f.read_page(60.0, 90.0),
    f.read_page(90.0, 120.0),
)
# Each element of results is a list of numpy arrays (one per signal).
```

The four reads decode in parallel on the tokio thread pool. The total wall time is close to the time of one page read, not four times that time.

While Rust decodes, other Python coroutines that do not need the extension continue to run. The decode does not block the event loop.

## Multi-channel access in async mode

The array proxies (`Proxy2D` and `Proxy3D`) are sync only. A proxy is an array view that reads from the file on access. Their purpose is synchronous numpy-style `proxy[ch, samp_range]` indexing. This indexing does not map well to `await` points. The ML preprocessing pipelines that use this access pattern are also sync.

For async reads of multiple signals, use `read_page`. It decodes signals in parallel on the tokio thread pool. If you need numpy-style indexing, open the file synchronously. The async file also has `signal_groups()` and `signal_group()` as plain sync getters. You can use them to find the signal groups.

## Waiting for annotations

Opening a file is fast because the annotation scan runs in the background. If
your code needs the full annotation index before it continues, wait for it:

```python
f = await aio.open("recording.edf")
# ... start reading signals immediately ...
await f.wait_for_annotations()
# Now f.annotations is guaranteed to be complete
for ann in f.annotations:
    print(ann.onset, ann.text)
```

To find out whether the scan is finished, read `f.annotations_ready` (sync). This
read does not block.

## Writing

The async API has the same two ways to write a file as the sync API.

To write a file in one call, use `write_edf`:

```python
import numpy as np

sig_def = aio.WriterSignal(
    label="EEG Fpz",
    physical_dimension="uV",
    physical_min=-3200.0,
    physical_max=3200.0,
    digital_min=-32768,
    digital_max=32767,
    samples_per_record=256,
)

data = np.zeros(256 * 4, dtype=np.float64)

await aio.write_edf(
    "out.edf",
    variant="EDF+C",
    record_duration=1.0,
    signals=[sig_def],
    data=[data],
)
```

To write a file as a stream, use `EdfWriter`. Its constructor is a classmethod.
Call `create()`. Do not call `EdfWriter(...)` directly:

```python
w = await aio.EdfWriter.create(
    "live.edf",
    variant="EDF+C",
    record_duration=1.0,
    signals=[sig_def],
)

for record in produce_records():
    await w.write_record([record])

await w.finish()
```

You can also use the async context manager:

```python
async with await aio.EdfWriter.create(
    "live.edf",
    variant="EDF+C",
    record_duration=1.0,
    signals=[sig_def],
) as w:
    for record in produce_records():
        await w.write_record([record])
```

`add_annotation` is sync. It puts an annotation in a queue for the next record:

```python
import edfarray

w.add_annotation(edfarray.Annotation(onset=5.0, text="event"))
```

## Transcoding

Transcoding writes an open file to a new path. You can change the variant at the
same time:

```python
f = await aio.open("input.edfd")
await f.write_to("output.edf", variant="EDF+C")
```

`write_to` copies only the ordinary signals. An ordinary signal is a signal that
is not the annotation channel. `write_to` encodes the annotations again from the
parsed index. It does not copy them byte for byte. The same [transcoding caveats](writing.md#round-tripping-an-existing-file)
apply as for the sync `write_to`:

- Transcoding EDF+D to a non-`+D` variant loses the EDF+D gaps. An EDF+D gap is
  a time gap between two records.
- Transcoding to a plain variant drops annotations.
- Transcoding to a smaller sample size loses precision.

The async `write_to` also accepts a `signals` subset selection. It also rejects
a destination that is open. Both behave the same as in the sync method.

## Performance and when to choose async

The async API is not always faster than the sync API. It adds a small overhead to each call. In return, it gives parallel decode and a responsive event loop. The benchmarks in `examples/` measure this on a 4 MB test file (`test_generator.edf`, 180 000 samples @ 200 Hz). Absolute numbers change with the file size and the machine.

### Parallel decode scales near-linearly up to core count

`benchmark_async_parallel_decode.py` compares N sequential awaits against
`asyncio.gather` of N concurrent reads on the same file:

| N | sequential | gather  | speedup | efficiency |
| - | ---------- | ------- | ------- | ---------- |
| 1 | 5.15 ms    | 5.18 ms | 0.99x   | 99%        |
| 2 | 10.47 ms   | 5.67 ms | 1.85x   | 92%        |
| 4 | 20.79 ms   | 5.70 ms | 3.65x   | 91%        |
| 8 | 41.11 ms   | 9.43 ms | 4.36x   | 55%        |

The speedup is close to N up to the number of available cores. Above that number, the speedup stops increasing.

### The event loop stays responsive under load

`benchmark_async_gil_release.py` runs a busy Python thread at the same time as 10 full decodes. Both APIs release the GIL during decode. With both APIs, the busy thread runs at about 95% of its free-running tick rate. But the wall-clock time for the decodes is very different:

| API   | Wall time (10 decodes, busy thread) |
| ----- | ----------------------------------- |
| sync  | 4296 ms                             |
| async | 325 ms (~13x faster)                |

The sync API runs decode on the calling thread. That thread releases the GIL
during each decode, but it must take the GIL back between decodes. So it competes
with the busy thread for the GIL and the CPU. The async API sends the work to a tokio
worker on a separate OS thread. Thus the Python side does only event-loop work.

### Single-shot overhead is small for bulk reads, noticeable for tiny ones

`benchmark_async_vs_sync.py` runs the same operations through both APIs
in series (concurrency = 1):

| Operation             | sync     | async    | overhead |
| --------------------- | -------- | -------- | -------- |
| `open`                | 158 us   | 237 us   | +50%     |
| full read (physical)  | 6.34 ms  | 6.07 ms  | ~0       |
| full read (digital)   | 4.46 ms  | 4.53 ms  | +1.5%    |
| 1-second slice        | 11 us    | 107 us   | +880%    |

A tokio dispatch and an event-loop hop have a fixed cost of about 100 us. This cost is very small compared with a decode of several milliseconds. But it is most of the time for a slice at microsecond scale.

### Recommendations

Use async (`edfarray.aio`) in these cases:

- You read multiple files or multiple regions concurrently and want true
  parallel decode. Examples are `asyncio.gather` and a server with multiple
  clients.
- You serve a UI or another event loop and must not stall it during long
  decodes.
- You already work inside an asyncio application.

Use sync (`edfarray.EdfFile`) in these cases:

- You run a serial pipeline of small reads (less than one millisecond each).
  In this pipeline, the overhead of about 100 us per call is important.
- The program is not async for other reasons, and you do not need concurrency.
- You write a one-off script. The sync API is simpler for that.

Both APIs use the same Rust decode path. Thus, for one bulk read, they finish in
almost the same wall time.

## GIL release and parallelism

The async runtime is a multi-threaded tokio executor. Every async method that
does decode or file I/O wraps the work in `tokio::task::spawn_blocking`. This
releases the Python GIL while the work runs. The results are:

- Decode of large signals does not block the event loop.
- Other Python-only coroutines can run while decode is in progress.
- Multiple reads on the same file are dispatched to separate OS threads and
  run in parallel.

The tests prove this with a busy-thread counter. A background thread increments
a counter while async reads run. The counter increases during decode. This result
shows that the decode releases the GIL.
