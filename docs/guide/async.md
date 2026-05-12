# Async API

The async API lives in `edfarray.aio` and mirrors the sync API. It is designed for
applications that run an event loop and cannot afford to block on mmap decode: a
desktop reader serving a browser UI, a local server feeding multiple clients, or
any pipeline where Python work must interleave with I/O-heavy signal decoding.

Every async method releases the GIL during decode via `tokio::task::spawn_blocking`
on a multi-threaded tokio runtime. Multiple concurrent reads on the same file run
truly in parallel.

## Opening files

Open a file asynchronously. The `open` call reads the header and starts a
background annotation scan, so it returns quickly:

```python
import asyncio
import edfarray.aio as aio

async def main():
    f = await aio.open("recording.edf")
    print(f.num_signals, f.duration)
```

For a quick metadata lookup without keeping the file open, use `inspect`:

```python
meta = await aio.inspect("recording.edf")
print(meta["variant"], meta["num_signals"], meta["duration"])
```

The async context manager form ensures the file is closed when the block exits:

```python
async with await aio.open("recording.edf") as f:
    data = await f.signal(0).read_physical(0, 1000)
```

Metadata getters (`num_signals`, `variant`, `start_datetime`, etc.) are **sync**
even on the async `EdfFile`. They read from an `Arc`-shared header that was
already loaded, so they return immediately without I/O.

## Reading signals

Get a `Signal` proxy with `f.signal()` (sync), then read data asynchronously:

```python
f = await aio.open("recording.edf")

sig = f.signal(0)                       # sync — returns a Signal proxy
chunk = await sig.read_physical(0, 1000) # async — decodes and returns numpy array
full  = await sig.to_numpy()            # async — entire signal as float64 array
raw   = await sig.to_digital()          # async — entire signal as int32 array
times = await sig.times()               # async — timestamp for each sample
```

Time-based reads map seconds to sample indices internally:

```python
data = await sig.read_at(0.0, 10.0)  # physical values in [0, 10) seconds
```

For repeated reads on the same signal, use `cache_capacity` to enable an LRU
cache of decoded records:

```python
sig = f.signal(0, cache_capacity=4)  # cache 4 decoded records
```

## Concurrent reads

The headline feature of the async API is truly parallel decoding. Multiple
coroutines reading different regions of the same file run concurrently on
separate OS threads:

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

The four reads decode in parallel on the tokio thread pool. Total wall time is
close to the time of a single page read, not four times that.

GIL release means that while Rust is decoding data, other Python coroutines
that do not need the Rust extension can continue running. The event loop is not
blocked.

## 2D access via ArrayProxy

The async `ArrayProxy` provides multi-channel 2D reads:

```python
ap = f.array_proxy()  # all ordinary signals at a common sample rate
print(ap.shape)       # (num_signals, total_samples)

# Read a 2D window: signals x samples
block = await ap.read_physical(0, 1000)
# block.shape == (num_signals, 1000)
```

All `ArrayProxy` methods (`read_physical`, `read_digital`, `get`,
`read_signals_at_sample`) are async and release the GIL.

## Waiting for annotations

Opening a file is fast because the annotation scan runs in the background. If
your code needs the full annotation index before proceeding, wait for it:

```python
f = await aio.open("recording.edf")
# ... start reading signals immediately ...
await f.wait_for_annotations()
# Now f.annotations is guaranteed to be complete
for ann in f.annotations:
    print(ann.onset, ann.text)
```

Check `f.annotations_ready` (sync) to see if the scan has finished without
blocking.

## Writing

Two writing paths are available, matching the sync API.

One-shot write using `write_edf`:

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

Streaming write using `EdfWriter`. The constructor is a classmethod -- call
`create()`, not `EdfWriter(...)` directly:

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

Or use the async context manager:

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

`add_annotation` is sync and queues an annotation for the next record:

```python
w.add_annotation(edfarray.Annotation(onset=5.0, text="event"))
```

## Transcoding

Write an open file to a new path, optionally changing the variant:

```python
f = await aio.open("input.edfd")
await f.write_to("output.edf", variant="EDF+C")
```

Only ordinary signals are copied. Annotations are re-encoded from the parsed
index rather than copied verbatim.

## Performance and when to choose async

The async API is not universally faster than the sync API — it trades a small
per-call overhead for the ability to run decodes in parallel and to keep the
event loop responsive. The benchmarks under `examples/` quantify this on a
4 MB fixture (`test_generator.edf`, 180 000 samples @ 200 Hz). Numbers will
vary by file size and machine, but the shape of the result is what matters.

### Parallel decode scales near-linearly up to core count

`benchmark_async_parallel_decode.py` compares N sequential awaits against
`asyncio.gather` of N concurrent reads on the same file:

| N | sequential | gather  | speedup | efficiency |
| - | ---------- | ------- | ------- | ---------- |
| 1 | 5.15 ms    | 5.18 ms | 0.99x   | 99%        |
| 2 | 10.47 ms   | 5.67 ms | 1.85x   | 92%        |
| 4 | 20.79 ms   | 5.70 ms | 3.65x   | 91%        |
| 8 | 41.11 ms   | 9.43 ms | 4.36x   | 55%        |

Speedup tracks N up to the available core count, then plateaus.

### The event loop stays responsive under load

`benchmark_async_gil_release.py` runs a busy Python thread alongside 10 full
decodes. Both APIs drop the GIL during decode (the busy thread runs at ~95%
of its free-running tick rate either way), but the wall-clock time for the
decodes themselves is dramatically different:

| API   | Wall time (10 decodes, busy thread) |
| ----- | ----------------------------------- |
| sync  | 4296 ms                             |
| async | 325 ms (~13x faster)                |

Sync runs decode on the calling thread, which then contends with the busy
thread for the GIL and the CPU. Async dispatches to a tokio worker on a
separate OS thread, so the Python side does only event-loop work.

### Single-shot overhead is small for bulk reads, noticeable for tiny ones

`benchmark_async_vs_sync.py` runs the same operations through both APIs
serially (concurrency = 1):

| Operation             | sync     | async    | overhead |
| --------------------- | -------- | -------- | -------- |
| `open`                | 158 µs   | 237 µs   | +50%     |
| full read (physical)  | 6.34 ms  | 6.07 ms  | ≈ 0      |
| full read (digital)   | 4.46 ms  | 4.53 ms  | +1.5%    |
| 1-second slice        | 11 µs    | 107 µs   | +880%    |

The fixed ~100 µs cost of a tokio dispatch + event-loop hop disappears into
a multi-ms decode but dominates a microsecond-scale slice.

### Recommendations

Use **async** (`edfarray.aio`) when:

- you read multiple files or multiple regions concurrently and want true
  parallel decode (`asyncio.gather`, multi-client server, etc.);
- you serve a UI or other event loop and cannot afford to stall it during
  long decodes;
- you are already inside an asyncio application.

Use **sync** (`edfarray.EdfFile`) when:

- you are doing a serial pipeline of small reads (sub-millisecond) where the
  ~100 µs per-call overhead matters;
- the program is not otherwise async and you don't need concurrency;
- you are writing a one-off script — the sync API is simpler.

Both APIs share the same Rust decode path, so for a single bulk read they
finish in essentially the same wall time.

## GIL release and parallelism

The async runtime is a multi-threaded tokio executor. Every async method that
performs decode or file I/O wraps the work in `tokio::task::spawn_blocking`,
which releases the Python GIL for the duration. This means:

- Decode of large signals does not block the event loop.
- Other Python-only coroutines can run while decode is in progress.
- Multiple reads on the same file are dispatched to separate OS threads and
  run in parallel.

Tests verify this with a busy-thread counter: a background thread increments a
counter while async reads are in flight. The counter advances during decode,
confirming the GIL is released.
