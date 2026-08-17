# API contracts

Behavior that callers can rely on, and a few places where edfarray differs from
what you might assume.

## Exceptions

Every error raised by edfarray derives from `edfarray.EdfError` *and* from the builtin
exception a caller would naturally reach for, so both styles work:

```python
try:
    f.signal("EEG Fpz-Cz")
except edfarray.SignalNotFoundError:  # specific
    ...
except KeyError:                       # idiomatic
    ...
except edfarray.EdfError:              # anything from edfarray
    ...
```

| Exception | Also inherits | Raised for |
| --- | --- | --- |
| `EdfError` | `Exception` | base class, never raised directly |
| `EdfFileError` | `OSError` | the file cannot be opened, mapped, or written |
| `InvalidFileError` | `ValueError` | not valid EDF/BDF, or an inconsistent header |
| `InvalidArgumentError` | `ValueError` | an argument the format or API does not allow |
| `OutOfRangeError` | `IndexError` | a record, signal, or sample index out of range |
| `SignalNotFoundError` | `KeyError` | no signal matched the requested label |
| `ClosedFileError` | `ValueError` | the file was used after `close()` |

Indexing with an unsupported *type* raises `TypeError`, matching numpy. `TypeError` is not
part of the `EdfError` hierarchy because it signals a programming error, not a data problem.

## Supported indexing

`Signal`, `Proxy2D`, and `Proxy3D` accept a subset of numpy indexing:

| Form | Supported |
| --- | --- |
| integer, including negative | yes |
| slice, any step (including negative) | yes on the sample axis |
| slice with step != 1 on the signal/record/channel axis | `ValueError` |
| list of integers on a proxy's signal axis | yes |
| empty slice | yes, returns an empty array with numpy's shape |
| `bool` index | `TypeError` |
| boolean mask array | `TypeError` |
| ndarray fancy indexing | `TypeError` |
| `None` / `np.newaxis` | `TypeError` |
| `Ellipsis` | `TypeError` |

`bool` is rejected rather than accepted as an integer. Python's `bool` subclasses `int`, so
`sig[True]` would otherwise silently return sample 1 while numpy would apply mask semantics --
a wrong answer rather than an unsupported one.

All three classes implement `__array__`, so `numpy.asarray(obj)` materializes the data in one
decode pass. Without it, numpy falls back to the sequence protocol and decodes one sample per
`__getitem__` call. `copy=False` is refused: samples are decoded on access, so a no-copy view
is impossible.

Scalar reads return a Python `float`, not a numpy scalar.

## Units

Method names carry the unit wherever it is ambiguous:

- `read_range(start, stop)` and `read_range_digital(start, stop)` -- **sample indices**
- `read_time_range(start_sec, end_sec)` -- **seconds**
- `read_page(start_sec, end_sec)` -- **seconds**
- `signal(idx, cache_capacity=N)` -- N is a count of **data records**

## Properties that do work

Most properties are cheap field reads. Two are not:

- `annotations` blocks until the annotation index is ready and builds a new list on every
  access. Bind it once (`anns = f.annotations`) rather than indexing it in a loop.
- `warnings` likewise waits for the scan.

`header()` is a method rather than a property precisely because it builds a fresh dict per
call.

## close()

`close()` releases this handle. Any `Signal`, `Proxy2D`, or `Proxy3D` already obtained from the
file keeps its own reference to the mapping and continues to work:

```python
sig = f.signal(0)
f.close()
sig.to_physical()   # still valid
f.num_signals       # ClosedFileError
```

The mapping is released once the file and every object derived from it are dropped.

## Threading

Reads release the GIL, so the sync API scales across Python threads. `EdfFile` and the proxies
are safe to read from multiple threads concurrently.

`EdfWriter` can be moved between threads, but not written to from two at once: a concurrent
`write_record()` raises `RuntimeError` (the object is already mutably borrowed) rather than
interleaving records. Serialize writes with your own lock if several threads produce data.

Calling `close()` on a file while another thread is reading it raises rather than corrupting
anything, but is not a supported pattern.

## Async

`edfarray.aio` runs blocking work on a thread pool. Two consequences:

- Cancelling a task does not cancel work already handed to the pool. A cancelled
  `write_record()` may still have written its record. Treat a cancelled writer as being in an
  unknown state and do not continue writing to it.
- `WriterSignal`, `Annotation`, and `SignalGroup` are the same classes as in the sync API, so
  specs and annotations pass freely between them.

Constructing a writer is the one place the two APIs differ in shape: the sync `EdfWriter` uses
its constructor, while `aio.EdfWriter` uses `await EdfWriter.create(...)`, because a Python
constructor cannot be awaited.

## Memory mapping

By default the file is read through a memory map.

- If the file is truncated by another process while open, touching the vanished pages raises
  `SIGBUS`, which terminates the process and cannot be caught as a Python exception. Do not
  read a file another process is rewriting in place.
- On a network filesystem every page fault is a network round trip. For NFS/SMB, or any file
  larger than RAM, prefer `strategy="stream"` (see the performance guide), which uses ordinary
  positional reads.

## Pickling

`Annotation`, `WriterSignal`, and `SignalGroup` are picklable and compare by value, so they
work with `multiprocessing` and as dict keys. `EdfFile`, `Signal`, and the proxies are not
picklable: they hold an open mapping. Send the path and reopen instead.
