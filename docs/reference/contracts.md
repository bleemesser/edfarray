# API contracts

This page lists the behavior that callers can rely on. It also lists some behavior of
edfarray that can surprise you.

## Exceptions

Every error that edfarray raises derives from `edfarray.EdfError`. Each error also derives
from the builtin exception that a caller usually catches for that problem. Thus both styles work:

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
| `EdfFileError` | `OSError` | the file cannot be opened, mapped, locked, or written |
| `InvalidFileError` | `ValueError` | not valid EDF/BDF, or an inconsistent header |
| `InvalidArgumentError` | `ValueError` | an argument the format or API does not allow |
| `OutOfRangeError` | `IndexError` | a record, signal, or sample index out of range |
| `SignalNotFoundError` | `KeyError` | no signal matched the requested label |
| `ClosedFileError` | `ValueError` | the file was used after `close()` |

An index of an unsupported type raises `TypeError`, as in numpy. `TypeError` is not part of
the `EdfError` hierarchy, because it shows a programming error, not a data problem.

## Supported indexing

A proxy is an object that reads samples only on access. `Signal`, `Proxy2D`, and `Proxy3D`
accept a subset of numpy indexing:

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

edfarray rejects `bool` and does not treat it as an integer. Python's `bool` subclasses `int`,
so an integer rule reads `sig[True]` as sample 1. numpy reads the same index as a mask. That
result is wrong, and no error shows it. Thus edfarray raises `TypeError` for a `bool` index.

All three classes implement `__array__`. Thus `numpy.asarray(obj)` decodes all the data in one
pass. Without `__array__`, numpy uses the sequence protocol and decodes one sample per
`__getitem__` call. edfarray rejects `copy=False`. edfarray decodes samples on access, so a
no-copy view is not possible.

Scalar reads return a Python `float`, not a numpy scalar.

## Units

Where the unit is not clear, the method names show it:

- `read_range(start, stop)` and `read_range_digital(start, stop)`: sample indices
- `read_time_range(start_sec, end_sec)`: seconds
- `read_page(start_sec, end_sec)`: seconds
- `signal(idx, cache_capacity=N)`: N is a count of data records (fixed-duration blocks of samples)

## Properties that do work

Most properties only read a field. Two properties do more work.

`annotations` blocks until the annotation index is ready. It also builds a new list on every
access. Bind it once (`anns = f.annotations`). Do not index `f.annotations` in a loop.
`warnings` also waits for the scan.

`header()` is a method, not a property, because it builds a new dict on each call.

## close()

`close()` releases this handle. Any `Signal`, `Proxy2D`, or `Proxy3D` already taken from the
file keeps its own reference to the mapping and continues to work:

```python
sig = f.signal(0)
f.close()
sig.to_physical()   # still valid
f.num_signals       # ClosedFileError
```

When you drop the file and every object taken from it, edfarray releases the mapping.

## Threading

Reads release the GIL (the global interpreter lock of Python). Thus the sync API scales across
Python threads. Multiple threads can read from `EdfFile` and the proxies at the same time.

`EdfWriter` can move between threads, but two threads cannot write to it at the same time. A
concurrent `write_record()` raises `RuntimeError` (the object is already mutably borrowed). It
does not interleave records. If several threads produce data, serialize the writes with your
own lock.

If you call `close()` on a file while another thread reads it, edfarray raises an exception and
corrupts nothing. edfarray does not support this pattern.

## Async

`edfarray.aio` runs blocking work on a thread pool. This has two effects.

If you cancel a task, the work that the pool already has continues. A cancelled
`write_record()` can still write its record. After a cancel, the state of the writer is
unknown. Do not write to that writer again.

`WriterSignal`, `Annotation`, and `SignalGroup` are the same classes as in the sync API. Thus
you can pass specs and annotations between the two APIs.

Writer construction is the only place where the two APIs differ in shape. The sync `EdfWriter`
uses its constructor. `aio.EdfWriter` uses `await EdfWriter.create(...)`, because Python cannot
await a constructor.

## Memory mapping

By default, edfarray reads the file through a memory map.

- An open file holds a shared advisory lock (a lock that other programs can ignore). The
  lock stays until you drop the last `EdfFile`, `Signal`, and proxy on the file. The
  edfarray writers take an exclusive lock before they truncate. Thus writing to an open file
  raises `EdfFileError` and does not corrupt the mapping.
- Unless `dry_run=True`, `edit_header` and `anonymize` take the same exclusive lock. Thus
  they cannot leave an open file with a stale header. If an edfarray writer is still writing
  a file, opening that file also raises `EdfFileError`.
- The lock is advisory. Other programs that ignore it can still truncate the file. On NFS, a
  process does not conflict with its own locks. If another program truncates the file while
  it is open, access to the lost pages raises `SIGBUS`. `SIGBUS` stops the process, and
  Python cannot catch it as an exception. Do not read a file that another program rewrites
  in place.
- On a network filesystem, every page fault (a load of a file page not in memory) is a
  network round trip. For NFS/SMB, or any file larger than RAM, use `strategy="stream"` (see
  the performance guide). This strategy uses ordinary positional reads.

## Pickling

`Annotation` can be pickled, is hashable, and compares by value. Thus it works with
`multiprocessing` and as a dict key. `WriterSignal` can be pickled and compares by value,
but it is not hashable.

`SignalGroup` is hashable and compares by value, but it cannot be pickled. To send a group to
another process, send its `indices` and call `EdfFile.signal_group(indices)` there.
`EdfFile`, `Signal`, and the proxies cannot be pickled, because they hold an open mapping.
Send the path and open the file again.
