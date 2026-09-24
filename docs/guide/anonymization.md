# Anonymization

Anonymize a recording before you share it. edfarray edits the identification fields in
place. For a 4 GiB recording, this costs a few hundred bytes of I/O, not a full rewrite. No
other part of the file moves. Thus the signal data, timings, and annotations of a release
candidate stay byte-identical to the version that you reviewed.

Three functions cover the workflow:

- `edfarray.edit_header(path, ...)` replaces the header fields that you choose.
- `edfarray.anonymize(path, ...)` does the usual anonymization in one call.
- `edfarray.audit(path)` reports where identity strings still appear.

## Editing header fields

```python
import datetime
import edfarray

diff = edfarray.edit_header(
    "recording.edf",
    patient_id="X F 02-MAR-1951 Subject-4MKCYYQJ",
    start_datetime=datetime.datetime(2020, 7, 4, 12, 0, 0),
)
# diff maps each changed field to {"before": ..., "after": ...}
```

To leave a field unchanged, pass `None` for it. Values must fit the fixed-width ASCII
fields. Thus each of the two id fields holds at most 80 bytes. `edit_header` checks every
value before it writes anything. Thus a rejected edit leaves the file byte-identical.

If another edfarray handle has the file open, `edit_header` and `anonymize` raise
`EdfFileError`. An open `EdfFile` keeps the header that it parsed at open time. Without
this check, that `EdfFile` will report the old identity after an edit. Before you edit a
file, close it. Then drop every signal and proxy taken from it. A proxy is a `Proxy2D` or
`Proxy3D` array view.

A dry run is a run that reports the result but writes nothing. To start a dry run, pass
`dry_run=True`. A dry run and `audit` only read the file, so they work on an open file.

## Anonymizing

```python
result = edfarray.anonymize(
    "recording.edf",
    seed="study-2026",      # reproducible pseudonym and date shift
    date_shift_days=None,   # or pin the shift yourself
    keep_sex=True,          # default
    keep_equipment=True,    # default
)
```

By default, `anonymize` makes these changes:

- It replaces the patient name with a pseudonym (`Subject-XXXXXXXX`). A pseudonym is a
  made-up name for the patient. With a `seed`, all files of one subject get the same
  pseudonym, so the recordings of a subject stay linkable across a corpus. Of the subject
  fields, the pseudonym uses only the patient name, code, and birthdate. Thus notes for
  each session in the free-text subfield do not split one subject into several
  pseudonyms. Without a seed, every call picks a new random pseudonym and a new date
  shift (the number of days that all dates move).
- It replaces the patient code, technician, admin code, and free-text subfields with `X`.
  It keeps sex and equipment by default. If you pass `keep_sex=False`, it does not keep
  sex. `keep_equipment=False` does the same for equipment.
- It shifts the birthdate, the recording date, and the header startdate by the same number
  of days. Age and time of day stay the same, but the calendar date changes. The two-digit
  year field can hold only the years 1985-2084. If a shift moves the startdate outside this
  range, `anonymize` raises an error. It does not clamp the date silently.

To see the changes without writing anything, add `dry_run=True`. A dry run makes the same
checks on the field values as a real run. A real run can still fail for two reasons.
Another edfarray handle can have the file open, or an I/O error can occur. To make the dry
run show the same pseudonym and date shift as the real run, pass a `seed`.

!!! danger "The seed is a re-identification key"
    Store the seed as you store a linking log (the list that maps pseudonyms to real
    identities). Keep it separate from the anonymized files, and never put it in the same
    release. A reused `seed` determines both the pseudonym and the date shift.

    edfarray makes each pseudonym with HMAC-SHA256, keyed on the seed and stretched over
    500,000 iterations. Without the seed, a pseudonym is not reversible. With the seed, a
    dictionary search over candidate names costs about 20 ms per guess, not zero time. This
    cost slows the search but does not stop it. The date shift has only 7,302 possible
    values, and this method cannot protect it at all.

## The audit gate

Header edits cannot change signal labels or annotation text. But some labs put patient
names in signal labels or event notes. Examples are the label `EEG Alice Smith` and the
note `Alice Smith restless`. `audit()` scans labels, transducers, prefilters, and annotation text. It returns
each hit with its location:

```python
report = edfarray.audit("recording.edf")
if not report["clean"]:
    for hit in report["hits"]:
        print(hit["location"], hit["term"], hit["excerpt"])
```

`audit()` takes its search terms from the patient name, patient code, technician, and
admin code in the current header. Thus you must run it before you anonymize the file. To
make sure that no removed term remains, run `audit()` again with the terms that `anonymize`
removed:

```python
result = edfarray.anonymize("recording.edf", seed="study-2026")
residual = edfarray.audit("recording.edf", terms=result["scrubbed_terms"])
```

A person must decide what to do with each remaining hit. To rename signals, write the file
again with `edfarray.write_edf()` from the decoded data. edfarray cannot rewrite annotation
text in place.

!!! warning "Anonymization is not a guarantee"
    Use the audit report as one check in a release process, not as the whole process.
    These functions remove only the identifiers that this library can see. They cannot find
    an identity in arbitrary binary payloads, in a filename, or in an export step that
    copies metadata to another location.
