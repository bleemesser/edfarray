# Anonymization

Sharing recordings? Scrub them first. edfarray edits the identification fields in place, so
a 4 GiB recording costs a few hundred bytes of I/O instead of a full rewrite. Nothing else
in the file moves, which means the signal data, timings, and annotations of a release
candidate stay byte-identical to what you validated.

Three functions cover the workflow:

- `edfarray.edit_header(path, ...)` replaces header fields you choose.
- `edfarray.anonymize(path, ...)` does the usual scrub in one call.
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

Pass `None` for a field to leave it alone. Values must fit the fixed-width ASCII fields, so
at most 80 bytes for the two id fields. Every value is checked before anything is written, so
a rejected edit leaves the file byte-identical. An `EdfFile` handle opened before the edit
keeps its stale parsed header; reopen it.

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

What it does, by default:

- Replaces the patient name with a pseudonym (`Subject-XXXXXXXX`). With a `seed`, the same
  subject's files always share one pseudonym, so a subject's recordings stay linkable across
  a corpus. The pseudonym is keyed on the patient name, code, and birthdate only, so
  per-session notes in the free-text subfield do not split one subject into several
  pseudonyms. Without a seed, the pseudonym is random per run.
- Replaces the patient code, technician, admin code, and free-text subfields with `X`.
  Sex and equipment survive unless you pass `keep_sex=False` / `keep_equipment=False`.
- Shifts birthdate, recording date, and the header startdate by the same number of days.
  Age and time-of-day survive, the calendar date does not. A shift that would push the
  startdate outside the 1985-2084 range the two-digit year field can hold raises instead
  of silently clamping.

Add `dry_run=True` to see what would change without writing anything. A dry run validates
everything a real run does, so an edit that passes as a dry run will not fail on write.

!!! danger "The seed is a re-identification key"
    A reused `seed` determines both the pseudonym and the date shift. Pseudonyms are HMAC-SHA256
    keyed on the seed and stretched over 500,000 iterations, so without the seed they are not
    reversible, and with it a dictionary sweep over candidate names costs about 20 ms per guess
    rather than being instant. That is a speed bump, not a wall: the date shift has only 7,303
    possible values and cannot be protected this way at all. Store the seed the way you would
    store a linking log -- separately from the anonymized files, and never in the same release.

## The audit gate

Header editing cannot touch signal labels or annotation text, and labs do write channel
names like `EEG Alice Smith` or event notes like `Alice Smith restless`. `audit()` scans
labels, transducers, prefilters, and annotation text, and returns each hit with its
location:

```python
report = edfarray.audit("recording.edf")
if not report["clean"]:
    for hit in report["hits"]:
        print(hit["location"], hit["term"], hit["excerpt"])
```

Terms come from the patient name, patient code, technician, and admin code currently in
the header, so run it before anonymizing. Afterward, re-check against the terms that were
scrubbed:

```python
result = edfarray.anonymize("recording.edf", seed="study-2026")
residual = edfarray.audit("recording.edf", terms=result["scrubbed_terms"])
```

Residual hits need a human decision. You can rename channels by rewriting the file with
`edfarray.write_edf()` from the decoded data, but there is no in-place text rewrite inside
annotation records.

!!! warning "Anonymization is not a guarantee"
    These functions remove the identifiers this library can see. They cannot know about
    an identity embedded in arbitrary binary payloads, a filename, or an export step that
    copies metadata somewhere else. Treat the audit report as one check in a release
    process, not as the whole process.
