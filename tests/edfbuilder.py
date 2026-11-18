"""Build minimal synthetic EDF+ files with precise control over TAL contents.

The shipped fixtures all start at onset 0 with a single annotation channel, which hides
timing and TAL-selection bugs. These builders let tests target those paths directly.
"""


def _fld(value, width):
    return str(value).ljust(width)[:width].encode("ascii")


def _sample(value, size):
    """Pack one little-endian signed digital sample of `size` bytes."""
    return int(value).to_bytes(size, "little", signed=True)


def build_edf_plus(
    path,
    *,
    record_onsets,
    annotations=(),
    signals=None,
    variant="EDF+C",
    samples_per_record=10,
    annotation_channels=1,
    annotation_bytes=60,
    annotation_label="EDF Annotations",
    record_duration=1.0,
    timekeeping=True,
    patient_id="X X X X",
    recording_id="Startdate 10-DEC-2009 X X X",
):
    """Write an EDF+ file whose record time-keeping onsets are exactly `record_onsets`.

    `annotations` is a sequence of `(record_idx, onset, text, channel)` tuples placed after
    the time-keeping TAL of the given annotation channel. A list of texts packs them all under
    one onset.

    `signals` is an optional list of ordinary-signal dicts, each `{"label", "rate", "values"}`
    where `values(record_idx, sample)` returns the digital value to write. Defaulting it keeps
    the historical single `Sig1` channel holding the record index, so existing callers are
    unaffected.

    `annotation_label` sets the label of every annotation channel. The BDF+ specification uses
    "BDF Annotations". A `variant` that starts with "BDF" writes 3-byte samples and the BDF
    version field.
    """
    num_records = len(record_onsets)
    bdf = variant.startswith("BDF")
    sample_size = 3 if bdf else 2
    if signals is None:
        signals = [{"label": "Sig1", "rate": samples_per_record, "values": lambda r, s: r}]
    n_ord = len(signals)
    num_signals = n_ord + annotation_channels
    ann_samples = annotation_bytes // sample_size

    header = b"".join(
        [
            b"\xffBIOSEMI" if bdf else _fld("0", 8),
            _fld(patient_id, 80),
            _fld(recording_id, 80),
            _fld("10.12.09", 8),
            _fld("12.44.02", 8),
            _fld(256 + 256 * num_signals, 8),
            _fld(variant, 44),
            _fld(num_records, 8),
            _fld(record_duration, 8),
            _fld(num_signals, 4),
        ]
    )

    labels = [s["label"] for s in signals] + [annotation_label] * annotation_channels
    per_signal = [
        (labels, 16),
        ([""] * num_signals, 80),
        (["uV"] * n_ord + [""] * annotation_channels, 8),
        ([-100] * n_ord + [-1] * annotation_channels, 8),
        ([100] * n_ord + [1] * annotation_channels, 8),
        ([-32768] * num_signals, 8),
        ([32767] * num_signals, 8),
        ([""] * num_signals, 80),
        ([s["rate"] for s in signals] + [ann_samples] * annotation_channels, 8),
        ([""] * num_signals, 32),
    ]
    for values, width in per_signal:
        header += b"".join(_fld(v, width) for v in values)

    body = b""
    for rec_idx, onset in enumerate(record_onsets):
        for sig in signals:
            rate = sig["rate"]
            body += b"".join(
                _sample(sig["values"](rec_idx, s), sample_size) for s in range(rate)
            )
        for channel in range(annotation_channels):
            tal = b""
            if channel == 0 and timekeeping:
                tal += b"%s\x14\x14\x00" % f"+{onset:g}".encode()
            for ann in annotations:
                ann_rec, ann_onset, text, ann_channel = ann
                if ann_rec == rec_idx and ann_channel == channel:
                    texts = text if isinstance(text, list) else [text]
                    joined = b"\x14".join(t.encode() for t in texts)
                    tal += b"%s\x14%s\x14\x00" % (f"+{ann_onset:g}".encode(), joined)
            if len(tal) > annotation_bytes:
                raise ValueError(f"TAL block {len(tal)} exceeds budget {annotation_bytes}")
            body += tal.ljust(annotation_bytes, b"\x00")

    path = str(path)
    with open(path, "wb") as fh:
        fh.write(header + body)
    return path
