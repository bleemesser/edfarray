"""Build minimal synthetic EDF+ files with precise control over TAL contents.

The shipped fixtures all start at onset 0 with a single annotation channel, which hides
timing and TAL-selection bugs. These builders let tests target those paths directly.
"""

import struct


def _fld(value, width):
    return str(value).ljust(width)[:width].encode("ascii")


def build_edf_plus(
    path,
    *,
    record_onsets,
    annotations=(),
    variant="EDF+C",
    samples_per_record=10,
    annotation_channels=1,
    annotation_bytes=60,
    record_duration=1.0,
    timekeeping=True,
):
    """Write an EDF+ file whose record time-keeping onsets are exactly `record_onsets`.

    `annotations` is a sequence of `(record_idx, onset, text, channel)` tuples placed after
    the time-keeping TAL of the given annotation channel.
    """
    num_records = len(record_onsets)
    num_signals = 1 + annotation_channels
    ann_samples = annotation_bytes // 2

    header = b"".join(
        [
            _fld("0", 8),
            _fld("X X X X", 80),
            _fld("Startdate 10-DEC-2009 X X X", 80),
            _fld("10.12.09", 8),
            _fld("12.44.02", 8),
            _fld(256 + 256 * num_signals, 8),
            _fld(variant, 44),
            _fld(num_records, 8),
            _fld(record_duration, 8),
            _fld(num_signals, 4),
        ]
    )

    labels = ["Sig1"] + ["EDF Annotations"] * annotation_channels
    per_signal = [
        (labels, 16),
        ([""] * num_signals, 80),
        (["uV"] + [""] * annotation_channels, 8),
        ([-100] + [-1] * annotation_channels, 8),
        ([100] + [1] * annotation_channels, 8),
        ([-32768] * num_signals, 8),
        ([32767] * num_signals, 8),
        ([""] * num_signals, 80),
        ([samples_per_record] + [ann_samples] * annotation_channels, 8),
        ([""] * num_signals, 32),
    ]
    for values, width in per_signal:
        header += b"".join(_fld(v, width) for v in values)

    body = b""
    for rec_idx, onset in enumerate(record_onsets):
        body += struct.pack("<%dh" % samples_per_record, *([rec_idx] * samples_per_record))
        for channel in range(annotation_channels):
            tal = b""
            if channel == 0 and timekeeping:
                tal += b"%s\x14\x14\x00" % f"+{onset:g}".encode()
            for ann in annotations:
                ann_rec, ann_onset, text, ann_channel = ann
                if ann_rec == rec_idx and ann_channel == channel:
                    tal += b"%s\x14%s\x14\x00" % (f"+{ann_onset:g}".encode(), text.encode())
            if len(tal) > annotation_bytes:
                raise ValueError(f"TAL block {len(tal)} exceeds budget {annotation_bytes}")
            body += tal.ljust(annotation_bytes, b"\x00")

    path = str(path)
    with open(path, "wb") as fh:
        fh.write(header + body)
    return path
