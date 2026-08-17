"""Generate a large synthetic EDF file for large-file benchmarking.

Usage: python scripts/gen_large_edf.py OUT.edf --gb 4 --channels 64 --rate 256
"""

import argparse
import struct
import sys


def field(value, width):
    return str(value).ljust(width)[:width].encode("ascii")


def generate(path, *, gb, channels, rate):
    record_size = channels * rate * 2
    num_records = max(1, int(gb * (1 << 30)) // record_size)
    num_signals = channels

    header = b"".join(
        [
            field("0", 8),
            field("X X X X", 80),
            field("Startdate 01-JAN-2020 X X X", 80),
            field("01.01.20", 8),
            field("00.00.00", 8),
            field(256 + 256 * num_signals, 8),
            field("", 44),
            field(num_records, 8),
            field(1, 8),
            field(num_signals, 4),
        ]
    )
    per_signal = [
        ([f"EEG{i}" for i in range(channels)], 16),
        ([""] * channels, 80),
        (["uV"] * channels, 8),
        ([-3200] * channels, 8),
        ([3200] * channels, 8),
        ([-32768] * channels, 8),
        ([32767] * channels, 8),
        ([""] * channels, 80),
        ([rate] * channels, 8),
        ([""] * channels, 32),
    ]
    for values, width in per_signal:
        header += b"".join(field(v, width) for v in values)

    # A deterministic per-channel pattern so reads can be verified, not just timed.
    record = bytearray()
    for ch in range(channels):
        record += struct.pack("<%dh" % rate, *[(ch * 7 + i) % 4096 - 2048 for i in range(rate)])
    record = bytes(record)

    total = len(header) + num_records * len(record)
    with open(path, "wb") as fh:
        fh.write(header)
        chunk = record * min(num_records, max(1, (8 << 20) // len(record)))
        per_chunk = len(chunk) // len(record)
        written = 0
        while written < num_records:
            n = min(per_chunk, num_records - written)
            fh.write(chunk[: n * len(record)])
            written += n
    print(
        f"wrote {path}: {total / (1 << 30):.2f} GiB, {num_records} records, "
        f"{channels} channels @ {rate} Hz, record_size={record_size} B"
    )
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--gb", type=float, default=4.0)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--rate", type=int, default=256)
    args = ap.parse_args(argv)
    generate(args.out, gb=args.gb, channels=args.channels, rate=args.rate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
