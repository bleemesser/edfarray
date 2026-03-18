# EDF Format

A brief overview of the EDF and EDF+ file formats, enough to understand how edfarray works internally. The full specifications are at [edfplus.info](https://www.edfplus.info/specs/).

## Structure

An EDF file consists of a fixed-size header followed by a sequence of data records.

```
[ Header: 256 + 256*ns bytes ][ Record 0 ][ Record 1 ] ... [ Record N-1 ]
```

`ns` is the number of signals.

## Header

The first 256 bytes contain the main header fields, all stored as ASCII text, left-justified and space-padded:

```
Offset  Size  Field
0       8     Version (always "0")
8       80    Patient identification
88      80    Recording identification
168     8     Start date (dd.mm.yy)
176     8     Start time (hh.mm.ss)
184     8     Header size in bytes
192     44    Reserved (EDF+ puts "EDF+C" or "EDF+D" here)
236     8     Number of data records (-1 if unknown)
244     8     Data record duration in seconds
252     4     Number of signals
```

After the main header, there are 256 bytes of per-signal metadata for each signal. These are stored in a transposed layout: all labels first (16 bytes each), then all transducer types (80 bytes each), and so on.

Per-signal fields: label (16), transducer type (80), physical dimension (8), physical minimum (8), physical maximum (8), digital minimum (8), digital maximum (8), prefiltering (80), number of samples per record (8), reserved (32).

## Data records

Each data record contains all signals sequentially. For each signal, there are `samples_per_record` samples stored as 16-bit signed integers in little-endian byte order.

```
[ Signal 0 samples ][ Signal 1 samples ] ... [ Signal ns-1 samples ]
```

The physical value of a sample is: `physical = gain * digital + offset`, where `gain = (physical_max - physical_min) / (digital_max - digital_min)` and `offset = physical_min - gain * digital_min`.

Different signals can have different sample counts per record, which means different sample rates.

## EDF+ extensions

EDF+ is backward-compatible with EDF. The differences:

The reserved field in the main header contains `"EDF+C"` for contiguous recordings or `"EDF+D"` for discontinuous recordings.

One or more signals are labeled `"EDF Annotations"`. These carry Time-stamped Annotation Lists (TALs) instead of signal data.

### Patient identification (EDF+)

The 80-byte patient identification field has structured subfields separated by spaces: `code sex birthdate name [additional]`. Unknown fields use `"X"`. Example: `"MCH-0234567 F 02-MAR-1951 Haagansen_Erlangen"`. Underscores replace spaces within names.

### Recording identification (EDF+)

Format: `"Startdate DD-MMM-YYYY admincode technician equipment [additional]"`. Example: `"Startdate 02-MAR-2002 PSG-1234 John_Doe Nihon_Kohden"`.

### Two-digit year clipping

The start date in the main header uses two-digit years. Per the EDF spec, years 85-99 map to 1985-1999 and 00-84 map to 2000-2084. EDF+ files can encode a four-digit year in the recording identification field.

### TAL format

Each annotation signal in a data record contains one or more TALs. A TAL has this byte structure:

```
+Onset[\x15Duration]\x14[Text\x14]*\x00
```

- Onset starts with `+` or `-`, in seconds from recording start.
- Duration is optional, separated by byte 0x15.
- Text segments are separated by byte 0x14.
- The TAL is terminated by byte 0x00.

The first TAL in the first annotation signal of each data record is the time-keeping annotation. It has empty text and its onset indicates the record's start time. In EDF+C, these onsets increase uniformly. In EDF+D, gaps between onsets indicate recording discontinuities.

The onset of the first time-keeping annotation in the file encodes the subsecond component of the recording start time, since the main header only stores integer seconds.
