# EDF Format

This page gives a short description of the EDF, EDF+, BDF, and BDF+ file formats. It gives enough detail to explain how edfarray works internally. The full specifications are at [edfplus.info](https://www.edfplus.info/specs/).

## Structure

An EDF file contains a fixed-size header and then a sequence of data records. A data record is one fixed-duration block of samples. The header comes first and is `256 + 256 * ns` bytes long. In this formula, `ns` is the number of signals. Records 0 to N-1 follow the header in order, with no space between them.

## Header

The first 256 bytes contain the main header fields. All fields are ASCII text, left-justified and padded with spaces. The one exception is the first byte of a BDF version field, 0xFF:

```text
Offset  Size  Field
0       8     Version ("0" for EDF, byte 0xFF then "BIOSEMI" for BDF)
8       80    Patient identification
88      80    Recording identification
168     8     Start date (dd.mm.yy)
176     8     Start time (hh.mm.ss)
184     8     Header size in bytes
192     44    Reserved ("EDF+C", "EDF+D", "BDF+C", "BDF+D", or "24BIT" for plain BDF)
236     8     Number of data records (-1 if unknown)
244     8     Data record duration in seconds
252     4     Number of signals
```

After the main header, the file has 256 bytes of metadata for each signal. The file stores this metadata in a transposed layout. All labels come first (16 bytes each), then all transducer types (80 bytes each), and so on.

Each signal has these fields, with the size in bytes:

- Label (16)
- Transducer type (80)
- Physical dimension (8)
- Physical minimum (8)
- Physical maximum (8)
- Digital minimum (8)
- Digital maximum (8)
- Prefiltering (80)
- Number of samples per record (8)
- Reserved (32)

## Data records

Each data record contains the samples of all signals, one signal after the other. Each signal has `samples_per_record` samples in the record. The file stores them as signed little-endian integers. EDF uses 16-bit samples and BDF uses 24-bit samples. The samples of signal 0 come first, then the samples of signal 1, and so on to signal `ns - 1`.

The physical value of a sample is `physical = gain * digital + offset`. In this formula, `gain = (physical_max - physical_min) / (digital_max - digital_min)` and `offset = physical_min - gain * digital_min`.

Different signals can have different sample counts per record. Thus they can have different sample rates.

## EDF+ extensions

EDF+ is backward-compatible with EDF. This section gives the differences.

The reserved field in the main header contains `"EDF+C"` for contiguous recordings or `"EDF+D"` for discontinuous recordings.

One or more signals have the label `"EDF Annotations"`. Each of these signals is an annotation channel. An annotation channel contains TALs (time-stamped annotation lists) instead of signal data.

### Patient identification (EDF+)

The 80-byte patient identification field contains subfields with a space between each two: `code sex birthdate name [additional]`. An unknown subfield contains `"X"`. For example: `"MCH-0234567 F 02-MAR-1951 Haagansen_Erlangen"`. In a name, the `_` character replaces each space.

### Recording identification (EDF+)

The field has the format `"Startdate DD-MMM-YYYY admincode technician equipment [additional]"`. For example: `"Startdate 02-MAR-2002 PSG-1234 John_Doe Nihon_Kohden"`.

### Two-digit year clipping

The start date in the main header uses two-digit years. The EDF specification maps years 85-99 to 1985-1999, and years 00-84 to 2000-2084. EDF+ files can encode a four-digit year in the recording identification field.

### TAL format

In a data record, each annotation channel contains one or more TALs. A TAL has this byte structure:

```text
+Onset[\x15Duration]\x14[Text\x14]*\x00
```

- The onset starts with `+` or `-`. It gives the time in seconds from the start of the recording.
- The duration is optional. Byte 0x15 separates it from the onset.
- Byte 0x14 separates the text segments.
- Byte 0x00 ends the TAL.

In each data record, the first TAL in the first annotation channel is the time-keeping annotation. Its text is empty. Its onset gives the start time of the record. In EDF+C, these onsets increase uniformly. In EDF+D, a gap between onsets shows a discontinuity in the recording. An EDF+D gap is a time span with no recorded data.

The main header stores the start time only in integer seconds. The onset of the first time-keeping annotation in the file encodes the subsecond part of the start time.
