//! Writing EDF/EDF+ and BDF/BDF+ files.
//!
//! Two entry points:
//!
//! - [`EdfWriter`] — streaming writer. Open a file, push records one at a time,
//!   call [`EdfWriter::finish`] to patch the `num_records` field in the header.
//!   Suitable for live recording or large files that don't fit in memory.
//!
//! - [`write_edf`] — one-shot helper. Pass a fully populated [`WriterSpec`] and
//!   all physical sample data; the function builds an [`EdfWriter`] and emits
//!   every record in one call. Suitable for editing or transcoding existing
//!   files where the data is already in memory.
//!
//! Annotation channel handling: for `+C`/`+D` variants the writer auto-creates
//! the `EDF Annotations` signal. The user does **not** include an annotation
//! channel in the signals list. Each record gets a time-keeping TAL plus any
//! user annotations whose onset falls within the record's time window. The
//! channel is sized to fit the largest record's annotations (or a configurable
//! minimum); annotations that don't fit cause an error rather than being
//! silently dropped.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use chrono::{Datelike, NaiveDateTime, Timelike};

use crate::annotation::Annotation;
use crate::error::{EdfError, Result};
use crate::header::EdfVariant;

/// Default minimum byte budget for the annotation channel per record.
/// Roughly enough room for a time-keeping TAL plus a few short user annotations.
const DEFAULT_MIN_ANNOTATION_BYTES: usize = 120;

/// One signal's worth of metadata for the writer. Mirrors [`SignalHeader`] but
/// owned and validated only at write time.
///
/// [`SignalHeader`]: crate::signal::SignalHeader
#[derive(Debug, Clone)]
pub struct WriterSignal {
    pub label: String,
    pub transducer: String,
    pub physical_dimension: String,
    pub physical_min: f64,
    pub physical_max: f64,
    pub digital_min: i32,
    pub digital_max: i32,
    pub prefiltering: String,
    /// Number of samples for this signal in one data record. Combined with
    /// the spec's `record_duration_secs` this implies the sample rate.
    pub samples_per_record: usize,
    pub reserved: String,
}

impl WriterSignal {
    /// Convenience constructor with empty optional fields.
    pub fn new(
        label: impl Into<String>,
        physical_dimension: impl Into<String>,
        physical_min: f64,
        physical_max: f64,
        digital_min: i32,
        digital_max: i32,
        samples_per_record: usize,
    ) -> Self {
        WriterSignal {
            label: label.into(),
            transducer: String::new(),
            physical_dimension: physical_dimension.into(),
            physical_min,
            physical_max,
            digital_min,
            digital_max,
            prefiltering: String::new(),
            samples_per_record,
            reserved: String::new(),
        }
    }

    fn validate(&self, index: usize, sample_size_bytes: usize) -> Result<()> {
        if self.samples_per_record == 0 {
            return Err(EdfError::InvalidSignalField {
                index,
                field: "samples_per_record",
                reason: "must be > 0".to_string(),
            });
        }
        if self.digital_min >= self.digital_max {
            return Err(EdfError::InvalidDigitalRange {
                index,
                min: self.digital_min,
                max: self.digital_max,
            });
        }
        if (self.physical_min - self.physical_max).abs() < f64::EPSILON {
            return Err(EdfError::InvalidPhysicalRange {
                index,
                min: self.physical_min,
                max: self.physical_max,
            });
        }
        let (dmin, dmax) = digital_bounds(sample_size_bytes);
        if (self.digital_min as i64) < dmin || (self.digital_max as i64) > dmax {
            return Err(EdfError::InvalidSignalField {
                index,
                field: "digital_min/max",
                reason: format!(
                    "digital range [{}, {}] outside {}-bit signed range [{dmin}, {dmax}]",
                    self.digital_min, self.digital_max,
                    sample_size_bytes * 8,
                ),
            });
        }
        Ok(())
    }

    fn gain_offset(&self) -> (f64, f64) {
        let gain = (self.physical_max - self.physical_min)
            / (self.digital_max as f64 - self.digital_min as f64);
        let offset = self.physical_min - gain * self.digital_min as f64;
        (gain, offset)
    }
}

/// Top-level write spec. Used both as the input to [`write_edf`] and to seed an
/// [`EdfWriter`].
#[derive(Debug, Clone)]
pub struct WriterSpec {
    pub variant: EdfVariant,
    pub patient_id: String,
    pub recording_id: String,
    pub start_datetime: NaiveDateTime,
    pub record_duration_secs: f64,
    pub signals: Vec<WriterSignal>,
    /// Minimum bytes reserved for the annotation channel per record.
    /// Ignored for plain EDF/BDF. Defaults to 120.
    pub annotation_bytes_per_record: Option<usize>,
}

impl WriterSpec {
    pub fn new(variant: EdfVariant, record_duration_secs: f64) -> Self {
        WriterSpec {
            variant,
            patient_id: String::from("X X X X"),
            recording_id: String::from("Startdate X X X X"),
            start_datetime: NaiveDateTime::default(),
            record_duration_secs,
            signals: Vec::new(),
            annotation_bytes_per_record: None,
        }
    }
}

/// Streaming EDF/BDF writer.
pub struct EdfWriter {
    path: PathBuf,
    inner: Option<BufWriter<File>>,
    spec: WriterSpec,
    /// Resolved annotation channel byte budget per record (0 for non-plus).
    ann_bytes_per_record: usize,
    /// Accumulated record count, written into the header on `finish`.
    num_records_written: u64,
    /// Annotations queued via `add_annotation` but not yet attached to a record.
    pending_annotations: Vec<Annotation>,
    /// Sub-second component of the recording start time, encoded into the first
    /// time-keeping annotation. Computed once at create() time.
    start_subsecond: f64,
    /// Header byte count, kept for the seek-back num_records patch.
    header_bytes: usize,
    finished: bool,
}

impl EdfWriter {
    /// Open `path` for writing and emit an initial header with `num_records = -1`.
    /// On `finish()`, seeks back and patches `num_records` with the true count.
    pub fn create(path: impl AsRef<Path>, spec: WriterSpec) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        if spec.signals.is_empty() {
            return Err(EdfError::NoSignals);
        }
        if !(spec.record_duration_secs > 0.0) {
            return Err(EdfError::InvalidArgument {
                name: "record_duration_secs",
                reason: format!("must be > 0, got {}", spec.record_duration_secs),
            });
        }
        let sample_size = spec.variant.sample_size_bytes();
        for (i, sig) in spec.signals.iter().enumerate() {
            sig.validate(i, sample_size)?;
            if sig.label.starts_with("EDF Annotations") {
                return Err(EdfError::InvalidSignalField {
                    index: i,
                    field: "label",
                    reason: "user signals must not be labelled 'EDF Annotations'; \
                             the writer adds the annotation channel automatically"
                        .to_string(),
                });
            }
        }

        let ann_bytes_per_record = if spec.variant.is_plus() {
            let bytes = spec
                .annotation_bytes_per_record
                .unwrap_or(DEFAULT_MIN_ANNOTATION_BYTES);
            // Round up to a multiple of sample_size.
            ((bytes + sample_size - 1) / sample_size) * sample_size
        } else {
            0
        };

        let start_subsecond = {
            let nanos = spec.start_datetime.and_utc().timestamp_subsec_nanos();
            (nanos as f64) / 1_000_000_000.0
        };

        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| EdfError::FileOpen {
                path: path.clone(),
                source: e,
            })?;

        let mut writer = BufWriter::new(file);

        let header_bytes = serialize_header(
            &spec,
            ann_bytes_per_record,
            /* num_records */ -1,
            &mut writer,
        )?;

        Ok(EdfWriter {
            path,
            inner: Some(writer),
            spec,
            ann_bytes_per_record,
            num_records_written: 0,
            pending_annotations: Vec::new(),
            start_subsecond,
            header_bytes,
            finished: false,
        })
    }

    /// Number of user-defined signals (excluding the auto-added annotation channel).
    pub fn num_signals(&self) -> usize {
        self.spec.signals.len()
    }

    /// Sample size in bytes (2 for EDF, 3 for BDF).
    pub fn sample_size_bytes(&self) -> usize {
        self.spec.variant.sample_size_bytes()
    }

    /// Queue an annotation to be emitted with the next call to `write_record` /
    /// `write_record_physical`. Useful for live-capture flows where annotations
    /// arrive between record boundaries.
    pub fn add_annotation(&mut self, ann: Annotation) {
        self.pending_annotations.push(ann);
    }

    /// Write one record from physical (f64) values. Each inner slice must have
    /// exactly `signals[i].samples_per_record` values.
    ///
    /// Pending annotations queued via `add_annotation` are flushed into this
    /// record's annotation channel.
    pub fn write_record_physical(&mut self, physical: &[&[f64]]) -> Result<()> {
        let extra = std::mem::take(&mut self.pending_annotations);
        self.write_record_with_annotations(physical, &extra)
    }

    /// Write one record from physical values, with explicit annotations to embed
    /// into this record's annotation channel (in addition to any pending ones).
    pub fn write_record_with_annotations(
        &mut self,
        physical: &[&[f64]],
        annotations: &[Annotation],
    ) -> Result<()> {
        if self.finished {
            return Err(EdfError::InvalidArgument {
                name: "writer",
                reason: "writer has been finished".to_string(),
            });
        }
        if physical.len() != self.spec.signals.len() {
            return Err(EdfError::InvalidArgument {
                name: "physical",
                reason: format!(
                    "expected {} signals, got {}",
                    self.spec.signals.len(),
                    physical.len()
                ),
            });
        }

        let sample_size = self.sample_size_bytes();
        let writer = self.inner.as_mut().expect("writer open");

        for (i, sig) in self.spec.signals.iter().enumerate() {
            let samples = physical[i];
            if samples.len() != sig.samples_per_record {
                return Err(EdfError::InvalidArgument {
                    name: "physical",
                    reason: format!(
                        "signal {i}: expected {} samples, got {}",
                        sig.samples_per_record,
                        samples.len()
                    ),
                });
            }
            let (gain, offset) = sig.gain_offset();
            let dmin = sig.digital_min;
            let dmax = sig.digital_max;

            for &phys in samples {
                let mut digital = ((phys - offset) / gain).round() as i64;
                if digital < dmin as i64 {
                    digital = dmin as i64;
                }
                if digital > dmax as i64 {
                    digital = dmax as i64;
                }
                write_sample(writer, digital as i32, sample_size)?;
            }
        }

        // Drain pending into a combined slice (preserving caller-supplied first).
        let mut combined: Vec<&Annotation> = Vec::with_capacity(annotations.len());
        for a in annotations {
            combined.push(a);
        }
        // Pull any remaining pending added _during_ this call (rare but possible).
        let pending = std::mem::take(&mut self.pending_annotations);
        let pending_refs: Vec<&Annotation> = pending.iter().collect();
        combined.extend(pending_refs);

        if self.spec.variant.is_plus() {
            write_annotation_channel(
                writer,
                self.num_records_written,
                self.spec.record_duration_secs,
                self.start_subsecond,
                &combined,
                self.ann_bytes_per_record,
            )?;
        } else if !combined.is_empty() {
            return Err(EdfError::InvalidArgument {
                name: "annotations",
                reason: "annotations not supported for plain EDF/BDF; use a +C/+D variant"
                    .to_string(),
            });
        }

        self.num_records_written += 1;
        Ok(())
    }

    /// Finalize the file: flush, seek back, and patch `num_records` in the header.
    pub fn finish(mut self) -> Result<()> {
        self.finish_in_place()
    }

    fn finish_in_place(&mut self) -> Result<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let mut writer = self.inner.take().expect("writer open");
        writer.flush().map_err(|e| EdfError::FileOpen {
            path: self.path.clone(),
            source: e,
        })?;
        let mut file = writer.into_inner().map_err(|e| EdfError::FileOpen {
            path: self.path.clone(),
            source: e.into_error(),
        })?;
        file.seek(SeekFrom::Start(236)).map_err(|e| EdfError::FileOpen {
            path: self.path.clone(),
            source: e,
        })?;
        let nrec_field = format_ascii_field(&self.num_records_written.to_string(), 8);
        file.write_all(&nrec_field).map_err(|e| EdfError::FileOpen {
            path: self.path.clone(),
            source: e,
        })?;
        file.flush().map_err(|e| EdfError::FileOpen {
            path: self.path.clone(),
            source: e,
        })?;
        // suppress unused field warning
        let _ = self.header_bytes;
        Ok(())
    }
}

impl Drop for EdfWriter {
    fn drop(&mut self) {
        if !self.finished {
            // Best-effort finalize on drop. Errors are suppressed because
            // panicking from Drop is unsound; callers should call `finish`
            // explicitly to surface IO errors.
            let _ = self.finish_in_place();
        }
    }
}

/// One-shot helper: write a complete file from a fully-populated spec, physical
/// sample data, and annotations.
///
/// `data[i]` must have length `num_records * signals[i].samples_per_record`,
/// where `num_records` is consistent across all signals (inferred from the
/// shortest signal, but all must agree). Annotations are distributed across
/// records by their onset.
pub fn write_edf(
    path: impl AsRef<Path>,
    spec: WriterSpec,
    data: &[&[f64]],
    annotations: &[Annotation],
) -> Result<()> {
    if data.len() != spec.signals.len() {
        return Err(EdfError::InvalidArgument {
            name: "data",
            reason: format!(
                "expected {} signal arrays, got {}",
                spec.signals.len(),
                data.len()
            ),
        });
    }

    let mut num_records: Option<usize> = None;
    for (i, sig) in spec.signals.iter().enumerate() {
        if sig.samples_per_record == 0 {
            return Err(EdfError::InvalidSignalField {
                index: i,
                field: "samples_per_record",
                reason: "must be > 0".to_string(),
            });
        }
        if data[i].len() % sig.samples_per_record != 0 {
            return Err(EdfError::InvalidArgument {
                name: "data",
                reason: format!(
                    "signal {i}: data length {} not a multiple of samples_per_record {}",
                    data[i].len(),
                    sig.samples_per_record
                ),
            });
        }
        let nr = data[i].len() / sig.samples_per_record;
        match num_records {
            None => num_records = Some(nr),
            Some(prev) if prev == nr => {}
            Some(prev) => {
                return Err(EdfError::InvalidArgument {
                    name: "data",
                    reason: format!(
                        "signal {i}: implies {nr} records, but earlier signals imply {prev}"
                    ),
                });
            }
        }
    }
    let num_records = num_records.unwrap_or(0);

    let mut writer = EdfWriter::create(path, spec)?;
    let signals_count = writer.spec.signals.len();
    let record_dur = writer.spec.record_duration_secs;

    // Group annotations by record window for distribution.
    let mut by_record: Vec<Vec<Annotation>> = (0..num_records).map(|_| Vec::new()).collect();
    for ann in annotations {
        let r = (ann.onset / record_dur).floor() as i64;
        let r = r.max(0) as usize;
        if r < num_records {
            by_record[r].push(ann.clone());
        } else if num_records > 0 {
            // Stash trailing annotations into the last record so they aren't lost.
            by_record[num_records - 1].push(ann.clone());
        } else {
            return Err(EdfError::InvalidArgument {
                name: "annotations",
                reason: "cannot write annotations to a file with zero records".to_string(),
            });
        }
    }

    for r in 0..num_records {
        let mut row: Vec<&[f64]> = Vec::with_capacity(signals_count);
        for i in 0..signals_count {
            let spr = writer.spec.signals[i].samples_per_record;
            let start = r * spr;
            row.push(&data[i][start..start + spr]);
        }
        writer.write_record_with_annotations(&row, &by_record[r])?;
    }

    writer.finish()?;
    Ok(())
}

/// Digital range bounds for `n` byte-wide signed samples. Returns (min, max).
fn digital_bounds(sample_size_bytes: usize) -> (i64, i64) {
    match sample_size_bytes {
        2 => (i16::MIN as i64, i16::MAX as i64),
        3 => (-(1i64 << 23), (1i64 << 23) - 1),
        _ => unreachable!("invalid sample size"),
    }
}

/// Format `value` as ASCII left-justified, space-padded, exactly `size` bytes.
/// Truncates if too long.
fn format_ascii_field(value: &str, size: usize) -> Vec<u8> {
    let bytes = value.as_bytes();
    let mut out = vec![b' '; size];
    let n = bytes.len().min(size);
    out[..n].copy_from_slice(&bytes[..n]);
    out
}

/// Format an f64 fitting into `size` bytes of ASCII. Tries decreasing precision
/// until it fits; falls back to a truncated representation.
fn format_f64_field(value: f64, size: usize) -> Vec<u8> {
    // Plain integer if possible.
    if value.fract() == 0.0 && value.abs() < 1e15 {
        let s = format!("{}", value as i64);
        if s.len() <= size {
            return format_ascii_field(&s, size);
        }
    }
    for precision in (0..=8).rev() {
        let s = format!("{:.*}", precision, value);
        if s.len() <= size {
            return format_ascii_field(&s, size);
        }
    }
    // Last resort: truncate.
    let s = format!("{}", value);
    format_ascii_field(&s, size)
}

fn serialize_header<W: Write>(
    spec: &WriterSpec,
    ann_bytes_per_record: usize,
    num_records: i64,
    w: &mut W,
) -> Result<usize> {
    let n_user = spec.signals.len();
    let n_total = if spec.variant.is_plus() {
        n_user + 1
    } else {
        n_user
    };
    let header_bytes = 256 + 256 * n_total;

    let mut main = Vec::with_capacity(256);

    // version (8): 0 for EDF, 0xFF + "BIOSEMI" + space for BDF
    match spec.variant {
        EdfVariant::Bdf | EdfVariant::BdfPlusC | EdfVariant::BdfPlusD => {
            main.push(0xFF);
            main.extend_from_slice(b"BIOSEMI");
        }
        _ => main.extend_from_slice(b"0       "),
    }
    debug_assert_eq!(main.len(), 8);

    main.extend(format_ascii_field(&spec.patient_id, 80));
    main.extend(format_ascii_field(&spec.recording_id, 80));

    // start_date (8) DD.MM.YY (year clipping per EDF: 85-99 = 1985-1999, else 2000-2084)
    let dt = spec.start_datetime;
    let yy = dt.year().rem_euclid(100);
    let date_str = format!("{:02}.{:02}.{:02}", dt.day(), dt.month(), yy);
    main.extend(format_ascii_field(&date_str, 8));

    let time_str = format!("{:02}.{:02}.{:02}", dt.hour(), dt.minute(), dt.second());
    main.extend(format_ascii_field(&time_str, 8));

    // header_bytes (8)
    main.extend(format_ascii_field(&header_bytes.to_string(), 8));

    // reserved (44) — encodes the variant for + files
    let reserved = match spec.variant {
        EdfVariant::Edf => "",
        EdfVariant::EdfPlusC => "EDF+C",
        EdfVariant::EdfPlusD => "EDF+D",
        EdfVariant::Bdf => "24BIT",
        EdfVariant::BdfPlusC => "BDF+C",
        EdfVariant::BdfPlusD => "BDF+D",
    };
    main.extend(format_ascii_field(reserved, 44));

    // num_records (8)
    main.extend(format_ascii_field(&num_records.to_string(), 8));

    // record_duration (8)
    main.extend(format_f64_field(spec.record_duration_secs, 8));

    // num_signals (4)
    main.extend(format_ascii_field(&n_total.to_string(), 4));

    debug_assert_eq!(main.len(), 256);
    w.write_all(&main).map_err(io_err)?;

    // Per-signal header is stored transposed: all labels together, then all transducers, etc.
    let mut all_signals: Vec<WriterSignal> = spec.signals.clone();
    if spec.variant.is_plus() {
        all_signals.push(annotation_signal(spec, ann_bytes_per_record));
    }

    let labels: Vec<_> = all_signals.iter().map(|s| s.label.clone()).collect();
    let transducers: Vec<_> = all_signals.iter().map(|s| s.transducer.clone()).collect();
    let dims: Vec<_> = all_signals.iter().map(|s| s.physical_dimension.clone()).collect();
    let pmins: Vec<_> = all_signals.iter().map(|s| s.physical_min).collect();
    let pmaxs: Vec<_> = all_signals.iter().map(|s| s.physical_max).collect();
    let dmins: Vec<_> = all_signals.iter().map(|s| s.digital_min).collect();
    let dmaxs: Vec<_> = all_signals.iter().map(|s| s.digital_max).collect();
    let prefilters: Vec<_> = all_signals.iter().map(|s| s.prefiltering.clone()).collect();
    let nsamps: Vec<_> = all_signals.iter().map(|s| s.samples_per_record).collect();
    let reserveds: Vec<_> = all_signals.iter().map(|s| s.reserved.clone()).collect();

    write_field_block(w, &labels, 16, |s| s.as_bytes().to_vec())?;
    write_field_block(w, &transducers, 80, |s| s.as_bytes().to_vec())?;
    write_field_block(w, &dims, 8, |s| s.as_bytes().to_vec())?;
    write_field_block(w, &pmins, 8, |v| format_f64_field(*v, 8))?;
    write_field_block(w, &pmaxs, 8, |v| format_f64_field(*v, 8))?;
    write_field_block(w, &dmins, 8, |v| v.to_string().into_bytes())?;
    write_field_block(w, &dmaxs, 8, |v| v.to_string().into_bytes())?;
    write_field_block(w, &prefilters, 80, |s| s.as_bytes().to_vec())?;
    write_field_block(w, &nsamps, 8, |v| v.to_string().into_bytes())?;
    write_field_block(w, &reserveds, 32, |s| s.as_bytes().to_vec())?;

    Ok(header_bytes)
}

fn write_field_block<W: Write, T, F: Fn(&T) -> Vec<u8>>(
    w: &mut W,
    values: &[T],
    size: usize,
    fmt: F,
) -> Result<()> {
    for v in values {
        let bytes = fmt(v);
        let s = std::str::from_utf8(&bytes).unwrap_or("");
        let field = format_ascii_field(s, size);
        w.write_all(&field).map_err(io_err)?;
    }
    Ok(())
}

fn annotation_signal(spec: &WriterSpec, ann_bytes_per_record: usize) -> WriterSignal {
    let sample_size = spec.variant.sample_size_bytes();
    let samples_per_record = ann_bytes_per_record / sample_size;
    let (dmin, dmax) = digital_bounds(sample_size);
    WriterSignal {
        label: "EDF Annotations".to_string(),
        transducer: String::new(),
        physical_dimension: String::new(),
        physical_min: -1.0,
        physical_max: 1.0,
        digital_min: dmin as i32,
        digital_max: dmax as i32,
        prefiltering: String::new(),
        samples_per_record,
        reserved: String::new(),
    }
}

fn write_sample<W: Write>(w: &mut W, value: i32, sample_size: usize) -> Result<()> {
    match sample_size {
        2 => {
            let v = value as i16;
            w.write_all(&v.to_le_bytes()).map_err(io_err)?;
        }
        3 => {
            let bytes = [
                (value & 0xFF) as u8,
                ((value >> 8) & 0xFF) as u8,
                ((value >> 16) & 0xFF) as u8,
            ];
            w.write_all(&bytes).map_err(io_err)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// TAL byte values
const TAL_SEPARATOR: u8 = 0x14;
const TAL_DURATION_MARKER: u8 = 0x15;
const TAL_TERMINATOR: u8 = 0x00;

fn write_annotation_channel<W: Write>(
    w: &mut W,
    record_idx: u64,
    record_duration: f64,
    start_subsecond: f64,
    annotations: &[&Annotation],
    byte_budget: usize,
) -> Result<()> {
    let mut buf = Vec::with_capacity(byte_budget);

    // Time-keeping TAL: +<record_onset>\x14\x14\x00
    let onset = record_idx as f64 * record_duration + start_subsecond;
    buf.extend_from_slice(format_tal_onset(onset).as_bytes());
    buf.push(TAL_SEPARATOR);
    buf.push(TAL_SEPARATOR);
    buf.push(TAL_TERMINATOR);

    for ann in annotations {
        let mut tal = Vec::new();
        tal.extend_from_slice(format_tal_onset(ann.onset + start_subsecond).as_bytes());
        if let Some(dur) = ann.duration {
            tal.push(TAL_DURATION_MARKER);
            tal.extend_from_slice(format_tal_duration(dur).as_bytes());
        }
        tal.push(TAL_SEPARATOR);
        tal.extend_from_slice(ann.text.as_bytes());
        tal.push(TAL_SEPARATOR);
        tal.push(TAL_TERMINATOR);

        if buf.len() + tal.len() > byte_budget {
            return Err(EdfError::InvalidArgument {
                name: "annotation_bytes_per_record",
                reason: format!(
                    "annotation channel overflow at record {record_idx}: \
                     budget {byte_budget} bytes; increase \
                     `annotation_bytes_per_record` in the spec"
                ),
            });
        }
        buf.extend(tal);
    }

    if buf.len() > byte_budget {
        return Err(EdfError::InvalidArgument {
            name: "annotation_bytes_per_record",
            reason: format!(
                "annotation channel overflow at record {record_idx}: \
                 timekeeping TAL alone needs {} bytes (budget {byte_budget})",
                buf.len()
            ),
        });
    }
    buf.resize(byte_budget, 0);
    w.write_all(&buf).map_err(io_err)?;
    Ok(())
}

/// Format an annotation onset with a leading sign per EDF+ TAL spec.
fn format_tal_onset(onset: f64) -> String {
    let s = format_tal_number(onset.abs());
    if onset.is_sign_negative() {
        format!("-{s}")
    } else {
        format!("+{s}")
    }
}

fn format_tal_duration(d: f64) -> String {
    format_tal_number(d.max(0.0))
}

/// Format a non-negative f64 as a TAL-compliant number (digits, optional single
/// decimal point, no leading/trailing dot).
fn format_tal_number(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    let s = format!("{:.6}", v);
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed.ends_with('.') {
        return format!("{}", v as i64);
    }
    trimmed.to_string()
}

fn io_err(e: std::io::Error) -> EdfError {
    EdfError::FileOpen {
        path: PathBuf::new(),
        source: e,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::EdfFile;
    use chrono::{NaiveDate, NaiveTime};
    use tempfile::TempDir;

    fn sample_spec(variant: EdfVariant) -> WriterSpec {
        WriterSpec {
            variant,
            patient_id: "X X X X".into(),
            recording_id: "Startdate 01-JAN-2026 X X X".into(),
            start_datetime: NaiveDateTime::new(
                NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                NaiveTime::from_hms_opt(12, 0, 0).unwrap(),
            ),
            record_duration_secs: 1.0,
            signals: vec![WriterSignal::new("EEG Fpz", "uV", -3200.0, 3200.0, -32768, 32767, 256)],
            annotation_bytes_per_record: None,
        }
    }

    fn ramp(samples: usize) -> Vec<f64> {
        (0..samples).map(|i| (i as f64) * 0.5 - 100.0).collect()
    }

    #[test]
    fn roundtrip_plain_edf() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("plain.edf");

        let spec = sample_spec(EdfVariant::Edf);
        let data: Vec<f64> = ramp(256 * 5);
        write_edf(&path, spec.clone(), &[&data], &[]).unwrap();

        let f = EdfFile::open(&path).unwrap();
        assert_eq!(f.variant(), EdfVariant::Edf);
        assert_eq!(f.num_records(), 5);
        assert_eq!(f.num_signals(), 1);
        let sig = f.signal(0).unwrap();
        let mut read = vec![0.0f64; sig.len()];
        sig.read_physical(0, sig.len(), &mut read).unwrap();
        assert_eq!(read.len(), data.len());
        for (a, b) in read.iter().zip(data.iter()) {
            assert!((a - b).abs() < 0.2, "got {a} expected {b}");
        }
    }

    #[test]
    fn roundtrip_edf_plus_c_with_annotations() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("plus.edf");

        let spec = sample_spec(EdfVariant::EdfPlusC);
        let data: Vec<f64> = ramp(256 * 4);
        let anns = vec![
            Annotation { onset: 0.5, duration: None, text: "start".into() },
            Annotation { onset: 1.25, duration: Some(2.0), text: "spike".into() },
            Annotation { onset: 3.9, duration: None, text: "end".into() },
        ];
        write_edf(&path, spec, &[&data], &anns).unwrap();

        let f = EdfFile::open(&path).unwrap();
        assert_eq!(f.variant(), EdfVariant::EdfPlusC);
        assert_eq!(f.num_records(), 4);
        // Note: user signals only, not the annotation channel
        assert_eq!(f.ordinary_signal_indices().len(), 1);

        let got_anns = f.annotations();
        assert_eq!(got_anns.len(), 3);
        assert!((got_anns[0].onset - 0.5).abs() < 1e-6);
        assert_eq!(got_anns[0].text, "start");
        assert!((got_anns[1].onset - 1.25).abs() < 1e-6);
        assert_eq!(got_anns[1].duration, Some(2.0));
        assert_eq!(got_anns[2].text, "end");
    }

    #[test]
    fn roundtrip_bdf() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.bdf");

        let mut spec = sample_spec(EdfVariant::Bdf);
        spec.signals[0].digital_min = -8_388_608;
        spec.signals[0].digital_max = 8_388_607;
        let data: Vec<f64> = ramp(256 * 3);
        write_edf(&path, spec, &[&data], &[]).unwrap();

        let f = EdfFile::open(&path).unwrap();
        assert_eq!(f.variant(), EdfVariant::Bdf);
        assert_eq!(f.num_records(), 3);
        let sig = f.signal(0).unwrap();
        let mut read = vec![0.0f64; sig.len()];
        sig.read_physical(0, sig.len(), &mut read).unwrap();
        for (a, b) in read.iter().zip(data.iter()) {
            assert!((a - b).abs() < 0.001);
        }
    }

    #[test]
    fn streaming_writer() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("stream.edf");

        let spec = sample_spec(EdfVariant::EdfPlusC);
        let mut w = EdfWriter::create(&path, spec).unwrap();
        for r in 0..3 {
            let chunk: Vec<f64> = (0..256).map(|i| (r * 256 + i) as f64).collect();
            w.add_annotation(Annotation {
                onset: r as f64 + 0.1,
                duration: None,
                text: format!("rec{r}"),
            });
            w.write_record_physical(&[&chunk]).unwrap();
        }
        w.finish().unwrap();

        let f = EdfFile::open(&path).unwrap();
        assert_eq!(f.num_records(), 3);
        let anns = f.annotations();
        assert_eq!(anns.len(), 3);
        assert_eq!(anns[0].text, "rec0");
        assert_eq!(anns[2].text, "rec2");
    }

    #[test]
    fn edffile_write_to_roundtrip() {
        let dir = TempDir::new().unwrap();
        let src = dir.path().join("src.edf");
        let dst = dir.path().join("dst.edf");

        let spec = sample_spec(EdfVariant::EdfPlusC);
        let data: Vec<f64> = ramp(256 * 3);
        let anns = vec![
            Annotation { onset: 0.5, duration: None, text: "a".into() },
            Annotation { onset: 2.0, duration: Some(0.5), text: "b".into() },
        ];
        write_edf(&src, spec, &[&data], &anns).unwrap();

        let f = EdfFile::open(&src).unwrap();
        f.write_to(&dst, None).unwrap();

        let g = EdfFile::open(&dst).unwrap();
        assert_eq!(g.num_records(), 3);
        assert_eq!(g.variant(), EdfVariant::EdfPlusC);
        let got = g.annotations();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].text, "a");
        assert_eq!(got[1].duration, Some(0.5));
    }

    #[test]
    fn rejects_user_annotation_label() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.edf");
        let mut spec = sample_spec(EdfVariant::EdfPlusC);
        spec.signals[0].label = "EDF Annotations".into();
        let err = match EdfWriter::create(&path, spec) {
            Ok(_) => panic!("expected error"),
            Err(e) => e,
        };
        assert!(matches!(err, EdfError::InvalidSignalField { .. }));
    }

    #[test]
    fn rejects_annotations_in_plain_edf() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("plain.edf");
        let spec = sample_spec(EdfVariant::Edf);
        let data = vec![0.0f64; 256];
        let anns = vec![Annotation { onset: 0.0, duration: None, text: "x".into() }];
        let err = write_edf(&path, spec, &[&data], &anns).unwrap_err();
        assert!(matches!(err, EdfError::InvalidArgument { .. }));
    }

    #[test]
    fn annotation_overflow_errors() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("overflow.edf");
        let mut spec = sample_spec(EdfVariant::EdfPlusC);
        spec.annotation_bytes_per_record = Some(40); // very small
        let data = vec![0.0f64; 256 * 1];
        let big_text = "X".repeat(200);
        let anns = vec![Annotation { onset: 0.0, duration: None, text: big_text }];
        let err = write_edf(&path, spec, &[&data], &anns).unwrap_err();
        assert!(matches!(err, EdfError::InvalidArgument { .. }));
    }
}
