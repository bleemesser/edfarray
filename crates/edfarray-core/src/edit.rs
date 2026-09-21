use std::fs::{File, OpenOptions};
use std::hash::BuildHasher;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use chrono::{Datelike, NaiveDate, NaiveDateTime, Timelike};
use hmac::{Hmac, KeyInit, Mac};
use sha2::{Digest, Sha256};

use crate::error::{EdfError, Result};
use crate::file::EdfFile;
use crate::header::{EdfHeader, MaybeDate, Sex, read_usize};
use crate::mmap::ScanMode;
use crate::writer::format_ascii_field;

/// Header field offsets and widths (from the EDF specification).
const PATIENT_ID_OFFSET: u64 = 8;
const RECORDING_ID_OFFSET: u64 = 88;
const START_DATE_OFFSET: u64 = 168;
const START_TIME_OFFSET: u64 = 176;
const ID_FIELD_SIZE: usize = 80;
const DATE_FIELD_SIZE: usize = 8;

/// SHA-256 iterations applied when deriving a pseudonym. The attack this defends against is
/// an adversary who holds the seed and hashes a roster of candidate names looking for a match,
/// so the stretch has to cover the name, not just the seed. It costs about 20 ms per call,
/// paid once per file and invisible next to opening the recording, while putting a six-figure
/// multiplier on a dictionary sweep. It is a cost knob, not a secret: changing it changes
/// every pseudonym, so it cannot be raised without re-anonymizing a corpus.
const PSEUDONYM_ITERATIONS: u32 = 500_000;

/// Years representable in the two-digit startdate field under the EDF 85-pivot rule.
const MIN_YEAR: i32 = 1985;
const MAX_YEAR: i32 = 2084;

/// A set of header field replacements. `None` fields are left untouched.
#[derive(Debug, Clone, Default)]
pub struct HeaderEdit {
    pub patient_id: Option<String>,
    pub recording_id: Option<String>,
    pub start_datetime: Option<NaiveDateTime>,
}

/// A `(before, after)` pair for one edited header field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldChange {
    pub before: String,
    pub after: String,
}

/// What an edit actually changed. Fields whose new value equals the current value are omitted.
#[derive(Debug, Clone, Default)]
pub struct HeaderDiff {
    pub patient_id: Option<FieldChange>,
    pub recording_id: Option<FieldChange>,
    /// Raw `dd.mm.yy hh.mm.ss` header fields, before and after.
    pub start_datetime: Option<FieldChange>,
}

impl HeaderDiff {
    /// Whether the edit changed nothing.
    pub fn is_empty(&self) -> bool {
        self.patient_id.is_none() && self.recording_id.is_none() && self.start_datetime.is_none()
    }
}

/// Replace header identification fields in place.
///
/// Only the header block is written; record data is never touched. Returns the fields that
/// actually changed. An `EdfFile` handle opened before this call keeps its stale parsed header.
///
/// Every field is validated before the first byte is written, so a rejected edit leaves the
/// file exactly as it was.
pub fn edit_header(path: impl AsRef<Path>, edit: &HeaderEdit) -> Result<HeaderDiff> {
    let path = path.as_ref();
    let (mut file, buf, header) = open_header(path, true)?;
    let planned = plan_edit(&buf, &header, edit)?;
    planned.apply(&mut file, path)?;
    Ok(planned.diff)
}

/// A fully validated set of header writes. Constructing one runs every check, so applying it
/// fails only on I/O, never on a field that turns out to be unrepresentable partway through.
struct PlannedEdit {
    /// `(offset, value, field width)`, in the order they are written.
    writes: Vec<(u64, String, usize)>,
    diff: HeaderDiff,
}

impl PlannedEdit {
    fn apply(&self, file: &mut File, path: &Path) -> Result<()> {
        for (offset, value, size) in &self.writes {
            write_field_at(file, path, *offset, value, *size)?;
        }
        // File::flush is a no-op; only sync_data pushes the edit past the page cache. An
        // anonymizer that reports success while the original name is still the only copy on
        // disk is worse than one that is slow.
        file.sync_data()
            .map_err(|e| io_err(path, "syncing the header", e))?;
        Ok(())
    }
}

fn plan_edit(buf: &[u8], header: &EdfHeader, edit: &HeaderEdit) -> Result<PlannedEdit> {
    let mut planned = PlannedEdit {
        writes: Vec::new(),
        diff: HeaderDiff::default(),
    };

    if let Some(value) = &edit.patient_id {
        let value = normalize_id_field(value, "patient_id")?;
        if value != header.patient_id {
            planned
                .writes
                .push((PATIENT_ID_OFFSET, value.clone(), ID_FIELD_SIZE));
            planned.diff.patient_id = Some(FieldChange {
                before: header.patient_id.clone(),
                after: value,
            });
        }
    }

    if let Some(value) = &edit.recording_id {
        let value = normalize_id_field(value, "recording_id")?;
        if value != header.recording_id {
            planned
                .writes
                .push((RECORDING_ID_OFFSET, value.clone(), ID_FIELD_SIZE));
            planned.diff.recording_id = Some(FieldChange {
                before: header.recording_id.clone(),
                after: value,
            });
        }
    }

    if let Some(dt) = edit.start_datetime {
        validate_start_year(dt.year())?;
        let new_date = format!(
            "{:02}.{:02}.{:02}",
            dt.day(),
            dt.month(),
            dt.year().rem_euclid(100)
        );
        let new_time = format!("{:02}.{:02}.{:02}", dt.hour(), dt.minute(), dt.second());
        let before_date = raw_field(buf, START_DATE_OFFSET as usize, DATE_FIELD_SIZE);
        let before_time = raw_field(buf, START_TIME_OFFSET as usize, DATE_FIELD_SIZE);
        if new_date != before_date || new_time != before_time {
            planned
                .writes
                .push((START_DATE_OFFSET, new_date.clone(), DATE_FIELD_SIZE));
            planned
                .writes
                .push((START_TIME_OFFSET, new_time.clone(), DATE_FIELD_SIZE));
            planned.diff.start_datetime = Some(FieldChange {
                before: format!("{before_date} {before_time}"),
                after: format!("{new_date} {new_time}"),
            });
        }
    }

    Ok(planned)
}

/// Options for [`anonymize`].
#[derive(Debug, Clone)]
pub struct AnonymizeOptions {
    /// Seed for the pseudonym and date shift. The same `(seed, subject)` always yields the
    /// same pseudonym, keyed on the patient name, code, and birthdate, so every recording of
    /// one subject links across a corpus even when their per-session notes differ. `None` uses
    /// a per-process random seed: repeatable within one run, unlinkable across runs.
    ///
    /// Treat a reused seed as a secret. It is the re-identification key: anyone holding it can
    /// recompute the pseudonym for a guessed name and recover the date shift. Do not publish it
    /// alongside the files it anonymized.
    pub seed: Option<String>,
    /// Explicit patient pseudonym. Defaults to a hash-derived `Subject-XXXXXXXX`.
    pub pseudonym: Option<String>,
    /// Explicit calendar shift applied to all dates. Defaults to a seed-derived shift in
    /// ±10 years. Both birthdate and recording dates move by the same amount, preserving age.
    pub date_shift_days: Option<i32>,
    /// Keep the sex subfield. Recommended: sex is rarely identifying and widely useful.
    pub keep_sex: bool,
    /// Keep the hospital patient code. Defaults to false; codes often re-identify via the
    /// hospital registry.
    pub keep_code: bool,
    /// Keep the recording technician field. Defaults to false.
    pub keep_technician: bool,
    /// Keep the equipment field. Defaults to true; equipment is about the device, not the person.
    pub keep_equipment: bool,
    /// Keep free-text `additional` subfields in both ids. Defaults to false.
    pub keep_additional: bool,
    /// Compute everything, write nothing.
    pub dry_run: bool,
}

impl Default for AnonymizeOptions {
    fn default() -> Self {
        AnonymizeOptions {
            seed: None,
            pseudonym: None,
            date_shift_days: None,
            keep_sex: true,
            keep_code: false,
            keep_technician: false,
            keep_equipment: true,
            keep_additional: false,
            dry_run: false,
        }
    }
}

/// What an [`anonymize`] call did (or, with [`AnonymizeOptions::dry_run`], would do).
#[derive(Debug, Clone)]
pub struct AnonymizeResult {
    pub pseudonym: String,
    pub date_shift_days: i32,
    pub patient_id_before: String,
    pub patient_id_after: String,
    pub recording_id_before: String,
    pub recording_id_after: String,
    /// Raw `dd.mm.yy hh.mm.ss` header startdate, before and after.
    pub start_datetime_before: String,
    pub start_datetime_after: String,
    /// Identity tokens (`audit`-style terms) that were replaced. Pass them to
    /// [`audit_with_terms`] to confirm no copy survives in signal labels or annotation text,
    /// which header-only anonymization cannot rewrite.
    pub scrubbed_terms: Vec<String>,
    /// The header diff that was written. Empty on dry runs.
    pub diff: HeaderDiff,
}

/// Scrub patient and recording identification in place.
///
/// Patient name becomes a pseudonym; code, technician, admin code, and free-text subfields are
/// replaced with `X` unless kept. Birthdate and all recording dates are shifted by the same
/// number of days, so age and within-file time-of-day survive while calendar dates do not.
///
/// Does not touch signal labels or annotation text, which may repeat identifying strings.
/// Use the returned [`AnonymizeResult::scrubbed_terms`] with [`audit_with_terms`] as the
/// final gate before shipping.
pub fn anonymize(path: impl AsRef<Path>, opts: &AnonymizeOptions) -> Result<AnonymizeResult> {
    let path = path.as_ref();
    let (mut file, buf, header) = open_header(path, !opts.dry_run)?;
    let scrubbed_terms = identity_terms(&header);

    let seed = opts.seed.clone().unwrap_or_else(random_seed);
    let shift = opts
        .date_shift_days
        .unwrap_or_else(|| derive_shift_days(&seed));

    let identity = subject_identity(&header);
    let pseudonym = match &opts.pseudonym {
        Some(p) => normalize_id_field(p, "pseudonym")?,
        None => generate_pseudonym(&seed, &identity),
    };

    let sex = if opts.keep_sex {
        header.patient.sex
    } else {
        None
    };
    let code = if opts.keep_code {
        header.patient.code.clone()
    } else {
        None
    };
    let birthdate = header
        .patient
        .birthdate
        .as_ref()
        .and_then(|d| shift_maybe_date(d, shift));
    let patient_additional = if opts.keep_additional {
        header.patient.additional.clone()
    } else {
        None
    };
    let new_patient_id = build_patient_id(
        code.as_deref(),
        sex,
        birthdate.as_ref(),
        &pseudonym,
        patient_additional.as_deref(),
    );

    let recording_start = header
        .recording
        .start_date
        .as_ref()
        .and_then(|d| shift_maybe_date(d, shift));
    let technician = if opts.keep_technician {
        header.recording.technician.as_deref()
    } else {
        None
    };
    let equipment = if opts.keep_equipment {
        header.recording.equipment.as_deref()
    } else {
        None
    };
    let recording_additional = if opts.keep_additional {
        header.recording.additional.as_deref()
    } else {
        None
    };
    let new_recording_id = build_recording_id(
        recording_start.as_ref(),
        technician,
        equipment,
        recording_additional,
    );

    // Shift the binary startdate too; a raw (unparseable) one is left as it is.
    let new_start = match header.start_datetime.as_datetime() {
        Some(dt) => {
            let shifted = shift_datetime(*dt, shift);
            if !(MIN_YEAR..=MAX_YEAR).contains(&shifted.year()) {
                return Err(EdfError::InvalidArgument {
                    name: "date_shift_days",
                    reason: format!(
                        "shifting the start date by {shift} days lands on year {}, outside the \
                         {MIN_YEAR}-{MAX_YEAR} range representable in the EDF startdate field; \
                         pass an explicit date_shift_days",
                        shifted.year()
                    ),
                });
            }
            Some(shifted)
        }
        None => None,
    };

    let before_date = raw_field(&buf, START_DATE_OFFSET as usize, DATE_FIELD_SIZE);
    let before_time = raw_field(&buf, START_TIME_OFFSET as usize, DATE_FIELD_SIZE);
    let start_before = format!("{before_date} {before_time}");
    let start_after = match new_start {
        Some(dt) => format!(
            "{:02}.{:02}.{:02} {:02}.{:02}.{:02}",
            dt.day(),
            dt.month(),
            dt.year().rem_euclid(100),
            dt.hour(),
            dt.minute(),
            dt.second()
        ),
        None => start_before.clone(),
    };

    let edit = HeaderEdit {
        patient_id: Some(new_patient_id.clone()),
        recording_id: Some(new_recording_id.clone()),
        start_datetime: new_start,
    };
    // Plan unconditionally, so a dry run rejects exactly what the real call would reject.
    let planned = plan_edit(&buf, &header, &edit)?;
    let diff = if opts.dry_run {
        HeaderDiff::default()
    } else {
        planned.apply(&mut file, path)?;
        planned.diff
    };

    Ok(AnonymizeResult {
        pseudonym,
        date_shift_days: shift,
        patient_id_before: header.patient_id.clone(),
        patient_id_after: new_patient_id,
        recording_id_before: header.recording_id.clone(),
        recording_id_after: new_recording_id,
        start_datetime_before: start_before,
        start_datetime_after: start_after,
        scrubbed_terms,
        diff,
    })
}

/// One identified occurrence of an original identity string outside the id fields.
#[derive(Debug, Clone)]
pub struct LeakHit {
    /// Human-readable location, e.g. `signal 3 label` or `annotation 12.500s`.
    pub location: String,
    /// Signal index for header-field hits, `None` for annotation text.
    pub signal_index: Option<usize>,
    /// The matched identity term, lowercased.
    pub term: String,
    /// The full string that contained the term.
    pub excerpt: String,
}

/// Result of [`audit`]: identity terms derived from the patient fields, and everywhere else
/// they appear.
#[derive(Debug, Clone, Default)]
pub struct AuditReport {
    pub terms: Vec<String>,
    pub hits: Vec<LeakHit>,
}

impl AuditReport {
    /// No identity terms leaked outside the identification fields.
    pub fn is_clean(&self) -> bool {
        self.hits.is_empty()
    }
}

/// Derive identity terms (lowercased) from the patient name, patient code, technician, and
/// admin code subfields.
fn identity_terms(header: &EdfHeader) -> Vec<String> {
    let mut terms: Vec<String> = Vec::new();
    let mut add_tokens = |value: Option<&str>, min_len: usize| {
        if let Some(value) = value {
            for token in value.split([' ', '_']) {
                let t = token.trim().to_lowercase();
                if t.len() >= min_len && t != "x" && !terms.contains(&t) {
                    terms.push(t);
                }
            }
        }
    };
    add_tokens(header.patient.name.as_deref(), 3);
    add_tokens(header.patient.code.as_deref(), 4);
    add_tokens(header.recording.technician.as_deref(), 3);
    add_tokens(header.recording.admin_code.as_deref(), 4);
    terms
}

/// Scan signal labels, transducers, prefilters, and annotation text for strings that repeat
/// the patient identity currently stored in the header fields. Read-only.
///
/// Run this *before* anonymizing: once the header is scrubbed it no longer knows what the
/// original identity was. After anonymizing, re-check with [`audit_with_terms`] and the
/// [`AnonymizeResult::scrubbed_terms`] captured at anonymization time.
pub fn audit(path: impl AsRef<Path>) -> Result<AuditReport> {
    let file = EdfFile::open_with_options(path, None, ScanMode::Lazy)?;
    let terms = identity_terms(file.header());
    audit_with_terms(&file, &terms)
}

/// Scan with an explicit term list (case-insensitive substring match) instead of deriving
/// terms from the header. Terms shorter than two characters are ignored.
pub fn audit_terms(path: impl AsRef<Path>, terms: &[String]) -> Result<AuditReport> {
    let file = EdfFile::open_with_options(path, None, ScanMode::Lazy)?;
    audit_with_terms(&file, terms)
}

fn audit_with_terms(file: &EdfFile, terms: &[String]) -> Result<AuditReport> {
    let terms: Vec<String> = terms
        .iter()
        .map(|t| t.trim().to_lowercase())
        .filter(|t| t.len() >= 2)
        .collect();

    let mut report = AuditReport {
        terms: terms.clone(),
        hits: Vec::new(),
    };

    let mut check = |location: String, signal_index: Option<usize>, haystack: &str| {
        let lower = haystack.to_lowercase();
        for term in &terms {
            if lower.contains(term.as_str()) {
                report.hits.push(LeakHit {
                    location: location.clone(),
                    signal_index,
                    term: term.clone(),
                    excerpt: haystack.to_string(),
                });
            }
        }
    };

    let header = file.header();
    for (i, sh) in header.signals.iter().enumerate() {
        check(format!("signal {i} label"), Some(i), &sh.label);
        check(format!("signal {i} transducer"), Some(i), &sh.transducer);
        check(
            format!("signal {i} prefiltering"),
            Some(i),
            &sh.prefiltering,
        );
    }
    for ann in file.annotations() {
        check(format!("annotation {:.3}s", ann.onset), None, &ann.text);
    }

    Ok(report)
}

fn open_header(path: &Path, write: bool) -> Result<(File, Vec<u8>, EdfHeader)> {
    let mut options = OpenOptions::new();
    options.read(true);
    if write {
        options.write(true);
    }
    let mut file = options.open(path).map_err(|e| EdfError::FileOpen {
        path: path.to_path_buf(),
        source: e,
    })?;

    let mut buf = vec![b' '; 256];
    file.read_exact(&mut buf)
        .map_err(|e| io_err(path, "reading the header", e))?;
    let num_signals = read_usize(&buf, 252, 4, "num_signals")?;
    if num_signals == 0 {
        return Err(EdfError::NoSignals);
    }
    let header_bytes = 256 + 256 * num_signals;
    buf.resize(header_bytes, b' ');
    file.read_exact(&mut buf[256..])
        .map_err(|e| io_err(path, "reading the signal headers", e))?;

    let header = EdfHeader::parse(&buf)?;
    Ok((file, buf, header))
}

/// Trim and reject anything that cannot sit in a fixed-width ASCII header field.
fn normalize_id_field(value: &str, name: &'static str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed.len() > ID_FIELD_SIZE {
        return Err(EdfError::InvalidArgument {
            name,
            reason: format!(
                "replacement is {} bytes, which does not fit the {ID_FIELD_SIZE}-byte EDF field",
                trimmed.len()
            ),
        });
    }
    for ch in trimmed.chars() {
        if !(ch.is_ascii_graphic() || ch == ' ') {
            return Err(EdfError::InvalidArgument {
                name,
                reason: format!(
                    "only printable ASCII is allowed in header fields; found {:?}",
                    ch
                ),
            });
        }
    }
    Ok(trimmed.to_string())
}

fn validate_start_year(year: i32) -> Result<()> {
    if !(MIN_YEAR..=MAX_YEAR).contains(&year) {
        return Err(EdfError::InvalidArgument {
            name: "start_datetime",
            reason: format!(
                "year {year} cannot be represented in the EDF startdate field; \
                 must be within {MIN_YEAR}..={MAX_YEAR}"
            ),
        });
    }
    Ok(())
}

fn write_field_at(
    file: &mut File,
    path: &Path,
    offset: u64,
    value: &str,
    size: usize,
) -> Result<()> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| io_err(path, "seeking in the header", e))?;
    file.write_all(&format_ascii_field(value, size))
        .map_err(|e| io_err(path, "writing the header", e))?;
    Ok(())
}

fn raw_field(buf: &[u8], offset: usize, size: usize) -> String {
    String::from_utf8_lossy(&buf[offset..offset + size])
        .trim()
        .to_string()
}

/// `code sex birthdate name [additional]` with `X` for unknowns and underscores for spaces,
/// the EDF+ patient id layout.
fn build_patient_id(
    code: Option<&str>,
    sex: Option<Sex>,
    birthdate: Option<&MaybeDate>,
    name: &str,
    additional: Option<&str>,
) -> String {
    let sex_str = match sex {
        Some(Sex::Male) => "M",
        Some(Sex::Female) => "F",
        None => "X",
    };
    let mut id = format!(
        "{} {sex_str} {} {}",
        token(code),
        maybe_date_token(birthdate),
        token(Some(name)),
    );
    push_additional(&mut id, additional);
    id
}

/// `Startdate date admincode technician equipment [additional]`, the EDF+ recording id layout.
/// The admin code is always `X`; callers anonymize through [`anonymize`], and there is no
/// option to keep it (it is an institution-registry pointer).
fn build_recording_id(
    start_date: Option<&MaybeDate>,
    technician: Option<&str>,
    equipment: Option<&str>,
    additional: Option<&str>,
) -> String {
    let mut id = format!(
        "Startdate {} X {} {}",
        maybe_date_token(start_date),
        token(technician),
        token(equipment),
    );
    push_additional(&mut id, additional);
    id
}

fn push_additional(id: &mut String, additional: Option<&str>) {
    if let Some(a) = additional.map(|a| a.trim())
        && !a.is_empty()
    {
        id.push(' ');
        id.push_str(&a.replace(' ', "_"));
    }
}

fn token(value: Option<&str>) -> String {
    value
        .map(|v| v.trim().replace(' ', "_"))
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| "X".to_string())
}

fn maybe_date_token(date: Option<&MaybeDate>) -> String {
    match date {
        Some(MaybeDate::Parsed(d)) => format_edf_plus_date(d),
        Some(MaybeDate::Raw(s)) => s.clone(),
        None => "X".to_string(),
    }
}

fn format_edf_plus_date(d: &NaiveDate) -> String {
    let months = [
        "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    ];
    format!(
        "{:02}-{}-{:04}",
        d.day(),
        months[(d.month0() as usize).min(11)],
        d.year()
    )
}

fn shift_date(d: NaiveDate, days: i32) -> Option<NaiveDate> {
    NaiveDate::from_num_days_from_ce_opt(d.num_days_from_ce() + days)
}

fn shift_maybe_date(date: &MaybeDate, days: i32) -> Option<MaybeDate> {
    match date {
        MaybeDate::Parsed(d) => shift_date(*d, days).map(MaybeDate::Parsed),
        // An unparseable date cannot be shifted, and emitting it verbatim would leak the
        // original. Dropping it (the caller renders `None` as `X`) is the safe failure.
        MaybeDate::Raw(_) => None,
    }
}

fn shift_datetime(dt: NaiveDateTime, days: i32) -> NaiveDateTime {
    let date = shift_date(dt.date(), days).unwrap_or(dt.date());
    date.and_time(dt.time())
}

/// The stable part of a subject's identity, used to key the pseudonym.
///
/// Deliberately excludes the free-text `additional` subfield and the raw field spacing: those
/// carry per-session notes that differ between recordings of one subject, and keying on them
/// would hand the same person a different pseudonym in every file.
fn subject_identity(header: &EdfHeader) -> String {
    let name = header.patient.name.as_deref().unwrap_or("").trim();
    let code = header.patient.code.as_deref().unwrap_or("").trim();
    let birth = match &header.patient.birthdate {
        Some(MaybeDate::Parsed(d)) => format_edf_plus_date(d),
        Some(MaybeDate::Raw(s)) => s.trim().to_string(),
        None => String::new(),
    };
    // Fall back to the whole field only when EDF+ subfield parsing found nothing to key on.
    if name.is_empty() && code.is_empty() && birth.is_empty() {
        return header.patient_id.trim().to_lowercase();
    }
    format!(
        "{}\u{1}{}\u{1}{}",
        name.to_lowercase(),
        code.to_lowercase(),
        birth.to_lowercase()
    )
}

/// HMAC-SHA256 keyed on `seed`, stretched over [`PSEUDONYM_ITERATIONS`], base32-encoded to an
/// 8-character suffix.
///
/// Keying on the seed means an adversary without it learns nothing; stretching means one who
/// has it still pays [`PSEUDONYM_ITERATIONS`] hashes per candidate name they want to test.
fn generate_pseudonym(seed: &str, subject: &str) -> String {
    let mut digest = hmac_sha256(seed.as_bytes(), b"pseudonym", subject.as_bytes());
    for _ in 0..PSEUDONYM_ITERATIONS {
        digest = Sha256::digest(digest).into();
    }
    const ALPHABET: &[u8; 32] = b"0123456789ABCDEFGHJKMNPQRSTVWXYZ";
    let mut value = u64::from_le_bytes(digest[..8].try_into().unwrap());
    let mut suffix = String::with_capacity(8);
    for _ in 0..8 {
        suffix.push(ALPHABET[(value & 0x1F) as usize] as char);
        value >>= 5;
    }
    format!("Subject-{suffix}")
}

/// Deliberately not stretched: the shift has only `span` possible values, so an attacker can
/// try them all regardless of how the seed is hashed. Stretching would buy nothing here and
/// would double the cost of every call.
fn derive_shift_days(seed: &str) -> i32 {
    let digest = hmac_sha256(seed.as_bytes(), b"dates", b"");
    // +/-10 years, skipping 0 (a zero shift would leave dates untouched).
    let span = 3652i64;
    let raw = i64::from_le_bytes(digest[..8].try_into().unwrap()) % span;
    if raw == 0 { 1 } else { raw as i32 }
}

/// Domain-separated HMAC, so the pseudonym and the date shift cannot be derived from one
/// another even though they share a seed.
fn hmac_sha256(key: &[u8], domain: &[u8], message: &[u8]) -> [u8; 32] {
    let mut mac = Hmac::<Sha256>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(domain);
    mac.update(&[0x01]);
    mac.update(message);
    mac.finalize().into_bytes().into()
}

fn random_seed() -> String {
    let a = std::hash::RandomState::new().hash_one("edfarray-anon-a");
    let b = std::hash::RandomState::new().hash_one("edfarray-anon-b");
    format!("{a:016x}{b:016x}")
}

fn io_err(path: &std::path::Path, op: &'static str, source: std::io::Error) -> EdfError {
    EdfError::Io {
        path: path.to_path_buf(),
        op,
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotation::Annotation;
    use crate::header::EdfVariant;
    use crate::writer::{WriterSignal, WriterSpec, write_edf};
    use chrono::{NaiveDate, NaiveTime};
    use tempfile::{TempDir, tempdir};

    fn build_file(dir: &TempDir, name: &str) -> std::path::PathBuf {
        let path = dir.path().join(name);
        let spec = WriterSpec {
            variant: EdfVariant::EdfPlusC,
            patient_id: "MCH-0234567 F 02-MAR-1951 Haagansen_Erlangen note_one".into(),
            recording_id: "Startdate 01-JAN-2026 PSG-9999 Jane_Tech Nihon_Kohden ward_b".into(),
            start_datetime: NaiveDateTime::new(
                NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                NaiveTime::from_hms_opt(12, 0, 0).unwrap(),
            ),
            record_duration_secs: 1.0,
            signals: vec![WriterSignal::new(
                "EEG Haagansen Fpz",
                "uV",
                -3200.0,
                3200.0,
                -32768,
                32767,
                64,
            )],
            annotation_bytes_per_record: None,
            record_onsets: None,
        };
        let data: Vec<f64> = (0..256).map(|i| (i as f64) * 0.25 - 30.0).collect();
        let anns = vec![Annotation {
            onset: 1.0,
            duration: None,
            text: "move artifacts: Haagansen restless".into(),
        }];
        write_edf(&path, spec, &[&data], &anns).unwrap();
        path
    }

    fn data_bytes(path: &Path) -> Vec<u8> {
        let all = std::fs::read(path).unwrap();
        all[(256 + 256 * 2)..].to_vec()
    }

    #[test]
    fn edit_header_replaces_fields_and_keeps_data() {
        let dir = tempdir().unwrap();
        let path = build_file(&dir, "edit.edf");
        let data_before = data_bytes(&path);

        let diff = edit_header(
            &path,
            &HeaderEdit {
                patient_id: Some("X X X Subject-001".into()),
                recording_id: None,
                start_datetime: None,
            },
        )
        .unwrap();

        assert!(diff.patient_id.is_some());
        assert!(diff.recording_id.is_none());
        let after = EdfFile::open(&path).unwrap();
        assert_eq!(after.header().patient_id, "X X X Subject-001");
        assert_eq!(after.patient().name.as_deref(), Some("Subject-001"));
        // Recording id and start datetime untouched.
        assert!(
            after
                .header()
                .recording_id
                .starts_with("Startdate 01-JAN-2026")
        );
        assert_eq!(data_bytes(&path), data_before);
    }

    #[test]
    fn edit_header_start_datetime_roundtrips() {
        let dir = tempdir().unwrap();
        let path = build_file(&dir, "dt.edf");
        let dt = NaiveDateTime::new(
            NaiveDate::from_ymd_opt(2020, 7, 4).unwrap(),
            NaiveTime::from_hms_opt(23, 59, 58).unwrap(),
        );
        let diff = edit_header(
            &path,
            &HeaderEdit {
                patient_id: None,
                recording_id: None,
                start_datetime: Some(dt),
            },
        )
        .unwrap();
        let change = diff.start_datetime.unwrap();
        assert_eq!(change.before, "01.01.26 12.00.00");
        assert_eq!(change.after, "04.07.20 23.59.58");

        let f = EdfFile::open(&path).unwrap();
        let got = f.header().start_datetime.as_datetime().unwrap();
        assert_eq!(*got, dt);
    }

    #[test]
    fn edit_rejects_oversized_and_non_ascii() {
        let dir = tempdir().unwrap();
        let path = build_file(&dir, "reject.edf");

        let err = edit_header(
            &path,
            &HeaderEdit {
                patient_id: Some("x".repeat(81)),
                recording_id: None,
                start_datetime: None,
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdfError::InvalidArgument {
                name: "patient_id",
                ..
            }
        ));

        let err = edit_header(
            &path,
            &HeaderEdit {
                patient_id: Some("badé name".into()),
                recording_id: None,
                start_datetime: None,
            },
        )
        .unwrap_err();
        assert!(matches!(err, EdfError::InvalidArgument { .. }));

        let err = edit_header(
            &path,
            &HeaderEdit {
                patient_id: None,
                recording_id: None,
                start_datetime: Some(
                    NaiveDate::from_ymd_opt(1900, 1, 1)
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap(),
                ),
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdfError::InvalidArgument {
                name: "start_datetime",
                ..
            }
        ));

        // None of the failed attempts modified the file.
        let f = EdfFile::open(&path).unwrap();
        assert!(f.header().patient_id.starts_with("MCH-0234567"));
    }

    #[test]
    fn no_op_edit_reports_no_changes() {
        let dir = tempdir().unwrap();
        let path = build_file(&dir, "noop.edf");
        let current = EdfFile::open(&path).unwrap().header().patient_id.clone();
        let diff = edit_header(
            &path,
            &HeaderEdit {
                patient_id: Some(current),
                recording_id: None,
                start_datetime: None,
            },
        )
        .unwrap();
        assert!(diff.is_empty());
    }

    #[test]
    fn anonymize_is_deterministic_per_seed() {
        let dir = tempdir().unwrap();
        let path = build_file(&dir, "anon.edf");
        let twin = build_file(&dir, "anon_twin.edf");
        let opts = AnonymizeOptions {
            seed: Some("study-x".into()),
            ..Default::default()
        };

        let a1 = anonymize(&path, &opts).unwrap();
        let f1 = EdfFile::open(&path).unwrap();
        let shift = a1.date_shift_days;
        assert!(shift != 0 && shift.abs() < 3652);
        assert_eq!(a1.pseudonym, f1.patient().name.clone().unwrap());
        assert!(a1.pseudonym.starts_with("Subject-"));
        assert!(!f1.header().patient_id.contains("Haagansen"));
        assert!(!f1.header().patient_id.contains("MCH-0234567"));
        assert!(!f1.header().recording_id.contains("Jane"));
        assert!(!f1.header().recording_id.contains("PSG-9999"));
        // Sex and equipment survive defaults.
        assert_eq!(f1.patient().sex, Some(Sex::Female));
        assert_eq!(f1.recording().equipment.as_deref(), Some("Nihon Kohden"));

        // Same seed, same original identity, separate file: same pseudonym and shift,
        // so one subject's recordings stay linkable across a corpus.
        let a2 = anonymize(&twin, &opts).unwrap();
        assert_eq!(a1.pseudonym, a2.pseudonym);
        assert_eq!(a1.date_shift_days, a2.date_shift_days);
    }

    #[test]
    fn anonymize_shifts_all_dates_equally() {
        let dir = tempdir().unwrap();
        let path = build_file(&dir, "dates.edf");
        let opts = AnonymizeOptions {
            seed: Some("s".into()),
            date_shift_days: Some(-400),
            ..Default::default()
        };
        anonymize(&path, &opts).unwrap();

        let f = EdfFile::open(&path).unwrap();
        let birth = f.patient().birthdate.as_ref().unwrap().as_date().unwrap();
        let rec_start = f
            .recording()
            .start_date
            .as_ref()
            .unwrap()
            .as_date()
            .unwrap();
        let header_start = f.header().start_datetime.as_datetime().unwrap().date();
        assert_eq!(
            birth.num_days_from_ce(),
            NaiveDate::from_ymd_opt(1951, 3, 2)
                .unwrap()
                .num_days_from_ce()
                - 400
        );
        // Age (birthdate→recording interval) preserved exactly.
        assert_eq!(
            rec_start.num_days_from_ce() - birth.num_days_from_ce(),
            NaiveDate::from_ymd_opt(2026, 1, 1)
                .unwrap()
                .num_days_from_ce()
                - NaiveDate::from_ymd_opt(1951, 3, 2)
                    .unwrap()
                    .num_days_from_ce()
        );
        assert_eq!(header_start, *rec_start);
    }

    #[test]
    fn anonymize_dry_run_writes_nothing() {
        let dir = tempdir().unwrap();
        let path = build_file(&dir, "dry.edf");
        let before = std::fs::read(&path).unwrap();

        let result = anonymize(
            &path,
            &AnonymizeOptions {
                seed: Some("s".into()),
                dry_run: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.diff.is_empty());
        assert!(!result.patient_id_after.is_empty());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn audit_finds_leaks_and_scrubbed_terms_recheck_catches_residuals() {
        let dir = tempdir().unwrap();
        let path = build_file(&dir, "audit.edf");

        let report = audit(&path).unwrap();
        assert!(!report.is_clean());
        assert!(report.terms.contains(&"haagansen".to_string()));

        let label_hits = report
            .hits
            .iter()
            .filter(|h| h.signal_index == Some(0) && h.location.ends_with("label"))
            .count();
        assert_eq!(label_hits, 1);

        let ann_hits = report
            .hits
            .iter()
            .filter(|h| h.signal_index.is_none())
            .count();
        assert_eq!(ann_hits, 1);

        let result = anonymize(
            &path,
            &AnonymizeOptions {
                seed: Some("s".into()),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(result.scrubbed_terms.contains(&"haagansen".to_string()));

        // Header-only anonymization cannot rewrite labels or annotation text. Re-checking
        // with the scrubbed terms still finds them: that residual list is the release gate.
        let residual = audit_terms(&path, &result.scrubbed_terms).unwrap();
        assert!(!residual.is_clean());
        assert!(residual.hits.iter().all(|h| h.term == "haagansen"));
        assert_eq!(residual.hits.len(), 2);
    }

    #[test]
    fn audit_clean_on_scrubbed_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("clean.edf");
        let spec = WriterSpec {
            variant: EdfVariant::EdfPlusC,
            patient_id: "X X X Subject-1234".into(),
            recording_id: "Startdate 01-JAN-2026 X X X".into(),
            start_datetime: NaiveDateTime::new(
                NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
            ),
            record_duration_secs: 1.0,
            signals: vec![WriterSignal::new(
                "EEG Fpz", "uV", -3200.0, 3200.0, -32768, 32767, 32,
            )],
            annotation_bytes_per_record: None,
            record_onsets: None,
        };
        write_edf(&path, spec, &[&vec![0.0f64; 64]], &[]).unwrap();

        let report = audit(&path).unwrap();
        assert!(report.is_clean(), "unexpected hits: {:?}", report.hits);
    }

    #[test]
    fn pseudonym_and_shift_helpers() {
        assert_eq!(
            generate_pseudonym("seed", "A B C"),
            generate_pseudonym("seed", "A B C")
        );
        assert_ne!(
            generate_pseudonym("seed", "A B C"),
            generate_pseudonym("seed", "X Y Z")
        );
        assert_ne!(
            generate_pseudonym("seed1", "A B C"),
            generate_pseudonym("seed2", "A B C")
        );
        assert!((derive_shift_days("s").abs() >= 1) && derive_shift_days("s").abs() < 3652);
    }

    #[test]
    fn failed_edit_leaves_no_partial_write() {
        let dir = tempdir().unwrap();
        let path = build_file(&dir, "partial.edf");
        let before = std::fs::read(&path).unwrap();

        // patient_id is valid, recording_id is not. Validation happens up front, so the
        // valid field must not reach the file.
        let err = edit_header(
            &path,
            &HeaderEdit {
                patient_id: Some("X X X Subject-OK".into()),
                recording_id: Some("R".repeat(81)),
                start_datetime: None,
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdfError::InvalidArgument {
                name: "recording_id",
                ..
            }
        ));
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[test]
    fn pseudonym_survives_differing_session_notes() {
        let dir = tempdir().unwrap();
        let one = dir.path().join("s1.edf");
        let two = dir.path().join("s2.edf");
        for (path, note) in [(&one, "session_one"), (&two, "session_two")] {
            let spec = WriterSpec {
                variant: EdfVariant::EdfPlusC,
                patient_id: format!("MCH-01 F 02-MAR-1951 Jane_Doe {note}"),
                recording_id: "Startdate 01-JAN-2026 X X X".into(),
                start_datetime: NaiveDateTime::new(
                    NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
                    NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
                ),
                record_duration_secs: 1.0,
                signals: vec![WriterSignal::new(
                    "EEG Fpz", "uV", -3200.0, 3200.0, -32768, 32767, 32,
                )],
                annotation_bytes_per_record: None,
                record_onsets: None,
            };
            write_edf(path, spec, &[&vec![0.0f64; 64]], &[]).unwrap();
        }
        let opts = AnonymizeOptions {
            seed: Some("study-2026".into()),
            ..Default::default()
        };
        let a = anonymize(&one, &opts).unwrap();
        let b = anonymize(&two, &opts).unwrap();
        // The per-session note is scrubbed anyway, so it must not split one subject in two.
        assert_eq!(a.pseudonym, b.pseudonym);
        assert_eq!(a.patient_id_after, b.patient_id_after);
    }

    #[test]
    fn dry_run_rejects_what_a_real_run_would_reject() {
        let dir = tempdir().unwrap();
        let path = build_file(&dir, "dryreject.edf");
        let opts = AnonymizeOptions {
            seed: Some("s".into()),
            pseudonym: Some("P".repeat(81)),
            dry_run: true,
            ..Default::default()
        };
        let err = anonymize(&path, &opts).unwrap_err();
        assert!(matches!(err, EdfError::InvalidArgument { .. }));
    }

    #[test]
    fn anonymize_rejects_out_of_representable_range() {
        let dir = tempdir().unwrap();
        let path = build_file(&dir, "range.edf");
        let err = anonymize(
            &path,
            &AnonymizeOptions {
                date_shift_days: Some(40_000),
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdfError::InvalidArgument {
                name: "date_shift_days",
                ..
            }
        ));
        // File untouched.
        let f = EdfFile::open(&path).unwrap();
        assert!(f.header().patient_id.starts_with("MCH-0234567"));
    }
}
