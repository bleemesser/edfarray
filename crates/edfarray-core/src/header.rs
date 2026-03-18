use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

use crate::error::{EdfError, Result};
use crate::signal::SignalHeader;

/// Fixed size of the main header block in bytes.
const MAIN_HEADER_SIZE: usize = 256;

/// Size of the per-signal header block for one signal.
const SIGNAL_HEADER_SIZE: usize = 256;

/// Identifies the file as EDF, EDF+C (contiguous), or EDF+D (discontinuous).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdfVariant {
    Edf,
    EdfPlusC,
    EdfPlusD,
}

impl EdfVariant {
    fn parse(reserved: &str) -> Self {
        if reserved.starts_with("EDF+C") {
            EdfVariant::EdfPlusC
        } else if reserved.starts_with("EDF+D") {
            EdfVariant::EdfPlusD
        } else {
            EdfVariant::Edf
        }
    }
}

impl std::fmt::Display for EdfVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdfVariant::Edf => write!(f, "EDF"),
            EdfVariant::EdfPlusC => write!(f, "EDF+C"),
            EdfVariant::EdfPlusD => write!(f, "EDF+D"),
        }
    }
}

/// Biological sex as specified in the EDF+ patient identification field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sex {
    Male,
    Female,
}

/// Parsed patient identification subfields from the EDF+ patient_id field.
///
/// The EDF+ format encodes patient info as space-separated subfields:
/// `"code sex birthdate name [additional...]"`. Fields set to "X" indicate
/// unknown values and are represented as `None`.
#[derive(Debug, Clone, Default)]
pub struct PatientInfo {
    pub code: Option<String>,
    pub sex: Option<Sex>,
    pub birthdate: Option<NaiveDate>,
    pub name: Option<String>,
    pub additional: Option<String>,
}

/// Parsed recording identification subfields from the EDF+ recording_id field.
///
/// The EDF+ format encodes recording info as space-separated subfields:
/// `"Startdate DD-MMM-YYYY admincode technician equipment [additional...]"`.
#[derive(Debug, Clone, Default)]
pub struct RecordingInfo {
    pub start_date: Option<NaiveDate>,
    pub admin_code: Option<String>,
    pub technician: Option<String>,
    pub equipment: Option<String>,
    pub additional: Option<String>,
}

/// The complete EDF/EDF+ file header.
#[derive(Debug, Clone)]
pub struct EdfHeader {
    pub version: String,
    pub patient_id: String,
    pub recording_id: String,
    pub start_datetime: NaiveDateTime,
    pub header_bytes: usize,
    pub variant: EdfVariant,
    pub num_records: i64,
    pub record_duration_secs: f64,
    pub num_signals: usize,
    pub signals: Vec<SignalHeader>,
    pub patient: PatientInfo,
    pub recording: RecordingInfo,
    pub warnings: Vec<String>,
}

impl EdfHeader {
    /// Parse a complete EDF header from the beginning of a byte slice.
    ///
    /// The slice must contain at least `256 + 256 * num_signals` bytes.
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < MAIN_HEADER_SIZE {
            return Err(EdfError::FileTooSmall {
                expected: MAIN_HEADER_SIZE,
                actual: data.len(),
            });
        }

        let mut warnings = Vec::new();

        let version = read_field(data, 0, 8, "version")?;
        let patient_id = read_field(data, 8, 80, "patient_id")?;
        let recording_id = read_field(data, 88, 80, "recording_id")?;
        let start_date_str = read_field(data, 168, 8, "start_date")?;
        let start_time_str = read_field(data, 176, 8, "start_time")?;
        let header_bytes = read_usize(data, 184, 8, "header_bytes")?;
        let reserved = read_field(data, 192, 44, "reserved")?;
        let num_records = read_i64(data, 236, 8, "num_records")?;
        let record_duration_secs = read_f64(data, 244, 8, "record_duration")?;
        let num_signals = read_usize(data, 252, 4, "num_signals")?;

        if num_signals == 0 {
            return Err(EdfError::NoSignals);
        }

        let expected_header_bytes = MAIN_HEADER_SIZE + SIGNAL_HEADER_SIZE * num_signals;
        if header_bytes != expected_header_bytes {
            return Err(EdfError::HeaderSizeMismatch {
                header_bytes,
                file_size: expected_header_bytes,
            });
        }

        if data.len() < header_bytes {
            return Err(EdfError::FileTooSmall {
                expected: header_bytes,
                actual: data.len(),
            });
        }

        let variant = EdfVariant::parse(&reserved);
        let start_datetime = parse_start_datetime(&start_date_str, &start_time_str)?;

        let signal_data = &data[MAIN_HEADER_SIZE..header_bytes];
        let mut signals = Vec::with_capacity(num_signals);
        for i in 0..num_signals {
            signals.push(SignalHeader::parse(signal_data, i, num_signals)?);
        }

        let patient = parse_patient_id(&patient_id, variant, &mut warnings);
        let recording = parse_recording_id(&recording_id, variant, &mut warnings);

        Ok(EdfHeader {
            version,
            patient_id,
            recording_id,
            start_datetime,
            header_bytes,
            variant,
            num_records,
            record_duration_secs,
            num_signals,
            signals,
            patient,
            recording,
            warnings,
        })
    }

    /// Byte offset where data records begin.
    pub fn data_offset(&self) -> usize {
        self.header_bytes
    }

    /// Size of one complete data record in bytes.
    pub fn record_size(&self) -> usize {
        self.signals.iter().map(|s| s.num_samples * 2).sum()
    }

    /// Total duration of the recording in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.num_records.max(0) as f64 * self.record_duration_secs
    }
}

fn read_field(data: &[u8], offset: usize, size: usize, name: &'static str) -> Result<String> {
    let bytes = data.get(offset..offset + size).ok_or(EdfError::InvalidHeaderField {
        field: name,
        reason: "header truncated".to_string(),
    })?;
    Ok(String::from_utf8_lossy(bytes).trim().to_string())
}

fn read_usize(data: &[u8], offset: usize, size: usize, name: &'static str) -> Result<usize> {
    let s = read_field(data, offset, size, name)?;
    s.parse::<usize>().map_err(|_| EdfError::InvalidHeaderField {
        field: name,
        reason: format!("not a valid unsigned integer: {:?}", s),
    })
}

fn read_i64(data: &[u8], offset: usize, size: usize, name: &'static str) -> Result<i64> {
    let s = read_field(data, offset, size, name)?;
    s.parse::<i64>().map_err(|_| EdfError::InvalidHeaderField {
        field: name,
        reason: format!("not a valid integer: {:?}", s),
    })
}

fn read_f64(data: &[u8], offset: usize, size: usize, name: &'static str) -> Result<f64> {
    let s = read_field(data, offset, size, name)?;
    s.parse::<f64>().map_err(|_| EdfError::InvalidHeaderField {
        field: name,
        reason: format!("not a valid number: {:?}", s),
    })
}

/// Parse the start date (dd.mm.yy) and time (hh.mm.ss) into a NaiveDateTime.
///
/// Per the EDF spec, two-digit years use 1985 as the clipping year:
/// 85-99 map to 1985-1999, 00-84 map to 2000-2084.
fn parse_start_datetime(date_str: &str, time_str: &str) -> Result<NaiveDateTime> {
    let date_parts: Vec<&str> = date_str.split('.').collect();
    if date_parts.len() != 3 {
        return Err(EdfError::InvalidHeaderField {
            field: "start_date",
            reason: format!("expected dd.mm.yy format, got {:?}", date_str),
        });
    }

    let day: u32 = date_parts[0].parse().map_err(|_| EdfError::InvalidHeaderField {
        field: "start_date",
        reason: format!("invalid day: {:?}", date_parts[0]),
    })?;
    let month: u32 = date_parts[1].parse().map_err(|_| EdfError::InvalidHeaderField {
        field: "start_date",
        reason: format!("invalid month: {:?}", date_parts[1]),
    })?;
    let year_2d: u32 = date_parts[2].parse().map_err(|_| EdfError::InvalidHeaderField {
        field: "start_date",
        reason: format!("invalid year: {:?}", date_parts[2]),
    })?;

    let year = if year_2d >= 85 {
        1900 + year_2d as i32
    } else {
        2000 + year_2d as i32
    };

    let date = NaiveDate::from_ymd_opt(year, month, day).ok_or_else(|| {
        EdfError::InvalidHeaderField {
            field: "start_date",
            reason: format!("invalid date: {year}-{month:02}-{day:02}"),
        }
    })?;

    let time_parts: Vec<&str> = time_str.split('.').collect();
    if time_parts.len() != 3 {
        return Err(EdfError::InvalidHeaderField {
            field: "start_time",
            reason: format!("expected hh.mm.ss format, got {:?}", time_str),
        });
    }

    let hour: u32 = time_parts[0].parse().map_err(|_| EdfError::InvalidHeaderField {
        field: "start_time",
        reason: format!("invalid hour: {:?}", time_parts[0]),
    })?;
    let minute: u32 = time_parts[1].parse().map_err(|_| EdfError::InvalidHeaderField {
        field: "start_time",
        reason: format!("invalid minute: {:?}", time_parts[1]),
    })?;
    let second: u32 = time_parts[2].parse().map_err(|_| EdfError::InvalidHeaderField {
        field: "start_time",
        reason: format!("invalid second: {:?}", time_parts[2]),
    })?;

    let time = NaiveTime::from_hms_opt(hour, minute, second).ok_or_else(|| {
        EdfError::InvalidHeaderField {
            field: "start_time",
            reason: format!("invalid time: {hour:02}:{minute:02}:{second:02}"),
        }
    })?;

    Ok(NaiveDateTime::new(date, time))
}

/// Parse EDF+ patient_id into structured subfields.
///
/// Format: `"code sex birthdate name [additional...]"`
/// where "X" means unknown. Underscores in names are replaced with spaces.
fn parse_patient_id(raw: &str, variant: EdfVariant, warnings: &mut Vec<String>) -> PatientInfo {
    if variant == EdfVariant::Edf {
        return PatientInfo::default();
    }

    let parts: Vec<&str> = raw.split_whitespace().collect();
    if parts.len() < 4 {
        warnings.push(format!(
            "patient_id has {} subfields (expected at least 4): {:?}",
            parts.len(),
            raw
        ));
        return PatientInfo::default();
    }

    let code = non_x(parts[0]);
    let sex = match parts[1].to_uppercase().as_str() {
        "M" => Some(Sex::Male),
        "F" => Some(Sex::Female),
        "X" => None,
        other => {
            warnings.push(format!("unrecognized sex value: {:?}", other));
            None
        }
    };
    let birthdate = if parts[2].eq_ignore_ascii_case("X") {
        None
    } else {
        parse_edf_plus_date(parts[2]).or_else(|| {
            warnings.push(format!("invalid patient birthdate: {:?}", parts[2]));
            None
        })
    };
    let name = non_x(parts[3]).map(|s| s.replace('_', " "));
    let additional = if parts.len() > 4 {
        Some(parts[4..].join(" ").replace('_', " "))
    } else {
        None
    };

    PatientInfo {
        code,
        sex,
        birthdate,
        name,
        additional,
    }
}

/// Parse EDF+ recording_id into structured subfields.
///
/// Format: `"Startdate DD-MMM-YYYY admincode technician equipment [additional...]"`
fn parse_recording_id(
    raw: &str,
    variant: EdfVariant,
    warnings: &mut Vec<String>,
) -> RecordingInfo {
    if variant == EdfVariant::Edf {
        return RecordingInfo::default();
    }

    let parts: Vec<&str> = raw.split_whitespace().collect();
    if parts.len() < 5 || !parts[0].eq_ignore_ascii_case("Startdate") {
        warnings.push(format!(
            "recording_id does not match EDF+ format: {:?}",
            raw
        ));
        return RecordingInfo::default();
    }

    let start_date = if parts[1].eq_ignore_ascii_case("X") {
        None
    } else {
        parse_edf_plus_date(parts[1]).or_else(|| {
            warnings.push(format!("invalid recording start date: {:?}", parts[1]));
            None
        })
    };
    let admin_code = non_x(parts[2]);
    let technician = non_x(parts[3]).map(|s| s.replace('_', " "));
    let equipment = non_x(parts[4]).map(|s| s.replace('_', " "));
    let additional = if parts.len() > 5 {
        Some(parts[5..].join(" ").replace('_', " "))
    } else {
        None
    };

    RecordingInfo {
        start_date,
        admin_code,
        technician,
        equipment,
        additional,
    }
}

/// Parse a date in EDF+ format: DD-MMM-YYYY (e.g., "02-MAR-1951").
fn parse_edf_plus_date(s: &str) -> Option<NaiveDate> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let day: u32 = parts[0].parse().ok()?;
    let month = match parts[1].to_uppercase().as_str() {
        "JAN" => 1,
        "FEB" => 2,
        "MAR" => 3,
        "APR" => 4,
        "MAY" => 5,
        "JUN" => 6,
        "JUL" => 7,
        "AUG" => 8,
        "SEP" => 9,
        "OCT" => 10,
        "NOV" => 11,
        "DEC" => 12,
        _ => return None,
    };
    let year: i32 = parts[2].parse().ok()?;
    NaiveDate::from_ymd_opt(year, month, day)
}

/// Return `None` if the field is "X" (unknown), otherwise `Some(value)`.
fn non_x(s: &str) -> Option<String> {
    if s.eq_ignore_ascii_case("X") {
        None
    } else {
        Some(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Datelike;

    /// Build a minimal valid EDF header with the given signal count.
    fn build_test_header(num_signals: usize) -> Vec<u8> {
        let header_bytes = MAIN_HEADER_SIZE + SIGNAL_HEADER_SIZE * num_signals;
        let mut buf = vec![b' '; header_bytes];

        write_field(&mut buf, 0, 8, "0");
        write_field(&mut buf, 8, 80, "X X X X");
        write_field(&mut buf, 88, 80, "Startdate X X X X");
        write_field(&mut buf, 168, 8, "01.01.00");
        write_field(&mut buf, 176, 8, "00.00.00");
        write_field(&mut buf, 184, 8, &header_bytes.to_string());
        write_field(&mut buf, 192, 44, "EDF+C");
        write_field(&mut buf, 236, 8, "10");
        write_field(&mut buf, 244, 8, "1");
        write_field(&mut buf, 252, 4, &num_signals.to_string());

        let sig_data = &mut buf[MAIN_HEADER_SIZE..];
        for i in 0..num_signals {
            write_signal_field(sig_data, i, num_signals, 0, 16, "EEG");
            write_signal_field(sig_data, i, num_signals, 16, 80, "AgAgCl");
            write_signal_field(sig_data, i, num_signals, 96, 8, "uV");
            write_signal_field(sig_data, i, num_signals, 104, 8, "-3200");
            write_signal_field(sig_data, i, num_signals, 112, 8, "3200");
            write_signal_field(sig_data, i, num_signals, 120, 8, "-32768");
            write_signal_field(sig_data, i, num_signals, 128, 8, "32767");
            write_signal_field(sig_data, i, num_signals, 136, 80, "");
            write_signal_field(sig_data, i, num_signals, 216, 8, "256");
            write_signal_field(sig_data, i, num_signals, 224, 32, "");
        }

        buf
    }

    fn write_field(buf: &mut [u8], offset: usize, size: usize, value: &str) {
        let bytes = value.as_bytes();
        let len = bytes.len().min(size);
        buf[offset..offset + len].copy_from_slice(&bytes[..len]);
    }

    fn write_signal_field(
        data: &mut [u8],
        index: usize,
        num_signals: usize,
        field_offset: usize,
        field_size: usize,
        value: &str,
    ) {
        let start = field_offset * num_signals + field_size * index;
        let bytes = value.as_bytes();
        let len = bytes.len().min(field_size);
        data[start..start + len].copy_from_slice(&bytes[..len]);
    }

    #[test]
    fn parse_minimal_header() {
        let data = build_test_header(1);
        let header = EdfHeader::parse(&data).unwrap();

        assert_eq!(header.version, "0");
        assert_eq!(header.variant, EdfVariant::EdfPlusC);
        assert_eq!(header.num_signals, 1);
        assert_eq!(header.num_records, 10);
        assert_eq!(header.record_duration_secs, 1.0);
        assert_eq!(header.start_datetime.year(), 2000);
        assert_eq!(header.duration_secs(), 10.0);
        assert_eq!(header.record_size(), 512); // 256 samples * 2 bytes
    }

    #[test]
    fn parse_multiple_signals() {
        let data = build_test_header(3);
        let header = EdfHeader::parse(&data).unwrap();
        assert_eq!(header.signals.len(), 3);
        assert_eq!(header.record_size(), 512 * 3);
    }

    #[test]
    fn year_clipping() {
        let mut data = build_test_header(1);
        write_field(&mut data, 168, 8, "01.01.85");
        let header = EdfHeader::parse(&data).unwrap();
        assert_eq!(header.start_datetime.year(), 1985);

        write_field(&mut data, 168, 8, "01.01.84");
        let header = EdfHeader::parse(&data).unwrap();
        assert_eq!(header.start_datetime.year(), 2084);
    }

    #[test]
    fn edf_variant_detection() {
        let mut data = build_test_header(1);

        write_field(&mut data, 192, 44, "");
        // Need to clear the field first
        for b in &mut data[192..236] {
            *b = b' ';
        }
        let header = EdfHeader::parse(&data).unwrap();
        assert_eq!(header.variant, EdfVariant::Edf);

        write_field(&mut data, 192, 44, "EDF+D");
        let header = EdfHeader::parse(&data).unwrap();
        assert_eq!(header.variant, EdfVariant::EdfPlusD);
    }

    #[test]
    fn parse_patient_info() {
        let mut data = build_test_header(1);
        write_field(&mut data, 8, 80, "MCH-0234567 F 02-MAR-1951 Haagansen_Erlangen extra_info");
        // Recalculate to ensure EDF+ variant
        write_field(&mut data, 192, 44, "EDF+C");

        let header = EdfHeader::parse(&data).unwrap();
        assert_eq!(header.patient.code.as_deref(), Some("MCH-0234567"));
        assert_eq!(header.patient.sex, Some(Sex::Female));
        assert_eq!(
            header.patient.birthdate,
            NaiveDate::from_ymd_opt(1951, 3, 2)
        );
        assert_eq!(header.patient.name.as_deref(), Some("Haagansen Erlangen"));
        assert_eq!(header.patient.additional.as_deref(), Some("extra info"));
    }

    #[test]
    fn parse_recording_info() {
        let mut data = build_test_header(1);
        write_field(
            &mut data,
            88,
            80,
            "Startdate 02-MAR-2002 PSG-1234 John_Doe Nihon_Kohden",
        );
        write_field(&mut data, 192, 44, "EDF+C");

        let header = EdfHeader::parse(&data).unwrap();
        assert_eq!(
            header.recording.start_date,
            NaiveDate::from_ymd_opt(2002, 3, 2)
        );
        assert_eq!(header.recording.admin_code.as_deref(), Some("PSG-1234"));
        assert_eq!(header.recording.technician.as_deref(), Some("John Doe"));
        assert_eq!(header.recording.equipment.as_deref(), Some("Nihon Kohden"));
    }

    #[test]
    fn unknown_patient_fields() {
        let mut data = build_test_header(1);
        write_field(&mut data, 8, 80, "X X X X");
        write_field(&mut data, 192, 44, "EDF+C");

        let header = EdfHeader::parse(&data).unwrap();
        assert!(header.patient.code.is_none());
        assert!(header.patient.sex.is_none());
        assert!(header.patient.birthdate.is_none());
        assert!(header.patient.name.is_none());
    }

    #[test]
    fn file_too_small() {
        let err = EdfHeader::parse(&[0u8; 100]).unwrap_err();
        assert!(matches!(err, EdfError::FileTooSmall { .. }));
    }

    #[test]
    fn header_size_mismatch() {
        let mut data = build_test_header(1);
        write_field(&mut data, 184, 8, "9999");
        let err = EdfHeader::parse(&data).unwrap_err();
        assert!(matches!(err, EdfError::HeaderSizeMismatch { .. }));
    }
}
