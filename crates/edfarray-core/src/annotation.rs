use crate::error::Result;
use crate::header::{EdfHeader, EdfVariant};
use crate::record::RecordLayout;

const TAL_SEPARATOR: u8 = 0x14;
const TAL_DURATION_MARKER: u8 = 0x15;
const TAL_TERMINATOR: u8 = 0x00;

/// A single annotation parsed from a TAL (Time-stamped Annotation List).
#[derive(Debug, Clone)]
pub struct Annotation {
    pub onset: f64,
    pub duration: Option<f64>,
    pub text: String,
}

/// Index of all annotations and record timing information built from a sequential
/// scan of the file's annotation signals.
#[derive(Debug, Clone)]
pub struct AnnotationIndex {
    /// All non-timekeeping annotations, sorted by onset.
    pub annotations: Vec<Annotation>,

    /// Actual onset time (in seconds from file start) for each data record.
    /// For EDF+C this increases uniformly; for EDF+D it may have gaps.
    pub record_onsets: Vec<f64>,

    /// Subsecond component of the recording start time, extracted from the
    /// first time-keeping annotation in the first data record. The EDF header
    /// only stores integer seconds; EDF+ encodes subsecond precision here.
    pub starttime_subsecond: f64,

    /// Warnings encountered during parsing (malformed TALs, etc.).
    pub warnings: Vec<String>,
}

impl AnnotationIndex {
    /// Build the annotation index by scanning all data records in the provided byte slice.
    ///
    /// `data` is the full file contents (or mmap). The header and layout describe
    /// where to find annotation signals within each data record.
    pub fn build(data: &[u8], header: &EdfHeader, layout: &RecordLayout) -> Result<Self> {
        let mut annotations = Vec::new();
        let mut record_onsets = Vec::new();
        let mut warnings = Vec::new();

        let annotation_indices: Vec<usize> = header
            .signals
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_annotation)
            .map(|(i, _)| i)
            .collect();

        if annotation_indices.is_empty() {
            let num_records = header.num_records.max(0) as usize;
            for i in 0..num_records {
                record_onsets.push(i as f64 * header.record_duration_secs);
            }
            return Ok(AnnotationIndex {
                annotations,
                record_onsets,
                starttime_subsecond: 0.0,
                warnings,
            });
        }

        let data_start = header.data_offset();
        let num_records = header.num_records.max(0) as usize;

        for rec_idx in 0..num_records {
            let rec_offset = data_start + rec_idx * layout.record_size;
            let rec_end = rec_offset + layout.record_size;

            if rec_end > data.len() {
                warnings.push(format!(
                    "data truncated at record {rec_idx}: expected {rec_end} bytes, file has {}",
                    data.len()
                ));
                break;
            }

            let record_data = &data[rec_offset..rec_end];
            let mut found_timekeeping = false;

            for &sig_idx in &annotation_indices {
                let sig_bytes = layout.signal_bytes(record_data, sig_idx)?;
                let tals = parse_tals(sig_bytes, rec_idx, &mut warnings);

                for (tal_idx, ann) in tals.into_iter().enumerate() {
                    if tal_idx == 0 && !found_timekeeping && ann.text.is_empty() {
                        record_onsets.push(ann.onset);
                        found_timekeeping = true;
                        continue;
                    }
                    if !ann.text.is_empty() {
                        annotations.push(ann);
                    }
                }
            }

            if !found_timekeeping {
                if header.variant != EdfVariant::Edf {
                    warnings.push(format!(
                        "missing time-keeping annotation in record {rec_idx}, using calculated onset"
                    ));
                }
                record_onsets.push(rec_idx as f64 * header.record_duration_secs);
            }
        }

        // The first time-keeping annotation's onset encodes the subsecond
        // component of the recording start time. The EDF header only stores
        // integer seconds, so EDF+ files use this to convey sub-second precision.
        // All onsets in the file are relative to the integer-second start, so we
        // subtract this offset to make them relative to the true start time.
        let starttime_subsecond = record_onsets.first().copied().unwrap_or(0.0);

        for onset in &mut record_onsets {
            *onset -= starttime_subsecond;
        }
        for ann in &mut annotations {
            ann.onset -= starttime_subsecond;
        }

        validate_record_onsets(
            &record_onsets,
            header.record_duration_secs,
            header.variant,
            &mut warnings,
        );

        annotations.sort_by(|a, b| {
            a.onset
                .partial_cmp(&b.onset)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(AnnotationIndex {
            annotations,
            record_onsets,
            starttime_subsecond,
            warnings,
        })
    }
}

/// Parse all TALs from a single annotation signal's bytes within one data record.
///
/// Follows edflib's defensive approach: on malformed TALs, emit a warning and
/// continue parsing rather than failing the entire file.
fn parse_tals(data: &[u8], record_idx: usize, warnings: &mut Vec<String>) -> Vec<Annotation> {
    let mut result = Vec::new();
    let mut pos = 0;

    while pos < data.len() {
        if data[pos] == TAL_TERMINATOR {
            pos += 1;
            continue;
        }

        match parse_single_tal(data, &mut pos, record_idx) {
            Ok(annotations) => result.extend(annotations),
            Err(msg) => {
                warnings.push(msg);
                skip_to_next_tal(data, &mut pos);
            }
        }
    }

    result
}

/// Parse one TAL starting at `pos`, advancing `pos` past it.
///
/// TAL format: `+Onset[\x15Duration]\x14[Text\x14]*\x00`
fn parse_single_tal(
    data: &[u8],
    pos: &mut usize,
    record_idx: usize,
) -> std::result::Result<Vec<Annotation>, String> {
    let tal_start = *pos;

    let onset_str = read_until(data, pos, &[TAL_SEPARATOR, TAL_DURATION_MARKER]);
    if onset_str.is_empty() {
        return Err(format!(
            "empty onset in TAL at record {record_idx}, byte offset {tal_start}"
        ));
    }

    let onset = parse_onset(&onset_str).map_err(|reason| {
        format!("invalid TAL onset at record {record_idx}, byte offset {tal_start}: {reason}")
    })?;

    let duration =
        if *pos > 0
            && *pos <= data.len()
            && data.get(pos.wrapping_sub(1)) == Some(&TAL_DURATION_MARKER)
        {
            let dur_str = read_until(data, pos, &[TAL_SEPARATOR]);
            parse_duration(&dur_str).map_err(|reason| format!(
            "invalid TAL duration at record {record_idx}, byte offset {tal_start}: {reason}"
        )).ok()
        } else {
            None
        };

    let mut annotations = Vec::new();

    loop {
        if *pos >= data.len() || data[*pos] == TAL_TERMINATOR {
            if *pos < data.len() {
                *pos += 1;
            }
            break;
        }

        let text_bytes = read_until_raw(data, pos, &[TAL_SEPARATOR, TAL_TERMINATOR]);
        let terminated_by = if *pos > 0 {
            data.get(pos.wrapping_sub(1)).copied()
        } else {
            None
        };

        match std::str::from_utf8(&text_bytes) {
            Ok(text) => {
                let text = text.to_string();
                annotations.push(Annotation {
                    onset,
                    duration,
                    text,
                });
            }
            Err(_) => {
                return Err(format!(
                    "non-UTF-8 annotation text at record {record_idx}, byte offset {tal_start}"
                ));
            }
        }

        if terminated_by == Some(TAL_TERMINATOR) {
            break;
        }
    }

    if annotations.is_empty() {
        annotations.push(Annotation {
            onset,
            duration,
            text: String::new(),
        });
    }

    Ok(annotations)
}

/// Validate onset string: must start with + or -, contain only digits and at most one dot.
fn parse_onset(s: &str) -> std::result::Result<f64, String> {
    if s.is_empty() {
        return Err("empty onset".to_string());
    }

    let first = s.as_bytes()[0];
    if first != b'+' && first != b'-' {
        return Err(format!("onset must start with + or -, got {:?}", s));
    }

    let number_part = &s[1..];
    validate_number(number_part, "onset")?;

    s.parse::<f64>()
        .map_err(|e| format!("onset parse error: {e}"))
}

/// Validate duration string: digits and at most one dot, no sign.
fn parse_duration(s: &str) -> std::result::Result<f64, String> {
    if s.is_empty() {
        return Err("empty duration".to_string());
    }
    validate_number(s, "duration")?;
    s.parse::<f64>()
        .map_err(|e| format!("duration parse error: {e}"))
}

fn validate_number(s: &str, label: &str) -> std::result::Result<(), String> {
    if s.is_empty() {
        return Err(format!("empty {label}"));
    }
    if s.starts_with('.') || s.ends_with('.') {
        return Err(format!("{label} cannot start or end with a dot: {:?}", s));
    }
    let mut dot_count = 0;
    for ch in s.chars() {
        if ch == '.' {
            dot_count += 1;
            if dot_count > 1 {
                return Err(format!("{label} has multiple decimal points: {:?}", s));
            }
        } else if !ch.is_ascii_digit() {
            return Err(format!("{label} contains invalid character: {:?}", s));
        }
    }
    Ok(())
}

/// Read bytes until one of the stop bytes is found, returning the content as a string.
/// Advances `pos` past the stop byte.
fn read_until(data: &[u8], pos: &mut usize, stop: &[u8]) -> String {
    let bytes = read_until_raw(data, pos, stop);
    String::from_utf8_lossy(&bytes).to_string()
}

/// Read bytes until one of the stop bytes is found, returning raw bytes.
/// Advances `pos` past the stop byte.
fn read_until_raw(data: &[u8], pos: &mut usize, stop: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();
    while *pos < data.len() {
        let b = data[*pos];
        *pos += 1;
        if stop.contains(&b) {
            return buf;
        }
        buf.push(b);
    }
    buf
}

/// Skip past the current TAL to the next one (find next 0x00 byte).
fn skip_to_next_tal(data: &[u8], pos: &mut usize) {
    while *pos < data.len() {
        if data[*pos] == TAL_TERMINATOR {
            *pos += 1;
            return;
        }
        *pos += 1;
    }
}

/// Check record onsets for consistency based on the file variant.
fn validate_record_onsets(
    onsets: &[f64],
    record_duration: f64,
    variant: EdfVariant,
    warnings: &mut Vec<String>,
) {
    if onsets.len() < 2 || record_duration <= 0.0 {
        return;
    }

    for i in 1..onsets.len() {
        let gap = onsets[i] - onsets[i - 1];
        let expected = record_duration;

        match variant {
            EdfVariant::EdfPlusC => {
                if (gap - expected).abs() > 0.001 {
                    warnings.push(format!(
                        "EDF+C record {i}: expected onset gap {expected}s, got {gap:.6}s"
                    ));
                }
            }
            EdfVariant::EdfPlusD => {
                if gap < expected - 0.001 {
                    warnings.push(format!(
                        "EDF+D record {i}: onset gap {gap:.6}s is less than record duration {expected}s"
                    ));
                }
            }
            EdfVariant::Edf => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::EdfHeader;

    fn make_tal(onset: &str, duration: Option<&str>, texts: &[&str]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(onset.as_bytes());
        if let Some(dur) = duration {
            buf.push(TAL_DURATION_MARKER);
            buf.extend_from_slice(dur.as_bytes());
        }
        buf.push(TAL_SEPARATOR);
        for text in texts {
            buf.extend_from_slice(text.as_bytes());
            buf.push(TAL_SEPARATOR);
        }
        buf.push(TAL_TERMINATOR);
        buf
    }

    #[test]
    fn parse_simple_tal() {
        let data = make_tal("+0", None, &["Lights off"]);
        let mut warnings = Vec::new();
        let tals = parse_tals(&data, 0, &mut warnings);

        assert!(warnings.is_empty());
        assert_eq!(tals.len(), 1);
        assert_eq!(tals[0].onset, 0.0);
        assert!(tals[0].duration.is_none());
        assert_eq!(tals[0].text, "Lights off");
    }

    #[test]
    fn parse_tal_with_duration() {
        let data = make_tal("+180.5", Some("30"), &["Sleep stage W"]);
        let mut warnings = Vec::new();
        let tals = parse_tals(&data, 0, &mut warnings);

        assert!(warnings.is_empty());
        assert_eq!(tals.len(), 1);
        assert!((tals[0].onset - 180.5).abs() < f64::EPSILON);
        assert_eq!(tals[0].duration, Some(30.0));
        assert_eq!(tals[0].text, "Sleep stage W");
    }

    #[test]
    fn parse_timekeeping_annotation() {
        let data = make_tal("+567", None, &[]);
        let mut warnings = Vec::new();
        let tals = parse_tals(&data, 0, &mut warnings);

        assert_eq!(tals.len(), 1);
        assert!((tals[0].onset - 567.0).abs() < f64::EPSILON);
        assert!(tals[0].text.is_empty());
    }

    #[test]
    fn parse_multiple_annotations_in_one_tal() {
        let data = make_tal("+10", None, &["Event A", "Event B"]);
        let mut warnings = Vec::new();
        let tals = parse_tals(&data, 0, &mut warnings);

        assert_eq!(tals.len(), 2);
        assert_eq!(tals[0].text, "Event A");
        assert_eq!(tals[1].text, "Event B");
        assert_eq!(tals[0].onset, tals[1].onset);
    }

    #[test]
    fn parse_multiple_tals() {
        let mut data = make_tal("+0", None, &[]);
        data.extend(make_tal("+30", Some("5"), &["Movement"]));
        data.extend(make_tal("+60", None, &["Arousal"]));

        let mut warnings = Vec::new();
        let tals = parse_tals(&data, 0, &mut warnings);

        assert!(warnings.is_empty());
        assert_eq!(tals.len(), 3);
        assert!(tals[0].text.is_empty());
        assert_eq!(tals[1].text, "Movement");
        assert_eq!(tals[2].text, "Arousal");
    }

    #[test]
    fn negative_onset() {
        let data = make_tal("-5.25", None, &["Pre-stimulus"]);
        let mut warnings = Vec::new();
        let tals = parse_tals(&data, 0, &mut warnings);

        assert_eq!(tals.len(), 1);
        assert!((tals[0].onset - (-5.25)).abs() < f64::EPSILON);
    }

    #[test]
    fn invalid_onset_warns() {
        let mut data = Vec::new();
        data.extend_from_slice(b"bad_onset");
        data.push(TAL_SEPARATOR);
        data.push(TAL_TERMINATOR);

        let mut warnings = Vec::new();
        let tals = parse_tals(&data, 0, &mut warnings);

        assert!(tals.is_empty());
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("onset must start with"));
    }

    #[test]
    fn malformed_tal_continues_parsing() {
        let mut data = Vec::new();
        data.extend_from_slice(b"bad");
        data.push(TAL_SEPARATOR);
        data.push(TAL_TERMINATOR);
        data.extend(make_tal("+10", None, &["Valid"]));

        let mut warnings = Vec::new();
        let tals = parse_tals(&data, 0, &mut warnings);

        assert_eq!(warnings.len(), 1);
        assert_eq!(tals.len(), 1);
        assert_eq!(tals[0].text, "Valid");
    }

    #[test]
    fn null_padded_data() {
        let mut data = make_tal("+0", None, &["Event"]);
        data.extend(vec![0u8; 20]);

        let mut warnings = Vec::new();
        let tals = parse_tals(&data, 0, &mut warnings);

        assert!(warnings.is_empty());
        assert_eq!(tals.len(), 1);
        assert_eq!(tals[0].text, "Event");
    }

    #[test]
    fn build_index_no_annotation_signals() {
        let header_data = build_edf_header(1, false);
        let mut full_data = header_data.clone();
        full_data.extend(vec![0u8; 512]); // one data record

        let header = EdfHeader::parse(&full_data).unwrap();
        let layout = RecordLayout::from_header(&header);
        let index = AnnotationIndex::build(&full_data, &header, &layout).unwrap();

        assert!(index.annotations.is_empty());
        assert_eq!(index.record_onsets.len(), 1);
        assert!((index.record_onsets[0]).abs() < f64::EPSILON);
    }

    #[test]
    fn build_index_with_annotations() {
        let (full_data, header) = build_file_with_annotations();
        let layout = RecordLayout::from_header(&header);
        let index = AnnotationIndex::build(&full_data, &header, &layout).unwrap();

        assert_eq!(index.record_onsets.len(), 2);
        assert!((index.record_onsets[0]).abs() < f64::EPSILON);
        assert!((index.record_onsets[1] - 1.0).abs() < f64::EPSILON);
        assert_eq!(index.annotations.len(), 1);
        assert_eq!(index.annotations[0].text, "TestEvent");
    }

    #[test]
    fn onset_validation_rejects_leading_dot() {
        assert!(parse_onset("+.5").is_err());
    }

    #[test]
    fn onset_validation_rejects_trailing_dot() {
        assert!(parse_onset("+5.").is_err());
    }

    #[test]
    fn onset_validation_rejects_multiple_dots() {
        assert!(parse_onset("+1.2.3").is_err());
    }

    fn build_edf_header(num_signals: usize, has_annotation: bool) -> Vec<u8> {
        let header_bytes = 256 + 256 * num_signals;
        let mut buf = vec![b' '; header_bytes];

        write_field(&mut buf, 0, 8, "0");
        write_field(&mut buf, 8, 80, "X X X X");
        write_field(&mut buf, 88, 80, "Startdate X X X X");
        write_field(&mut buf, 168, 8, "01.01.00");
        write_field(&mut buf, 176, 8, "00.00.00");
        write_field(&mut buf, 184, 8, &header_bytes.to_string());
        write_field(&mut buf, 192, 44, if has_annotation { "EDF+C" } else { "" });
        write_field(&mut buf, 236, 8, "1");
        write_field(&mut buf, 244, 8, "1");
        write_field(&mut buf, 252, 4, &num_signals.to_string());

        let sig_data = &mut buf[256..];
        for i in 0..num_signals {
            let label = if has_annotation && i == num_signals - 1 {
                "EDF Annotations"
            } else {
                "EEG"
            };
            write_sig(sig_data, i, num_signals, 0, 16, label);
            write_sig(sig_data, i, num_signals, 16, 80, "");
            write_sig(sig_data, i, num_signals, 96, 8, "uV");
            write_sig(sig_data, i, num_signals, 104, 8, "-3200");
            write_sig(sig_data, i, num_signals, 112, 8, "3200");
            write_sig(sig_data, i, num_signals, 120, 8, "-32768");
            write_sig(sig_data, i, num_signals, 128, 8, "32767");
            write_sig(sig_data, i, num_signals, 136, 80, "");
            write_sig(sig_data, i, num_signals, 216, 8, "30");
            write_sig(sig_data, i, num_signals, 224, 32, "");
        }
        buf
    }

    fn build_file_with_annotations() -> (Vec<u8>, EdfHeader) {
        let num_signals = 2;
        let mut header_data = build_edf_header(num_signals, true);
        write_field(&mut header_data, 236, 8, "2"); // 2 records

        let header = EdfHeader::parse(&header_data).unwrap();
        let layout = RecordLayout::from_header(&header);

        let mut file_data = header_data;

        for rec_idx in 0..2 {
            let eeg_bytes = vec![0u8; 30 * 2];
            file_data.extend_from_slice(&eeg_bytes);

            let mut ann_bytes = Vec::new();
            let onset = format!("+{}", rec_idx);
            ann_bytes.extend(make_tal(&onset, None, &[]));
            if rec_idx == 0 {
                ann_bytes.extend(make_tal("+0.5", None, &["TestEvent"]));
            }
            ann_bytes.resize(30 * 2, 0);
            file_data.extend_from_slice(&ann_bytes);
        }

        assert_eq!(
            file_data.len(),
            header.data_offset() + 2 * layout.record_size
        );
        let header = EdfHeader::parse(&file_data[..header.header_bytes]).unwrap();
        (file_data, header)
    }

    fn write_field(buf: &mut [u8], offset: usize, size: usize, value: &str) {
        let bytes = value.as_bytes();
        let len = bytes.len().min(size);
        buf[offset..offset + len].copy_from_slice(&bytes[..len]);
    }

    fn write_sig(
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
}

#[cfg(test)]
mod proptest_fuzz {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// The TAL parser must never panic on arbitrary byte sequences.
        #[test]
        fn parse_tals_never_panics(data in proptest::collection::vec(any::<u8>(), 0..1024)) {
            let mut warnings = Vec::new();
            let _ = parse_tals(&data, 0, &mut warnings);
        }

        /// Onset validation must never panic on arbitrary strings.
        #[test]
        fn parse_onset_never_panics(s in "\\PC{0,64}") {
            let _ = parse_onset(&s);
        }

        /// Duration validation must never panic on arbitrary strings.
        #[test]
        fn parse_duration_never_panics(s in "\\PC{0,64}") {
            let _ = parse_duration(&s);
        }

        /// Well-formed TALs should always round-trip: if we construct a valid TAL
        /// from known-good components, parsing should recover the same values.
        #[test]
        fn well_formed_tal_roundtrips(
            onset_sign in prop_oneof![Just("+"), Just("-")],
            onset_int in 0u32..100_000,
            onset_frac in 0u32..1_000_000,
            has_duration in any::<bool>(),
            duration_int in 0u32..10_000,
            text in "[a-zA-Z0-9 ]{0,50}",
        ) {
            let onset_str = if onset_frac > 0 {
                format!("{onset_sign}{onset_int}.{onset_frac:06}")
            } else {
                format!("{onset_sign}{onset_int}")
            };

            let expected_onset: f64 = onset_str.parse().unwrap();

            let mut tal = Vec::new();
            tal.extend_from_slice(onset_str.as_bytes());

            let expected_duration = if has_duration {
                let dur_str = format!("{duration_int}");
                tal.push(TAL_DURATION_MARKER);
                tal.extend_from_slice(dur_str.as_bytes());
                Some(duration_int as f64)
            } else {
                None
            };

            tal.push(TAL_SEPARATOR);
            if !text.is_empty() {
                tal.extend_from_slice(text.as_bytes());
                tal.push(TAL_SEPARATOR);
            }
            tal.push(TAL_TERMINATOR);

            let mut warnings = Vec::new();
            let result = parse_tals(&tal, 0, &mut warnings);

            prop_assert!(warnings.is_empty(), "unexpected warnings: {:?}", warnings);
            prop_assert!(!result.is_empty(), "expected at least one annotation");

            let ann = &result[0];
            prop_assert!(
                (ann.onset - expected_onset).abs() < 1e-6,
                "onset mismatch: got {}, expected {}",
                ann.onset,
                expected_onset
            );
            prop_assert_eq!(ann.duration, expected_duration);
            prop_assert_eq!(&ann.text, &text);
        }

        /// Arbitrary bytes appended after a valid TAL should not corrupt parsing
        /// of the valid TAL.
        #[test]
        fn valid_tal_with_trailing_garbage(
            garbage in proptest::collection::vec(any::<u8>(), 0..256),
        ) {
            let mut data = Vec::new();
            data.extend_from_slice(b"+0");
            data.push(TAL_SEPARATOR);
            data.extend_from_slice(b"TestEvent");
            data.push(TAL_SEPARATOR);
            data.push(TAL_TERMINATOR);
            data.extend_from_slice(&garbage);

            let mut warnings = Vec::new();
            let result = parse_tals(&data, 0, &mut warnings);

            // The first TAL should always parse correctly
            let valid_anns: Vec<_> = result.iter().filter(|a| a.text == "TestEvent").collect();
            prop_assert!(!valid_anns.is_empty(), "valid TAL was lost");
            prop_assert!((valid_anns[0].onset - 0.0).abs() < 1e-10);
        }
    }
}
