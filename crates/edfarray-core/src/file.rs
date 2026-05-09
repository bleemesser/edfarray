use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use regex::RegexBuilder;
use rayon::prelude::*;

use crate::annotation::Annotation;
use crate::array_proxy::ArrayProxy;
use crate::error::{EdfError, Result};
use crate::header::{EdfHeader, EdfVariant, PatientInfo, RecordingInfo};
use crate::mmap::MappedFile;
use crate::proxy::SignalProxy;
use crate::signal::SignalHeader;

/// Top-level handle for an open EDF/EDF+ file.
///
/// Provides access to the file header, signal data (via `SignalProxy`),
/// and annotations. The underlying file is memory-mapped and remains
/// open for the lifetime of this struct.
pub struct EdfFile {
    file: Arc<MappedFile>,
}

impl EdfFile {
    /// Open an EDF/EDF+ file at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = MappedFile::open(path.as_ref())?;
        Ok(EdfFile { file })
    }

    /// The parsed file header.
    pub fn header(&self) -> &EdfHeader {
        &self.file.header
    }

    /// File variant: EDF, EDF+C, or EDF+D.
    pub fn variant(&self) -> EdfVariant {
        self.file.header.variant
    }

    /// Number of signals in the file.
    pub fn num_signals(&self) -> usize {
        self.file.header.num_signals
    }

    /// Number of data records.
    pub fn num_records(&self) -> usize {
        self.file.header.num_records.max(0) as usize
    }

    /// Duration of each data record in seconds.
    pub fn record_duration(&self) -> f64 {
        self.file.header.record_duration_secs
    }

    /// Total recording duration in seconds.
    pub fn duration(&self) -> f64 {
        self.file.header.duration_secs()
    }

    /// Parsed patient identification info (EDF+ only).
    pub fn patient(&self) -> &PatientInfo {
        &self.file.header.patient
    }

    /// Parsed recording identification info (EDF+ only).
    pub fn recording(&self) -> &RecordingInfo {
        &self.file.header.recording
    }

    /// All non-timekeeping annotations, sorted by onset.
    /// Blocks until the annotation scan is complete.
    pub fn annotations(&self) -> Vec<Annotation> {
        self.file
            .with_annotations(|idx| idx.annotations.clone())
    }

    /// Return annotations with onset strictly before `t`.
    /// Uses binary search (partition_point) for efficiency.
    /// Blocks until the annotation scan is complete.
    pub fn annotations_before(&self, t: f64) -> Vec<Annotation> {
        self.file.with_annotations(|idx| {
            let split = idx.annotations.partition_point(|a| a.onset < t);
            idx.annotations[..split].to_vec()
        })
    }

    /// Return annotations with onset greater than or equal to `t`.
    /// Uses binary search (partition_point) for efficiency.
    /// Blocks until the annotation scan is complete.
    pub fn annotations_after(&self, t: f64) -> Vec<Annotation> {
        self.file.with_annotations(|idx| {
            let split = idx.annotations.partition_point(|a| a.onset < t);
            idx.annotations[split..].to_vec()
        })
    }

    /// Return annotations with onset in the half-open interval `[start, end)`.
    /// Uses binary search for efficiency.
    /// Blocks until the annotation scan is complete.
    pub fn annotations_in_range(&self, start: f64, end: f64) -> Vec<Annotation> {
        self.file.with_annotations(|idx| {
            let lo = idx.annotations.partition_point(|a| a.onset < start);
            let hi = idx.annotations[lo..].partition_point(|a| a.onset < end) + lo;
            idx.annotations[lo..hi].to_vec()
        })
    }

    /// Filter annotations by text content.
    ///
    /// If `regex` is `false`, returns annotations whose text contains the query
    /// as a case-insensitive substring.
    ///
    /// If `regex` is `true`, returns annotations whose text matches the query
    /// as a case-insensitive regex pattern.
    ///
    /// Returns `EdfError::InvalidArgument` if the regex pattern is invalid.
    /// Blocks until the annotation scan is complete.
    pub fn filter_annotations(&self, query: &str, regex: bool) -> Result<Vec<Annotation>> {
        self.file.with_annotations(|idx| {
            let annotations = &idx.annotations;
            if regex {
                let re = RegexBuilder::new(query)
                    .case_insensitive(true)
                    .build()
                    .map_err(|_| EdfError::InvalidArgument {
                        name: "regex pattern",
                        reason: format!("invalid regex pattern: {query}"),
                    })?;
                Ok(annotations
                    .iter()
                    .filter(|a| re.is_match(&a.text))
                    .cloned()
                    .collect())
            } else {
                let lower = query.to_lowercase();
                Ok(annotations
                    .iter()
                    .filter(|a| a.text.to_lowercase().contains(&lower))
                    .cloned()
                    .collect())
            }
        })
    }

    /// Return annotations whose text exactly matches `text` (case-sensitive).
    /// Blocks until the annotation scan is complete.
    pub fn annotations_by_text(&self, text: &str) -> Vec<Annotation> {
        self.file.with_annotations(|idx| {
            idx.annotations
                .iter()
                .filter(|a| a.text == text)
                .cloned()
                .collect()
        })
    }

    /// Parse warnings accumulated during file open (malformed TALs, etc.).
    /// Blocks until the annotation scan is complete.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = self.file.header.warnings.clone();
        self.file.with_annotations(|idx| {
            w.extend(idx.warnings.iter().cloned());
        });
        w
    }

    /// Progress of the background annotation scan: (records_scanned, total_records).
    pub fn scan_progress(&self) -> (usize, usize) {
        self.file.scan_progress()
    }

    /// Whether the background annotation scan has completed.
    pub fn annotations_ready(&self) -> bool {
        self.file.annotations_ready()
    }

    /// Get a signal proxy by index.
    pub fn signal(&self, idx: usize) -> Result<SignalProxy> {
        SignalProxy::new(Arc::clone(&self.file), idx)
    }

    /// Labels of all signals in the file.
    pub fn signal_labels(&self) -> Vec<&str> {
        self.file
            .header
            .signals
            .iter()
            .map(|s| s.label.as_str())
            .collect()
    }

    /// Indices of all non-annotation (ordinary) signals.
    pub fn ordinary_signal_indices(&self) -> Vec<usize> {
        self.file
            .header
            .signals
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.is_annotation)
            .map(|(i, _)| i)
            .collect()
    }

    /// Read a page of physical data for the given signals over a time range.
    ///
    /// Returns a vec of f64 buffers, one per signal index. Each buffer contains
    /// the physical samples for that signal in the time range `[start_sec, end_sec)`.
    /// Signals may have different sample rates, so buffers may have different lengths.
    ///
    /// **EDF+D note:** when `use_time` is false (default), time parameters are
    /// converted to flat sample indices (`(time * sample_rate) as usize`), not
    /// physical time. For discontinuous recordings, set `use_time=true` to resolve
    /// the time range to actual sample indices using record onset times.
    pub fn read_page(
        &self,
        signal_indices: &[usize],
        start_sec: f64,
        end_sec: f64,
        use_time: bool,
    ) -> Result<Vec<Vec<f64>>> {
        self.advise_time_range(start_sec, end_sec);
        let file = &self.file;
        signal_indices
            .par_iter()
            .map(|&idx| {
                let proxy = SignalProxy::new(Arc::clone(file), idx)?;
                let (s_start, s_end) = if use_time {
                    self.file.sample_range_for_time(&proxy, start_sec, end_sec)
                } else {
                    let sr = proxy.sample_rate();
                    let s_start = (start_sec.max(0.0) * sr) as usize;
                    let s_end = ((end_sec.max(0.0) * sr) as usize).min(proxy.len());
                    (s_start, s_end)
                };
                self.read_samples(&proxy, s_start, s_end)
            })
            .collect()
    }

    /// Read a page of digital data for the given signals over a time range.
    pub fn read_page_digital(
        &self,
        signal_indices: &[usize],
        start_sec: f64,
        end_sec: f64,
        use_time: bool,
    ) -> Result<Vec<Vec<i16>>> {
        self.advise_time_range(start_sec, end_sec);
        let file = &self.file;
        signal_indices
            .par_iter()
            .map(|&idx| {
                let proxy = SignalProxy::new(Arc::clone(file), idx)?;
                let (s_start, s_end) = if use_time {
                    self.file.sample_range_for_time(&proxy, start_sec, end_sec)
                } else {
                    let sr = proxy.sample_rate();
                    let s_start = (start_sec.max(0.0) * sr) as usize;
                    let s_end = ((end_sec.max(0.0) * sr) as usize).min(proxy.len());
                    (s_start, s_end)
                };
                self.read_digital_samples(&proxy, s_start, s_end)
            })
            .collect()
    }

    /// Hint to the OS that we'll need the data records covering the given time range.
    #[cfg(unix)]
    fn advise_time_range(&self, start_sec: f64, end_sec: f64) {
        let dur = self.file.header.record_duration_secs;
        if dur <= 0.0 {
            return;
        }
        let first = (start_sec / dur) as usize;
        let last = ((end_sec / dur).ceil() as usize).min(self.num_records());
        self.file.advise_willneed(first, last);
    }

    #[cfg(not(unix))]
    fn advise_time_range(&self, _start_sec: f64, _end_sec: f64) {}

    fn read_samples(&self, proxy: &SignalProxy, s_start: usize, s_end: usize) -> Result<Vec<f64>> {
        if s_start >= proxy.len() || s_start >= s_end {
            return Ok(Vec::new());
        }
        let count = s_end - s_start;
        let mut buf = vec![0.0f64; count];
        proxy.read_physical(s_start, s_end, &mut buf)?;
        Ok(buf)
    }

    fn read_digital_samples(&self, proxy: &SignalProxy, s_start: usize, s_end: usize) -> Result<Vec<i16>> {
        if s_start >= proxy.len() || s_start >= s_end {
            return Ok(Vec::new());
        }
        let count = s_end - s_start;
        let mut buf = vec![0i16; count];
        proxy.read_digital(s_start, s_end, &mut buf)?;
        Ok(buf)
    }

    /// Create a 2D array proxy over the given signal indices (or all ordinary signals).
    ///
    /// All signals must have the same sample rate. Returns an error if rates differ.
    pub fn array_proxy(&self, signal_indices: Option<&[usize]>) -> Result<ArrayProxy> {
        let indices = match signal_indices {
            Some(idx) => idx.to_vec(),
            None => self.ordinary_signal_indices(),
        };
        ArrayProxy::new(Arc::clone(&self.file), &indices)
    }

    /// Group ordinary signal indices by sample rate (Hz, rounded to integer).
    ///
    /// Returns a map from rate to signal indices. Useful for creating `ArrayProxy`
    /// instances when the file has mixed sample rates.
    pub fn signal_indices_by_rate(&self) -> HashMap<u64, Vec<usize>> {
        let mut map: HashMap<u64, Vec<usize>> = HashMap::new();
        for idx in self.ordinary_signal_indices() {
            let rate = self.file.header.signals[idx]
                .sample_rate(self.file.header.record_duration_secs);
            let key = rate.to_bits();
            map.entry(key).or_default().push(idx);
        }
        map.into_iter()
            .map(|(bits, indices)| (f64::from_bits(bits).round() as u64, indices))
            .fold(HashMap::new(), |mut acc, (rate, indices)| {
                acc.entry(rate).or_default().extend(indices);
                acc
            })
    }

    /// Get a signal proxy by label (first match).
    pub fn signal_by_label(&self, label: &str) -> Result<SignalProxy> {
        let idx = self
            .file
            .header
            .signals
            .iter()
            .position(|s| s.label == label)
            .ok_or_else(|| EdfError::SignalNotFound {
                label: label.to_string(),
            })?;
        self.signal(idx)
    }

    /// Return all signal indices whose label matches `label`.
    ///
    /// If `exact` is `false` (default), performs a case-insensitive substring match.
    /// If `exact` is `true`, performs a case-sensitive exact equality match.
    ///
    /// Searches all signals including annotation signals.
    pub fn find_all_signals(&self, label: &str, exact: bool) -> Vec<usize> {
        let query = label.to_lowercase();
        self.file
            .header
            .signals
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                if exact {
                    s.label == label
                } else {
                    s.label.to_lowercase().contains(&query)
                }
            })
            .map(|(i, _)| i)
            .collect()
    }
}

fn read_usize(data: &[u8], offset: usize, size: usize, name: &'static str) -> Result<usize> {
    let s = read_field(data, offset, size, name)?;
    s.parse::<usize>()
        .map_err(|_| EdfError::InvalidHeaderField {
            field: name,
            reason: format!("not a valid unsigned integer: {s:?}"),
        })
}

fn read_field(data: &[u8], offset: usize, size: usize, name: &'static str) -> Result<String> {
    let bytes = data
        .get(offset..offset + size)
        .ok_or(EdfError::InvalidHeaderField {
            field: name,
            reason: "header truncated".to_string(),
        })?;
    Ok(String::from_utf8_lossy(bytes).trim().to_string())
}

/// Lightweight metadata extracted from an EDF/EDF+ file header without
/// scanning data records or building an annotation index.
#[derive(Debug, Clone)]
pub struct EdfMetadata {
    pub variant: EdfVariant,
    pub num_signals: usize,
    pub num_records: i64,
    pub record_duration: f64,
    pub duration: f64,
    pub patient_id: String,
    pub recording_id: String,
    pub signal_labels: Vec<String>,
    pub sample_rates: Vec<f64>,
}

impl EdfFile {
    /// Read only the header; no annotation scan, no persistent mmap.
    ///
    /// This is a cheap operation suitable for batch inspection of many files.
    /// It does not memory-map the file, does not spawn background threads,
    /// and does not read any data records.
    pub fn inspect(path: impl AsRef<Path>) -> Result<EdfMetadata> {
        let mut file = std::fs::File::open(path.as_ref()).map_err(|e| EdfError::FileOpen {
            path: path.as_ref().to_path_buf(),
            source: e,
        })?;

        // EDF header = 256 bytes + 256 bytes per signal.
        // We need to read the main header to know num_signals, then read
        // enough to get all signal headers.
        let mut buf = vec![b' '; 256];
        file.read_exact(&mut buf).map_err(|e| EdfError::FileOpen {
            path: path.as_ref().to_path_buf(),
            source: e,
        })?;

        let num_signals = read_usize(&buf, 252, 4, "num_signals")?;
        if num_signals == 0 {
            return Err(EdfError::NoSignals);
        }

        let expected_header_bytes = 256 + 256 * num_signals;
        buf.resize(expected_header_bytes, 0);
        file.read_exact(&mut buf[256..])
            .map_err(|e| EdfError::FileOpen {
                path: path.as_ref().to_path_buf(),
                source: e,
            })?;

        let header = EdfHeader::parse(&buf)?;

        let signal_labels: Vec<String> =
            header.signals.iter().map(|s| s.label.clone()).collect();
        let sample_rates: Vec<f64> = header
            .signals
            .iter()
            .map(|s: &SignalHeader| s.sample_rate(header.record_duration_secs))
            .collect();

        let duration = (header.num_records.max(0) as f64) * header.record_duration_secs;

        Ok(EdfMetadata {
            variant: header.variant,
            num_signals: header.num_signals,
            num_records: header.num_records,
            record_duration: header.record_duration_secs,
            duration,
            patient_id: header.patient_id,
            recording_id: header.recording_id,
            signal_labels,
            sample_rates,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn edf_file_api() {
        let file = build_test_file();
        let edf = EdfFile::open(file.path()).unwrap();

        assert_eq!(edf.num_signals(), 1);
        assert_eq!(edf.num_records(), 2);
        assert_eq!(edf.record_duration(), 1.0);
        assert_eq!(edf.duration(), 2.0);
        assert_eq!(edf.variant(), EdfVariant::Edf);

        let sig = edf.signal(0).unwrap();
        assert_eq!(sig.len(), 8);

        let val = sig.get_physical(0).unwrap();
        assert!((val - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn array_proxy_single_signal() {
        let file = build_test_file();
        let edf = EdfFile::open(file.path()).unwrap();
        let proxy = edf.array_proxy(None).unwrap();
        assert_eq!(proxy.shape(), (1, 8));
        assert_eq!(proxy.sample_rate(), 4.0);
        let val = proxy.get(0, 3).unwrap();
        assert!((val - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn array_proxy_explicit_indices() {
        let file = build_test_file();
        let edf = EdfFile::open(file.path()).unwrap();
        let proxy = edf.array_proxy(Some(&[0])).unwrap();
        assert_eq!(proxy.shape(), (1, 8));
    }

    #[test]
    fn signal_indices_by_rate_groups() {
        let file = build_test_file();
        let edf = EdfFile::open(file.path()).unwrap();
        let by_rate = edf.signal_indices_by_rate();
        assert_eq!(by_rate.len(), 1);
        let indices: Vec<usize> = by_rate.values().next().unwrap().clone();
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn annotations_ready_plain_edf() {
        let file = build_test_file();
        let edf = EdfFile::open(file.path()).unwrap();
        assert!(edf.annotations_ready());
    }

    #[test]
    fn scan_progress_plain_edf() {
        let file = build_test_file();
        let edf = EdfFile::open(file.path()).unwrap();
        let (done, total) = edf.scan_progress();
        assert_eq!(done, total);
    }

    #[test]
    fn annotations_returns_vec() {
        let file = build_test_file();
        let edf = EdfFile::open(file.path()).unwrap();
        let anns = edf.annotations();
        assert!(anns.is_empty());
    }

    #[test]
    fn warnings_includes_header_and_annotation_warnings() {
        let file = build_test_file();
        let edf = EdfFile::open(file.path()).unwrap();
        let w = edf.warnings();
        assert!(w.is_empty());
    }

    #[test]
    fn signal_by_label() {
        let file = build_test_file();
        let edf = EdfFile::open(file.path()).unwrap();

        let sig = edf.signal_by_label("EEG").unwrap();
        assert_eq!(sig.len(), 8);

        let err = edf.signal_by_label("NONEXISTENT").unwrap_err();
        assert!(matches!(err, EdfError::SignalNotFound { .. }));
    }

    fn build_test_file() -> NamedTempFile {
        let num_signals = 1;
        let header_bytes = 256 + 256 * num_signals;
        let samples_per_record = 4;
        let num_records = 2;
        let mut buf = vec![b' '; header_bytes];

        write_hdr(&mut buf, 0, 8, "0");
        write_hdr(&mut buf, 8, 80, "X X X X");
        write_hdr(&mut buf, 88, 80, "Startdate X X X X");
        write_hdr(&mut buf, 168, 8, "01.01.00");
        write_hdr(&mut buf, 176, 8, "00.00.00");
        write_hdr(&mut buf, 184, 8, &header_bytes.to_string());
        write_hdr(&mut buf, 192, 44, "");
        write_hdr(&mut buf, 236, 8, &num_records.to_string());
        write_hdr(&mut buf, 244, 8, "1");
        write_hdr(&mut buf, 252, 4, &num_signals.to_string());

        let sig = &mut buf[256..];
        write_sig(sig, 0, 1, 0, 16, "EEG");
        write_sig(sig, 0, 1, 16, 80, "");
        write_sig(sig, 0, 1, 96, 8, "uV");
        write_sig(sig, 0, 1, 104, 8, "-100");
        write_sig(sig, 0, 1, 112, 8, "100");
        write_sig(sig, 0, 1, 120, 8, "-100");
        write_sig(sig, 0, 1, 128, 8, "100");
        write_sig(sig, 0, 1, 136, 80, "");
        write_sig(sig, 0, 1, 216, 8, &samples_per_record.to_string());
        write_sig(sig, 0, 1, 224, 32, "");

        for i in 0..(num_records * samples_per_record) {
            buf.extend_from_slice(&(i as i16).to_le_bytes());
        }

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    fn write_hdr(buf: &mut [u8], offset: usize, size: usize, value: &str) {
        let bytes = value.as_bytes();
        buf[offset..offset + bytes.len().min(size)]
            .copy_from_slice(&bytes[..bytes.len().min(size)]);
    }

    fn write_sig(data: &mut [u8], index: usize, ns: usize, fo: usize, fs: usize, value: &str) {
        let start = fo * ns + fs * index;
        let bytes = value.as_bytes();
        data[start..start + bytes.len().min(fs)].copy_from_slice(&bytes[..bytes.len().min(fs)]);
    }
}

#[cfg(test)]
mod fixture_tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name)
    }

    #[test]
    fn parse_test_generator() {
        let edf = EdfFile::open(fixture_path("test_generator.edf")).unwrap();
        assert_eq!(edf.variant(), EdfVariant::Edf);
        assert_eq!(edf.num_signals(), 16);
        assert_eq!(edf.duration(), 900.0);
    }

    #[test]
    fn annotations_ready_after_access() {
        let edf = EdfFile::open(fixture_path("test_generator_2.edf")).unwrap();
        let _ = edf.annotations();
        assert!(edf.annotations_ready());
    }

    #[test]
    fn async_scan_edf_plus_c() {
        let edf = EdfFile::open(fixture_path("test_generator_2.edf")).unwrap();
        assert_eq!(edf.variant(), EdfVariant::EdfPlusC);
        let anns = edf.annotations();
        assert!(!anns.is_empty());
        let (done, total) = edf.scan_progress();
        assert_eq!(done, total);
    }

    #[test]
    fn async_scan_edf_plus_d() {
        let edf = EdfFile::open(fixture_path("edfPlusD.edf")).unwrap();
        assert_eq!(edf.variant(), EdfVariant::EdfPlusD);
        let anns = edf.annotations();
        let _ = anns;
        let (done, total) = edf.scan_progress();
        assert_eq!(done, total);
    }

    #[test]
    fn signal_indices_by_rate_mixed() {
        let edf = EdfFile::open(fixture_path("test_generator.edf")).unwrap();
        let by_rate = edf.signal_indices_by_rate();
        assert!(by_rate.len() >= 1);
        let total: usize = by_rate.values().map(|v| v.len()).sum();
        assert_eq!(total, edf.ordinary_signal_indices().len());
    }

    #[test]
    fn array_proxy_mixed_rates_error() {
        let edf = EdfFile::open(fixture_path("test_generator.edf")).unwrap();
        let result = edf.array_proxy(None);
        assert!(result.is_err());
    }

    #[test]
    fn array_proxy_same_rate_group() {
        let edf = EdfFile::open(fixture_path("test_generator.edf")).unwrap();
        let by_rate = edf.signal_indices_by_rate();
        let group = by_rate.values().next().unwrap();
        let proxy = edf.array_proxy(Some(group)).unwrap();
        assert_eq!(proxy.shape().0, group.len());
        assert!(proxy.shape().1 > 0);
    }

    #[test]
    fn annotations_in_range_includes_start_boundary() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let all = edf.annotations();
        let target = all[0].onset;
        let anns = edf.annotations_in_range(target, target + 1.0);
        assert!(
            anns.iter().any(|a| (a.onset - target).abs() < f64::EPSILON),
            "annotation at start boundary should be included"
        );
    }

    #[test]
    fn annotations_in_range_excludes_end_boundary() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let all = edf.annotations();
        let target = all[0].onset;
        let anns = edf.annotations_in_range(target - 1.0, target);
        assert!(
            !anns.iter().any(|a| (a.onset - target).abs() < f64::EPSILON),
            "annotation at end boundary should be excluded"
        );
    }

    #[test]
    fn annotations_before_excludes_exact_onset() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let all = edf.annotations();
        let target = all[0].onset;
        let anns = edf.annotations_before(target);
        assert!(
            !anns.iter().any(|a| (a.onset - target).abs() < f64::EPSILON),
            "annotation at exact onset should be excluded from annotations_before"
        );
    }

    #[test]
    fn annotations_before_includes_all_before_target() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let all = edf.annotations();
        let target = all[0].onset;
        let anns = edf.annotations_before(target + 1.0);
        assert!(
            anns.iter().any(|a| (a.onset - target).abs() < f64::EPSILON),
            "annotation just before target+1 should be included"
        );
    }

    #[test]
    fn annotations_after_includes_exact_onset() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let all = edf.annotations();
        let target = all[0].onset;
        let anns = edf.annotations_after(target);
        assert!(
            anns.iter().any(|a| (a.onset - target).abs() < f64::EPSILON),
            "annotation at exact onset should be included in annotations_after"
        );
    }

    #[test]
    fn annotations_after_excludes_before_onset() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let all = edf.annotations();
        let target = all[0].onset;
        let anns = edf.annotations_after(target + 0.001);
        assert!(
            !anns.iter().any(|a| (a.onset - target).abs() < f64::EPSILON),
            "annotation at target should be excluded when querying after target+0.001"
        );
    }

    #[test]
    fn filter_annotations_regex_case_insensitive() {
        // Fixture contains "RECORD START" — lowercase pattern must match.
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let anns = edf.filter_annotations("record", true).unwrap();
        assert!(
            anns.iter().any(|a| a.text == "RECORD START"),
            "case-insensitive regex should match uppercase fixture text"
        );
    }

    #[test]
    fn filter_annotations_substring_case_insensitive() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let anns = edf.filter_annotations("record", false).unwrap();
        assert!(anns.iter().any(|a| a.text == "RECORD START"));
    }

    #[test]
    fn filter_annotations_substring_no_match() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let anns = edf.filter_annotations("zzzzz_no_such_text", false).unwrap();
        assert!(anns.is_empty());
    }

    #[test]
    fn filter_annotations_regex_no_match() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let anns = edf.filter_annotations(r"^zzzzz", true).unwrap();
        assert!(anns.is_empty());
    }

    #[test]
    fn annotations_by_text_exact_match() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let anns = edf.annotations_by_text("RECORD START");
        assert_eq!(anns.len(), 1);
        assert_eq!(anns[0].text, "RECORD START");
    }

    #[test]
    fn annotations_by_text_case_sensitive() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let anns = edf.annotations_by_text("record start");
        assert!(anns.is_empty(), "exact match should be case-sensitive");
    }

    #[test]
    fn annotations_by_text_no_match() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let anns = edf.annotations_by_text("Nonexistent");
        assert!(anns.is_empty());
    }

    #[test]
    fn filter_annotations_invalid_regex_uses_invalid_argument() {
        let edf = EdfFile::open(fixture_path("edfPlusC.edf")).unwrap();
        let err = edf.filter_annotations("[invalid", true).unwrap_err();
        assert!(matches!(err, EdfError::InvalidArgument { .. }));
    }

    #[test]
    fn find_all_signals_partial_case_insensitive() {
        let edf = EdfFile::open(fixture_path("test_generator.edf")).unwrap();
        // "dc" is a substring of DC01, DC04, DC03, DC02 (indices 12, 13, 14, 15)
        let indices = edf.find_all_signals("dc", false);
        assert!(indices.len() >= 1, "partial match should find at least one signal");
        // Verify all returned indices have labels containing "dc" (case-insensitive)
        for &idx in &indices {
            let label_lower = edf.header().signals[idx].label.to_lowercase();
            assert!(
                label_lower.contains("dc"),
                "signal index {} label '{}' should contain 'dc' case-insensitively",
                idx,
                edf.header().signals[idx].label
            );
        }
        assert_eq!(indices, vec![12, 13, 14, 15]);
    }

    #[test]
    fn find_all_signals_exact_match() {
        let edf = EdfFile::open(fixture_path("test_generator.edf")).unwrap();
        // Exact match for "F4" should return exactly index 0
        let indices = edf.find_all_signals("F4", true);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn find_all_signals_exact_case_sensitive() {
        let edf = EdfFile::open(fixture_path("test_generator.edf")).unwrap();
        // "f4" (lowercase) should NOT match "F4" with exact=true
        let indices = edf.find_all_signals("f4", true);
        assert!(indices.is_empty(), "exact match should be case-sensitive");
    }

    #[test]
    fn find_all_signals_no_match() {
        let edf = EdfFile::open(fixture_path("test_generator.edf")).unwrap();
        let indices = edf.find_all_signals("zzzzz_no_such_label", false);
        assert!(indices.is_empty(), "non-existent label should return empty vec");
    }

    #[test]
    fn inspect_returns_metadata_without_scan() {
        let meta = EdfFile::inspect(fixture_path("edfPlusD.edf")).unwrap();
        assert_eq!(meta.variant, EdfVariant::EdfPlusD);
        assert!(meta.num_signals > 0, "should have at least one signal");
        assert_eq!(meta.signal_labels.len(), meta.num_signals);
        assert_eq!(meta.sample_rates.len(), meta.num_signals);
        assert!(meta.signal_labels.iter().any(|l| !l.is_empty()));
    }

    #[test]
    fn inspect_plain_edf() {
        let meta = EdfFile::inspect(fixture_path("test_generator.edf")).unwrap();
        assert_eq!(meta.variant, EdfVariant::Edf);
        assert!(meta.num_signals > 0);
    }

    #[test]
    fn inspect_nonexistent_path() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does_not_exist.edf");
        let err = EdfFile::inspect(&missing).unwrap_err();
        assert!(matches!(err, EdfError::FileOpen { .. }));
    }

    #[test]
    fn inspect_signal_labels_and_rates_align() {
        let meta = EdfFile::inspect(fixture_path("test_generator.edf")).unwrap();
        assert_eq!(meta.signal_labels.len(), meta.num_signals);
        assert_eq!(meta.sample_rates.len(), meta.num_signals);
        assert_eq!(meta.signal_labels.len(), meta.sample_rates.len());
    }
}
