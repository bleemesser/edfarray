use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;

use crate::annotation::Annotation;
use crate::error::{EdfError, Result};
use crate::header::{EdfHeader, EdfVariant, PatientInfo, RecordingInfo};
use crate::mmap::MappedFile;
use crate::proxy::SignalProxy;

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
    pub fn annotations(&self) -> &[Annotation] {
        &self.file.annotations.annotations
    }

    /// Parse warnings accumulated during file open (malformed TALs, etc.).
    pub fn warnings(&self) -> Vec<String> {
        let mut w = self.file.header.warnings.clone();
        w.extend(self.file.annotations.warnings.iter().cloned());
        w
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
    pub fn read_page(
        &self,
        signal_indices: &[usize],
        start_sec: f64,
        end_sec: f64,
    ) -> Result<Vec<Vec<f64>>> {
        self.advise_time_range(start_sec, end_sec);
        let file = &self.file;
        signal_indices
            .par_iter()
            .map(|&idx| {
                let proxy = SignalProxy::new(Arc::clone(file), idx)?;
                let sr = proxy.sample_rate();
                let s_start = (start_sec * sr) as usize;
                let s_end = ((end_sec * sr) as usize).min(proxy.len());
                if s_start >= proxy.len() || s_start >= s_end {
                    return Ok(Vec::new());
                }
                let count = s_end - s_start;
                let mut buf = vec![0.0f64; count];
                proxy.read_physical(s_start, s_end, &mut buf)?;
                Ok(buf)
            })
            .collect()
    }

    /// Read a page of digital data for the given signals over a time range.
    pub fn read_page_digital(
        &self,
        signal_indices: &[usize],
        start_sec: f64,
        end_sec: f64,
    ) -> Result<Vec<Vec<i16>>> {
        self.advise_time_range(start_sec, end_sec);
        let file = &self.file;
        signal_indices
            .par_iter()
            .map(|&idx| {
                let proxy = SignalProxy::new(Arc::clone(file), idx)?;
                let sr = proxy.sample_rate();
                let s_start = (start_sec * sr) as usize;
                let s_end = ((end_sec * sr) as usize).min(proxy.len());
                if s_start >= proxy.len() || s_start >= s_end {
                    return Ok(Vec::new());
                }
                let count = s_end - s_start;
                let mut buf = vec![0i16; count];
                proxy.read_digital(s_start, s_end, &mut buf)?;
                Ok(buf)
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
    fn parse_short_psg() {
        let edf = EdfFile::open(fixture_path("short_psg.edf")).unwrap();
        eprintln!("variant: {:?}", edf.variant());
        eprintln!("patient_id: {:?}", edf.header().patient_id);
        eprintln!("patient: {:?}", edf.patient());
        assert_eq!(edf.patient().sex, Some(crate::header::Sex::Female));
        assert_eq!(edf.patient().name.as_deref(), Some("Female 33yr"));
    }
}
