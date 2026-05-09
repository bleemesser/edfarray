use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;

use memmap2::Mmap;

use crate::annotation::AnnotationIndex;
use crate::error::{EdfError, Result};
use crate::header::{EdfHeader, EdfVariant};
use crate::proxy::SignalProxy;
use crate::record::RecordLayout;

/// Internal state machine for the background annotation scan.
enum AnnotationState {
    NotStarted,
    Scanning {
        progress: Arc<AtomicUsize>,
        total: usize,
    },
    Complete(AnnotationIndex),
}

/// A memory-mapped EDF file with parsed header, record layout, and deferred annotation index.
///
/// On open, the file is mapped into memory and the header and record layout are parsed
/// synchronously (fast, fixed-size). For files with annotation signals, the annotation
/// scan runs in a background thread so the file can be used immediately for signal reads.
///
/// For plain EDF files (no annotation signals), the annotation index is trivially
/// computed at open time (uniform record spacing) with no background work.
pub struct MappedFile {
    mmap: Mmap,
    pub header: EdfHeader,
    pub layout: RecordLayout,
    annotations: RwLock<AnnotationState>,
    scan_done: (Mutex<bool>, Condvar),
}

impl std::fmt::Debug for MappedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedFile")
            .field("len", &self.mmap.len())
            .field("header", &self.header)
            .field("layout", &self.layout)
            .finish()
    }
}

impl MappedFile {
    /// Open and parse an EDF/EDF+ file.
    ///
    /// Parses the header and record layout synchronously, then spawns a background
    /// thread to build the annotation index (for files with annotation signals).
    /// Returns immediately — signal data can be read before the scan finishes.
    pub fn open(path: &Path) -> Result<Arc<Self>> {
        let file = std::fs::File::open(path).map_err(|e| EdfError::FileOpen {
            path: path.to_path_buf(),
            source: e,
        })?;

        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| EdfError::MmapFailed {
            path: path.to_path_buf(),
            source: e,
        })?;

        let header = EdfHeader::parse(&mmap)?;
        let layout = RecordLayout::from_header(&header);

        let has_annotations = header.signals.iter().any(|s| s.is_annotation);

        let initial_state = if !has_annotations {
            let num_records = header.num_records.max(0) as usize;
            let record_onsets: Vec<f64> = (0..num_records)
                .map(|i| i as f64 * header.record_duration_secs)
                .collect();
            AnnotationState::Complete(AnnotationIndex {
                annotations: Vec::new(),
                record_onsets,
                starttime_subsecond: 0.0,
                warnings: Vec::new(),
            })
        } else {
            AnnotationState::NotStarted
        };

        let mapped = Arc::new(MappedFile {
            mmap,
            header,
            layout,
            annotations: RwLock::new(initial_state),
            scan_done: (Mutex::new(!has_annotations), Condvar::new()),
        });

        if has_annotations {
            mapped.start_annotation_scan();
        }

        Ok(mapped)
    }

    /// Spawn a background thread to scan all data records and build the annotation index.
    fn start_annotation_scan(self: &Arc<Self>) {
        let num_records = self.header.num_records.max(0) as usize;
        let progress = Arc::new(AtomicUsize::new(0));

        {
            let mut state = self.annotations.write().unwrap();
            *state = AnnotationState::Scanning {
                progress: Arc::clone(&progress),
                total: num_records,
            };
        }

        let file = Arc::clone(self);
        thread::spawn(move || {
            let index = AnnotationIndex::build_with_progress(
                &file.mmap,
                &file.header,
                &file.layout,
                &progress,
            );

            match index {
                Ok(idx) => {
                    let mut state = file.annotations.write().unwrap();
                    *state = AnnotationState::Complete(idx);
                }
                Err(e) => {
                    let mut state = file.annotations.write().unwrap();
                    *state = AnnotationState::Complete(AnnotationIndex {
                        annotations: Vec::new(),
                        record_onsets: (0..num_records)
                            .map(|i| i as f64 * file.header.record_duration_secs)
                            .collect(),
                        starttime_subsecond: 0.0,
                        warnings: vec![format!("annotation scan failed: {e}")],
                    });
                }
            }

            let (lock, cvar) = &file.scan_done;
            let mut done = lock.lock().unwrap();
            *done = true;
            cvar.notify_all();
        });
    }

    /// Block until the background annotation scan has completed.
    ///
    /// This is a no-op if the scan is already done (plain EDF, or scan finished).
    pub fn wait_for_annotations(&self) {
        let (lock, cvar) = &self.scan_done;
        let mut done = lock.lock().unwrap();
        while !*done {
            done = cvar.wait(done).unwrap();
        }
    }

    /// Check whether the annotation scan has completed without blocking.
    pub fn annotations_ready(&self) -> bool {
        let done = self.scan_done.0.lock().unwrap();
        *done
    }

    /// Returns `(records_scanned, total_records)` for the background annotation scan.
    ///
    /// Non-blocking. Can be polled to show progress for large files.
    pub fn scan_progress(&self) -> (usize, usize) {
        let state = self.annotations.read().unwrap();
        match &*state {
            AnnotationState::NotStarted => (0, 0),
            AnnotationState::Scanning { progress, total } => {
                (progress.load(Ordering::Relaxed), *total)
            }
            AnnotationState::Complete(idx) => {
                let n = idx.record_onsets.len();
                (n, n)
            }
        }
    }

    /// Block until the annotation scan is complete, then call `f` with the index.
    pub fn with_annotations<T>(&self, f: impl FnOnce(&AnnotationIndex) -> T) -> T {
        self.wait_for_annotations();
        let state = self.annotations.read().unwrap();
        match &*state {
            AnnotationState::Complete(idx) => f(idx),
            _ => unreachable!("annotations must be complete after wait"),
        }
    }

    /// Onset time (in seconds) for the given data record.
    ///
    /// For EDF and EDF+C, this is computed directly as `rec_idx * record_duration`
    /// without blocking. For EDF+D, this blocks until the annotation scan completes
    /// to get the actual (potentially non-uniform) onset from the TALs.
    pub fn record_onset(&self, rec_idx: usize) -> f64 {
        if self.header.variant != EdfVariant::EdfPlusD {
            return rec_idx as f64 * self.header.record_duration_secs;
        }
        self.with_annotations(|idx| {
            idx.record_onsets
                .get(rec_idx)
                .copied()
                .unwrap_or(rec_idx as f64 * self.header.record_duration_secs)
        })
    }

    /// Find the first and last sample indices for a signal that fall within a time range.
    ///
    /// Uses record onsets to resolve the time range to actual sample indices,
    /// accounting for gaps in EDF+D files.
    pub fn sample_range_for_time(&self, proxy: &SignalProxy, start_sec: f64, end_sec: f64) -> (usize, usize) {
        let num_records = self.header.num_records.max(0) as usize;
        if num_records == 0 {
            return (0, 0);
        }

        let samples_per_record = proxy.header().num_samples;
        let sample_rate = proxy.sample_rate();
        let total_samples = num_records * samples_per_record;

        // Clamp time range
        let start_sec = start_sec.max(0.0);
        let end_sec = end_sec.max(0.0);

        // Empty range
        if start_sec >= end_sec {
            return (0, 0);
        }

        // Get record onsets (blocks for EDF+D)
        let onsets: Vec<f64> = self.with_annotations(|idx| {
            idx.record_onsets.clone()
        });

        if onsets.is_empty() {
            // Fallback: uniform spacing
            let s_start = (start_sec * sample_rate) as usize;
            let s_end = ((end_sec * sample_rate) as usize).min(total_samples);
            return (s_start, s_end);
        }

        // Binary search: find first record whose samples could extend past start_sec
        let first_rec = onsets.partition_point(|&o| o + self.header.record_duration_secs <= start_sec);
        // Binary search: find first record where onset < end_sec
        let end_rec_pt = onsets.partition_point(|&o| o < end_sec);

        // If no record starts before end_sec, no samples fall in the range
        if end_rec_pt == 0 {
            return (0, 0);
        }

        // last_rec is the last record that starts before end_sec
        let last_rec = end_rec_pt.saturating_sub(1).min(num_records - 1);
        // first_rec is the first record whose onset >= start_sec
        let first_rec = first_rec.min(num_records - 1);

        if first_rec > last_rec {
            return (0, 0);
        }

        // O(1) arithmetic for s_start: first sample whose time >= start_sec
        let rec_onset = onsets[first_rec];
        let start_offset = (((start_sec - rec_onset) * sample_rate).ceil() as isize)
            .clamp(0, samples_per_record as isize) as usize;
        let s_start = first_rec * samples_per_record + start_offset;

        // O(1) arithmetic for s_end: exclusive upper bound, first sample whose time >= end_sec
        let last_rec_onset = onsets[last_rec];
        let end_offset = (((end_sec - last_rec_onset) * sample_rate).ceil() as isize)
            .clamp(0, samples_per_record as isize) as usize;
        let s_end = last_rec * samples_per_record + end_offset;

        let s_start = s_start.min(total_samples);
        let s_end = s_end.min(total_samples);

        (s_start, s_end)
    }

    /// Raw bytes of the entire file (the mmap backing).
    pub fn data(&self) -> &[u8] {
        &self.mmap
    }

    /// Extract the raw bytes of a single data record.
    pub fn record_bytes(&self, rec_idx: usize) -> Result<&[u8]> {
        let num_records = self.header.num_records.max(0) as usize;
        if rec_idx >= num_records {
            return Err(EdfError::RecordOutOfRange {
                index: rec_idx,
                count: num_records,
            });
        }
        let start = self.header.data_offset() + rec_idx * self.layout.record_size;
        let end = start + self.layout.record_size;
        self.mmap.get(start..end).ok_or(EdfError::RecordOutOfRange {
            index: rec_idx,
            count: num_records,
        })
    }

    /// Hint to the OS that we'll soon need the bytes for the given record range.
    #[cfg(unix)]
    pub fn advise_willneed(&self, start_record: usize, end_record: usize) {
        let data_offset = self.header.data_offset();
        let byte_start = data_offset + start_record * self.layout.record_size;
        let byte_end = data_offset + end_record * self.layout.record_size;
        let byte_end = byte_end.min(self.mmap.len());
        if byte_start < byte_end
            && let Some(slice) = self.mmap.get(byte_start..byte_end)
        {
            let _ = unsafe {
                libc::madvise(
                    slice.as_ptr() as *mut libc::c_void,
                    byte_end - byte_start,
                    libc::MADV_WILLNEED,
                )
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::NamedTempFile;

    fn build_test_file() -> (NamedTempFile, usize, usize) {
        let num_signals = 1;
        let header_bytes = 256 + 256 * num_signals;
        let num_records = 3;
        let samples_per_record = 4;
        let record_size = samples_per_record * 2;

        let mut buf = vec![b' '; header_bytes];
        write_field(&mut buf, 0, 8, "0");
        write_field(&mut buf, 8, 80, "X X X X");
        write_field(&mut buf, 88, 80, "Startdate X X X X");
        write_field(&mut buf, 168, 8, "01.01.00");
        write_field(&mut buf, 176, 8, "00.00.00");
        write_field(&mut buf, 184, 8, &header_bytes.to_string());
        write_field(&mut buf, 192, 44, "");
        write_field(&mut buf, 236, 8, &num_records.to_string());
        write_field(&mut buf, 244, 8, "1");
        write_field(&mut buf, 252, 4, &num_signals.to_string());

        let sig_data = &mut buf[256..];
        write_sig(sig_data, 0, 1, 0, 16, "EEG");
        write_sig(sig_data, 0, 1, 16, 80, "");
        write_sig(sig_data, 0, 1, 96, 8, "uV");
        write_sig(sig_data, 0, 1, 104, 8, "-3200");
        write_sig(sig_data, 0, 1, 112, 8, "3200");
        write_sig(sig_data, 0, 1, 120, 8, "-32768");
        write_sig(sig_data, 0, 1, 128, 8, "32767");
        write_sig(sig_data, 0, 1, 136, 80, "");
        write_sig(sig_data, 0, 1, 216, 8, &samples_per_record.to_string());
        write_sig(sig_data, 0, 1, 224, 32, "");

        for rec in 0..num_records {
            for sample in 0..samples_per_record {
                let val = (rec * samples_per_record + sample) as i16;
                buf.extend_from_slice(&val.to_le_bytes());
            }
        }

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        (file, samples_per_record, record_size)
    }

    #[test]
    fn open_and_read_records() {
        let (file, _, record_size) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();

        assert_eq!(mapped.header.num_signals, 1);
        assert_eq!(mapped.header.num_records, 3);
        assert_eq!(mapped.layout.record_size, record_size);

        let rec0 = mapped.record_bytes(0).unwrap();
        assert_eq!(rec0.len(), record_size);

        let rec2 = mapped.record_bytes(2).unwrap();
        let first_sample = i16::from_le_bytes([rec2[0], rec2[1]]);
        assert_eq!(first_sample, 8);
    }

    #[test]
    fn record_out_of_range() {
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        assert!(mapped.record_bytes(3).is_err());
    }

    #[test]
    fn open_nonexistent_file() {
        let result = MappedFile::open(Path::new("/tmp/nonexistent_edf_file.edf"));
        assert!(matches!(result.unwrap_err(), EdfError::FileOpen { .. }));
    }

    #[test]
    fn scan_progress_no_annotations() {
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        let (done, total) = mapped.scan_progress();
        assert_eq!(done, total);
    }

    #[test]
    fn with_annotations_plain_edf() {
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        mapped.with_annotations(|idx| {
            assert!(idx.annotations.is_empty());
            assert_eq!(idx.record_onsets.len(), 3);
            assert!((idx.record_onsets[0] - 0.0).abs() < f64::EPSILON);
            assert!((idx.record_onsets[1] - 1.0).abs() < f64::EPSILON);
            assert!((idx.record_onsets[2] - 2.0).abs() < f64::EPSILON);
        });
    }

    #[test]
    fn record_onset_plain_edf() {
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        assert!((mapped.record_onset(0) - 0.0).abs() < f64::EPSILON);
        assert!((mapped.record_onset(1) - 1.0).abs() < f64::EPSILON);
        assert!((mapped.record_onset(2) - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn annotations_ready_plain_edf() {
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        assert!(mapped.annotations_ready());
    }

    #[test]
    fn wait_for_annotations_idempotent() {
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        mapped.wait_for_annotations();
        mapped.wait_for_annotations();
        let (done, total) = mapped.scan_progress();
        assert_eq!(done, total);
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

    #[test]
    fn sample_range_for_time_plain_edf_start_at_zero() {
        // Regression test for sentinel bug: start_sec=0.0 must include sample 0
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        let (s_start, s_end) = mapped.sample_range_for_time(&proxy, 0.0, 1.0);
        assert_eq!(s_start, 0, "sample 0 must be included when start_sec=0.0");
        assert_eq!(s_end, 4);
    }

    #[test]
    fn sample_range_for_time_all_onsets_after_end() {
        // Regression test for underflow: all onsets >= end_sec must return (0,0) without panic
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        // end_sec=0.0, all onsets (0.0, 1.0, 2.0) are >= 0.0, so partition_point(|o| o < 0.0) = 0
        let (s_start, s_end) = mapped.sample_range_for_time(&proxy, -1.0, 0.0);
        assert_eq!(s_start, 0);
        assert_eq!(s_end, 0);
    }

    #[test]
    fn sample_range_for_time_start_ge_end() {
        // start_sec >= end_sec returns empty range
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        let (s_start, s_end) = mapped.sample_range_for_time(&proxy, 5.0, 3.0);
        assert_eq!(s_start, 0);
        assert_eq!(s_end, 0);
        let (s_start, s_end) = mapped.sample_range_for_time(&proxy, 3.0, 3.0);
        assert_eq!(s_start, 0);
        assert_eq!(s_end, 0);
    }

    #[test]
    fn sample_range_for_time_edfd_gap_spanning() {
        // Use the existing edfPlusD.edf fixture which has non-uniform onsets
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/edfPlusD.edf");
        let mapped = MappedFile::open(&fixture).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        mapped.wait_for_annotations();

        // Verify onsets are non-uniform (has gaps)
        mapped.with_annotations(|idx| {
            assert!(idx.record_onsets.len() >= 3);
            let has_gap = idx.record_onsets.windows(2).any(|w| {
                (w[1] - w[0] - mapped.header.record_duration_secs).abs() > 0.001
            });
            assert!(has_gap, "fixture should have non-uniform onsets");
        });

        // Range spanning the gap: should skip the gap
        let (s_start, s_end) = mapped.sample_range_for_time(&proxy, 0.0, 10.0);
        assert_eq!(s_start, 0);
        let total_uniform = mapped.header.num_records as usize * proxy.header().num_samples;
        assert!(s_end < total_uniform, "gap should reduce sample count");
    }

    #[test]
    fn sample_range_for_time_edfd_inside_gap() {
        // Range entirely inside a gap returns empty
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/edfPlusD.edf");
        let mapped = MappedFile::open(&fixture).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        mapped.wait_for_annotations();

        // Find a gap region and query it
        mapped.with_annotations(|idx| {
            let onsets = &idx.record_onsets;
            for w in onsets.windows(2) {
                let gap_start = w[0] + mapped.header.record_duration_secs;
                let gap_end = w[1];
                if gap_end - gap_start > 0.5 {
                    let mid = (gap_start + gap_end) / 2.0;
                    let (s_start, s_end) = mapped.sample_range_for_time(&proxy, mid - 0.1, mid + 0.1);
                    assert_eq!(s_start, 0, "range inside gap should return (0,0)");
                    assert_eq!(s_end, 0);
                    break;
                }
            }
        });
    }

    #[test]
    fn sample_range_for_time_edfd_mid_record_range() {
        // Range that starts mid-record
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/edfPlusD.edf");
        let mapped = MappedFile::open(&fixture).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        mapped.wait_for_annotations();

        // Start at 0.5s which is mid-record 0
        let (s_start, s_end) = mapped.sample_range_for_time(&proxy, 0.5, 1.0);
        assert!(s_start > 0, "should skip the first portion of record 0");
        assert!(s_end > s_start);
    }

    #[test]
    fn sample_range_for_time_mid_record() {
        // Range starting mid-record
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        // start at 0.5s (midway through record 0), end at 1.5s (midway through record 1)
        let (s_start, s_end) = mapped.sample_range_for_time(&proxy, 0.5, 1.5);
        assert_eq!(s_start, 2); // sample index 2 is at time 0.5s
        assert_eq!(s_end, 6);   // exclusive: samples 2,3,4,5
    }

    #[test]
    fn sample_range_for_time_edfd_underflow_all_onsets_after_end() {
        // Regression test: all onsets >= end_sec should return (0,0) without panic
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/edfPlusD.edf");
        let mapped = MappedFile::open(&fixture).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        mapped.wait_for_annotations();

        // Query a time range entirely before the first record onset
        mapped.with_annotations(|idx| {
            let first_onset = idx.record_onsets.first().copied().unwrap_or(0.0);
            let before_first = first_onset - 10.0;
            let (s_start, s_end) = mapped.sample_range_for_time(&proxy, before_first, first_onset - 0.001);
            assert_eq!(s_start, 0);
            assert_eq!(s_end, 0);
        });
    }

    #[test]
    fn sample_range_for_time_edfd_mid_record() {
        // Range that starts mid-record
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/edfPlusD.edf");
        let mapped = MappedFile::open(&fixture).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        mapped.wait_for_annotations();

        // Start at 0.5s which is mid-record 0
        let (s_start, s_end) = mapped.sample_range_for_time(&proxy, 0.5, 1.0);
        assert!(s_start > 0, "should skip the first portion of record 0");
        assert!(s_end > s_start);
    }

}
