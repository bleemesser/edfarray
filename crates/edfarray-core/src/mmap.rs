use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;

use memmap2::Mmap;

use crate::annotation::AnnotationIndex;
use crate::error::{EdfError, Result};
use crate::header::{EdfHeader, EdfVariant};
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
}
