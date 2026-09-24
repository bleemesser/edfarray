use std::fs::TryLockError;
use std::ops::Deref;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::{self, JoinHandle};

use memmap2::Mmap;

use crate::annotation::AnnotationIndex;
use crate::error::{EdfError, Result};
use crate::grid::first_sample_at_or_after;
use crate::header::{EdfHeader, EdfVariant};
use crate::proxy::SignalProxy;
use crate::record::RecordLayout;

/// Upper bound on a single `WillNeed` hint. If a hint covers more than the kernel can keep,
/// the kernel evicts pages that the caller still uses.
#[cfg(unix)]
const MAX_WILLNEED_BYTES: usize = 64 << 20;

/// Access-pattern hint for a range of records.
///
/// The hint is advisory only. `memmap2` gives advice only on unix. On other platforms, every
/// variant is a no-op, but the variants stay in the type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Advice {
    /// No special treatment. This undoes an earlier hint.
    Normal,
    /// The range will be read front to back once.
    Sequential,
    /// The range will be needed soon.
    WillNeed,
}

#[cfg(unix)]
impl From<Advice> for memmap2::Advice {
    fn from(advice: Advice) -> Self {
        match advice {
            Advice::Normal => memmap2::Advice::Normal,
            Advice::Sequential => memmap2::Advice::Sequential,
            Advice::WillNeed => memmap2::Advice::WillNeed,
        }
    }
}

/// Internal state machine for the background annotation scan.
enum AnnotationState {
    NotStarted,
    Scanning {
        progress: Arc<AtomicUsize>,
        total: usize,
    },
    Complete(AnnotationIndex),
}

#[cfg(unix)]
fn page_size() -> usize {
    // SAFETY: sysconf takes an integer and returns one, with no memory or lifetime contract.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }.max(1)
}

/// Positional read that fills `buf` entirely.
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0usize;
        while done < buf.len() {
            let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "short read",
                ));
            }
            done += n;
        }
        Ok(())
    }
}

/// Lock accessors that tolerate poisoning. A scan thread that panics must not make every later
/// annotation access panic too.
fn write_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

fn read_lock<T>(lock: &RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

/// Marks the annotation scan as finished when dropped, also during a panic unwind.
struct ScanCompletion {
    file: Arc<MappedData>,
}

impl Drop for ScanCompletion {
    fn drop(&mut self) {
        {
            let mut state = write_lock(&self.file.annotations);
            if !matches!(*state, AnnotationState::Complete(_)) {
                *state = AnnotationState::Complete(
                    self.file
                        .fallback_index("annotation scan panicked".to_string()),
                );
            }
        }
        let (lock, cvar) = &self.file.scan_done;
        let mut done = lock.lock().unwrap_or_else(|e| e.into_inner());
        *done = true;
        cvar.notify_all();
    }
}

/// Memory-mapped EDF file with parsed header, record layout, and deferred annotation index.
///
/// While any `MappedFile` for a path is alive, the file holds a shared advisory lock, and
/// [`EdfWriter`](crate::writer::EdfWriter) rejects an overwrite of it. Dropping the last handle
/// stops the background annotation scan before the mapping and the lock are released.
pub struct MappedFile {
    data: Arc<MappedData>,
    scan: Mutex<Option<JoinHandle<()>>>,
}

/// Shared state behind a [`MappedFile`]. The background annotation scan holds this, not the
/// `MappedFile`. Thus the mapping ends when the last user handle is dropped.
pub struct MappedData {
    mmap: Mmap,
    /// Kept open alongside the mapping for positional reads on the streaming path.
    file: std::fs::File,
    pub header: EdfHeader,
    pub layout: RecordLayout,
    annotations: RwLock<AnnotationState>,
    scan_done: (Mutex<bool>, Condvar),
    /// Serializes on-demand index construction when the eager scan is disabled.
    scan_guard: Mutex<()>,
    cancel_scan: AtomicBool,
}

/// How the annotation index is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScanMode {
    /// Scan in the background as soon as the file is opened.
    #[default]
    Eager,
    /// Build the index on first annotation access, on the calling thread.
    Lazy,
}

impl Deref for MappedFile {
    type Target = MappedData;

    fn deref(&self) -> &MappedData {
        &self.data
    }
}

impl Drop for MappedFile {
    fn drop(&mut self) {
        self.data.cancel_scan.store(true, Ordering::Relaxed);
        let handle = self
            .scan
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(handle) = handle {
            // A panicked scan already recorded its fallback index; nothing is left to report.
            let _ = handle.join();
        }
    }
}

impl std::fmt::Debug for MappedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}

impl std::fmt::Debug for MappedData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MappedFile")
            .field("len", &self.mmap.len())
            .field("header", &self.header)
            .field("layout", &self.layout)
            .finish()
    }
}

impl MappedFile {
    /// Open an EDF file. For a file with an annotation channel, starts a background annotation
    /// scan.
    pub fn open(path: &Path) -> Result<Arc<Self>> {
        Self::open_with_variant(path, None)
    }

    /// Open a file, and optionally force the EDF/BDF variant instead of the variant that the
    /// header gives. The override controls only the plain/`+C`/`+D` distinction. The EDF-vs-BDF
    /// sample size always comes from the version field, so the function rejects an override
    /// with a different sample-size family.
    pub fn open_with_variant(path: &Path, variant: Option<EdfVariant>) -> Result<Arc<Self>> {
        Self::open_with_options(path, variant, ScanMode::default())
    }

    /// Open a file, choosing when the annotation index is built.
    ///
    /// [`ScanMode::Lazy`] does not read every record when the file opens. This is important for
    /// large files, where the scan otherwise competes with the reads of the caller for page
    /// cache.
    pub fn open_with_options(
        path: &Path,
        variant: Option<EdfVariant>,
        scan_mode: ScanMode,
    ) -> Result<Arc<Self>> {
        let file = std::fs::File::open(path).map_err(|e| EdfError::FileOpen {
            path: path.to_path_buf(),
            source: e,
        })?;

        // Filesystems without lock support return an error rather than WouldBlock. The lock is
        // advisory, so opening proceeds without it there.
        if let Err(TryLockError::WouldBlock) = file.try_lock_shared() {
            return Err(EdfError::Io {
                path: path.to_path_buf(),
                op: "opening",
                source: std::io::Error::new(
                    std::io::ErrorKind::ResourceBusy,
                    "an EdfWriter is still writing this file",
                ),
            });
        }

        // SAFETY: mapping a file is unsafe because another process can change its contents or
        // length underneath us. Concurrent writes can produce torn reads, and truncation makes
        // touching a vanished page raise SIGBUS, which no Rust code can catch. Callers are told
        // not to read a file being rewritten in place; see docs/reference/contracts.md.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| EdfError::MmapFailed {
            path: path.to_path_buf(),
            source: e,
        })?;

        let mut header = EdfHeader::parse(&mmap)?;
        if let Some(forced) = variant {
            let detected = header.variant;
            if forced.sample_size_bytes() != detected.sample_size_bytes() {
                return Err(EdfError::InvalidArgument {
                    name: "variant",
                    reason: format!(
                        "cannot override {detected} as {forced}: EDF/BDF sample size is \
                         determined by the version field and cannot be overridden"
                    ),
                });
            }
            if forced != detected {
                header.warnings.push(format!(
                    "variant overridden from detected {detected} to {forced}"
                ));
            }
            header.variant = forced;
        }
        header.reconcile_num_records_with_file_size(mmap.len());
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

        let data = Arc::new(MappedData {
            mmap,
            file,
            header,
            layout,
            annotations: RwLock::new(initial_state),
            scan_done: (Mutex::new(!has_annotations), Condvar::new()),
            scan_guard: Mutex::new(()),
            cancel_scan: AtomicBool::new(false),
        });

        let scan =
            (has_annotations && scan_mode == ScanMode::Eager).then(|| data.start_annotation_scan());

        Ok(Arc::new(MappedFile {
            data,
            scan: Mutex::new(scan),
        }))
    }
}

impl MappedData {
    /// Spawn a background thread to scan all data records and build the annotation index.
    fn start_annotation_scan(self: &Arc<Self>) -> JoinHandle<()> {
        let num_records = self.header.num_records.max(0) as usize;
        let progress = Arc::new(AtomicUsize::new(0));

        {
            let mut state = write_lock(&self.annotations);
            *state = AnnotationState::Scanning {
                progress: Arc::clone(&progress),
                total: num_records,
            };
        }

        let file = Arc::clone(self);
        thread::spawn(move || {
            // Signals completion even if the scan panics. Otherwise every later annotation
            // access will block forever on scan_done.
            let guard = ScanCompletion {
                file: Arc::clone(&file),
            };

            let resolved = file.build_annotation_index(&progress);
            *write_lock(&file.annotations) = AnnotationState::Complete(resolved);

            drop(guard);
        })
    }

    /// Scan every record's annotation channel and build the index.
    ///
    /// The scan reads the whole file once. Thus it tells the kernel to read far ahead and drop
    /// pages behind the cursor, so the page cache does not grow by the file size.
    fn build_annotation_index(&self, progress: &AtomicUsize) -> AnnotationIndex {
        let num_records = self.header.num_records.max(0) as usize;
        self.advise_records(0, num_records, Advice::Sequential);

        let index = AnnotationIndex::build_cancellable(
            &self.mmap,
            &self.header,
            &self.layout,
            progress,
            &self.cancel_scan,
        );

        // Leave the mapping without a lingering sequential hint, since callers read randomly.
        self.advise_records(0, num_records, Advice::Normal);

        match index {
            Ok(idx) => idx,
            Err(e) => self.fallback_index(format!("annotation scan failed: {e}")),
        }
    }

    /// Synthetic index used when the scan cannot produce a real one: uniform record onsets and
    /// no annotations.
    fn fallback_index(&self, warning: String) -> AnnotationIndex {
        let num_records = self.header.num_records.max(0) as usize;
        AnnotationIndex {
            annotations: Vec::new(),
            record_onsets: (0..num_records)
                .map(|i| i as f64 * self.header.record_duration_secs)
                .collect(),
            starttime_subsecond: 0.0,
            warnings: vec![warning],
        }
    }

    /// Block until the annotation index is available, building it here if the scan was deferred.
    pub fn wait_for_annotations(&self) {
        if self.annotations_ready() {
            return;
        }

        {
            // Only one thread builds a deferred index; the rest fall through to the wait below.
            let _guard = self.scan_guard.lock().unwrap_or_else(|e| e.into_inner());
            let deferred = matches!(&*read_lock(&self.annotations), AnnotationState::NotStarted);
            if deferred && !self.annotations_ready() {
                let progress = AtomicUsize::new(0);
                let index = self.build_annotation_index(&progress);
                *write_lock(&self.annotations) = AnnotationState::Complete(index);
                let (lock, cvar) = &self.scan_done;
                *lock.lock().unwrap_or_else(|e| e.into_inner()) = true;
                cvar.notify_all();
                return;
            }
        }

        let (lock, cvar) = &self.scan_done;
        let mut done = lock.lock().unwrap_or_else(|e| e.into_inner());
        while !*done {
            done = cvar.wait(done).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Whether the annotation scan is complete. Does not block.
    pub fn annotations_ready(&self) -> bool {
        *self.scan_done.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Annotation scan progress: `(records_scanned, total_records)`. Non-blocking.
    pub fn scan_progress(&self) -> (usize, usize) {
        let state = read_lock(&self.annotations);
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
        let state = read_lock(&self.annotations);
        match &*state {
            AnnotationState::Complete(idx) => f(idx),
            // ScanCompletion guarantees a Complete state once scan_done is set, but fall back
            // rather than panic if that ever fails.
            _ => f(&self.fallback_index("annotation scan did not complete".to_string())),
        }
    }

    /// Record onset time in seconds. For EDF+D, blocks until the non-uniform onsets are known.
    pub fn record_onset(&self, rec_idx: usize) -> f64 {
        if !self.header.variant.is_plus_d() {
            return rec_idx as f64 * self.header.record_duration_secs;
        }
        self.with_annotations(|idx| {
            idx.record_onsets
                .get(rec_idx)
                .copied()
                .unwrap_or(rec_idx as f64 * self.header.record_duration_secs)
        })
    }

    /// Resolve a time range to sample indices with the record onsets, which include EDF+D gaps.
    pub fn sample_range_for_time(
        &self,
        proxy: &SignalProxy,
        start_sec: f64,
        end_sec: f64,
    ) -> (usize, usize) {
        let num_records = self.header.num_records.max(0) as usize;
        if num_records == 0 {
            return (0, 0);
        }

        let samples_per_record = proxy.header().num_samples;
        let sample_rate = proxy.sample_rate();
        let total_samples = num_records.saturating_mul(samples_per_record);

        let start_sec = start_sec.max(0.0);
        let end_sec = end_sec.max(0.0);

        if start_sec >= end_sec {
            return (0, 0);
        }

        // Borrowed, not cloned: read_page calls this once per channel, and the onset table has
        // one entry per record.
        self.with_annotations(|idx| {
            let onsets = &idx.record_onsets;

            if onsets.is_empty() {
                let to_index = |t: f64| {
                    let idx = first_sample_at_or_after(t * sample_rate).max(0) as u64;
                    usize::try_from(idx).map_or(total_samples, |i| i.min(total_samples))
                };
                return (to_index(start_sec), to_index(end_sec));
            }

            let first_rec =
                onsets.partition_point(|&o| o + self.header.record_duration_secs <= start_sec);
            let end_rec_pt = onsets.partition_point(|&o| o < end_sec);

            if end_rec_pt == 0 {
                return (0, 0);
            }

            let last_rec = end_rec_pt.saturating_sub(1).min(num_records - 1);
            let first_rec = first_rec.min(num_records - 1);

            if first_rec > last_rec {
                return (0, 0);
            }

            let spr = i64::try_from(samples_per_record).unwrap_or(i64::MAX);
            let start_offset =
                first_sample_at_or_after((start_sec - onsets[first_rec]) * sample_rate)
                    .clamp(0, spr) as usize;
            let s_start = first_rec * samples_per_record + start_offset;

            let end_offset = first_sample_at_or_after((end_sec - onsets[last_rec]) * sample_rate)
                .clamp(0, spr) as usize;
            let s_end = last_rec * samples_per_record + end_offset;

            (s_start.min(total_samples), s_end.min(total_samples))
        })
    }

    /// Size of the mapping in bytes.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Whether the mapping is empty.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
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
        let start = self
            .header
            .data_offset()
            .saturating_add(rec_idx.saturating_mul(self.layout.record_size));
        let end = start.saturating_add(self.layout.record_size);
        self.mmap.get(start..end).ok_or(EdfError::RecordOutOfRange {
            index: rec_idx,
            count: num_records,
        })
    }

    /// Read `count` records starting at `start_record` into `buf` with a positional read.
    ///
    /// This is the streaming alternative to faulting the records in through the mapping. It
    /// costs one syscall for each chunk instead of one page fault for each record. Its resident
    /// set is only `buf`.
    pub fn read_records_into(
        &self,
        start_record: usize,
        count: usize,
        buf: &mut [u8],
    ) -> Result<()> {
        let Some((start, end)) = self.record_byte_range(start_record, start_record + count) else {
            return Ok(());
        };
        let want = (end - start).min(buf.len());
        read_exact_at(&self.file, &mut buf[..want], start as u64).map_err(|e| {
            EdfError::InvalidArgument {
                name: "record range",
                reason: format!("failed to read records at byte {start}: {e}"),
            }
        })
    }

    /// Sampled fraction of the pages for a record range that are already resident.
    ///
    /// Returns 1.0 if the residency is unknown. Thus callers keep the mmap path by default and
    /// do not stream on a guess.
    pub fn residency(&self, start_record: usize, end_record: usize) -> f64 {
        #[cfg(unix)]
        {
            let Some((start, end)) = self.record_byte_range(start_record, end_record) else {
                return 1.0;
            };
            let page = page_size();
            let pages = (end - start).div_ceil(page);
            if pages == 0 {
                return 1.0;
            }
            const SAMPLES: usize = 64;
            let step = (pages / SAMPLES).max(1);
            let base = self.mmap.as_ptr();
            let mut checked = 0usize;
            let mut resident = 0usize;
            let mut page_idx = 0usize;
            while page_idx < pages && checked < SAMPLES {
                let offset = start + page_idx * page;
                // Align to a page boundary; mincore rejects unaligned addresses.
                let aligned = offset - (base as usize + offset) % page;
                let mut vec = [0u8; 1];
                // SAFETY: `aligned` is page-aligned and within the mapping, `page` is one
                // page, and `vec` has room for the one byte mincore writes per page.
                let rc = unsafe {
                    libc::mincore(
                        base.add(aligned) as *mut libc::c_void,
                        page,
                        vec.as_mut_ptr() as *mut _,
                    )
                };
                if rc != 0 {
                    return 1.0;
                }
                resident += (vec[0] & 1) as usize;
                checked += 1;
                page_idx += step;
            }
            if checked == 0 {
                return 1.0;
            }
            resident as f64 / checked as f64
        }
        #[cfg(not(unix))]
        {
            let _ = (start_record, end_record);
            1.0
        }
    }

    /// Byte range spanned by a record range, clamped to the mapping.
    fn record_byte_range(&self, start_record: usize, end_record: usize) -> Option<(usize, usize)> {
        let data_offset = self.header.data_offset();
        let start = data_offset.checked_add(start_record.checked_mul(self.layout.record_size)?)?;
        let end = data_offset
            .checked_add(end_record.checked_mul(self.layout.record_size)?)?
            .min(self.mmap.len());
        (start < end).then_some((start, end))
    }

    /// Apply an access-pattern hint to the records in `[start_record, end_record)`.
    ///
    /// The hint is advisory only. The function ignores failures, and the call is a no-op on
    /// platforms other than Unix. `Advice::WillNeed` is capped at 64 MiB. A larger span asks
    /// the kernel to fault in more than it can keep, which makes reads slower under memory
    /// pressure.
    pub fn advise_records(&self, start_record: usize, end_record: usize, advice: Advice) {
        #[cfg(unix)]
        {
            let Some((start, mut end)) = self.record_byte_range(start_record, end_record) else {
                return;
            };
            if matches!(advice, Advice::WillNeed) {
                end = end.min(start.saturating_add(MAX_WILLNEED_BYTES));
            }
            // memmap2 aligns the address down to a page boundary; a hand-rolled madvise on an
            // unaligned address fails with EINVAL.
            let _ = self.mmap.advise_range(advice.into(), start, end - start);
        }
        #[cfg(not(unix))]
        {
            let _ = (start_record, end_record, advice);
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
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        let (s_start, s_end) = mapped.sample_range_for_time(&proxy, 0.0, 1.0);
        assert_eq!(s_start, 0, "sample 0 must be included when start_sec=0.0");
        assert_eq!(s_end, 4);
    }

    #[test]
    fn sample_range_for_time_all_onsets_after_end() {
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
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/edfPlusD.edf");
        let mapped = MappedFile::open(&fixture).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        mapped.wait_for_annotations();

        // Verify onsets are non-uniform (has gaps)
        mapped.with_annotations(|idx| {
            assert!(idx.record_onsets.len() >= 3);
            let has_gap = idx
                .record_onsets
                .windows(2)
                .any(|w| (w[1] - w[0] - mapped.header.record_duration_secs).abs() > 0.001);
            assert!(has_gap, "fixture should have non-uniform onsets");
        });

        // Range spanning the gap: must skip the gap
        let (s_start, s_end) = mapped.sample_range_for_time(&proxy, 0.0, 10.0);
        assert_eq!(s_start, 0);
        let total_uniform = mapped.header.num_records as usize * proxy.header().num_samples;
        assert!(s_end < total_uniform, "gap should reduce sample count");
    }

    #[test]
    fn sample_range_for_time_edfd_inside_gap() {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/edfPlusD.edf");
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
                    let (s_start, s_end) =
                        mapped.sample_range_for_time(&proxy, mid - 0.1, mid + 0.1);
                    assert_eq!(s_start, 0, "range inside gap should return (0,0)");
                    assert_eq!(s_end, 0);
                    break;
                }
            }
        });
    }

    #[test]
    fn sample_range_for_time_edfd_mid_record_range() {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/edfPlusD.edf");
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
        let (file, _, _) = build_test_file();
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        // start at 0.5s (midway through record 0), end at 1.5s (midway through record 1)
        let (s_start, s_end) = mapped.sample_range_for_time(&proxy, 0.5, 1.5);
        assert_eq!(s_start, 2); // sample index 2 is at time 0.5s
        assert_eq!(s_end, 6); // exclusive: samples 2,3,4,5
    }

    #[test]
    fn sample_range_for_time_edfd_underflow_all_onsets_after_end() {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/edfPlusD.edf");
        let mapped = MappedFile::open(&fixture).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        mapped.wait_for_annotations();

        // Query a time range entirely before the first record onset
        mapped.with_annotations(|idx| {
            let first_onset = idx.record_onsets.first().copied().unwrap_or(0.0);
            let before_first = first_onset - 10.0;
            let (s_start, s_end) =
                mapped.sample_range_for_time(&proxy, before_first, first_onset - 0.001);
            assert_eq!(s_start, 0);
            assert_eq!(s_end, 0);
        });
    }

    #[test]
    fn sample_range_for_time_edfd_mid_record() {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/edfPlusD.edf");
        let mapped = MappedFile::open(&fixture).unwrap();
        let proxy = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        mapped.wait_for_annotations();

        // Start at 0.5s which is mid-record 0
        let (s_start, s_end) = mapped.sample_range_for_time(&proxy, 0.5, 1.0);
        assert!(s_start > 0, "should skip the first portion of record 0");
        assert!(s_end > s_start);
    }
}
