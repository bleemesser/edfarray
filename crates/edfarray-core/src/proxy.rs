use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{EdfError, Result};
use crate::grid::first_sample_at_or_after;
use crate::mmap::MappedFile;
use crate::signal::SignalHeader;

/// Minimum record-span size before streaming is considered. Below this the mmap path wins on
/// syscall overhead alone.
const STREAM_MIN_BYTES: usize = 32 << 20;

/// Bytes read per positional read on the streaming path.
const STREAM_CHUNK_BYTES: usize = 8 << 20;

/// Sampled residency above which the data is treated as already cached, so faulting it in is
/// nearly free and the mapping is the faster path.
const RESIDENT_ENOUGH: f64 = 0.5;

/// How often a long mmap read re-checks residency of the remaining span.
const RESIDENCY_RECHECK_RECORDS: usize = 1024;

/// How a read gets its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReadStrategy {
    /// Stream large non-resident reads, use the mapping otherwise.
    #[default]
    Auto,
    /// Always read through the memory mapping.
    Mmap,
    /// Always read sequentially through a bounded buffer.
    Stream,
}

/// LRU cache of decoded physical records, keyed by record index.
///
/// Recency is tracked by a monotonic counter rather than a queue, so lookups and insertions do
/// not scan. Eviction scans once per insertion, which only happens on a miss at capacity.
#[derive(Debug)]
struct LruCache {
    capacity: usize,
    map: HashMap<usize, CacheEntry>,
    clock: u64,
}

#[derive(Debug)]
struct CacheEntry {
    data: Vec<f64>,
    used: u64,
}

impl LruCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::with_capacity(capacity),
            clock: 0,
        }
    }

    fn get(&mut self, rec_idx: usize) -> Option<&[f64]> {
        self.clock += 1;
        let clock = self.clock;
        let entry = self.map.get_mut(&rec_idx)?;
        entry.used = clock;
        Some(&entry.data)
    }

    fn put(&mut self, rec_idx: usize, data: Vec<f64>) {
        self.clock += 1;
        if !self.map.contains_key(&rec_idx)
            && self.map.len() >= self.capacity
            && let Some(&lru) = self.map.iter().min_by_key(|(_, e)| e.used).map(|(k, _)| k)
        {
            self.map.remove(&lru);
        }
        self.map.insert(
            rec_idx,
            CacheEntry {
                data,
                used: self.clock,
            },
        );
    }

    /// Reusable scratch buffer for a decode that is about to be cached.
    fn take_buffer(&mut self, len: usize) -> Vec<f64> {
        let mut buf = if self.map.len() >= self.capacity {
            self.map
                .iter()
                .min_by_key(|(_, e)| e.used)
                .map(|(k, _)| *k)
                .and_then(|lru| self.map.remove(&lru))
                .map(|e| e.data)
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        buf.clear();
        buf.resize(len, 0.0);
        buf
    }
}

/// Array-like view of one signal.
///
/// Decodes samples on access. Bytes come from the memory map or a streaming read depending on
/// [`ReadStrategy`]; an optional LRU cache holds decoded records for repeated physical reads.
#[derive(Debug)]
pub struct SignalProxy {
    signal_idx: usize,
    file: Arc<MappedFile>,
    total_samples: usize,
    samples_per_record: usize,
    cache: Option<Mutex<LruCache>>,
    strategy: ReadStrategy,
}

impl SignalProxy {
    /// Create a proxy for the signal at the given index.
    pub fn new(file: Arc<MappedFile>, signal_idx: usize) -> Result<Self> {
        if signal_idx >= file.header.num_signals {
            return Err(EdfError::SignalOutOfRange {
                index: signal_idx,
                count: file.header.num_signals,
            });
        }
        let samples_per_record = file.header.signals[signal_idx].num_samples;
        let num_records = file.header.num_records.max(0) as usize;
        let total_samples = num_records.saturating_mul(samples_per_record);

        Ok(SignalProxy {
            signal_idx,
            file,
            total_samples,
            samples_per_record,
            cache: None,
            strategy: ReadStrategy::default(),
        })
    }

    /// Total number of samples for this signal across all records.
    pub fn len(&self) -> usize {
        self.total_samples
    }

    pub fn is_empty(&self) -> bool {
        self.total_samples == 0
    }

    /// Metadata for this signal.
    pub fn header(&self) -> &SignalHeader {
        &self.file.header.signals[self.signal_idx]
    }

    /// Computed sample rate in Hz.
    pub fn sample_rate(&self) -> f64 {
        self.header()
            .sample_rate(self.file.header.record_duration_secs)
    }

    /// Force how reads fetch bytes, overriding the automatic choice.
    pub fn with_strategy(mut self, strategy: ReadStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable an LRU cache of decoded records for physical reads.
    ///
    /// `capacity` counts records, not bytes or samples. Digital reads bypass the cache.
    /// Capacity 0 leaves it disabled.
    pub fn with_cache(mut self, capacity: usize) -> Self {
        if capacity == 0 {
            return self;
        }
        self.cache = Some(Mutex::new(LruCache::new(capacity)));
        self
    }

    /// Read a single sample as a physical (f64) value.
    pub fn get_physical(&self, idx: usize) -> Result<f64> {
        if idx >= self.total_samples {
            return Err(EdfError::SampleOutOfRange {
                index: idx,
                count: self.total_samples,
            });
        }
        let (rec_idx, offset) = self.resolve_index(idx);

        // Check warm cache. On miss, do not populate for single-sample access.
        if let Some(cache) = &self.cache
            && let Some(cached) = cache.lock().unwrap().get(rec_idx)
            && offset < cached.len()
        {
            return Ok(cached[offset]);
        }

        let record_data = self.file.record_bytes(rec_idx)?;
        let sig_bytes = self
            .file
            .layout
            .signal_bytes(record_data, self.signal_idx)?;
        let bytes = self.file.layout.sample_size_bytes;
        let byte_offset = offset * bytes;
        let raw =
            sig_bytes
                .get(byte_offset..byte_offset + bytes)
                .ok_or(EdfError::SampleOutOfRange {
                    index: idx,
                    count: self.total_samples,
                })?;
        let digital: i32 = match bytes {
            2 => i16::from_le_bytes([raw[0], raw[1]]) as i32,
            3 => {
                let raw24 = (raw[0] as i32) | ((raw[1] as i32) << 8) | ((raw[2] as i32) << 16);
                (raw24 << 8) >> 8
            }
            _ => unreachable!("invalid sample size"),
        };
        let h = self.header();
        Ok(h.gain * digital as f64 + h.offset)
    }

    /// Read physical samples into `out`. Decodes digital to f64 from mmap.
    pub fn read_physical(&self, start: usize, end: usize, out: &mut [f64]) -> Result<()> {
        self.validate_range(start, end, out.len())?;
        if self.cache.is_none() {
            let bytes = self.file.layout.sample_size_bytes;
            return self.read_range_inner(
                start,
                end,
                move |sig_bytes, offset, count, out_slice| {
                    let h = self.header();
                    self.file.layout.decode_physical(
                        &sig_bytes[offset * bytes..(offset + count) * bytes],
                        h.gain,
                        h.offset,
                        out_slice,
                    );
                },
                out,
            );
        }
        self.read_physical_cached(start, end, out)
    }

    fn read_physical_cached(&self, start: usize, end: usize, out: &mut [f64]) -> Result<()> {
        let mut remaining_start = start;
        let mut out_pos = 0;

        while remaining_start < end {
            let (rec_idx, offset) = self.resolve_index(remaining_start);
            let available = self.samples_per_record - offset;
            let needed = end - remaining_start;
            let count = available.min(needed);

            let record_data = self.file.record_bytes(rec_idx)?;
            let sig_bytes = self
                .file
                .layout
                .signal_bytes(record_data, self.signal_idx)?;

            // One lock acquisition per record: the hit path returns under it, and the miss path
            // takes an evicted buffer to decode into so the hot loop does not allocate.
            let mut full_decoded = {
                let mut cache = self.cache.as_ref().unwrap().lock().unwrap();
                if let Some(cached) = cache.get(rec_idx) {
                    out[out_pos..out_pos + count].copy_from_slice(&cached[offset..offset + count]);
                    out_pos += count;
                    remaining_start += count;
                    continue;
                }
                cache.take_buffer(self.samples_per_record)
            };

            {
                let h = self.header();
                self.file
                    .layout
                    .decode_physical(sig_bytes, h.gain, h.offset, &mut full_decoded);
            }

            out[out_pos..out_pos + count].copy_from_slice(&full_decoded[offset..offset + count]);
            out_pos += count;
            remaining_start += count;

            self.cache
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .put(rec_idx, full_decoded);
        }

        Ok(())
    }

    /// Read a range of samples as raw digital (i32) values into a pre-allocated buffer.
    pub fn read_digital(&self, start: usize, end: usize, out: &mut [i32]) -> Result<()> {
        self.validate_range(start, end, out.len())?;
        let bytes = self.file.layout.sample_size_bytes;
        self.read_range_inner(
            start,
            end,
            move |sig_bytes, offset, count, out_slice| {
                self.file.layout.decode_digital(
                    &sig_bytes[offset * bytes..(offset + count) * bytes],
                    out_slice,
                );
            },
            out,
        )
    }

    /// Physical time in seconds for sample at `idx`. Blocks for EDF+D to resolve onsets.
    pub fn sample_time(&self, idx: usize) -> f64 {
        let (rec_idx, offset) = self.resolve_index(idx);
        let record_onset = self.file.record_onset(rec_idx);
        let rate = self.sample_rate();
        if rate <= 0.0 || !rate.is_finite() {
            return record_onset;
        }
        record_onset + offset as f64 / rate
    }

    /// Fill a buffer with timestamps for a range of samples.
    pub fn read_times(&self, start: usize, end: usize, out: &mut [f64]) -> Result<()> {
        self.validate_range(start, end, out.len())?;
        for (i, idx) in (start..end).enumerate() {
            out[i] = self.sample_time(idx);
        }
        Ok(())
    }

    /// Sample range covering `[start_sec, end_sec)` for a uniformly timed recording.
    ///
    /// Half-open, with both ends resolved by `first_sample_at_or_after`, matching the EDF+D
    /// path in `MappedFile::sample_range_for_time` so both variants return the same count.
    pub fn uniform_sample_range(&self, start_sec: f64, end_sec: f64) -> (usize, usize) {
        let sr = self.sample_rate();
        if sr <= 0.0 || !sr.is_finite() {
            return (0, 0);
        }
        let to_index = |t: f64| -> usize {
            let idx = first_sample_at_or_after(t.max(0.0) * sr).max(0) as u64;
            usize::try_from(idx).map_or(self.total_samples, |i| i.min(self.total_samples))
        };
        (to_index(start_sec), to_index(end_sec))
    }

    /// Sample range covering `[start_sec, end_sec)`, accounting for EDF+D gaps.
    pub fn sample_range_for_time(&self, start_sec: f64, end_sec: f64) -> (usize, usize) {
        if self.file.header.variant.is_plus_d() {
            self.file.sample_range_for_time(self, start_sec, end_sec)
        } else {
            self.uniform_sample_range(start_sec, end_sec)
        }
    }

    /// Read physical samples in `[start_sec, end_sec)`. Accounts for EDF+D gaps.
    pub fn read_at(&self, start_sec: f64, end_sec: f64) -> Result<Vec<f64>> {
        let (s_start, s_end) = self.sample_range_for_time(start_sec, end_sec);
        if s_start >= s_end {
            return Ok(Vec::new());
        }
        let count = s_end - s_start;
        let mut buf = vec![0.0f64; count];
        self.read_physical(s_start, s_end, &mut buf)?;
        Ok(buf)
    }

    fn resolve_index(&self, idx: usize) -> (usize, usize) {
        // A header may declare zero samples per record; that signal has no samples to resolve.
        if self.samples_per_record == 0 {
            return (0, 0);
        }
        let rec_idx = idx / self.samples_per_record;
        let offset = idx % self.samples_per_record;
        (rec_idx, offset)
    }

    fn validate_range(&self, start: usize, end: usize, buf_len: usize) -> Result<()> {
        if end > self.total_samples {
            return Err(EdfError::SampleOutOfRange {
                index: end.saturating_sub(1),
                count: self.total_samples,
            });
        }
        if start > end {
            return Err(EdfError::SampleOutOfRange {
                index: start,
                count: self.total_samples,
            });
        }
        if end - start != buf_len {
            return Err(EdfError::BufferSizeMismatch {
                expected: end - start,
                actual: buf_len,
            });
        }
        Ok(())
    }

    /// Read samples across record boundaries using `decode_fn`.
    fn read_range_inner<T>(
        &self,
        start: usize,
        end: usize,
        decode_fn: impl Fn(&[u8], usize, usize, &mut [T]),
        out: &mut [T],
    ) -> Result<()> {
        if self.should_stream(start, end) {
            return self.read_range_streaming(start, end, decode_fn, out);
        }

        let mut remaining_start = start;
        let mut out_pos = 0;
        let mut since_check = 0usize;

        while remaining_start < end {
            let (rec_idx, offset) = self.resolve_index(remaining_start);
            let available = self.samples_per_record - offset;
            let needed = end - remaining_start;
            let count = available.min(needed);

            // Residency can collapse mid-read when the machine is under memory pressure: pages
            // faulted in at the start get evicted before the read finishes. Re-check
            // periodically and switch the remainder to streaming rather than paying a fault
            // per record for the rest of the file.
            since_check += 1;
            if since_check >= RESIDENCY_RECHECK_RECORDS {
                since_check = 0;
                if self.should_stream(remaining_start, end) {
                    return self.read_range_streaming(
                        remaining_start,
                        end,
                        decode_fn,
                        &mut out[out_pos..],
                    );
                }
            }

            let record_data = self.file.record_bytes(rec_idx)?;
            let sig_bytes = self
                .file
                .layout
                .signal_bytes(record_data, self.signal_idx)?;

            decode_fn(sig_bytes, offset, count, &mut out[out_pos..out_pos + count]);

            remaining_start += count;
            out_pos += count;
        }

        Ok(())
    }

    /// Whether a read should stream through a bounded buffer instead of the mapping.
    ///
    /// EDF interleaves channels within a record, so reading one channel touches a slice of
    /// every record: the kernel must fault in the whole span to hand back a fraction of it.
    /// When that span is large and not already resident, one page fault per record costs far
    /// more than reading the same bytes sequentially.
    fn should_stream(&self, start: usize, end: usize) -> bool {
        match self.strategy {
            ReadStrategy::Mmap => return false,
            ReadStrategy::Stream => return true,
            ReadStrategy::Auto => {}
        }
        if self.samples_per_record == 0 || !cfg!(unix) {
            return false;
        }
        let (first_rec, _) = self.resolve_index(start);
        let (last_rec, _) = self.resolve_index(end.saturating_sub(1));
        let span_bytes = (last_rec - first_rec + 1).saturating_mul(self.file.layout.record_size);
        if span_bytes < STREAM_MIN_BYTES {
            return false;
        }
        self.file.residency(first_rec, last_rec + 1) < RESIDENT_ENOUGH
    }

    /// Decode a sample range by reading records sequentially into a bounded buffer.
    fn read_range_streaming<T>(
        &self,
        start: usize,
        end: usize,
        decode_fn: impl Fn(&[u8], usize, usize, &mut [T]),
        out: &mut [T],
    ) -> Result<()> {
        let record_size = self.file.layout.record_size;
        let num_records = self.file.header.num_records.max(0) as usize;
        if record_size == 0 {
            return Ok(());
        }

        let chunk_records = (STREAM_CHUNK_BYTES / record_size).max(1);
        let mut buf = vec![0u8; chunk_records * record_size];

        let mut remaining_start = start;
        let mut out_pos = 0;

        while remaining_start < end {
            let (rec_idx, _) = self.resolve_index(remaining_start);
            let (last_rec, _) = self.resolve_index(end - 1);
            let n_records = chunk_records
                .min(last_rec + 1 - rec_idx)
                .min(num_records.saturating_sub(rec_idx));
            if n_records == 0 {
                break;
            }

            let filled = &mut buf[..n_records * record_size];
            self.file.read_records_into(rec_idx, n_records, filled)?;

            for i in 0..n_records {
                let (cur_rec, offset) = self.resolve_index(remaining_start);
                debug_assert_eq!(cur_rec, rec_idx + i);
                let available = self.samples_per_record - offset;
                let count = available.min(end - remaining_start);

                let record_data = &filled[i * record_size..(i + 1) * record_size];
                let sig_bytes = self
                    .file
                    .layout
                    .signal_bytes(record_data, self.signal_idx)?;
                decode_fn(sig_bytes, offset, count, &mut out[out_pos..out_pos + count]);

                remaining_start += count;
                out_pos += count;
                if remaining_start >= end {
                    break;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn build_test_file(num_records: usize, samples_per_record: usize) -> NamedTempFile {
        let num_signals = 1;
        let header_bytes = 256 + 256 * num_signals;
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
        write_sig(sig_data, 0, 1, 104, 8, "-100");
        write_sig(sig_data, 0, 1, 112, 8, "100");
        write_sig(sig_data, 0, 1, 120, 8, "-100");
        write_sig(sig_data, 0, 1, 128, 8, "100");
        write_sig(sig_data, 0, 1, 136, 80, "");
        write_sig(sig_data, 0, 1, 216, 8, &samples_per_record.to_string());
        write_sig(sig_data, 0, 1, 224, 32, "");

        for i in 0..(num_records * samples_per_record) {
            let val = i as i16;
            buf.extend_from_slice(&val.to_le_bytes());
        }

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn get_single_sample() {
        let file = build_test_file(3, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = SignalProxy::new(mapped, 0).unwrap();

        assert_eq!(proxy.len(), 12);

        // gain=1.0, offset=0.0 because phys range == digital range
        let val = proxy.get_physical(0).unwrap();
        assert!((val - 0.0).abs() < f64::EPSILON);

        let val = proxy.get_physical(5).unwrap();
        assert!((val - 5.0).abs() < f64::EPSILON);

        let val = proxy.get_physical(11).unwrap();
        assert!((val - 11.0).abs() < f64::EPSILON);
    }

    #[test]
    fn streaming_matches_mmap_for_every_subrange() {
        let file = build_test_file(37, 5);
        let mapped = MappedFile::open(file.path()).unwrap();
        let via_mmap = SignalProxy::new(Arc::clone(&mapped), 0)
            .unwrap()
            .with_strategy(ReadStrategy::Mmap);
        let via_stream = SignalProxy::new(mapped, 0)
            .unwrap()
            .with_strategy(ReadStrategy::Stream);

        let total = via_mmap.len();
        for start in [0, 1, 4, 5, 6, 17, total - 1] {
            for end in [start, start + 1, start + 7, total] {
                if end > total {
                    continue;
                }
                let n = end - start;
                let mut a = vec![0.0f64; n];
                let mut b = vec![0.0f64; n];
                via_mmap.read_physical(start, end, &mut a).unwrap();
                via_stream.read_physical(start, end, &mut b).unwrap();
                assert_eq!(a, b, "physical mismatch for {start}..{end}");

                let mut c = vec![0i32; n];
                let mut d = vec![0i32; n];
                via_mmap.read_digital(start, end, &mut c).unwrap();
                via_stream.read_digital(start, end, &mut d).unwrap();
                assert_eq!(c, d, "digital mismatch for {start}..{end}");
            }
        }
    }

    #[test]
    fn read_physical_range() {
        let file = build_test_file(3, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = SignalProxy::new(mapped, 0).unwrap();

        let mut out = [0.0f64; 6];
        proxy.read_physical(2, 8, &mut out).unwrap();
        for (i, &val) in out.iter().enumerate() {
            assert!((val - (i + 2) as f64).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn read_physical_across_records() {
        let file = build_test_file(3, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = SignalProxy::new(mapped, 0).unwrap();

        let mut out = [0.0f64; 12];
        proxy.read_physical(0, 12, &mut out).unwrap();
        for (i, &val) in out.iter().enumerate() {
            assert!((val - i as f64).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn read_digital_range() {
        let file = build_test_file(2, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = SignalProxy::new(mapped, 0).unwrap();

        let mut out = [0i32; 4];
        proxy.read_digital(2, 6, &mut out).unwrap();
        assert_eq!(out, [2, 3, 4, 5]);
    }

    #[test]
    fn sample_out_of_range() {
        let file = build_test_file(2, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = SignalProxy::new(mapped, 0).unwrap();

        assert!(proxy.get_physical(8).is_err());
        let mut out = [0.0f64; 2];
        assert!(proxy.read_physical(7, 9, &mut out).is_err());
    }

    #[test]
    fn sample_times() {
        let file = build_test_file(3, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = SignalProxy::new(mapped, 0).unwrap();

        assert!((proxy.sample_time(0) - 0.0).abs() < f64::EPSILON);
        assert!((proxy.sample_time(4) - 1.0).abs() < f64::EPSILON);
        assert!((proxy.sample_time(2) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn signal_out_of_range() {
        let file = build_test_file(1, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        assert!(SignalProxy::new(mapped, 1).is_err());
    }

    #[test]
    fn lru_get_miss_returns_none() {
        let mut c = LruCache::new(2);
        assert!(c.get(0).is_none());
    }

    #[test]
    fn lru_put_get_roundtrip() {
        let mut c = LruCache::new(2);
        c.put(0, vec![1.0, 2.0]);
        let got = c.get(0).unwrap();
        assert_eq!(got, &[1.0, 2.0]);
    }

    #[test]
    fn lru_evicts_least_recent() {
        let mut c = LruCache::new(2);
        c.put(0, vec![0.0]);
        c.put(1, vec![1.0]);
        c.put(2, vec![2.0]); // evicts 0
        assert!(c.get(0).is_none());
        assert!(c.get(1).is_some());
        assert!(c.get(2).is_some());
    }

    #[test]
    fn lru_get_promotes_to_front() {
        let mut c = LruCache::new(2);
        c.put(0, vec![0.0]);
        c.put(1, vec![1.0]);
        let _ = c.get(0); // promote 0
        c.put(2, vec![2.0]); // should evict 1, not 0
        assert!(c.get(0).is_some());
        assert!(c.get(1).is_none());
        assert!(c.get(2).is_some());
    }

    #[test]
    fn lru_put_existing_key_replaces() {
        let mut c = LruCache::new(2);
        c.put(0, vec![0.0]);
        c.put(0, vec![99.0]);
        assert_eq!(c.get(0).unwrap(), &[99.0]);
        c.put(1, vec![1.0]);
        c.put(2, vec![2.0]); // if 0 was duplicated in `order`, this would not evict 0
        assert!(c.get(0).is_none());
    }

    #[test]
    fn lru_zero_capacity_holds_one_record() {
        let mut cache = LruCache::new(0);
        cache.put(0, vec![1.0, 2.0]);
        assert_eq!(cache.get(0), Some([1.0, 2.0].as_slice()));
    }

    #[test]
    fn lru_evicts_least_recently_used() {
        let mut cache = LruCache::new(2);
        cache.put(0, vec![0.0]);
        cache.put(1, vec![1.0]);
        assert!(cache.get(0).is_some());
        cache.put(2, vec![2.0]);
        // record 1 was the least recently used, so it goes; record 0 was just touched
        assert!(cache.get(1).is_none());
        assert!(cache.get(0).is_some());
        assert!(cache.get(2).is_some());
    }

    #[test]
    fn cache_disabled_by_default() {
        let file = build_test_file(3, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let sig = SignalProxy::new(mapped, 0).unwrap();
        let mut buf = vec![0.0; sig.len()];
        sig.read_physical(0, sig.len(), &mut buf).unwrap();
        assert_eq!(buf.len(), 12);
    }

    #[test]
    fn cache_enabled_returns_same_values() {
        let file = build_test_file(3, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let sig_a = SignalProxy::new(mapped.clone(), 0).unwrap();
        let sig_b = SignalProxy::new(mapped, 0).unwrap().with_cache(2);

        let mut buf_a = vec![0.0; sig_a.len()];
        let mut buf_b = vec![0.0; sig_b.len()];
        sig_a.read_physical(0, sig_a.len(), &mut buf_a).unwrap();
        sig_b.read_physical(0, sig_b.len(), &mut buf_b).unwrap();
        assert_eq!(buf_a, buf_b);
    }

    #[test]
    fn cache_hit_on_repeated_read() {
        let file = build_test_file(3, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let sig = SignalProxy::new(mapped, 0).unwrap().with_cache(4);
        let mut buf = vec![0.0; 4];
        sig.read_physical(0, 4, &mut buf).unwrap();
        let first = buf.clone();
        sig.read_physical(0, 4, &mut buf).unwrap();
        assert_eq!(first, buf);
    }

    #[test]
    fn cache_get_physical_uses_warm_cache() {
        let file = build_test_file(3, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let sig = SignalProxy::new(mapped, 0).unwrap().with_cache(4);
        let mut buf = vec![0.0; 4];
        sig.read_physical(0, 4, &mut buf).unwrap();
        let v = sig.get_physical(2).unwrap();
        assert!((v - buf[2]).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_unaligned_read_matches_uncached() {
        // Regression: read_physical_cached used to slice
        // sig_bytes[offset*bytes..(offset+spr)*bytes] which panics when offset > 0.
        // Verify a non-record-boundary read produces the same values cached vs uncached.
        let file = build_test_file(3, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let sig_cold = SignalProxy::new(Arc::clone(&mapped), 0).unwrap();
        let sig_warm = SignalProxy::new(mapped, 0).unwrap().with_cache(4);

        let mut cold = vec![0.0; 6];
        let mut warm = vec![0.0; 6];
        sig_cold.read_physical(2, 8, &mut cold).unwrap();
        sig_warm.read_physical(2, 8, &mut warm).unwrap();
        assert_eq!(cold, warm);

        // Second read on the warm proxy should hit cache and still match.
        let mut warm2 = vec![0.0; 6];
        sig_warm.read_physical(2, 8, &mut warm2).unwrap();
        assert_eq!(cold, warm2);
    }

    #[test]
    fn cache_with_capacity_zero_is_disabled() {
        let file = build_test_file(3, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let sig = SignalProxy::new(mapped, 0).unwrap().with_cache(0);
        let mut buf = vec![0.0; 4];
        sig.read_physical(0, 4, &mut buf).unwrap();
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
