use std::sync::Arc;

use crate::error::{EdfError, Result};
use crate::mmap::MappedFile;
use crate::record::RecordLayout;
use crate::signal::SignalHeader;

/// Array-like view of a single signal across the entire recording.
///
/// Translates global sample indices into record/offset pairs and decodes
/// samples directly from the memory-mapped file on each access. No
/// application-level caching is performed — the OS page cache handles this.
#[derive(Debug)]
pub struct SignalProxy {
    signal_idx: usize,
    file: Arc<MappedFile>,
    total_samples: usize,
    samples_per_record: usize,
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
        let total_samples = num_records * samples_per_record;

        Ok(SignalProxy {
            signal_idx,
            file,
            total_samples,
            samples_per_record,
        })
    }

    /// Total number of samples for this signal across all records.
    pub fn len(&self) -> usize {
        self.total_samples
    }

    /// Returns true if this signal has zero samples.
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

    /// Read a single sample as a physical (f64) value.
    pub fn get_physical(&self, idx: usize) -> Result<f64> {
        if idx >= self.total_samples {
            return Err(EdfError::SampleOutOfRange {
                index: idx,
                count: self.total_samples,
            });
        }
        let (rec_idx, offset) = self.resolve_index(idx);
        let record_data = self.file.record_bytes(rec_idx)?;
        let sig_bytes = self
            .file
            .layout
            .signal_bytes(record_data, self.signal_idx)?;
        let byte_offset = offset * 2;
        let raw =
            sig_bytes
                .get(byte_offset..byte_offset + 2)
                .ok_or(EdfError::SampleOutOfRange {
                    index: idx,
                    count: self.total_samples,
                })?;
        let digital = i16::from_le_bytes([raw[0], raw[1]]);
        let h = self.header();
        Ok(h.gain * digital as f64 + h.offset)
    }

    /// Read a range of samples as physical (f64) values into a pre-allocated buffer.
    ///
    /// This is the primary hot path. It resolves which data records are needed,
    /// decodes i16 -> f64 directly from the mmap, and writes into `out`.
    pub fn read_physical(&self, start: usize, end: usize, out: &mut [f64]) -> Result<()> {
        self.validate_range(start, end, out.len())?;
        self.read_range_inner(
            start,
            end,
            |sig_bytes, offset, count, out_slice| {
                let h = self.header();
                RecordLayout::decode_physical(
                    &sig_bytes[offset * 2..(offset + count) * 2],
                    h.gain,
                    h.offset,
                    out_slice,
                );
            },
            out,
        )
    }

    /// Read a range of samples as raw digital (i16) values into a pre-allocated buffer.
    pub fn read_digital(&self, start: usize, end: usize, out: &mut [i16]) -> Result<()> {
        self.validate_range(start, end, out.len())?;
        self.read_range_inner(
            start,
            end,
            |sig_bytes, offset, count, out_slice| {
                RecordLayout::decode_digital(
                    &sig_bytes[offset * 2..(offset + count) * 2],
                    out_slice,
                );
            },
            out,
        )
    }

    /// Physical time in seconds for the sample at the given global index.
    ///
    /// For EDF+D files, this accounts for gaps between records using the
    /// record onset times from the annotation index (blocks until scan completes).
    /// For EDF and EDF+C, record onsets are computed directly without waiting.
    pub fn sample_time(&self, idx: usize) -> f64 {
        let (rec_idx, offset) = self.resolve_index(idx);
        let record_onset = self.file.record_onset(rec_idx);
        let sample_offset = offset as f64 / self.sample_rate();
        record_onset + sample_offset
    }

    /// Fill a buffer with timestamps for a range of samples.
    pub fn read_times(&self, start: usize, end: usize, out: &mut [f64]) -> Result<()> {
        self.validate_range(start, end, out.len())?;
        for (i, idx) in (start..end).enumerate() {
            out[i] = self.sample_time(idx);
        }
        Ok(())
    }

    fn resolve_index(&self, idx: usize) -> (usize, usize) {
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
            return Err(EdfError::SampleOutOfRange {
                index: end,
                count: self.total_samples,
            });
        }
        Ok(())
    }

    /// Generic inner loop for reading a range of samples across record boundaries.
    ///
    /// `decode_fn` receives (signal_bytes, sample_offset_in_record, sample_count, output_slice)
    /// and writes decoded values into the output slice.
    fn read_range_inner<T>(
        &self,
        start: usize,
        end: usize,
        decode_fn: impl Fn(&[u8], usize, usize, &mut [T]),
        out: &mut [T],
    ) -> Result<()> {
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

            decode_fn(sig_bytes, offset, count, &mut out[out_pos..out_pos + count]);

            remaining_start += count;
            out_pos += count;
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

        let mut out = [0i16; 4];
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
