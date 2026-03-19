use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{EdfError, Result};
use crate::mmap::MappedFile;
use crate::proxy::SignalProxy;

/// A 2D array view over multiple signals with the same sample rate.
///
/// Presents ordinary (non-annotation) signals as a rectangular array with shape
/// `(num_signals, num_samples)`. All signals in the proxy must share the same
/// sample rate; use `EdfFile::signal_indices_by_rate()` to group signals first
/// if the file has mixed rates.
///
/// Data is read on demand from the memory-mapped file — this struct holds no
/// sample data itself.
#[derive(Debug)]
pub struct ArrayProxy {
    file: Arc<MappedFile>,
    signal_indices: Vec<usize>,
    samples_per_signal: usize,
    sample_rate: f64,
}

impl ArrayProxy {
    /// Create a new array proxy over the given signal indices.
    ///
    /// Returns `MixedSampleRates` if the signals don't all share the same rate.
    /// An empty `signal_indices` slice is allowed and produces a 0x0 proxy.
    pub fn new(file: Arc<MappedFile>, signal_indices: &[usize]) -> Result<Self> {
        if signal_indices.is_empty() {
            return Ok(ArrayProxy {
                file,
                signal_indices: Vec::new(),
                samples_per_signal: 0,
                sample_rate: 0.0,
            });
        }

        let record_duration = file.header.record_duration_secs;

        let first_idx = signal_indices[0];
        if first_idx >= file.header.num_signals {
            return Err(EdfError::SignalOutOfRange {
                index: first_idx,
                count: file.header.num_signals,
            });
        }
        let first_rate = file.header.signals[first_idx].sample_rate(record_duration);
        let first_spr = file.header.signals[first_idx].num_samples;

        for &idx in &signal_indices[1..] {
            if idx >= file.header.num_signals {
                return Err(EdfError::SignalOutOfRange {
                    index: idx,
                    count: file.header.num_signals,
                });
            }
            let rate = file.header.signals[idx].sample_rate(record_duration);
            if (rate - first_rate).abs() > 1e-9 {
                return Err(EdfError::MixedSampleRates {
                    reason: format!(
                        "signal {} has rate {}Hz but signal {} has rate {}Hz. \
                         Use signal_indices_by_rate() to group signals by rate.",
                        first_idx, first_rate, idx, rate
                    ),
                });
            }
        }

        let num_records = file.header.num_records.max(0) as usize;
        let samples_per_signal = num_records * first_spr;

        Ok(ArrayProxy {
            file,
            signal_indices: signal_indices.to_vec(),
            samples_per_signal,
            sample_rate: first_rate,
        })
    }

    /// Shape of the 2D view: `(num_signals, total_samples_per_signal)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.signal_indices.len(), self.samples_per_signal)
    }

    /// Common sample rate (Hz) of all signals in this proxy.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// The underlying file-level signal indices this proxy covers.
    pub fn signal_indices(&self) -> &[usize] {
        &self.signal_indices
    }

    /// Read a single physical sample at `(signal, sample)` in proxy coordinates.
    pub fn get(&self, signal: usize, sample: usize) -> Result<f64> {
        let sig_idx = self.resolve_signal(signal)?;
        let proxy = SignalProxy::new(Arc::clone(&self.file), sig_idx)?;
        proxy.get_physical(sample)
    }

    /// Read physical samples for the given proxy-coordinate signal indices and sample range.
    ///
    /// This is the core read method. `signal_indices` are proxy-coordinate indices
    /// (not file-level). Parallelized across signals with rayon.
    pub fn read_physical(
        &self,
        signal_indices: &[usize],
        samples: Range<usize>,
    ) -> Result<Vec<Vec<f64>>> {
        let file = &self.file;
        let sample_start = samples.start;
        let sample_end = samples.end;

        signal_indices
            .par_iter()
            .map(|&s| {
                let sig_idx = self.resolve_signal(s)?;
                let proxy = SignalProxy::new(Arc::clone(file), sig_idx)?;
                let end = sample_end.min(proxy.len());
                let start = sample_start.min(end);
                let count = end - start;
                let mut buf = vec![0.0f64; count];
                if count > 0 {
                    proxy.read_physical(start, end, &mut buf)?;
                }
                Ok(buf)
            })
            .collect()
    }

    /// Read a rectangular block of physical samples. Parallelized across signals.
    ///
    /// `signals` and `samples` are ranges in proxy coordinates.
    /// Returns one `Vec<f64>` per signal in the range.
    pub fn read_slice(
        &self,
        signals: Range<usize>,
        samples: Range<usize>,
    ) -> Result<Vec<Vec<f64>>> {
        let indices: Vec<usize> = signals.collect();
        self.read_physical(&indices, samples)
    }

    /// Read digital (i16) samples for the given proxy-coordinate signal indices and sample range.
    ///
    /// Parallelized across signals with rayon.
    pub fn read_digital(
        &self,
        signal_indices: &[usize],
        samples: Range<usize>,
    ) -> Result<Vec<Vec<i16>>> {
        let file = &self.file;
        let sample_start = samples.start;
        let sample_end = samples.end;

        signal_indices
            .par_iter()
            .map(|&s| {
                let sig_idx = self.resolve_signal(s)?;
                let proxy = SignalProxy::new(Arc::clone(file), sig_idx)?;
                let end = sample_end.min(proxy.len());
                let start = sample_start.min(end);
                let count = end - start;
                let mut buf = vec![0i16; count];
                if count > 0 {
                    proxy.read_digital(start, end, &mut buf)?;
                }
                Ok(buf)
            })
            .collect()
    }

    /// Read a rectangular block of raw digital (i16) samples. Parallelized across signals.
    pub fn read_slice_digital(
        &self,
        signals: Range<usize>,
        samples: Range<usize>,
    ) -> Result<Vec<Vec<i16>>> {
        let indices: Vec<usize> = signals.collect();
        self.read_digital(&indices, samples)
    }

    /// Read one sample from each of the specified signals (proxy-coordinate indices).
    pub fn read_signals_at_sample(
        &self,
        signal_indices: &[usize],
        sample: usize,
    ) -> Result<Vec<f64>> {
        signal_indices
            .iter()
            .map(|&s| self.get(s, sample))
            .collect()
    }

    /// Map a proxy-coordinate signal index to the underlying file signal index.
    fn resolve_signal(&self, proxy_idx: usize) -> Result<usize> {
        self.signal_indices.get(proxy_idx).copied().ok_or(
            EdfError::SignalOutOfRange {
                index: proxy_idx,
                count: self.signal_indices.len(),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn build_test_file(num_signals: usize, num_records: usize, spr: usize) -> NamedTempFile {
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
        for i in 0..num_signals {
            write_sig(sig_data, i, num_signals, 0, 16, &format!("EEG{}", i));
            write_sig(sig_data, i, num_signals, 16, 80, "");
            write_sig(sig_data, i, num_signals, 96, 8, "uV");
            write_sig(sig_data, i, num_signals, 104, 8, "-100");
            write_sig(sig_data, i, num_signals, 112, 8, "100");
            write_sig(sig_data, i, num_signals, 120, 8, "-100");
            write_sig(sig_data, i, num_signals, 128, 8, "100");
            write_sig(sig_data, i, num_signals, 136, 80, "");
            write_sig(sig_data, i, num_signals, 216, 8, &spr.to_string());
            write_sig(sig_data, i, num_signals, 224, 32, "");
        }

        for _rec in 0..num_records {
            for sig in 0..num_signals {
                for s in 0..spr {
                    let val = (sig * 100 + s) as i16;
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
        }

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn array_proxy_shape() {
        let file = build_test_file(3, 2, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = ArrayProxy::new(mapped, &[0, 1, 2]).unwrap();
        assert_eq!(proxy.shape(), (3, 8));
    }

    #[test]
    fn array_proxy_get() {
        let file = build_test_file(2, 2, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = ArrayProxy::new(mapped, &[0, 1]).unwrap();
        let val = proxy.get(0, 0).unwrap();
        assert!((val - 0.0).abs() < f64::EPSILON);
        let val = proxy.get(1, 0).unwrap();
        assert!((val - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn array_proxy_read_slice() {
        let file = build_test_file(2, 2, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = ArrayProxy::new(mapped, &[0, 1]).unwrap();
        let data = proxy.read_slice(0..2, 0..4).unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].len(), 4);
        assert!((data[0][0] - 0.0).abs() < f64::EPSILON);
        assert!((data[1][0] - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn array_proxy_empty() {
        let file = build_test_file(2, 2, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = ArrayProxy::new(mapped, &[]).unwrap();
        assert_eq!(proxy.shape(), (0, 0));
    }

    #[test]
    fn array_proxy_read_slice_digital() {
        let file = build_test_file(2, 2, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = ArrayProxy::new(mapped, &[0, 1]).unwrap();
        let data = proxy.read_slice_digital(0..2, 0..4).unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0], vec![0, 1, 2, 3]);
        assert_eq!(data[1], vec![100, 101, 102, 103]);
    }

    #[test]
    fn array_proxy_read_signals_at_sample() {
        let file = build_test_file(3, 2, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = ArrayProxy::new(mapped, &[0, 1, 2]).unwrap();
        let vals = proxy.read_signals_at_sample(&[0, 2], 1).unwrap();
        assert_eq!(vals.len(), 2);
        assert!((vals[0] - 1.0).abs() < f64::EPSILON);
        assert!((vals[1] - 201.0).abs() < f64::EPSILON);
    }

    #[test]
    fn array_proxy_read_physical() {
        let file = build_test_file(3, 2, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = ArrayProxy::new(mapped, &[0, 1, 2]).unwrap();
        let data = proxy.read_physical(&[0, 2], 0..3).unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0].len(), 3);
        assert!((data[0][0] - 0.0).abs() < f64::EPSILON);
        assert!((data[1][0] - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn array_proxy_mixed_rates_error() {
        let num_signals = 2;
        let header_bytes = 256 + 256 * num_signals;
        let mut buf = vec![b' '; header_bytes];
        write_field(&mut buf, 0, 8, "0");
        write_field(&mut buf, 8, 80, "X X X X");
        write_field(&mut buf, 88, 80, "Startdate X X X X");
        write_field(&mut buf, 168, 8, "01.01.00");
        write_field(&mut buf, 176, 8, "00.00.00");
        write_field(&mut buf, 184, 8, &header_bytes.to_string());
        write_field(&mut buf, 192, 44, "");
        write_field(&mut buf, 236, 8, "1");
        write_field(&mut buf, 244, 8, "1");
        write_field(&mut buf, 252, 4, &num_signals.to_string());

        let sig_data = &mut buf[256..];
        write_sig(sig_data, 0, num_signals, 0, 16, "EEG0");
        write_sig(sig_data, 0, num_signals, 16, 80, "");
        write_sig(sig_data, 0, num_signals, 96, 8, "uV");
        write_sig(sig_data, 0, num_signals, 104, 8, "-100");
        write_sig(sig_data, 0, num_signals, 112, 8, "100");
        write_sig(sig_data, 0, num_signals, 120, 8, "-100");
        write_sig(sig_data, 0, num_signals, 128, 8, "100");
        write_sig(sig_data, 0, num_signals, 136, 80, "");
        write_sig(sig_data, 0, num_signals, 216, 8, "4");
        write_sig(sig_data, 0, num_signals, 224, 32, "");

        write_sig(sig_data, 1, num_signals, 0, 16, "EEG1");
        write_sig(sig_data, 1, num_signals, 16, 80, "");
        write_sig(sig_data, 1, num_signals, 96, 8, "uV");
        write_sig(sig_data, 1, num_signals, 104, 8, "-100");
        write_sig(sig_data, 1, num_signals, 112, 8, "100");
        write_sig(sig_data, 1, num_signals, 120, 8, "-100");
        write_sig(sig_data, 1, num_signals, 128, 8, "100");
        write_sig(sig_data, 1, num_signals, 136, 80, "");
        write_sig(sig_data, 1, num_signals, 216, 8, "8");
        write_sig(sig_data, 1, num_signals, 224, 32, "");

        for _ in 0..4 { buf.extend_from_slice(&0i16.to_le_bytes()); }
        for _ in 0..8 { buf.extend_from_slice(&0i16.to_le_bytes()); }

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();

        let mapped = MappedFile::open(file.path()).unwrap();
        let err = ArrayProxy::new(mapped, &[0, 1]).unwrap_err();
        assert!(matches!(err, EdfError::MixedSampleRates { .. }));
    }

    #[test]
    fn array_proxy_signal_out_of_range() {
        let file = build_test_file(2, 2, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let err = ArrayProxy::new(mapped, &[0, 5]).unwrap_err();
        assert!(matches!(err, EdfError::SignalOutOfRange { .. }));
    }

    #[test]
    fn array_proxy_get_out_of_range() {
        let file = build_test_file(2, 2, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = ArrayProxy::new(mapped, &[0, 1]).unwrap();
        assert!(proxy.get(2, 0).is_err());
        assert!(proxy.get(0, 100).is_err());
    }

    #[test]
    fn array_proxy_sample_rate_and_indices() {
        let file = build_test_file(2, 2, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = ArrayProxy::new(mapped, &[0, 1]).unwrap();
        assert_eq!(proxy.sample_rate(), 4.0);
        assert_eq!(proxy.signal_indices(), &[0, 1]);
    }

    #[test]
    fn array_proxy_read_across_records() {
        let file = build_test_file(2, 3, 4);
        let mapped = MappedFile::open(file.path()).unwrap();
        let proxy = ArrayProxy::new(mapped, &[0]).unwrap();
        assert_eq!(proxy.shape(), (1, 12));
        let data = proxy.read_slice(0..1, 0..12).unwrap();
        assert_eq!(data[0].len(), 12);
        for i in 0..12 {
            assert!((data[0][i] - (i % 4) as f64).abs() < f64::EPSILON);
        }
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
