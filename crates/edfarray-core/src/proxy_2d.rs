use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{EdfError, Result};
use crate::group::{PadMode, SignalGroup};
use crate::mmap::MappedFile;
use crate::proxy::SignalProxy;

/// 2D view over signal channels. Decodes on demand from mmap; holds no sample data.
#[derive(Debug)]
pub struct Proxy2D {
    file: Arc<MappedFile>,
    group: SignalGroup,
    valid_lengths: Vec<usize>,
    pad_mode: PadMode,
}

impl Proxy2D {
    /// Build a 2D proxy from a `SignalGroup` with the given pad policy.
    /// Returns `EdfError::InvalidArgument` if the group is empty.
    pub fn new(file: Arc<MappedFile>, group: SignalGroup, pad_mode: PadMode) -> Result<Self> {
        if group.indices.is_empty() {
            return Err(EdfError::InvalidArgument {
                name: "group",
                reason: "Proxy2D requires a non-empty signal group".into(),
            });
        }

        let num_records = file.header.num_records.max(0) as usize;
        let valid_lengths: Vec<usize> = group
            .indices
            .iter()
            .map(|&i| num_records * file.header.signals[i].num_samples)
            .collect();

        Ok(Proxy2D {
            file,
            group,
            valid_lengths,
            pad_mode,
        })
    }

    /// Shape of the 2D view: `(num_signals, max_samples)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.group.len(), self.group.max_samples)
    }

    /// Common sample rate (Hz), or `None` if the group is `Open` (mixed rates).
    pub fn sample_rate(&self) -> Option<f64> {
        self.group.sample_rate
    }

    /// The group this proxy was built from.
    pub fn group(&self) -> &SignalGroup {
        &self.group
    }

    /// Pad policy applied to reads past a channel's valid length.
    pub fn pad_mode(&self) -> PadMode {
        self.pad_mode
    }

    /// Per-channel valid sample counts in proxy-coordinate order.
    pub fn valid_lengths(&self) -> &[usize] {
        &self.valid_lengths
    }

    /// Read a single physical sample at `(signal, sample)` in proxy coordinates.
    pub fn get(&self, signal: usize, sample: usize) -> Result<f64> {
        let sig_idx = self.resolve_signal(signal)?;
        let valid = self.valid_lengths[signal];
        if sample >= valid {
            return match self.pad_mode {
                PadMode::Raise => Err(EdfError::SampleOutOfRange {
                    index: sample,
                    count: valid,
                }),
                PadMode::Nan => Ok(f64::NAN),
                PadMode::Zero => Ok(0.0),
                PadMode::Value(v) => Ok(v),
                PadMode::Edge => {
                    if valid == 0 {
                        Err(EdfError::SampleOutOfRange {
                            index: sample,
                            count: valid,
                        })
                    } else {
                        let proxy = SignalProxy::new(Arc::clone(&self.file), sig_idx)?;
                        proxy.get_physical(valid - 1)
                    }
                }
            };
        }
        let proxy = SignalProxy::new(Arc::clone(&self.file), sig_idx)?;
        proxy.get_physical(sample)
    }

    /// Read physical samples for signal indices and sample range. Parallel across signals.
    pub fn read_physical(
        &self,
        signal_indices: &[usize],
        samples: Range<usize>,
    ) -> Result<Vec<Vec<f64>>> {
        let file = &self.file;
        let pad = self.pad_mode;
        let s_start = samples.start;
        let s_end = samples.end;

        signal_indices
            .par_iter()
            .map(|&s| {
                let sig_idx = self.resolve_signal(s)?;
                let valid = self.valid_lengths[s];
                read_physical_with_pad(file, sig_idx, valid, s_start, s_end, pad)
            })
            .collect()
    }

    /// Read a rectangular block of physical samples. Parallelized across signals.
    pub fn read_slice(
        &self,
        signals: Range<usize>,
        samples: Range<usize>,
    ) -> Result<Vec<Vec<f64>>> {
        let indices: Vec<usize> = signals.collect();
        self.read_physical(&indices, samples)
    }

    /// Read digital (i32) samples for the given proxy-coordinate signal indices and sample range.
    pub fn read_digital(
        &self,
        signal_indices: &[usize],
        samples: Range<usize>,
    ) -> Result<Vec<Vec<i32>>> {
        let file = &self.file;
        let pad = self.pad_mode;
        let s_start = samples.start;
        let s_end = samples.end;

        signal_indices
            .par_iter()
            .map(|&s| {
                let sig_idx = self.resolve_signal(s)?;
                let valid = self.valid_lengths[s];
                read_digital_with_pad(file, sig_idx, valid, s_start, s_end, pad)
            })
            .collect()
    }

    /// Read a rectangular block of raw digital (i32) samples. Parallelized across signals.
    pub fn read_slice_digital(
        &self,
        signals: Range<usize>,
        samples: Range<usize>,
    ) -> Result<Vec<Vec<i32>>> {
        let indices: Vec<usize> = signals.collect();
        self.read_digital(&indices, samples)
    }

    /// Read one physical sample from each of the specified signals (proxy-coordinate indices).
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

    fn resolve_signal(&self, proxy_idx: usize) -> Result<usize> {
        self.group
            .indices
            .get(proxy_idx)
            .copied()
            .ok_or(EdfError::SignalOutOfRange {
                index: proxy_idx,
                count: self.group.len(),
            })
    }
}

fn read_physical_with_pad(
    file: &Arc<MappedFile>,
    sig_idx: usize,
    valid: usize,
    s_start: usize,
    s_end: usize,
    pad: PadMode,
) -> Result<Vec<f64>> {
    let proxy = SignalProxy::new(Arc::clone(file), sig_idx)?;
    let req_end = s_end;
    let req_start = s_start.min(req_end);

    if matches!(pad, PadMode::Raise) {
        if req_end > valid {
            return Err(EdfError::SampleOutOfRange {
                index: req_end - 1,
                count: valid,
            });
        }
        let count = req_end - req_start;
        let mut buf = vec![0.0f64; count];
        if count > 0 {
            proxy.read_physical(req_start, req_end, &mut buf)?;
        }
        return Ok(buf);
    }

    let count = req_end.saturating_sub(req_start);
    let mut buf = vec![0.0f64; count];
    if count == 0 {
        return Ok(buf);
    }

    let real_end = req_end.min(valid);
    let real_start = req_start.min(real_end);
    let real_count = real_end - real_start;
    if real_count > 0 {
        proxy.read_physical(real_start, real_end, &mut buf[..real_count])?;
    }

    if real_count < count {
        let fill = match pad {
            PadMode::Nan => f64::NAN,
            PadMode::Zero => 0.0,
            PadMode::Value(v) => v,
            PadMode::Edge => {
                if valid == 0 {
                    return Err(EdfError::SampleOutOfRange {
                        index: req_end - 1,
                        count: valid,
                    });
                }
                if real_count > 0 {
                    buf[real_count - 1]
                } else {
                    proxy.get_physical(valid - 1)?
                }
            }
            PadMode::Raise => unreachable!(),
        };
        for slot in &mut buf[real_count..] {
            *slot = fill;
        }
    }

    Ok(buf)
}

fn read_digital_with_pad(
    file: &Arc<MappedFile>,
    sig_idx: usize,
    valid: usize,
    s_start: usize,
    s_end: usize,
    pad: PadMode,
) -> Result<Vec<i32>> {
    if matches!(pad, PadMode::Nan) {
        return Err(EdfError::InvalidArgument {
            name: "pad_mode",
            reason: "PadMode::Nan is not valid for digital reads".into(),
        });
    }

    let proxy = SignalProxy::new(Arc::clone(file), sig_idx)?;
    let req_end = s_end;
    let req_start = s_start.min(req_end);

    if matches!(pad, PadMode::Raise) {
        if req_end > valid {
            return Err(EdfError::SampleOutOfRange {
                index: req_end - 1,
                count: valid,
            });
        }
        let count = req_end - req_start;
        let mut buf = vec![0i32; count];
        if count > 0 {
            proxy.read_digital(req_start, req_end, &mut buf)?;
        }
        return Ok(buf);
    }

    let count = req_end.saturating_sub(req_start);
    let mut buf = vec![0i32; count];
    if count == 0 {
        return Ok(buf);
    }

    let real_end = req_end.min(valid);
    let real_start = req_start.min(real_end);
    let real_count = real_end - real_start;
    if real_count > 0 {
        proxy.read_digital(real_start, real_end, &mut buf[..real_count])?;
    }

    if real_count < count {
        let fill: i32 = match pad {
            PadMode::Zero => 0,
            PadMode::Value(v) => {
                if !v.is_finite() || v < i32::MIN as f64 || v > i32::MAX as f64 {
                    return Err(EdfError::InvalidArgument {
                        name: "pad_mode",
                        reason: format!("PadMode::Value({v}) out of i32 range for digital read"),
                    });
                }
                v.trunc() as i32
            }
            PadMode::Edge => {
                if valid == 0 {
                    return Err(EdfError::SampleOutOfRange {
                        index: req_end - 1,
                        count: valid,
                    });
                }
                if real_count > 0 {
                    buf[real_count - 1]
                } else {
                    let mut one = [0i32; 1];
                    proxy.read_digital(valid - 1, valid, &mut one)?;
                    one[0]
                }
            }
            PadMode::Nan | PadMode::Raise => unreachable!(),
        };
        for slot in &mut buf[real_count..] {
            *slot = fill;
        }
    }

    Ok(buf)
}

impl Proxy2D {
    /// Build a 2D proxy from file-level signal indices.
    pub fn from_indices(
        file: Arc<MappedFile>,
        indices: &[usize],
        pad_mode: PadMode,
    ) -> Result<Self> {
        let group = SignalGroup::from_indices(&file.header, indices)?;
        Self::new(file, group, pad_mode)
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

    fn open(file: &NamedTempFile) -> Arc<MappedFile> {
        MappedFile::open(file.path()).unwrap()
    }

    #[test]
    fn shape_from_group() {
        let f = build_test_file(3, 2, 4);
        let mapped = open(&f);
        let p = Proxy2D::from_indices(mapped, &[0, 1, 2], PadMode::Raise).unwrap();
        assert_eq!(p.shape(), (3, 8));
        assert_eq!(p.sample_rate(), Some(4.0));
        assert_eq!(p.valid_lengths(), &[8, 8, 8]);
    }

    #[test]
    fn empty_group_errors() {
        let f = build_test_file(2, 2, 4);
        let mapped = open(&f);
        let header = &mapped.header;
        let err = SignalGroup::from_indices(header, &[]).unwrap_err();
        assert!(matches!(err, EdfError::InvalidArgument { .. }));
    }

    #[test]
    fn read_slice_raises_oob_by_default() {
        let f = build_test_file(2, 2, 4);
        let mapped = open(&f);
        let p = Proxy2D::from_indices(mapped, &[0, 1], PadMode::Raise).unwrap();
        let err = p.read_slice(0..2, 0..10).unwrap_err();
        assert!(matches!(err, EdfError::SampleOutOfRange { .. }));
    }

    #[test]
    fn read_slice_pad_nan_fills_past_end() {
        let f = build_test_file(2, 2, 4);
        let mapped = open(&f);
        let p = Proxy2D::from_indices(mapped, &[0, 1], PadMode::Nan).unwrap();
        let data = p.read_slice(0..1, 6..10).unwrap();
        assert_eq!(data[0].len(), 4);
        assert!(data[0][0].is_finite());
        assert!(data[0][1].is_finite());
        assert!(data[0][2].is_nan());
        assert!(data[0][3].is_nan());
    }

    #[test]
    fn read_slice_pad_zero_digital() {
        let f = build_test_file(2, 2, 4);
        let mapped = open(&f);
        let p = Proxy2D::from_indices(mapped, &[0], PadMode::Zero).unwrap();
        let data = p.read_slice_digital(0..1, 6..10).unwrap();
        // Signal 0 spr=4 cycles [0,1,2,3] per record; samples 6,7 = 2,3; 8,9 pad.
        assert_eq!(data[0], vec![2, 3, 0, 0]);
    }

    #[test]
    fn read_slice_pad_edge_replicates_last() {
        let f = build_test_file(2, 2, 4);
        let mapped = open(&f);
        let p = Proxy2D::from_indices(mapped, &[0], PadMode::Edge).unwrap();
        let data = p.read_slice(0..1, 6..10).unwrap();
        let last = data[0][1];
        assert_eq!(data[0][2], last);
        assert_eq!(data[0][3], last);
    }

    #[test]
    fn nan_pad_rejected_for_digital() {
        let f = build_test_file(2, 2, 4);
        let mapped = open(&f);
        let p = Proxy2D::from_indices(mapped, &[0], PadMode::Nan).unwrap();
        let err = p.read_slice_digital(0..1, 6..10).unwrap_err();
        assert!(matches!(err, EdfError::InvalidArgument { .. }));
    }

    #[test]
    fn get_pad_value() {
        let f = build_test_file(2, 2, 4);
        let mapped = open(&f);
        let p = Proxy2D::from_indices(mapped, &[0], PadMode::Value(-7.5)).unwrap();
        let v = p.get(0, 100).unwrap();
        assert_eq!(v, -7.5);
    }

    #[test]
    fn read_slice_via_group() {
        let f = build_test_file(2, 2, 4);
        let mapped = open(&f);
        let group = SignalGroup::from_indices(&mapped.header, &[0, 1]).unwrap();
        let p = Proxy2D::new(mapped, group, PadMode::Raise).unwrap();
        let data = p.read_slice(0..2, 0..4).unwrap();
        assert_eq!(data.len(), 2);
        assert!((data[0][0] - 0.0).abs() < f64::EPSILON);
        assert!((data[1][0] - 100.0).abs() < f64::EPSILON);
    }
}
