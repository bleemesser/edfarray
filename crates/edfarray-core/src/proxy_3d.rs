use std::ops::Range;
use std::sync::Arc;

use rayon::prelude::*;

use crate::error::{EdfError, Result};
use crate::group::{GroupKind, SignalGroup};
use crate::mmap::MappedFile;
use crate::proxy::SignalProxy;

/// 3D view over a rectangular `SignalGroup`. Shape: `(num_records, num_channels, samples_per_record)`.
#[derive(Debug)]
pub struct Proxy3D {
    file: Arc<MappedFile>,
    group: SignalGroup,
    num_records: usize,
    samples_per_record: usize,
}

impl Proxy3D {
    /// Build a 3D proxy. Requires a `Rectangular` group (all channels share `samples_per_record`).
    pub fn new(file: Arc<MappedFile>, group: SignalGroup) -> Result<Self> {
        if !matches!(group.kind, GroupKind::Rectangular) {
            return Err(EdfError::InvalidArgument {
                name: "group",
                reason: "Proxy3D requires a Rectangular signal group; use signal_groups() \
                         to pick a same-rate group, or build Proxy2D instead"
                    .into(),
            });
        }
        let num_records = file.header.num_records.max(0) as usize;
        let samples_per_record = group
            .samples_per_record
            .expect("Rectangular group always has samples_per_record");

        Ok(Proxy3D {
            file,
            group,
            num_records,
            samples_per_record,
        })
    }

    /// Shape of the 3D view: `(num_records, num_channels, samples_per_record)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.num_records, self.group.len(), self.samples_per_record)
    }

    /// Common sample rate (Hz) of the group's channels.
    pub fn sample_rate(&self) -> f64 {
        self.group
            .sample_rate
            .expect("Rectangular group always has sample_rate")
    }

    pub fn group(&self) -> &SignalGroup {
        &self.group
    }

    /// Read a single physical sample at `(record, channel, sample)`.
    pub fn get(&self, record: usize, channel: usize, sample: usize) -> Result<f64> {
        let sig_idx = self.resolve_channel(channel)?;
        if record >= self.num_records {
            return Err(EdfError::RecordOutOfRange {
                index: record,
                count: self.num_records,
            });
        }
        if sample >= self.samples_per_record {
            return Err(EdfError::SampleOutOfRange {
                index: sample,
                count: self.samples_per_record,
            });
        }
        let proxy = SignalProxy::new(Arc::clone(&self.file), sig_idx)?;
        proxy.get_physical(record * self.samples_per_record + sample)
    }

    /// Read a contiguous block of physical samples. Returns a flat Vec in record-major order.
    pub fn read_physical_block(
        &self,
        records: Range<usize>,
        channels: Range<usize>,
    ) -> Result<Vec<f64>> {
        self.read_block(records, channels, 0.0, |proxy, start, end, buf| {
            proxy.read_physical(start, end, buf)
        })
    }

    /// Read a contiguous block of digital (i32) samples in record-major order.
    pub fn read_digital_block(
        &self,
        records: Range<usize>,
        channels: Range<usize>,
    ) -> Result<Vec<i32>> {
        self.read_block(records, channels, 0, |proxy, start, end, buf| {
            proxy.read_digital(start, end, buf)
        })
    }

    /// Read every requested channel over a record range and interleave into record-major order.
    fn read_block<T, F>(
        &self,
        records: Range<usize>,
        channels: Range<usize>,
        fill: T,
        read: F,
    ) -> Result<Vec<T>>
    where
        T: Copy + Send + Sync,
        F: Fn(&SignalProxy, usize, usize, &mut [T]) -> Result<()> + Send + Sync,
    {
        self.check_record_range(&records)?;
        self.check_channel_range(&channels)?;
        let n_rec = records.len();
        let n_ch = channels.len();
        let spr = self.samples_per_record;
        let mut out = vec![fill; n_rec * n_ch * spr];
        if n_rec == 0 || n_ch == 0 {
            return Ok(out);
        }

        let file = &self.file;
        let group = &self.group;
        let rec_start = records.start;
        let ch_start = channels.start;

        let per_channel: Result<Vec<Vec<T>>> = (0..n_ch)
            .into_par_iter()
            .map(|ci| {
                let sig_idx = group.indices[ch_start + ci];
                let proxy = SignalProxy::new(Arc::clone(file), sig_idx)?;
                let mut buf = vec![fill; n_rec * spr];
                read(&proxy, rec_start * spr, (rec_start + n_rec) * spr, &mut buf)?;
                Ok(buf)
            })
            .collect();
        let per_channel = per_channel?;

        for ri in 0..n_rec {
            for (ci, channel) in per_channel.iter().enumerate() {
                let dst_base = (ri * n_ch + ci) * spr;
                let src_base = ri * spr;
                out[dst_base..dst_base + spr].copy_from_slice(&channel[src_base..src_base + spr]);
            }
        }
        Ok(out)
    }

    /// Metadata for a zero-copy strided view over the raw int16 mmap.
    ///
    /// Returns `Some` only for 2-byte EDF files with contiguous, non-annotation channels.
    pub fn stride_info(&self) -> Option<StrideInfo> {
        if self.file.layout.sample_size_bytes != 2 {
            return None;
        }
        let n_ch = self.group.len();
        if n_ch == 0 {
            return None;
        }

        // Channels must be a contiguous index range.
        let first = self.group.indices[0];
        for (i, &idx) in self.group.indices.iter().enumerate() {
            if idx != first + i {
                return None;
            }
        }

        // Annotation channels corrupt the stride view.
        for idx in first..first + n_ch {
            if self.file.header.signals[idx].is_annotation {
                return None;
            }
        }

        let data_offset = self.file.header.data_offset();
        let record_size = self.file.layout.record_size;
        let channel_offset_in_record = self.file.layout.signal_offsets[first];
        let spr = self.samples_per_record;

        let info = StrideInfo {
            base_offset: data_offset + channel_offset_in_record,
            record_stride_bytes: record_size,
            channel_stride_bytes: spr * 2,
            sample_stride_bytes: 2,
            shape: (self.num_records, n_ch, spr),
        };

        // Consumers build a zero-copy view from these numbers, so they must be bounded by the
        // real mapping rather than by what the header claims.
        let last_byte = info
            .base_offset
            .checked_add(self.num_records.checked_sub(1)?.checked_mul(record_size)?)?
            .checked_add((n_ch - 1).checked_mul(spr * 2)?)?
            .checked_add(spr.checked_mul(2)?)?;
        if last_byte > self.file.len() {
            return None;
        }

        Some(info)
    }

    fn resolve_channel(&self, proxy_idx: usize) -> Result<usize> {
        self.group
            .indices
            .get(proxy_idx)
            .copied()
            .ok_or(EdfError::SignalOutOfRange {
                index: proxy_idx,
                count: self.group.len(),
            })
    }

    fn check_record_range(&self, r: &Range<usize>) -> Result<()> {
        if r.end > self.num_records {
            return Err(EdfError::RecordOutOfRange {
                index: r.end.saturating_sub(1),
                count: self.num_records,
            });
        }
        Ok(())
    }

    fn check_channel_range(&self, r: &Range<usize>) -> Result<()> {
        if r.end > self.group.len() {
            return Err(EdfError::SignalOutOfRange {
                index: r.end.saturating_sub(1),
                count: self.group.len(),
            });
        }
        Ok(())
    }
}

/// Byte-level metadata for a zero-copy strided view of a 3D proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrideInfo {
    pub base_offset: usize,
    pub record_stride_bytes: usize,
    pub channel_stride_bytes: usize,
    pub sample_stride_bytes: usize,
    pub shape: (usize, usize, usize),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::PadMode;
    use crate::proxy_2d::Proxy2D;
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

    #[test]
    fn shape_records_channels_spr() {
        let f = build_test_file(3, 5, 4);
        let mapped = MappedFile::open(f.path()).unwrap();
        let group = SignalGroup::from_indices(&mapped.header, &[0, 1, 2]).unwrap();
        let p = Proxy3D::new(mapped, group).unwrap();
        assert_eq!(p.shape(), (5, 3, 4));
        assert_eq!(p.sample_rate(), 4.0);
    }

    #[test]
    fn rejects_open_group() {
        // Build a file with two different sprs.
        let header_bytes = 256 + 256 * 2;
        let mut buf = vec![b' '; header_bytes];
        write_field(&mut buf, 0, 8, "0");
        write_field(&mut buf, 8, 80, "X X X X");
        write_field(&mut buf, 88, 80, "Startdate X X X X");
        write_field(&mut buf, 168, 8, "01.01.00");
        write_field(&mut buf, 176, 8, "00.00.00");
        write_field(&mut buf, 184, 8, &header_bytes.to_string());
        write_field(&mut buf, 192, 44, "");
        write_field(&mut buf, 236, 8, "2");
        write_field(&mut buf, 244, 8, "1");
        write_field(&mut buf, 252, 4, "2");
        let sig_data = &mut buf[256..];
        let sprs = [4, 8];
        for (i, spr) in sprs.iter().enumerate() {
            write_sig(sig_data, i, 2, 0, 16, &format!("EEG{}", i));
            write_sig(sig_data, i, 2, 96, 8, "uV");
            write_sig(sig_data, i, 2, 104, 8, "-100");
            write_sig(sig_data, i, 2, 112, 8, "100");
            write_sig(sig_data, i, 2, 120, 8, "-100");
            write_sig(sig_data, i, 2, 128, 8, "100");
            write_sig(sig_data, i, 2, 216, 8, &spr.to_string());
        }
        for _ in 0..2 {
            for spr in sprs {
                for _ in 0..spr {
                    buf.extend_from_slice(&0i16.to_le_bytes());
                }
            }
        }
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&buf).unwrap();
        file.flush().unwrap();

        let mapped = MappedFile::open(file.path()).unwrap();
        let group = SignalGroup::from_indices(&mapped.header, &[0, 1]).unwrap();
        assert_eq!(group.kind, GroupKind::Open);
        let err = Proxy3D::new(mapped, group).unwrap_err();
        assert!(matches!(err, EdfError::InvalidArgument { .. }));
    }

    #[test]
    fn read_block_matches_2d() {
        let f = build_test_file(3, 4, 5);
        let mapped = MappedFile::open(f.path()).unwrap();
        let group = SignalGroup::from_indices(&mapped.header, &[0, 1, 2]).unwrap();
        let p3 = Proxy3D::new(Arc::clone(&mapped), group.clone()).unwrap();
        let p2 = Proxy2D::new(mapped, group, PadMode::Raise).unwrap();

        let block = p3.read_physical_block(0..4, 0..3).unwrap();
        for ri in 0..4 {
            for ci in 0..3 {
                for si in 0..5 {
                    let v3 = block[(ri * 3 + ci) * 5 + si];
                    let v2 = p2.get(ci, ri * 5 + si).unwrap();
                    assert!(
                        (v3 - v2).abs() < f64::EPSILON,
                        "mismatch at ({ri},{ci},{si})"
                    );
                }
            }
        }
    }

    #[test]
    fn get_indexes_correctly() {
        let f = build_test_file(2, 2, 4);
        let mapped = MappedFile::open(f.path()).unwrap();
        let group = SignalGroup::from_indices(&mapped.header, &[0, 1]).unwrap();
        let p = Proxy3D::new(mapped, group).unwrap();
        // signal sig writes sig*100 + s for sample s within each record.
        assert_eq!(p.get(0, 0, 0).unwrap(), 0.0);
        assert_eq!(p.get(0, 1, 3).unwrap(), 103.0);
        assert_eq!(p.get(1, 0, 2).unwrap(), 2.0);
    }

    #[test]
    fn stride_info_full_data_file() {
        // No annotation channels; group covers all signals, so contiguous.
        let f = build_test_file(3, 4, 5);
        let mapped = MappedFile::open(f.path()).unwrap();
        let group = SignalGroup::from_indices(&mapped.header, &[0, 1, 2]).unwrap();
        let p = Proxy3D::new(mapped, group).unwrap();
        let info = p
            .stride_info()
            .expect("contiguous data -> stride view available");
        assert_eq!(info.shape, (4, 3, 5));
        assert_eq!(info.sample_stride_bytes, 2);
        assert_eq!(info.channel_stride_bytes, 5 * 2);
        assert_eq!(info.record_stride_bytes, 3 * 5 * 2);
    }

    #[test]
    fn stride_info_skipping_channels_returns_none() {
        let f = build_test_file(4, 2, 4);
        let mapped = MappedFile::open(f.path()).unwrap();
        // Pick non-contiguous channels -> no stride view.
        let group = SignalGroup::from_indices(&mapped.header, &[0, 2]).unwrap();
        let p = Proxy3D::new(mapped, group).unwrap();
        assert!(p.stride_info().is_none());
    }

    #[test]
    fn record_out_of_range_errors() {
        let f = build_test_file(2, 2, 4);
        let mapped = MappedFile::open(f.path()).unwrap();
        let group = SignalGroup::from_indices(&mapped.header, &[0, 1]).unwrap();
        let p = Proxy3D::new(mapped, group).unwrap();
        let err = p.read_physical_block(0..5, 0..2).unwrap_err();
        assert!(matches!(err, EdfError::RecordOutOfRange { .. }));
    }
}
