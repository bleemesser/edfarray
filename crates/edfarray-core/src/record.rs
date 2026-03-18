use crate::error::{EdfError, Result};
use crate::header::EdfHeader;

/// Describes the byte layout of signals within a single data record.
///
/// EDF data records contain all signals sequentially: all samples for signal 0,
/// then all samples for signal 1, etc. Each sample is a 2-byte little-endian
/// signed integer.
#[derive(Debug, Clone)]
pub struct RecordLayout {
    pub record_size: usize,
    pub signal_offsets: Vec<usize>,
    pub signal_sample_counts: Vec<usize>,
}

impl RecordLayout {
    /// Build the record layout from a parsed header.
    pub fn from_header(header: &EdfHeader) -> Self {
        let mut offsets = Vec::with_capacity(header.num_signals);
        let mut counts = Vec::with_capacity(header.num_signals);
        let mut offset = 0usize;

        for sig in &header.signals {
            offsets.push(offset);
            counts.push(sig.num_samples);
            offset += sig.num_samples * 2;
        }

        RecordLayout {
            record_size: offset,
            signal_offsets: offsets,
            signal_sample_counts: counts,
        }
    }

    /// Extract the raw bytes for a specific signal within a data record.
    ///
    /// `record_data` must be exactly `record_size` bytes (one complete data record).
    pub fn signal_bytes<'a>(
        &self,
        record_data: &'a [u8],
        signal_idx: usize,
    ) -> Result<&'a [u8]> {
        if signal_idx >= self.signal_offsets.len() {
            return Err(EdfError::SignalOutOfRange {
                index: signal_idx,
                count: self.signal_offsets.len(),
            });
        }
        let start = self.signal_offsets[signal_idx];
        let byte_count = self.signal_sample_counts[signal_idx] * 2;
        record_data
            .get(start..start + byte_count)
            .ok_or(EdfError::RecordOutOfRange {
                index: 0,
                count: 0,
            })
    }

    /// Decode raw little-endian i16 bytes into physical f64 values.
    ///
    /// Uses a two-pass approach to help the compiler autovectorize:
    /// first widen i16→f64, then apply gain and offset as a uniform f64 pass.
    pub fn decode_physical(raw: &[u8], gain: f64, offset: f64, out: &mut [f64]) {
        for (i, chunk) in raw.chunks_exact(2).enumerate() {
            out[i] = i16::from_le_bytes([chunk[0], chunk[1]]) as f64;
        }
        for val in out.iter_mut() {
            *val = *val * gain + offset;
        }
    }

    /// Decode raw little-endian i16 bytes into digital i16 values.
    pub fn decode_digital(raw: &[u8], out: &mut [i16]) {
        for (i, chunk) in raw.chunks_exact(2).enumerate() {
            out[i] = i16::from_le_bytes([chunk[0], chunk[1]]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_physical_values() {
        let raw = [
            0x00, 0x00, // 0
            0xFF, 0x7F, // 32767
            0x00, 0x80, // -32768
        ];
        let gain = 6400.0 / 65535.0;
        let offset = -3200.0 - gain * -32768.0;
        let mut out = [0.0f64; 3];

        RecordLayout::decode_physical(&raw, gain, offset, &mut out);

        assert!((out[0]).abs() < 0.1);
        assert!((out[1] - 3200.0).abs() < 0.1);
        assert!((out[2] - (-3200.0)).abs() < 0.1);
    }

    #[test]
    fn decode_digital_values() {
        let raw = [
            0x00, 0x00, // 0
            0x01, 0x00, // 1
            0xFF, 0xFF, // -1
        ];
        let mut out = [0i16; 3];
        RecordLayout::decode_digital(&raw, &mut out);
        assert_eq!(out, [0, 1, -1]);
    }

    #[test]
    fn layout_from_header() {
        let header_bytes = build_test_header_for_layout();
        let header = EdfHeader::parse(&header_bytes).unwrap();
        let layout = RecordLayout::from_header(&header);

        assert_eq!(layout.signal_offsets.len(), 2);
        assert_eq!(layout.signal_offsets[0], 0);
        assert_eq!(layout.signal_offsets[1], 512); // 256 samples * 2 bytes
        assert_eq!(layout.record_size, 1024); // 2 signals * 256 samples * 2 bytes
    }

    fn build_test_header_for_layout() -> Vec<u8> {
        let num_signals = 2;
        let header_bytes = 256 + 256 * num_signals;
        let mut buf = vec![b' '; header_bytes];

        write_field(&mut buf, 0, 8, "0");
        write_field(&mut buf, 8, 80, "X X X X");
        write_field(&mut buf, 88, 80, "Startdate X X X X");
        write_field(&mut buf, 168, 8, "01.01.00");
        write_field(&mut buf, 176, 8, "00.00.00");
        write_field(&mut buf, 184, 8, &header_bytes.to_string());
        write_field(&mut buf, 192, 44, "EDF+C");
        write_field(&mut buf, 236, 8, "10");
        write_field(&mut buf, 244, 8, "1");
        write_field(&mut buf, 252, 4, &num_signals.to_string());

        let sig_data = &mut buf[256..];
        for i in 0..num_signals {
            write_sig(sig_data, i, num_signals, 0, 16, "EEG");
            write_sig(sig_data, i, num_signals, 16, 80, "");
            write_sig(sig_data, i, num_signals, 96, 8, "uV");
            write_sig(sig_data, i, num_signals, 104, 8, "-3200");
            write_sig(sig_data, i, num_signals, 112, 8, "3200");
            write_sig(sig_data, i, num_signals, 120, 8, "-32768");
            write_sig(sig_data, i, num_signals, 128, 8, "32767");
            write_sig(sig_data, i, num_signals, 136, 80, "");
            write_sig(sig_data, i, num_signals, 216, 8, "256");
            write_sig(sig_data, i, num_signals, 224, 32, "");
        }
        buf
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
