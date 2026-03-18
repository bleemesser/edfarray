use crate::error::{EdfError, Result};

/// Label used by EDF+ to identify annotation signals.
const EDF_ANNOTATIONS_LABEL: &str = "EDF Annotations";

/// Metadata for a single signal, parsed from the per-signal header fields.
#[derive(Debug, Clone)]
pub struct SignalHeader {
    pub label: String,
    pub transducer: String,
    pub physical_dimension: String,
    pub physical_min: f64,
    pub physical_max: f64,
    pub digital_min: i16,
    pub digital_max: i16,
    pub prefiltering: String,
    pub num_samples: usize,
    pub reserved: String,
    pub gain: f64,
    pub offset: f64,
    pub is_annotation: bool,
}

impl SignalHeader {
    /// Parse the header fields for signal at `index` from the per-signal header bytes.
    ///
    /// The EDF format stores per-signal fields in a transposed layout: all labels
    /// come first (16 bytes × ns), then all transducer types (80 bytes × ns), etc.
    /// The caller provides the full per-signal header block and the total signal count.
    pub fn parse(data: &[u8], index: usize, num_signals: usize) -> Result<Self> {
        let label = read_signal_field(data, index, num_signals, 0, 16)?;
        let transducer = read_signal_field(data, index, num_signals, 16, 80)?;
        let physical_dimension = read_signal_field(data, index, num_signals, 96, 8)?;

        let physical_min = parse_signal_f64(data, index, num_signals, 104, 8, "physical_min")?;
        let physical_max = parse_signal_f64(data, index, num_signals, 112, 8, "physical_max")?;
        let digital_min = parse_signal_i16(data, index, num_signals, 120, 8, "digital_min")?;
        let digital_max = parse_signal_i16(data, index, num_signals, 128, 8, "digital_max")?;

        let prefiltering = read_signal_field(data, index, num_signals, 136, 80)?;
        let num_samples = parse_signal_usize(data, index, num_signals, 216, 8, "num_samples")?;
        let reserved = read_signal_field(data, index, num_signals, 224, 32)?;

        if digital_min >= digital_max {
            return Err(EdfError::InvalidDigitalRange {
                index,
                min: digital_min,
                max: digital_max,
            });
        }

        if (physical_min - physical_max).abs() < f64::EPSILON {
            return Err(EdfError::InvalidPhysicalRange {
                index,
                min: physical_min,
                max: physical_max,
            });
        }

        let gain = (physical_max - physical_min) / (digital_max as f64 - digital_min as f64);
        let offset = physical_min - gain * digital_min as f64;
        let is_annotation = label.starts_with(EDF_ANNOTATIONS_LABEL);

        Ok(SignalHeader {
            label,
            transducer,
            physical_dimension,
            physical_min,
            physical_max,
            digital_min,
            digital_max,
            prefiltering,
            num_samples,
            reserved,
            gain,
            offset,
            is_annotation,
        })
    }

    /// Convert a raw digital sample value to its physical value.
    pub fn digital_to_physical(&self, digital: i16) -> f64 {
        self.gain * digital as f64 + self.offset
    }

    /// Computed sample rate given the data record duration in seconds.
    pub fn sample_rate(&self, record_duration_secs: f64) -> f64 {
        if record_duration_secs == 0.0 {
            0.0
        } else {
            self.num_samples as f64 / record_duration_secs
        }
    }
}

/// Read a trimmed ASCII string field from the transposed per-signal header layout.
///
/// In EDF, per-signal fields are stored contiguously for all signals:
/// field_start + field_size * signal_index gives the offset for a specific signal.
fn read_signal_field(
    data: &[u8],
    index: usize,
    num_signals: usize,
    field_offset: usize,
    field_size: usize,
) -> Result<String> {
    let start = field_offset * num_signals + field_size * index;
    let end = start + field_size;
    let bytes = data
        .get(start..end)
        .ok_or_else(|| EdfError::InvalidSignalField {
            index,
            field: "header",
            reason: format!("signal header truncated at byte {start}"),
        })?;
    Ok(String::from_utf8_lossy(bytes).trim().to_string())
}

fn parse_signal_f64(
    data: &[u8],
    index: usize,
    num_signals: usize,
    field_offset: usize,
    field_size: usize,
    field_name: &'static str,
) -> Result<f64> {
    let s = read_signal_field(data, index, num_signals, field_offset, field_size)?;
    s.parse::<f64>().map_err(|_| EdfError::InvalidSignalField {
        index,
        field: field_name,
        reason: format!("not a valid number: {:?}", s),
    })
}

fn parse_signal_i16(
    data: &[u8],
    index: usize,
    num_signals: usize,
    field_offset: usize,
    field_size: usize,
    field_name: &'static str,
) -> Result<i16> {
    let s = read_signal_field(data, index, num_signals, field_offset, field_size)?;
    s.parse::<i16>().map_err(|_| EdfError::InvalidSignalField {
        index,
        field: field_name,
        reason: format!("not a valid i16: {:?}", s),
    })
}

fn parse_signal_usize(
    data: &[u8],
    index: usize,
    num_signals: usize,
    field_offset: usize,
    field_size: usize,
    field_name: &'static str,
) -> Result<usize> {
    let s = read_signal_field(data, index, num_signals, field_offset, field_size)?;
    s.parse::<usize>()
        .map_err(|_| EdfError::InvalidSignalField {
            index,
            field: field_name,
            reason: format!("not a valid unsigned integer: {:?}", s),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_signal_header_bytes(num_signals: usize, fields: &[(&[u8], usize)]) -> Vec<u8> {
        let total: usize = fields.iter().map(|(_, size)| size * num_signals).sum();
        let mut buf = vec![b' '; total];
        for (field_idx, (values, field_size)) in fields.iter().enumerate() {
            let field_offset: usize = fields[..field_idx]
                .iter()
                .map(|(_, s)| s * num_signals)
                .sum();
            for sig in 0..num_signals {
                let start = field_offset + field_size * sig;
                let len = values.len().min(*field_size);
                buf[start..start + len].copy_from_slice(&values[..len]);
            }
        }
        buf
    }

    #[test]
    fn parse_valid_signal() {
        let fields: Vec<(&[u8], usize)> = vec![
            (b"EEG Fp1", 16),  // label
            (b"AgAgCl", 80),   // transducer
            (b"uV", 8),        // physical_dimension
            (b"-3200", 8),     // physical_min
            (b"3200", 8),      // physical_max
            (b"-32768", 8),    // digital_min
            (b"32767", 8),     // digital_max
            (b"HP:0.1Hz", 80), // prefiltering
            (b"256", 8),       // num_samples
            (b"", 32),         // reserved
        ];
        let data = build_signal_header_bytes(1, &fields);
        let sig = SignalHeader::parse(&data, 0, 1).unwrap();

        assert_eq!(sig.label, "EEG Fp1");
        assert_eq!(sig.transducer, "AgAgCl");
        assert_eq!(sig.physical_dimension, "uV");
        assert_eq!(sig.physical_min, -3200.0);
        assert_eq!(sig.physical_max, 3200.0);
        assert_eq!(sig.digital_min, -32768);
        assert_eq!(sig.digital_max, 32767);
        assert_eq!(sig.num_samples, 256);
        assert!(!sig.is_annotation);
        assert!((sig.gain - (6400.0 / 65535.0)).abs() < 1e-10);
    }

    #[test]
    fn annotation_signal_detected() {
        let fields: Vec<(&[u8], usize)> = vec![
            (b"EDF Annotations", 16),
            (b"", 80),
            (b"", 8),
            (b"0", 8),
            (b"1", 8),
            (b"0", 8),
            (b"1", 8),
            (b"", 80),
            (b"30", 8),
            (b"", 32),
        ];
        let data = build_signal_header_bytes(1, &fields);
        let sig = SignalHeader::parse(&data, 0, 1).unwrap();
        assert!(sig.is_annotation);
    }

    #[test]
    fn invalid_digital_range_rejected() {
        let fields: Vec<(&[u8], usize)> = vec![
            (b"EEG", 16),
            (b"", 80),
            (b"uV", 8),
            (b"-100", 8),
            (b"100", 8),
            (b"100", 8), // digital_min > digital_max
            (b"-100", 8),
            (b"", 80),
            (b"256", 8),
            (b"", 32),
        ];
        let data = build_signal_header_bytes(1, &fields);
        let err = SignalHeader::parse(&data, 0, 1).unwrap_err();
        assert!(matches!(err, EdfError::InvalidDigitalRange { .. }));
    }

    #[test]
    fn digital_to_physical_conversion() {
        let fields: Vec<(&[u8], usize)> = vec![
            (b"EEG", 16),
            (b"", 80),
            (b"uV", 8),
            (b"-3200", 8),
            (b"3200", 8),
            (b"-32768", 8),
            (b"32767", 8),
            (b"", 80),
            (b"256", 8),
            (b"", 32),
        ];
        let data = build_signal_header_bytes(1, &fields);
        let sig = SignalHeader::parse(&data, 0, 1).unwrap();

        let phys = sig.digital_to_physical(0);
        assert!(phys.abs() < 0.1); // near zero for midpoint
    }
}
