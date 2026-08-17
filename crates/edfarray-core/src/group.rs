use crate::error::{EdfError, Result};
use crate::header::EdfHeader;

/// Structural classification of a `SignalGroup`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupKind {
    /// Shared sample rate and total sample count.
    Rectangular,
    /// Mixed sample rates (2D proxy only).
    Open,
}

/// Fill-value policy when 2D proxy reads extend past a channel's valid length.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PadMode {
    /// Raise `SampleOutOfRange` on overflow. Default.
    #[default]
    Raise,
    /// Pad with `f64::NAN`. Physical reads only.
    Nan,
    /// Pad with `0.0` (physical) or `0` (digital).
    Zero,
    /// Pad with caller-supplied value. Truncated to `i32` on digital reads.
    Value(f64),
    /// Replicate last valid sample per channel.
    Edge,
}

/// Signal indices grouped for proxy construction with supporting metadata.
#[derive(Debug, Clone)]
pub struct SignalGroup {
    /// File-level signal indices.
    pub indices: Vec<usize>,
    /// Structural classification.
    pub kind: GroupKind,
    /// Common sample rate in Hz. `Some` only for `Rectangular`.
    pub sample_rate: Option<f64>,
    /// Common samples-per-record. `Some` only for `Rectangular`.
    pub samples_per_record: Option<usize>,
    /// Min total sample count across channels.
    pub min_samples: usize,
    /// Max total sample count across channels.
    pub max_samples: usize,
    /// True if group contains every ordinary signal. Set only by `EdfFile::signal_groups`.
    pub covers_all_ordinary: bool,
    /// True if `indices.len() == 1`.
    pub is_singleton: bool,
}

impl SignalGroup {
    /// Classify signal indices into a group. Does not reject annotation channels.
    pub fn from_indices(header: &EdfHeader, indices: &[usize]) -> Result<Self> {
        if indices.is_empty() {
            return Err(EdfError::InvalidArgument {
                name: "indices",
                reason: "signal group must contain at least one index".into(),
            });
        }

        let n = header.num_signals;
        for &i in indices {
            if i >= n {
                return Err(EdfError::SignalOutOfRange { index: i, count: n });
            }
        }

        let rd = header.record_duration_secs;
        let num_records = header.num_records.max(0) as usize;

        let first = &header.signals[indices[0]];
        let first_rate = first.sample_rate(rd);
        let first_spr = first.num_samples;
        let first_total = num_records * first_spr;

        let mut all_same_rate = true;
        let mut min_samples = first_total;
        let mut max_samples = first_total;

        for &i in &indices[1..] {
            let s = &header.signals[i];
            let rate = s.sample_rate(rd);
            if (rate - first_rate).abs() > first_rate.abs().max(rate.abs()) * 1e-9 {
                all_same_rate = false;
            }
            let total = num_records * s.num_samples;
            if total < min_samples {
                min_samples = total;
            }
            if total > max_samples {
                max_samples = total;
            }
        }

        let kind = if all_same_rate {
            GroupKind::Rectangular
        } else {
            GroupKind::Open
        };

        let (sample_rate, samples_per_record) = match kind {
            GroupKind::Rectangular => (Some(first_rate), Some(first_spr)),
            GroupKind::Open => (None, None),
        };

        Ok(SignalGroup {
            indices: indices.to_vec(),
            kind,
            sample_rate,
            samples_per_record,
            min_samples,
            max_samples,
            covers_all_ordinary: false,
            is_singleton: indices.len() == 1,
        })
    }

    /// Number of channels in the group.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// `true` if a 3D proxy can be built from this group.
    pub fn is_rectangular(&self) -> bool {
        matches!(self.kind, GroupKind::Rectangular)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{EdfVariant, MaybeDateTime};
    use crate::signal::SignalHeader;

    fn sig(label: &str, num_samples: usize, is_annotation: bool) -> SignalHeader {
        SignalHeader {
            label: label.into(),
            transducer: String::new(),
            physical_dimension: String::new(),
            physical_min: -100.0,
            physical_max: 100.0,
            digital_min: -32768,
            digital_max: 32767,
            prefiltering: String::new(),
            num_samples,
            reserved: String::new(),
            gain: 1.0,
            offset: 0.0,
            is_annotation,
        }
    }

    fn header(signals: Vec<SignalHeader>, num_records: i64, rd: f64) -> EdfHeader {
        EdfHeader {
            version: "0".into(),
            patient_id: String::new(),
            recording_id: String::new(),
            start_datetime: MaybeDateTime::Raw {
                date: String::new(),
                time: String::new(),
            },
            header_bytes: 256,
            variant: EdfVariant::Edf,
            num_records,
            record_duration_secs: rd,
            num_signals: signals.len(),
            signals,
            patient: Default::default(),
            recording: Default::default(),
            warnings: Vec::new(),
        }
    }

    #[test]
    fn rectangular_classification() {
        let h = header(
            vec![sig("EEG0", 256, false), sig("EEG1", 256, false)],
            10,
            1.0,
        );
        let g = SignalGroup::from_indices(&h, &[0, 1]).unwrap();
        assert_eq!(g.kind, GroupKind::Rectangular);
        assert_eq!(g.sample_rate, Some(256.0));
        assert_eq!(g.samples_per_record, Some(256));
        assert_eq!(g.min_samples, 2560);
        assert_eq!(g.max_samples, 2560);
        assert!(!g.is_singleton);
        assert!(g.is_rectangular());
    }

    #[test]
    fn open_classification_mixed_rates() {
        let h = header(
            vec![sig("EEG0", 256, false), sig("RESP", 32, false)],
            4,
            1.0,
        );
        let g = SignalGroup::from_indices(&h, &[0, 1]).unwrap();
        assert_eq!(g.kind, GroupKind::Open);
        assert_eq!(g.sample_rate, None);
        assert_eq!(g.samples_per_record, None);
        assert_eq!(g.min_samples, 128);
        assert_eq!(g.max_samples, 1024);
    }

    #[test]
    fn singleton() {
        let h = header(vec![sig("EEG0", 256, false)], 4, 1.0);
        let g = SignalGroup::from_indices(&h, &[0]).unwrap();
        assert!(g.is_singleton);
        assert_eq!(g.kind, GroupKind::Rectangular);
    }

    #[test]
    fn empty_indices_errors() {
        let h = header(vec![sig("EEG0", 256, false)], 4, 1.0);
        let err = SignalGroup::from_indices(&h, &[]).unwrap_err();
        assert!(matches!(err, EdfError::InvalidArgument { .. }));
    }

    #[test]
    fn out_of_range_errors() {
        let h = header(vec![sig("EEG0", 256, false)], 4, 1.0);
        let err = SignalGroup::from_indices(&h, &[0, 5]).unwrap_err();
        assert!(matches!(err, EdfError::SignalOutOfRange { .. }));
    }

    #[test]
    fn covers_all_ordinary_default_false() {
        let h = header(vec![sig("EEG0", 256, false)], 4, 1.0);
        let g = SignalGroup::from_indices(&h, &[0]).unwrap();
        assert!(!g.covers_all_ordinary);
    }
}
