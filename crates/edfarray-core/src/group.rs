use crate::error::{EdfError, Result};
use crate::header::EdfHeader;

/// Structural classification of a `SignalGroup`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupKind {
    /// Shared sample rate and total sample count.
    Rectangular,
    /// Mixed sample rates. Only a 2D proxy can use this kind of group.
    Open,
}

/// Fill-value policy for a 2D proxy read that goes past the valid length of a channel.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PadMode {
    /// Fail with `SampleOutOfRange` when a read goes past the valid length. This is the default.
    #[default]
    Raise,
    /// Pad with `f64::NAN`. Physical reads only.
    Nan,
    /// Pad with `0.0` (physical) or `0` (digital).
    Zero,
    /// Pad with a value from the caller. Digital reads truncate the value to `i32`.
    Value(f64),
    /// Repeat the last valid sample of each channel.
    Edge,
}

/// Signal indices grouped to build a proxy, with supporting metadata.
///
/// The fields are crate-visible, not public, because invariants connect them. `Proxy3D` relies
/// on a `Rectangular` group that always has a sample rate and samples-per-record. Build one
/// with [`SignalGroup::from_indices`].
#[derive(Debug, Clone)]
pub struct SignalGroup {
    /// File-level signal indices.
    pub(crate) indices: Vec<usize>,
    /// Structural classification.
    pub(crate) kind: GroupKind,
    /// Common sample rate in Hz. `Some` only for `Rectangular`.
    pub(crate) sample_rate: Option<f64>,
    /// Common samples-per-record. `Some` only for `Rectangular`.
    pub(crate) samples_per_record: Option<usize>,
    /// Smallest total sample count across channels.
    pub(crate) min_samples: usize,
    /// Largest total sample count across channels.
    pub(crate) max_samples: usize,
    /// True if the group contains every ordinary signal. Only `EdfFile::signal_groups` sets it.
    pub(crate) covers_all_ordinary: bool,
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
        })
    }

    /// File-level signal indices in the group.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Structural classification of the group.
    pub fn kind(&self) -> GroupKind {
        self.kind
    }

    /// Common sample rate in Hz. `Some` only for `Rectangular` groups.
    pub fn sample_rate(&self) -> Option<f64> {
        self.sample_rate
    }

    /// Common samples-per-record. `Some` only for `Rectangular` groups.
    pub fn samples_per_record(&self) -> Option<usize> {
        self.samples_per_record
    }

    /// Smallest total sample count across the group's channels.
    pub fn min_samples(&self) -> usize {
        self.min_samples
    }

    /// Largest total sample count across the group's channels.
    pub fn max_samples(&self) -> usize {
        self.max_samples
    }

    /// Whether the group contains every ordinary signal in the file.
    pub fn covers_all_ordinary(&self) -> bool {
        self.covers_all_ordinary
    }

    /// Whether the group holds exactly one channel.
    pub fn is_singleton(&self) -> bool {
        self.indices.len() == 1
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
        assert!(!g.is_singleton());
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
        assert!(g.is_singleton());
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
