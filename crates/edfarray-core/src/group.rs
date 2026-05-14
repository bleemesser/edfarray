use crate::error::{EdfError, Result};
use crate::header::EdfHeader;

/// Structural classification of a [`SignalGroup`].
///
/// Within a single EDF file, `record_duration_secs` and `num_records` are
/// file-wide and `num_samples` is per-signal, so `rate = num_samples /
/// record_duration` implies same-rate channels always have the same total
/// sample count. The two reachable cases for a single-file group are
/// therefore:
/// - `Rectangular` -> shared sample rate; upgradable to a 3D strided view.
/// - `Open` -> mixed sample rates; 2D only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GroupKind {
    /// All channels share a sample rate and total sample count.
    Rectangular,
    /// Channels differ in sample rate (and hence in total sample count).
    Open,
}

/// Fill-value policy applied when a read on a 2D proxy extends past a
/// channel's valid length (i.e. when the group is `Open` and shorter channels
/// are read past their end).
///
/// Padding is interpreted in the *domain of the read*: a physical read sees an
/// `f64` fill, a digital read sees an `i32` fill. The fill is a logical
/// placeholder, not a synthetic digital sample run through per-channel scaling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PadMode {
    /// Raise `SampleOutOfRange` on any read past a channel's valid length.
    /// This is the default and matches the pre-`PadMode` behavior.
    Raise,
    /// Pad with `f64::NAN`. Physical reads only — combining with a digital read
    /// raises [`EdfError::InvalidArgument`].
    Nan,
    /// Pad with `0.0` (physical) or `0` (digital).
    Zero,
    /// Pad with a caller-supplied value, interpreted in the read domain.
    /// On digital reads the value is truncated to `i32` and range-checked.
    Value(f64),
    /// Replicate the last valid sample for each channel (numpy-style "edge").
    Edge,
}

impl Default for PadMode {
    fn default() -> Self {
        PadMode::Raise
    }
}

/// A set of signal indices grouped for proxy construction, along with the
/// metadata needed to reason about which proxy types it supports.
///
/// Construct via [`EdfFile::signal_groups`](crate::file::EdfFile::signal_groups)
/// for the file's natural grouping by sample rate, or via
/// [`SignalGroup::from_indices`] to classify an arbitrary subset.
#[derive(Debug, Clone)]
pub struct SignalGroup {
    /// File-level signal indices.
    pub indices: Vec<usize>,
    /// Structural classification.
    pub kind: GroupKind,
    /// Common sample rate in Hz. `Some` iff `kind` is `Rectangular`.
    pub sample_rate: Option<f64>,
    /// Common samples-per-record. `Some` iff `kind` is `Rectangular`.
    pub samples_per_record: Option<usize>,
    /// Minimum total sample count across the group's channels.
    pub min_samples: usize,
    /// Maximum total sample count across the group's channels.
    ///
    /// Equal to `min_samples` iff `kind` is `Rectangular`.
    pub max_samples: usize,
    /// `true` iff this group contains every ordinary (non-annotation) signal
    /// in the file. Only set by [`EdfFile::signal_groups`]; hand-built groups
    /// always have this `false`.
    pub covers_all_ordinary: bool,
    /// `true` iff `indices.len() == 1`.
    pub is_singleton: bool,
}

impl SignalGroup {
    /// Classify an arbitrary set of file-level signal indices.
    ///
    /// Annotation channels are not rejected here — callers that want to exclude
    /// them should filter first via
    /// [`EdfFile::ordinary_signal_indices`](crate::file::EdfFile::ordinary_signal_indices).
    /// Returns [`EdfError::SignalOutOfRange`] for any index past the header's
    /// signal count, and [`EdfError::InvalidArgument`] for an empty slice.
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
                return Err(EdfError::SignalOutOfRange {
                    index: i,
                    count: n,
                });
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
            if (rate - first_rate).abs() > 1e-9 {
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
