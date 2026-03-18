use std::path::PathBuf;

/// All fallible operations in edfarray return `Result<T, EdfError>`.
#[derive(Debug, thiserror::Error)]
pub enum EdfError {
    #[error("failed to open file: {}", path.display())]
    FileOpen {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to create memory map: {}", path.display())]
    MmapFailed {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("file too small: expected at least {expected} bytes, got {actual}")]
    FileTooSmall { expected: usize, actual: usize },

    #[error("invalid header field `{field}`: {reason}")]
    InvalidHeaderField { field: &'static str, reason: String },

    #[error("header declares {header_bytes} header bytes, but file is only {file_size} bytes")]
    HeaderSizeMismatch {
        header_bytes: usize,
        file_size: usize,
    },

    #[error("signal count is zero")]
    NoSignals,

    #[error("signal {index} has invalid `{field}`: {reason}")]
    InvalidSignalField {
        index: usize,
        field: &'static str,
        reason: String,
    },

    #[error("signal {index} has digital_min ({min}) >= digital_max ({max})")]
    InvalidDigitalRange { index: usize, min: i16, max: i16 },

    #[error("signal {index} has physical_min ({min}) == physical_max ({max})")]
    InvalidPhysicalRange { index: usize, min: f64, max: f64 },

    #[error("invalid TAL at record {record}, byte offset {offset}: {reason}")]
    InvalidTal {
        record: usize,
        offset: usize,
        reason: String,
    },

    #[error("annotation onset is not valid UTF-8 at record {record}")]
    InvalidAnnotationEncoding { record: usize },

    #[error("missing time-keeping annotation in record {record}")]
    MissingTimekeepingAnnotation { record: usize },

    #[error("record index {index} out of range (file has {count} records)")]
    RecordOutOfRange { index: usize, count: usize },

    #[error("signal index {index} out of range (file has {count} signals)")]
    SignalOutOfRange { index: usize, count: usize },

    #[error("sample index {index} out of range (signal has {count} samples)")]
    SampleOutOfRange { index: usize, count: usize },

    #[error("no signal with label `{label}`")]
    SignalNotFound { label: String },
}

pub type Result<T> = std::result::Result<T, EdfError>;
