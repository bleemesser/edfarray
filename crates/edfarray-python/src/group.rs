use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use edfarray_core::group::{GroupKind, SignalGroup};

/// A set of signal indices for a proxy, and the metadata that shows if a 2D or 3D proxy
/// is possible.
///
/// A proxy is an array-like object that reads samples from the file only when you index it.
/// `EdfFile.signal_groups()` makes these groups. In a single EDF file, every group that this
/// method returns is "rectangular": all its signals share a sample rate and a total sample
/// count.
#[gen_stub_pyclass]
#[pyclass(name = "SignalGroup", module = "edfarray._core", frozen)]
pub struct PySignalGroup {
    inner: SignalGroup,
}

impl PySignalGroup {
    pub fn new(inner: SignalGroup) -> Self {
        PySignalGroup { inner }
    }

    pub fn inner(&self) -> &SignalGroup {
        &self.inner
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PySignalGroup {
    /// The file-level signal indices in this group.
    #[getter]
    fn indices(&self) -> Vec<usize> {
        self.inner.indices().to_vec()
    }

    /// The structural kind: `"rectangular"` (all signals share a sample rate)
    /// or `"open"` (mixed sample rates, 2D proxy only).
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind() {
            GroupKind::Rectangular => "rectangular",
            GroupKind::Open => "open",
        }
    }

    /// The common sample rate in Hz, or `None` if the group is `"open"`.
    #[getter]
    fn sample_rate(&self) -> Option<f64> {
        self.inner.sample_rate()
    }

    /// The common samples-per-record, or `None` if the group is `"open"`.
    #[getter]
    fn samples_per_record(&self) -> Option<usize> {
        self.inner.samples_per_record()
    }

    /// The minimum total sample count of the signals in the group.
    #[getter]
    fn min_samples(&self) -> usize {
        self.inner.min_samples()
    }

    /// The maximum total sample count of the signals in the group.
    /// If the group is `"rectangular"`, this value is equal to `min_samples`. If not, it is
    /// larger, unless the file has no records.
    #[getter]
    fn max_samples(&self) -> usize {
        self.inner.max_samples()
    }

    /// `True` if this group contains every ordinary signal in the file, and `False` if not.
    /// An ordinary signal is a signal that is not an annotation channel.
    #[getter]
    fn covers_all_ordinary(&self) -> bool {
        self.inner.covers_all_ordinary()
    }

    /// `True` if the group contains exactly one signal, and `False` if not.
    #[getter]
    fn is_singleton(&self) -> bool {
        self.inner.is_singleton()
    }

    /// `True` if you can build a 3D proxy from this group, and `False` if not.
    #[getter]
    fn is_rectangular(&self) -> bool {
        self.inner.is_rectangular()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        let rate = match self.inner.sample_rate() {
            Some(r) => format!("{}Hz", r),
            None => "mixed".into(),
        };
        let span = if self.inner.min_samples() == self.inner.max_samples() {
            format!("samples={}", self.inner.min_samples())
        } else {
            format!(
                "samples={}..{}",
                self.inner.min_samples(),
                self.inner.max_samples()
            )
        };
        format!(
            "SignalGroup(kind={:?}, n={}, rate={}, {}, covers_all={})",
            self.kind(),
            self.inner.len(),
            rate,
            span,
            self.inner.covers_all_ordinary(),
        )
    }

    /// Iterate over the file-level signal indices of the group.
    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let indices = slf.inner.indices().to_vec();
        Ok(pyo3::types::PyList::new(py, indices)?
            .as_any()
            .try_iter()?
            .into_any()
            .unbind())
    }

    /// Two groups are equal when they hold the same indices in the same order.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyRef<'_, PySignalGroup>>() {
            Ok(other) => self.inner.indices() == other.inner.indices(),
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.indices().hash(&mut hasher);
        hasher.finish()
    }
}
