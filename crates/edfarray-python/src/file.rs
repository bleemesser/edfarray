use chrono::{Datelike, Timelike};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use edfarray_core::file::EdfFile;
use edfarray_core::header::Sex;

use crate::annotations::PyAnnotation;
use crate::array_proxy::PyArrayProxy;
use crate::errors::to_py_err;
use crate::signal::PySignal;

/// An open EDF/EDF+ file.
#[gen_stub_pyclass]
#[pyclass(name = "EdfFile")]
pub struct PyEdfFile {
    inner: Option<EdfFile>,
}

impl PyEdfFile {
    fn get(&self) -> &EdfFile {
        self.inner
            .as_ref()
            .expect("operation on closed EdfFile")
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEdfFile {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = EdfFile::open(path).map_err(to_py_err)?;
        Ok(PyEdfFile { inner: Some(inner) })
    }

    fn __enter__<'py>(slf: PyRef<'py, Self>) -> PyRef<'py, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, _args: Bound<'_, pyo3::types::PyTuple>) {
        self.inner = None;
    }

    /// Explicitly release the underlying memory-mapped file.
    ///
    /// After calling `close()`, any further method calls on this `EdfFile`
    /// will raise. Existing `Signal` and `ArrayProxy` objects keep their own
    /// references and remain usable. Idempotent.
    fn close(&mut self) {
        self.inner = None;
    }

    /// Whether `close()` has been called.
    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_none()
    }

    fn __repr__(&self) -> String {
        let Some(inner) = self.inner.as_ref() else {
            return "EdfFile(closed)".to_string();
        };
        format!(
            "EdfFile(variant={:?}, signals={}, records={}, duration={}s)",
            inner.variant().to_string(),
            inner.num_signals(),
            inner.num_records(),
            inner.duration()
        )
    }

    /// Total number of signals, including annotation channels.
    #[getter]
    fn num_signals(&self) -> usize {
        self.get().num_signals()
    }

    /// Number of data records.
    #[getter]
    fn num_records(&self) -> usize {
        self.get().num_records()
    }

    /// Duration of each data record in seconds.
    #[getter]
    fn record_duration(&self) -> f64 {
        self.get().record_duration()
    }

    /// Total recording duration in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        self.get().duration()
    }

    /// File variant: "EDF", "EDF+C", or "EDF+D".
    #[getter]
    fn variant(&self) -> String {
        self.get().variant().to_string()
    }

    /// Raw 80-byte patient identification field.
    #[getter]
    fn patient_id(&self) -> &str {
        &self.get().header().patient_id
    }

    /// Raw 80-byte recording identification field.
    #[getter]
    fn recording_id(&self) -> &str {
        &self.get().header().recording_id
    }

    /// Returns `datetime.datetime` if the header date/time could be parsed,
    /// or a string like `"04.04.yy 12.57.02"` if it was anonymized.
    #[getter]
    fn start_datetime<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let mdt = &self.get().header().start_datetime;
        match mdt.as_datetime() {
            Some(dt) => {
                let datetime_mod = py.import("datetime")?;
                let datetime_cls = datetime_mod.getattr("datetime")?;
                let result = datetime_cls.call1((
                    dt.year(),
                    dt.month(),
                    dt.day(),
                    dt.hour(),
                    dt.minute(),
                    dt.second(),
                ))?;
                Ok(result.unbind())
            }
            None => {
                let s = format!("{} {}", mdt.raw_date(), mdt.raw_time());
                Ok(s.into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    /// Patient name parsed from the identification field, or None.
    #[getter]
    fn patient_name(&self) -> Option<&str> {
        self.get().patient().name.as_deref()
    }

    /// Hospital patient code, or None.
    #[getter]
    fn patient_code(&self) -> Option<&str> {
        self.get().patient().code.as_deref()
    }

    /// "M" or "F", or None if unknown.
    #[getter]
    fn patient_sex(&self) -> Option<&str> {
        self.get().patient().sex.map(|s| match s {
            Sex::Male => "M",
            Sex::Female => "F",
        })
    }

    /// Returns `datetime.date` if parseable, a raw string if anonymized, or `None` if absent.
    #[getter]
    fn patient_birthdate<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        use edfarray_core::header::MaybeDate;
        match &self.get().patient().birthdate {
            Some(MaybeDate::Parsed(date)) => {
                let datetime_mod = py.import("datetime")?;
                let date_cls = datetime_mod.getattr("date")?;
                let result = date_cls.call1((date.year(), date.month(), date.day()))?;
                Ok(Some(result.unbind()))
            }
            Some(MaybeDate::Raw(s)) => Ok(Some(s.clone().into_pyobject(py)?.into_any().unbind())),
            None => Ok(None),
        }
    }

    /// Additional patient information, or None.
    #[getter]
    fn patient_additional(&self) -> Option<&str> {
        self.get().patient().additional.as_deref()
    }

    /// Hospital administration code, or None.
    #[getter]
    fn admin_code(&self) -> Option<&str> {
        self.get().recording().admin_code.as_deref()
    }

    /// Technician or investigator code, or None.
    #[getter]
    fn technician(&self) -> Option<&str> {
        self.get().recording().technician.as_deref()
    }

    /// Equipment code, or None.
    #[getter]
    fn equipment(&self) -> Option<&str> {
        self.get().recording().equipment.as_deref()
    }

    /// Additional recording information, or None.
    #[getter]
    fn recording_additional(&self) -> Option<&str> {
        self.get().recording().additional.as_deref()
    }

    /// All non-timekeeping annotations, sorted by onset.
    #[getter]
    fn annotations(&self) -> Vec<PyAnnotation> {
        self.get()
            .annotations()
            .iter()
            .map(PyAnnotation::from)
            .collect()
    }

    /// Annotations with onset strictly before `t`.
    /// Uses binary search for efficiency.
    pub fn annotations_before(&self, t: f64) -> Vec<PyAnnotation> {
        self.get()
            .annotations_before(t)
            .iter()
            .map(PyAnnotation::from)
            .collect()
    }

    /// Annotations with onset >= `t`.
    /// Uses binary search for efficiency.
    pub fn annotations_after(&self, t: f64) -> Vec<PyAnnotation> {
        self.get()
            .annotations_after(t)
            .iter()
            .map(PyAnnotation::from)
            .collect()
    }

    /// Annotations with onset in [start, end).
    /// Uses binary search for efficiency.
    pub fn annotations_in_range(&self, start: f64, end: f64) -> Vec<PyAnnotation> {
        self.get()
            .annotations_in_range(start, end)
            .iter()
            .map(PyAnnotation::from)
            .collect()
    }

    /// Filter annotations by text content.
    ///
    /// If `regex` is False, returns annotations whose text contains the query
    /// as a case-insensitive substring.
    ///
    /// If `regex` is True, returns annotations whose text matches the query
    /// as a case-insensitive regex pattern.
    ///
    /// Raises `ValueError` if the regex pattern is invalid.
    #[pyo3(signature = (query, regex=false))]
    fn filter_annotations(&self, query: &str, regex: bool) -> PyResult<Vec<PyAnnotation>> {
        let anns = self
            .get()
            .filter_annotations(query, regex)
            .map_err(to_py_err)?;
        Ok(anns.iter().map(PyAnnotation::from).collect())
    }

    /// Annotations whose text exactly matches `text` (case-sensitive).
    pub fn annotations_by_text(&self, text: &str) -> Vec<PyAnnotation> {
        self.get()
            .annotations_by_text(text)
            .iter()
            .map(PyAnnotation::from)
            .collect()
    }

    /// Return all signals whose label matches `label`.
    ///
    /// If `exact` is `False` (default), performs a case-insensitive substring match.
    /// If `exact` is `True`, performs a case-sensitive exact equality match.
    ///
    /// Searches all signals including annotation signals.
    #[pyo3(signature = (label, exact=false))]
    fn find_all_signals(&self, label: &str, exact: bool) -> PyResult<Vec<PySignal>> {
        let indices = self.get().find_all_signals(label, exact);
        let mut result = Vec::with_capacity(indices.len());
        for idx in indices {
            let proxy = self.get().signal(idx).map_err(to_py_err)?;
            result.push(PySignal::new(proxy));
        }
        Ok(result)
    }

    /// Parse warnings accumulated during file open.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.get().warnings()
    }

    /// Dictionary with basic header fields.
    #[getter]
    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("version", &self.get().header().version)?;
        dict.set_item("patient_id", &self.get().header().patient_id)?;
        dict.set_item("recording_id", &self.get().header().recording_id)?;
        dict.set_item("num_signals", self.get().num_signals())?;
        dict.set_item("num_records", self.get().num_records())?;
        dict.set_item("record_duration", self.get().record_duration())?;
        dict.set_item("duration", self.get().duration())?;
        dict.set_item("variant", self.get().variant().to_string())?;
        Ok(dict)
    }

    /// Get a signal by index or label.
    fn signal(&self, idx_or_label: &Bound<'_, PyAny>) -> PyResult<PySignal> {
        if let Ok(idx) = idx_or_label.extract::<usize>() {
            let proxy = self.get().signal(idx).map_err(to_py_err)?;
            Ok(PySignal::new(proxy))
        } else if let Ok(label) = idx_or_label.extract::<String>() {
            let proxy = self.get().signal_by_label(&label).map_err(to_py_err)?;
            Ok(PySignal::new(proxy))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "signal() argument must be int or str",
            ))
        }
    }

    /// Labels of all signals in the file.
    fn signal_labels(&self) -> Vec<&str> {
        self.get().signal_labels()
    }

    /// Indices of all non-annotation (ordinary) signals.
    fn ordinary_signal_indices(&self) -> Vec<usize> {
        self.get().ordinary_signal_indices()
    }

    /// Read a page of physical data for multiple signals over a time range.
    ///
    /// Returns a list of numpy arrays, one per signal. Signals with different
    /// sample rates will produce arrays of different lengths.
    ///
    /// If `signal_indices` is None, reads all ordinary (non-annotation) signals.
    ///
    /// When `use_time` is false (default), time parameters are converted to flat
    /// sample indices. For EDF+D files with gaps, set `use_time=true` to resolve
    /// the time range using actual record onset times.
    #[pyo3(signature = (start_sec, end_sec, signal_indices=None, use_time=false))]
    fn read_page<'py>(
        &self,
        py: Python<'py>,
        start_sec: f64,
        end_sec: f64,
        signal_indices: Option<Vec<usize>>,
        use_time: bool,
    ) -> PyResult<Vec<Bound<'py, numpy::PyArray1<f64>>>> {
        let indices = signal_indices.unwrap_or_else(|| self.get().ordinary_signal_indices());
        let buffers = self
            .get()
            .read_page(&indices, start_sec, end_sec, use_time)
            .map_err(to_py_err)?;
        let mut arrays = Vec::with_capacity(buffers.len());
        for buf in buffers {
            let array = numpy::PyArray1::from_vec(py, buf);
            arrays.push(array);
        }
        Ok(arrays)
    }

    /// Whether the background annotation scan has completed.
    #[getter]
    fn annotations_ready(&self) -> bool {
        self.get().annotations_ready()
    }

    /// Progress of the background annotation scan: (records_scanned, total_records).
    #[getter]
    fn scan_progress(&self) -> (usize, usize) {
        self.get().scan_progress()
    }

    /// Create a 2D array proxy for numpy-style indexing.
    ///
    /// All selected signals must have the same sample rate.
    /// If `signal_indices` is None, uses all ordinary (non-annotation) signals.
    #[pyo3(signature = (signal_indices=None))]
    fn array_proxy(&self, signal_indices: Option<Vec<usize>>) -> PyResult<PyArrayProxy> {
        let proxy = self
            .get()
            .array_proxy(signal_indices.as_deref())
            .map_err(to_py_err)?;
        Ok(PyArrayProxy::new(proxy))
    }

    /// Group ordinary signal indices by sample rate.
    ///
    /// Returns a dict mapping sample rate (as int Hz) to a list of signal indices.
    fn signal_indices_by_rate<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let map = self.get().signal_indices_by_rate();
        let dict = PyDict::new(py);
        for (rate, indices) in map {
            dict.set_item(rate, indices)?;
        }
        Ok(dict)
    }

    /// Read a page of digital (raw int16) data for multiple signals over a time range.
    ///
    /// If `signal_indices` is None, reads all ordinary (non-annotation) signals.
    ///
    /// When `use_time` is false (default), time parameters are converted to flat
    /// sample indices. For EDF+D files with gaps, set `use_time=true` to resolve
    /// the time range using actual record onset times.
    #[pyo3(signature = (start_sec, end_sec, signal_indices=None, use_time=false))]
    fn read_page_digital<'py>(
        &self,
        py: Python<'py>,
        start_sec: f64,
        end_sec: f64,
        signal_indices: Option<Vec<usize>>,
        use_time: bool,
    ) -> PyResult<Vec<Bound<'py, numpy::PyArray1<i16>>>> {
        let indices = signal_indices.unwrap_or_else(|| self.get().ordinary_signal_indices());
        let buffers = self
            .get()
            .read_page_digital(&indices, start_sec, end_sec, use_time)
            .map_err(to_py_err)?;
        let mut arrays = Vec::with_capacity(buffers.len());
        for buf in buffers {
            let array = numpy::PyArray1::from_vec(py, buf);
            arrays.push(array);
        }
        Ok(arrays)
    }
}

/// Lightweight metadata extracted from an EDF/EDF+ file header without
/// scanning data records or building an annotation index.
///
/// Returns a dict with keys:
/// `variant`, `num_signals`, `num_records`, `record_duration`, `duration`,
/// `patient_id`, `recording_id`, `signal_labels`, `sample_rates`.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn inspect<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let meta = EdfFile::inspect(path).map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("variant", meta.variant.to_string())?;
    dict.set_item("num_signals", meta.num_signals)?;
    dict.set_item("num_records", meta.num_records)?;
    dict.set_item("record_duration", meta.record_duration)?;
    dict.set_item("duration", meta.duration)?;
    dict.set_item("patient_id", meta.patient_id)?;
    dict.set_item("recording_id", meta.recording_id)?;
    dict.set_item("signal_labels", meta.signal_labels)?;
    dict.set_item("sample_rates", meta.sample_rates)?;
    Ok(dict)
}
