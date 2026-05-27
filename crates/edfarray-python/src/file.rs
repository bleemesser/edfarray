use chrono::{Datelike, Timelike};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use edfarray_core::file::EdfFile;
use edfarray_core::header::Sex;

use crate::annotations::PyAnnotation;
use crate::errors::to_py_err;
use crate::group::PySignalGroup;
use crate::proxy_2d::{PyProxy2D, parse_pad_mode};
use crate::proxy_3d::PyProxy3D;
use crate::signal::PySignal;

/// An open EDF/EDF+ file.
#[gen_stub_pyclass]
#[pyclass(name = "EdfFile")]
pub struct PyEdfFile {
    inner: Option<EdfFile>,
}

impl PyEdfFile {
    fn get(&self) -> &EdfFile {
        self.inner.as_ref().expect("operation on closed EdfFile")
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

    /// Release the underlying memory-mapped file. Idempotent.
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

    /// Recording start time as `datetime.datetime`, or raw string if anonymized.
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

    /// Build a 2D proxy from a `SignalGroup`.
    ///
    /// `pad_mode` controls reads past a channel's valid length when the group
    /// is `"open"` (mixed sample rates). Accepts the string `"raise"` (default),
    /// `"nan"`, `"zero"`, `"edge"`, or a numeric scalar (interpreted as
    /// `Value(x)`). `None` is treated as `"raise"`.
    #[pyo3(signature = (group, pad_mode=None))]
    fn proxy_2d(
        &self,
        group: &PySignalGroup,
        pad_mode: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyProxy2D> {
        let mode = parse_pad_mode(pad_mode.as_ref())?;
        let proxy = self
            .get()
            .proxy_2d(group.inner().clone(), mode)
            .map_err(to_py_err)?;
        Ok(PyProxy2D::new(proxy))
    }

    /// Build a 3D proxy from a rectangular `SignalGroup`.
    ///
    /// Requires `group.kind == "rectangular"` (all channels share a sample
    /// rate). Use `signal_groups()` to discover eligible groups, or
    /// `signal_group(...)` to construct one from specific indices.
    fn proxy_3d(&self, group: &PySignalGroup) -> PyResult<PyProxy3D> {
        let proxy = self
            .get()
            .proxy_3d(group.inner().clone())
            .map_err(to_py_err)?;
        Ok(PyProxy3D::new(proxy))
    }

    /// Classify an arbitrary list of file-level signal indices into a
    /// `SignalGroup`. Use this when you want a group that's a subset of (or
    /// crosses) the file's natural rate-based groupings.
    fn signal_group(&self, indices: Vec<usize>) -> PyResult<PySignalGroup> {
        let g = edfarray_core::group::SignalGroup::from_indices(self.get().header(), &indices)
            .map_err(to_py_err)?;
        Ok(PySignalGroup::new(g))
    }

    /// Partition all ordinary signals into groups by sample rate.
    ///
    /// Returns a list of `SignalGroup` objects, each carrying its sample rate,
    /// structural kind, sample-count range, and whether it covers every
    /// ordinary signal in the file. Sub-Hz precision is preserved.
    fn signal_groups(&self) -> Vec<PySignalGroup> {
        self.get()
            .signal_groups()
            .into_iter()
            .map(PySignalGroup::new)
            .collect()
    }

    /// Write this file to `path`, optionally transcoding to a different variant.
    ///
    /// Annotations and ordinary signals are copied; the destination's annotation
    /// channel is rebuilt from parsed annotations rather than copied verbatim.
    /// `variant` may be one of "EDF", "EDF+C", "EDF+D", "BDF", "BDF+C", "BDF+D";
    /// if omitted, uses the source variant.
    #[pyo3(signature = (path, variant=None))]
    fn write_to(&self, path: &str, variant: Option<&str>) -> PyResult<()> {
        use edfarray_core::header::EdfVariant;
        let target = match variant {
            None => None,
            Some("EDF") => Some(EdfVariant::Edf),
            Some("EDF+C") => Some(EdfVariant::EdfPlusC),
            Some("EDF+D") => Some(EdfVariant::EdfPlusD),
            Some("BDF") => Some(EdfVariant::Bdf),
            Some("BDF+C") => Some(EdfVariant::BdfPlusC),
            Some("BDF+D") => Some(EdfVariant::BdfPlusD),
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown variant {:?}",
                    other
                )));
            }
        };
        self.get().write_to(path, target).map_err(to_py_err)
    }

    /// Read a page of digital (raw int32) data for multiple signals over a time range.
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
    ) -> PyResult<Vec<Bound<'py, numpy::PyArray1<i32>>>> {
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
