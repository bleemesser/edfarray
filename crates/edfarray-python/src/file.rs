use chrono::{Datelike, Timelike};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use edfarray_core::file::EdfFile;
use edfarray_core::header::Sex;

use crate::annotations::PyAnnotation;
use crate::errors::to_py_err;
use crate::signal::PySignal;

/// An open EDF/EDF+ file.
#[gen_stub_pyclass]
#[pyclass(name = "EdfFile")]
pub struct PyEdfFile {
    inner: EdfFile,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEdfFile {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = EdfFile::open(path).map_err(to_py_err)?;
        Ok(PyEdfFile { inner })
    }

    fn __enter__<'py>(slf: PyRef<'py, Self>) -> PyRef<'py, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: Bound<'_, pyo3::types::PyTuple>) {}

    fn __repr__(&self) -> String {
        format!(
            "EdfFile(variant={:?}, signals={}, records={}, duration={}s)",
            self.inner.variant().to_string(),
            self.inner.num_signals(),
            self.inner.num_records(),
            self.inner.duration()
        )
    }

    /// Total number of signals, including annotation channels.
    #[getter]
    fn num_signals(&self) -> usize {
        self.inner.num_signals()
    }

    /// Number of data records.
    #[getter]
    fn num_records(&self) -> usize {
        self.inner.num_records()
    }

    /// Duration of each data record in seconds.
    #[getter]
    fn record_duration(&self) -> f64 {
        self.inner.record_duration()
    }

    /// Total recording duration in seconds.
    #[getter]
    fn duration(&self) -> f64 {
        self.inner.duration()
    }

    /// File variant: "EDF", "EDF+C", or "EDF+D".
    #[getter]
    fn variant(&self) -> String {
        self.inner.variant().to_string()
    }

    /// Raw 80-byte patient identification field.
    #[getter]
    fn patient_id(&self) -> &str {
        &self.inner.header().patient_id
    }

    /// Raw 80-byte recording identification field.
    #[getter]
    fn recording_id(&self) -> &str {
        &self.inner.header().recording_id
    }

    /// Returns `datetime.datetime` if the header date/time could be parsed,
    /// or a string like `"04.04.yy 12.57.02"` if it was anonymized.
    #[getter]
    fn start_datetime<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let mdt = &self.inner.header().start_datetime;
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
        self.inner.patient().name.as_deref()
    }

    /// Hospital patient code, or None.
    #[getter]
    fn patient_code(&self) -> Option<&str> {
        self.inner.patient().code.as_deref()
    }

    /// "M" or "F", or None if unknown.
    #[getter]
    fn patient_sex(&self) -> Option<&str> {
        self.inner.patient().sex.map(|s| match s {
            Sex::Male => "M",
            Sex::Female => "F",
        })
    }

    /// Returns `datetime.date` if parseable, a raw string if anonymized, or `None` if absent.
    #[getter]
    fn patient_birthdate<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        use edfarray_core::header::MaybeDate;
        match &self.inner.patient().birthdate {
            Some(MaybeDate::Parsed(date)) => {
                let datetime_mod = py.import("datetime")?;
                let date_cls = datetime_mod.getattr("date")?;
                let result = date_cls.call1((
                    date.year(),
                    date.month(),
                    date.day(),
                ))?;
                Ok(Some(result.unbind()))
            }
            Some(MaybeDate::Raw(s)) => {
                Ok(Some(s.clone().into_pyobject(py)?.into_any().unbind()))
            }
            None => Ok(None),
        }
    }

    /// Additional patient information, or None.
    #[getter]
    fn patient_additional(&self) -> Option<&str> {
        self.inner.patient().additional.as_deref()
    }

    /// Hospital administration code, or None.
    #[getter]
    fn admin_code(&self) -> Option<&str> {
        self.inner.recording().admin_code.as_deref()
    }

    /// Technician or investigator code, or None.
    #[getter]
    fn technician(&self) -> Option<&str> {
        self.inner.recording().technician.as_deref()
    }

    /// Equipment code, or None.
    #[getter]
    fn equipment(&self) -> Option<&str> {
        self.inner.recording().equipment.as_deref()
    }

    /// Additional recording information, or None.
    #[getter]
    fn recording_additional(&self) -> Option<&str> {
        self.inner.recording().additional.as_deref()
    }

    /// All non-timekeeping annotations, sorted by onset.
    #[getter]
    fn annotations(&self) -> Vec<PyAnnotation> {
        self.inner.annotations().iter().map(PyAnnotation::from).collect()
    }

    /// Parse warnings accumulated during file open.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner.warnings()
    }

    /// Dictionary with basic header fields.
    #[getter]
    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("version", &self.inner.header().version)?;
        dict.set_item("patient_id", &self.inner.header().patient_id)?;
        dict.set_item("recording_id", &self.inner.header().recording_id)?;
        dict.set_item("num_signals", self.inner.num_signals())?;
        dict.set_item("num_records", self.inner.num_records())?;
        dict.set_item("record_duration", self.inner.record_duration())?;
        dict.set_item("duration", self.inner.duration())?;
        dict.set_item("variant", self.inner.variant().to_string())?;
        Ok(dict)
    }

    /// Get a signal by index or label.
    fn signal(&self, idx_or_label: &Bound<'_, PyAny>) -> PyResult<PySignal> {
        if let Ok(idx) = idx_or_label.extract::<usize>() {
            let proxy = self.inner.signal(idx).map_err(to_py_err)?;
            Ok(PySignal::new(proxy))
        } else if let Ok(label) = idx_or_label.extract::<String>() {
            let proxy = self.inner.signal_by_label(&label).map_err(to_py_err)?;
            Ok(PySignal::new(proxy))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "signal() argument must be int or str",
            ))
        }
    }

    /// Indices of all non-annotation (ordinary) signals.
    fn ordinary_signal_indices(&self) -> Vec<usize> {
        self.inner.ordinary_signal_indices()
    }

    /// Read a page of physical data for multiple signals over a time range.
    ///
    /// Returns a list of numpy arrays, one per signal. Signals with different
    /// sample rates will produce arrays of different lengths.
    ///
    /// If `signal_indices` is None, reads all ordinary (non-annotation) signals.
    #[pyo3(signature = (start_sec, end_sec, signal_indices=None))]
    fn read_page<'py>(
        &self,
        py: Python<'py>,
        start_sec: f64,
        end_sec: f64,
        signal_indices: Option<Vec<usize>>,
    ) -> PyResult<Vec<Bound<'py, numpy::PyArray1<f64>>>> {
        let indices = signal_indices.unwrap_or_else(|| self.inner.ordinary_signal_indices());
        let buffers = self
            .inner
            .read_page(&indices, start_sec, end_sec)
            .map_err(to_py_err)?;
        let mut arrays = Vec::with_capacity(buffers.len());
        for buf in buffers {
            let array = numpy::PyArray1::from_vec(py, buf);
            arrays.push(array);
        }
        Ok(arrays)
    }

    /// Read a page of digital (raw int16) data for multiple signals over a time range.
    ///
    /// If `signal_indices` is None, reads all ordinary (non-annotation) signals.
    #[pyo3(signature = (start_sec, end_sec, signal_indices=None))]
    fn read_page_digital<'py>(
        &self,
        py: Python<'py>,
        start_sec: f64,
        end_sec: f64,
        signal_indices: Option<Vec<usize>>,
    ) -> PyResult<Vec<Bound<'py, numpy::PyArray1<i16>>>> {
        let indices = signal_indices.unwrap_or_else(|| self.inner.ordinary_signal_indices());
        let buffers = self
            .inner
            .read_page_digital(&indices, start_sec, end_sec)
            .map_err(to_py_err)?;
        let mut arrays = Vec::with_capacity(buffers.len());
        for buf in buffers {
            let array = numpy::PyArray1::from_vec(py, buf);
            arrays.push(array);
        }
        Ok(arrays)
    }
}
