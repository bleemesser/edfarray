use std::sync::{Arc, Mutex as StdMutex};

use chrono::{Datelike, Timelike};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use edfarray_core::annotation::Annotation as CoreAnnotation;
use edfarray_core::file::EdfFile;
use edfarray_core::header::Sex;
use edfarray_core::writer::{EdfWriter, write_edf};

use crate::annotations::PyAnnotation;
use crate::errors::to_py_err;
use crate::group::PySignalGroup;
use crate::writer::{anns_to_core, build_spec, parse_variant};

use numpy::{PyArray1, PyReadonlyArray1};

#[pyclass(name = "WriterSignal", module = "edfarray._core.aio", from_py_object)]
#[derive(Clone)]
pub struct PyAsyncWriterSignal {
    inner: edfarray_core::writer::WriterSignal,
}

#[pymethods]
impl PyAsyncWriterSignal {
    #[new]
    #[pyo3(signature = (
        label,
        physical_dimension,
        physical_min,
        physical_max,
        digital_min,
        digital_max,
        samples_per_record,
        transducer = String::new(),
        prefiltering = String::new(),
        reserved = String::new(),
    ))]
    fn new(
        label: String,
        physical_dimension: String,
        physical_min: f64,
        physical_max: f64,
        digital_min: i32,
        digital_max: i32,
        samples_per_record: usize,
        transducer: String,
        prefiltering: String,
        reserved: String,
    ) -> Self {
        Self {
            inner: edfarray_core::writer::WriterSignal {
                label,
                transducer,
                physical_dimension,
                physical_min,
                physical_max,
                digital_min,
                digital_max,
                prefiltering,
                samples_per_record,
                reserved,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "<edfarray.aio.WriterSignal label={:?} samples_per_record={}>",
            self.inner.label, self.inner.samples_per_record
        )
    }
}

#[pyclass(name = "EdfFile", module = "edfarray._core.aio")]
pub struct PyAsyncEdfFile {
    inner: Option<Arc<EdfFile>>,
}

impl PyAsyncEdfFile {
    fn get(&self) -> PyResult<&Arc<EdfFile>> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("operation on closed EdfFile"))
    }
}

#[pymethods]
impl PyAsyncEdfFile {
    fn __aenter__<'py>(slf: Py<Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(slf) })
    }

    #[pyo3(signature = (*_args))]
    fn __aexit__<'py>(
        &mut self,
        py: Python<'py>,
        _args: Bound<'_, pyo3::types::PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner = None;
        pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(()) })
    }

    fn close(&mut self) {
        self.inner = None;
    }

    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_none()
    }

    fn __repr__(&self) -> String {
        let Some(inner) = self.inner.as_ref() else {
            return "<edfarray.aio.EdfFile closed>".to_string();
        };
        format!(
            "<edfarray.aio.EdfFile variant={:?} signals={} records={} duration={}s>",
            inner.variant().to_string(),
            inner.num_signals(),
            inner.num_records(),
            inner.duration()
        )
    }

    #[getter]
    fn num_signals(&self) -> usize {
        self.get().unwrap().num_signals()
    }

    #[getter]
    fn num_records(&self) -> usize {
        self.get().unwrap().num_records()
    }

    #[getter]
    fn record_duration(&self) -> f64 {
        self.get().unwrap().record_duration()
    }

    #[getter]
    fn duration(&self) -> f64 {
        self.get().unwrap().duration()
    }

    #[getter]
    fn variant(&self) -> String {
        self.get().unwrap().variant().to_string()
    }

    #[getter]
    fn patient_id(&self) -> String {
        self.get().unwrap().header().patient_id.clone()
    }

    #[getter]
    fn recording_id(&self) -> String {
        self.get().unwrap().header().recording_id.clone()
    }

    #[getter]
    fn start_datetime<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mdt = &self.get().unwrap().header().start_datetime;
        match mdt.as_datetime() {
            Some(dt) => {
                let datetime_mod = py.import("datetime")?;
                let datetime_cls = datetime_mod.getattr("datetime")?;
                datetime_cls.call1((
                    dt.year(),
                    dt.month(),
                    dt.day(),
                    dt.hour(),
                    dt.minute(),
                    dt.second(),
                ))
            }
            None => {
                let s = format!("{} {}", mdt.raw_date(), mdt.raw_time());
                Ok(s.into_pyobject(py).unwrap().into_any())
            }
        }
    }

    #[getter]
    fn patient_name(&self) -> Option<String> {
        self.get().unwrap().patient().name.clone()
    }

    #[getter]
    fn patient_code(&self) -> Option<String> {
        self.get().unwrap().patient().code.clone()
    }

    #[getter]
    fn patient_sex(&self) -> Option<String> {
        self.get().unwrap().patient().sex.map(|s| match s {
            Sex::Male => "M".to_string(),
            Sex::Female => "F".to_string(),
        })
    }

    #[getter]
    fn patient_birthdate<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        use edfarray_core::header::MaybeDate;
        match &self.get().unwrap().patient().birthdate {
            Some(MaybeDate::Parsed(date)) => {
                let datetime_mod = py.import("datetime")?;
                let date_cls = datetime_mod.getattr("date")?;
                let result = date_cls.call1((date.year(), date.month(), date.day()))?;
                Ok(Some(result))
            }
            Some(MaybeDate::Raw(s)) => {
                let result = s.clone().into_pyobject(py).unwrap();
                Ok(Some(result.into_any()))
            }
            None => Ok(None),
        }
    }

    #[getter]
    fn patient_additional(&self) -> Option<String> {
        self.get().unwrap().patient().additional.clone()
    }

    #[getter]
    fn admin_code(&self) -> Option<String> {
        self.get().unwrap().recording().admin_code.clone()
    }

    #[getter]
    fn technician(&self) -> Option<String> {
        self.get().unwrap().recording().technician.clone()
    }

    #[getter]
    fn equipment(&self) -> Option<String> {
        self.get().unwrap().recording().equipment.clone()
    }

    #[getter]
    fn recording_additional(&self) -> Option<String> {
        self.get().unwrap().recording().additional.clone()
    }

    #[getter]
    fn annotations(&self) -> Vec<PyAnnotation> {
        self.get()
            .unwrap()
            .annotations()
            .iter()
            .map(PyAnnotation::from)
            .collect()
    }

    fn annotations_before(&self, t: f64) -> Vec<PyAnnotation> {
        self.get()
            .unwrap()
            .annotations_before(t)
            .iter()
            .map(PyAnnotation::from)
            .collect()
    }

    fn annotations_after(&self, t: f64) -> Vec<PyAnnotation> {
        self.get()
            .unwrap()
            .annotations_after(t)
            .iter()
            .map(PyAnnotation::from)
            .collect()
    }

    fn annotations_in_range(&self, start: f64, end: f64) -> Vec<PyAnnotation> {
        self.get()
            .unwrap()
            .annotations_in_range(start, end)
            .iter()
            .map(PyAnnotation::from)
            .collect()
    }

    #[pyo3(signature = (query, regex=false))]
    fn filter_annotations(&self, query: &str, regex: bool) -> PyResult<Vec<PyAnnotation>> {
        let anns = self
            .get()
            .unwrap()
            .filter_annotations(query, regex)
            .map_err(to_py_err)?;
        Ok(anns.iter().map(PyAnnotation::from).collect())
    }

    fn annotations_by_text(&self, text: &str) -> Vec<PyAnnotation> {
        self.get()
            .unwrap()
            .annotations_by_text(text)
            .iter()
            .map(PyAnnotation::from)
            .collect()
    }

    #[pyo3(signature = (label, exact=false))]
    fn find_all_signals(&self, label: &str, exact: bool) -> Vec<usize> {
        self.get().unwrap().find_all_signals(label, exact)
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.get().unwrap().warnings()
    }

    #[getter]
    fn annotations_ready(&self) -> bool {
        self.get().unwrap().annotations_ready()
    }

    #[getter]
    fn scan_progress(&self) -> (usize, usize) {
        self.get().unwrap().scan_progress()
    }

    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inner = self.get().unwrap();
        let dict = PyDict::new(py);
        dict.set_item("version", &inner.header().version)?;
        dict.set_item("patient_id", &inner.header().patient_id)?;
        dict.set_item("recording_id", &inner.header().recording_id)?;
        dict.set_item("num_signals", inner.num_signals())?;
        dict.set_item("num_records", inner.num_records())?;
        dict.set_item("record_duration", inner.record_duration())?;
        dict.set_item("duration", inner.duration())?;
        dict.set_item("variant", inner.variant().to_string())?;
        Ok(dict)
    }

    fn signal_labels(&self) -> Vec<String> {
        self.get()
            .unwrap()
            .signal_labels()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    fn ordinary_signal_indices(&self) -> Vec<usize> {
        self.get().unwrap().ordinary_signal_indices()
    }

    fn signal_group(&self, indices: Vec<usize>) -> PyResult<PySignalGroup> {
        let inner = self.get()?;
        let g = edfarray_core::group::SignalGroup::from_indices(inner.header(), &indices)
            .map_err(to_py_err)?;
        Ok(PySignalGroup::new(g))
    }

    fn signal_groups(&self) -> PyResult<Vec<PySignalGroup>> {
        Ok(self
            .get()?
            .signal_groups()
            .into_iter()
            .map(PySignalGroup::new)
            .collect())
    }

    fn wait_for_annotations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get()?.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || inner.wait_for_annotations())
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?;
            Ok(())
        })
    }

    /// Read physical data for multiple signals over a time range. Returns list of numpy arrays.
    #[pyo3(signature = (start_sec, end_sec, signal_indices=None, use_time=false))]
    fn read_page<'py>(
        &self,
        py: Python<'py>,
        start_sec: f64,
        end_sec: f64,
        signal_indices: Option<Vec<usize>>,
        use_time: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get()?.clone();
        let indices = signal_indices.unwrap_or_else(|| inner.ordinary_signal_indices());
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let buffers = tokio::task::spawn_blocking(move || {
                inner.read_page(&indices, start_sec, end_sec, use_time)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            Python::attach(|py| -> PyResult<Py<pyo3::types::PyList>> {
                let list = pyo3::types::PyList::empty(py);
                for buf in buffers {
                    list.append(PyArray1::from_vec(py, buf))?;
                }
                Ok(list.unbind())
            })
        })
    }

    /// Read digital (int32) data for multiple signals over a time range.
    #[pyo3(signature = (start_sec, end_sec, signal_indices=None, use_time=false))]
    fn read_page_digital<'py>(
        &self,
        py: Python<'py>,
        start_sec: f64,
        end_sec: f64,
        signal_indices: Option<Vec<usize>>,
        use_time: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get()?.clone();
        let indices = signal_indices.unwrap_or_else(|| inner.ordinary_signal_indices());
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let buffers = tokio::task::spawn_blocking(move || {
                inner.read_page_digital(&indices, start_sec, end_sec, use_time)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            Python::attach(|py| -> PyResult<Py<pyo3::types::PyList>> {
                let list = pyo3::types::PyList::empty(py);
                for buf in buffers {
                    list.append(PyArray1::from_vec(py, buf))?;
                }
                Ok(list.unbind())
            })
        })
    }

    /// Write to `path`, optionally transcoding to a different variant.
    ///
    /// Transcoding caveats:
    /// - Records are streamed contiguously, so transcoding from EDF+D to any
    ///   non-EDF+D variant discards the discontinuity: the original per-record
    ///   onsets/gaps are replaced by uniform `record_idx * record_duration` timing.
    /// - Because the annotation channel is rebuilt from parsed annotations,
    ///   transcoding to a plain (non-"+") EDF/BDF variant drops all annotations,
    ///   since plain variants have no annotation channel.
    /// - Downconverting sample size (e.g. BDF 24-bit to EDF 16-bit) clamps the
    ///   digital range and re-encodes from physical values, losing precision.
    #[pyo3(signature = (path, variant=None))]
    fn write_to<'py>(
        &self,
        py: Python<'py>,
        path: String,
        variant: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.get()?.clone();
        let target = match variant.as_deref() {
            None => None,
            Some(s) => Some(parse_variant(s)?),
        };
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || inner.write_to(&path, target))
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
                .map_err(to_py_err)?;
            Ok(())
        })
    }

    /// Get a signal by index or label.
    ///
    /// `cache_capacity` enables an LRU cache of decoded physical records for
    /// this signal. The unit is a count of EDF data records (not samples or
    /// bytes); one cached record holds `samples_per_record` float64 values, so
    /// the cache costs roughly `cache_capacity * samples_per_record * 8` bytes.
    /// 0 (the default) disables it.
    ///
    /// Leave it at 0 for one-pass or strictly forward reads -- the OS page cache
    /// already serves the raw bytes, so a cache only pays off when you re-decode
    /// the *same* records (overlapping windows, back-and-forth seeks, repeated
    /// slices). A good starting capacity is a few records more than your largest
    /// repeated window spans, i.e. `ceil(window_samples / samples_per_record) + 2`.
    /// The cache only accelerates physical reads -- `read_digital()` always
    /// re-decodes from the memory map. Caching is per-`Signal`: re-fetching from
    /// `signal()` starts fresh.
    #[pyo3(signature = (idx_or_label, cache_capacity=0))]
    fn signal(
        &self,
        idx_or_label: &Bound<'_, PyAny>,
        cache_capacity: usize,
    ) -> PyResult<PyAsyncSignal> {
        let inner = self.get()?;
        let proxy = if let Ok(idx) = idx_or_label.extract::<usize>() {
            inner.signal(idx).map_err(to_py_err)?
        } else if let Ok(label) = idx_or_label.extract::<String>() {
            inner.signal_by_label(&label).map_err(to_py_err)?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "signal() argument must be int or str",
            ));
        };
        let proxy = if cache_capacity > 0 {
            proxy.with_cache(cache_capacity)
        } else {
            proxy
        };
        Ok(PyAsyncSignal::new(proxy))
    }
}

#[pyclass(name = "Signal", module = "edfarray._core.aio")]
pub struct PyAsyncSignal {
    inner: Arc<edfarray_core::proxy::SignalProxy>,
}

impl PyAsyncSignal {
    pub fn new(proxy: edfarray_core::proxy::SignalProxy) -> Self {
        PyAsyncSignal {
            inner: Arc::new(proxy),
        }
    }
}

#[pymethods]
impl PyAsyncSignal {
    #[getter]
    fn label(&self) -> String {
        self.inner.header().label.clone()
    }

    #[getter]
    fn transducer(&self) -> String {
        self.inner.header().transducer.clone()
    }

    #[getter]
    fn physical_dimension(&self) -> String {
        self.inner.header().physical_dimension.clone()
    }

    #[getter]
    fn prefiltering(&self) -> String {
        self.inner.header().prefiltering.clone()
    }

    #[getter]
    fn sample_rate(&self) -> f64 {
        self.inner.sample_rate()
    }

    #[getter]
    fn samples_per_record(&self) -> usize {
        self.inner.header().num_samples
    }

    #[getter]
    fn physical_min(&self) -> f64 {
        self.inner.header().physical_min
    }

    #[getter]
    fn physical_max(&self) -> f64 {
        self.inner.header().physical_max
    }

    #[getter]
    fn digital_min(&self) -> i32 {
        self.inner.header().digital_min
    }

    #[getter]
    fn digital_max(&self) -> i32 {
        self.inner.header().digital_max
    }

    #[getter]
    fn num_samples(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "<edfarray.aio.Signal label={:?} samples={} rate={}Hz>",
            self.inner.header().label,
            self.inner.len(),
            self.inner.sample_rate()
        )
    }

    fn read_physical<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        stop: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let buf: Vec<f64> = tokio::task::spawn_blocking(
                move || -> Result<Vec<f64>, edfarray_core::error::EdfError> {
                    let count = stop.saturating_sub(start);
                    let mut buf = vec![0.0; count];
                    if count > 0 {
                        proxy.read_physical(start, stop, &mut buf)?;
                    }
                    Ok(buf)
                },
            )
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            Python::attach(|py| -> PyResult<Py<PyArray1<f64>>> {
                Ok(PyArray1::from_vec(py, buf).unbind())
            })
        })
    }

    fn read_digital<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        stop: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let buf: Vec<i32> = tokio::task::spawn_blocking(
                move || -> Result<Vec<i32>, edfarray_core::error::EdfError> {
                    let count = stop.saturating_sub(start);
                    let mut buf = vec![0i32; count];
                    if count > 0 {
                        proxy.read_digital(start, stop, &mut buf)?;
                    }
                    Ok(buf)
                },
            )
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            Python::attach(|py| -> PyResult<Py<PyArray1<i32>>> {
                Ok(PyArray1::from_vec(py, buf).unbind())
            })
        })
    }

    fn read_at<'py>(
        &self,
        py: Python<'py>,
        start_sec: f64,
        end_sec: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let buf = tokio::task::spawn_blocking(move || proxy.read_at(start_sec, end_sec))
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
                .map_err(to_py_err)?;
            Python::attach(|py| -> PyResult<Py<PyArray1<f64>>> {
                Ok(PyArray1::from_vec(py, buf).unbind())
            })
        })
    }

    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.inner.clone();
        let len = self.inner.len();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let buf = tokio::task::spawn_blocking(
                move || -> Result<Vec<f64>, edfarray_core::error::EdfError> {
                    let mut buf = vec![0.0; len];
                    if len > 0 {
                        proxy.read_physical(0, len, &mut buf)?;
                    }
                    Ok(buf)
                },
            )
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            Python::attach(|py| -> PyResult<Py<PyArray1<f64>>> {
                Ok(PyArray1::from_vec(py, buf).unbind())
            })
        })
    }

    fn to_digital<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.inner.clone();
        let len = self.inner.len();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let buf = tokio::task::spawn_blocking(
                move || -> Result<Vec<i32>, edfarray_core::error::EdfError> {
                    let mut buf = vec![0i32; len];
                    if len > 0 {
                        proxy.read_digital(0, len, &mut buf)?;
                    }
                    Ok(buf)
                },
            )
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            Python::attach(|py| -> PyResult<Py<PyArray1<i32>>> {
                Ok(PyArray1::from_vec(py, buf).unbind())
            })
        })
    }

    fn times<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.inner.clone();
        let len = self.inner.len();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let buf = tokio::task::spawn_blocking(
                move || -> Result<Vec<f64>, edfarray_core::error::EdfError> {
                    let mut buf = vec![0.0; len];
                    if len > 0 {
                        proxy.read_times(0, len, &mut buf)?;
                    }
                    Ok(buf)
                },
            )
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            Python::attach(|py| -> PyResult<Py<PyArray1<f64>>> {
                Ok(PyArray1::from_vec(py, buf).unbind())
            })
        })
    }
}

#[pyclass(name = "EdfWriter", module = "edfarray._core.aio")]
pub struct PyAsyncEdfWriter {
    inner: Arc<StdMutex<Option<EdfWriter>>>,
}

/// Lock the writer mutex, mapping a poisoned mutex to a Python exception.
fn lock_writer(
    m: &StdMutex<Option<EdfWriter>>,
) -> PyResult<std::sync::MutexGuard<'_, Option<EdfWriter>>> {
    m.lock()
        .map_err(|_| PyRuntimeError::new_err("EdfWriter is poisoned by a previous failed write"))
}

#[pymethods]
impl PyAsyncEdfWriter {
    #[classmethod]
    #[pyo3(signature = (
        path,
        *,
        variant,
        record_duration,
        signals,
        start_datetime = None,
        patient_id = None,
        recording_id = None,
        annotation_bytes_per_record = None,
    ))]
    fn create<'py>(
        _cls: &Bound<'_, pyo3::types::PyType>,
        py: Python<'py>,
        path: String,
        variant: String,
        record_duration: f64,
        signals: Vec<PyAsyncWriterSignal>,
        start_datetime: Option<&Bound<'_, PyAny>>,
        patient_id: Option<String>,
        recording_id: Option<String>,
        annotation_bytes_per_record: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let spec = build_spec(
            &variant,
            record_duration,
            signals.into_iter().map(|s| s.inner).collect(),
            start_datetime,
            patient_id,
            recording_id,
            annotation_bytes_per_record,
        )?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let writer = tokio::task::spawn_blocking(move || EdfWriter::create(&path, spec))
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
                .map_err(to_py_err)?;
            Python::attach(|py| {
                Py::new(
                    py,
                    PyAsyncEdfWriter {
                        inner: Arc::new(StdMutex::new(Some(writer))),
                    },
                )
            })
        })
    }

    fn add_annotation(&self, annotation: PyAnnotation) -> PyResult<()> {
        let mut guard = lock_writer(&self.inner)?;
        let w = guard
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("EdfWriter has been finished"))?;
        w.add_annotation(CoreAnnotation {
            onset: annotation.onset,
            duration: annotation.duration,
            text: annotation.text,
        });
        Ok(())
    }

    #[pyo3(signature = (physical, annotations=None))]
    fn write_record<'py>(
        &self,
        py: Python<'py>,
        physical: Vec<PyReadonlyArray1<f64>>,
        annotations: Option<Vec<PyAnnotation>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let owned: Vec<Vec<f64>> = physical
            .iter()
            .map(|arr| arr.as_slice().map(|s| s.to_vec()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                pyo3::exceptions::PyTypeError::new_err(format!(
                    "physical arrays must be contiguous: {e}"
                ))
            })?;
        let anns_owned: Vec<CoreAnnotation> =
            annotations.map(|a| anns_to_core(&a)).unwrap_or_default();
        let writer = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || -> PyResult<()> {
                let mut guard = lock_writer(&writer)?;
                let w = guard
                    .as_mut()
                    .ok_or_else(|| PyValueError::new_err("EdfWriter has been finished"))?;
                let slices: Vec<&[f64]> = owned.iter().map(|v| v.as_slice()).collect();
                w.write_record_with_annotations(&slices, &anns_owned)
                    .map_err(to_py_err)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))??;
            Ok(())
        })
    }

    fn finish<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let writer = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || -> PyResult<()> {
                let mut guard = lock_writer(&writer)?;
                let w = guard
                    .take()
                    .ok_or_else(|| PyValueError::new_err("EdfWriter has been finished"))?;
                w.finish().map_err(to_py_err)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))??;
            Ok(())
        })
    }

    fn __repr__(&self) -> String {
        let open = match self.inner.lock() {
            Ok(g) => g.is_some(),
            Err(p) => p.into_inner().is_some(),
        };
        if open {
            "<edfarray.aio.EdfWriter open>".to_string()
        } else {
            "<edfarray.aio.EdfWriter finished>".to_string()
        }
    }

    fn __aenter__<'py>(slf: Py<Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(slf) })
    }

    #[pyo3(signature = (*_args))]
    fn __aexit__<'py>(
        &self,
        py: Python<'py>,
        _args: Bound<'_, pyo3::types::PyTuple>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let writer = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || -> PyResult<()> {
                let mut guard = lock_writer(&writer)?;
                if let Some(w) = guard.take() {
                    w.finish().map_err(to_py_err)?;
                }
                Ok(())
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))??;
            Ok(())
        })
    }
}

#[pyfunction]
#[pyo3(name = "write_edf", signature = (
    path,
    *,
    variant,
    record_duration,
    signals,
    data,
    annotations = None,
    start_datetime = None,
    patient_id = None,
    recording_id = None,
    annotation_bytes_per_record = None,
))]
#[allow(clippy::too_many_arguments)]
fn write_edf_async<'py>(
    py: Python<'py>,
    path: String,
    variant: String,
    record_duration: f64,
    signals: Vec<PyAsyncWriterSignal>,
    data: Vec<PyReadonlyArray1<f64>>,
    annotations: Option<Vec<PyAnnotation>>,
    start_datetime: Option<&Bound<'_, PyAny>>,
    patient_id: Option<String>,
    recording_id: Option<String>,
    annotation_bytes_per_record: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let spec = build_spec(
        &variant,
        record_duration,
        signals.into_iter().map(|s| s.inner).collect(),
        start_datetime,
        patient_id,
        recording_id,
        annotation_bytes_per_record,
    )?;
    let owned: Vec<Vec<f64>> = data
        .iter()
        .map(|arr| arr.as_slice().map(|s| s.to_vec()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("data arrays must be contiguous: {e}"))
        })?;
    let anns_owned: Vec<CoreAnnotation> = annotations.map(|a| anns_to_core(&a)).unwrap_or_default();
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        tokio::task::spawn_blocking(move || -> PyResult<()> {
            let slices: Vec<&[f64]> = owned.iter().map(|v| v.as_slice()).collect();
            write_edf(&path, spec, &slices, &anns_owned).map_err(to_py_err)
        })
        .await
        .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))??;
        Ok(())
    })
}

/// Open an EDF/EDF+/BDF file.
///
/// `variant` forces the file variant instead of trusting the auto-detected
/// one, for files that omit or misreport the EDF+ "+C"/"+D" marker. It only
/// controls the plain/"+C"/"+D" distinction; an override that changes the
/// EDF-vs-BDF sample size (set by the version field) raises `ValueError`.
#[pyfunction]
#[pyo3(name = "open")]
#[pyo3(signature = (path, variant=None))]
fn open_async<'py>(
    py: Python<'py>,
    path: String,
    variant: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let forced = variant.map(parse_variant).transpose()?;
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let edf = tokio::task::spawn_blocking(move || match forced {
            None => EdfFile::open(&path),
            Some(v) => EdfFile::open_with_variant(&path, v),
        })
        .await
        .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
        .map_err(to_py_err)?;
        Python::attach(|py| {
            Py::new(
                py,
                PyAsyncEdfFile {
                    inner: Some(Arc::new(edf)),
                },
            )
        })
    })
}

#[pyfunction]
#[pyo3(name = "inspect")]
fn inspect_async<'py>(py: Python<'py>, path: String) -> PyResult<Bound<'py, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let meta = tokio::task::spawn_blocking(move || EdfFile::inspect(&path))
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
        Python::attach(|py| -> PyResult<Py<PyDict>> {
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
            Ok(dict.unbind())
        })
    })
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let aio = PyModule::new(py, "aio")?;
    aio.add_class::<PyAsyncWriterSignal>()?;
    aio.add_class::<PyAsyncEdfFile>()?;
    aio.add_class::<PyAsyncSignal>()?;
    aio.add_class::<PyAsyncEdfWriter>()?;
    aio.add_function(wrap_pyfunction!(open_async, &aio)?)?;
    aio.add_function(wrap_pyfunction!(inspect_async, &aio)?)?;
    aio.add_function(wrap_pyfunction!(write_edf_async, &aio)?)?;
    parent.add_submodule(&aio)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("edfarray._core.aio", &aio)?;
    Ok(())
}
