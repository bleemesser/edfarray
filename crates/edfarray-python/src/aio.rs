use std::sync::Arc;

use chrono::{Datelike, Timelike};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyDict;

use edfarray_core::file::EdfFile;
use edfarray_core::header::Sex;

use crate::annotations::PyAnnotation;
use crate::errors::to_py_err;

use numpy::{PyArray1, PyArray2, PyArrayMethods};

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

    fn signal_indices_by_rate<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let map = self.get().unwrap().signal_indices_by_rate();
        let dict = PyDict::new(py);
        for (rate, indices) in map {
            dict.set_item(rate, indices)?;
        }
        Ok(dict)
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

    /// Read a page of physical (f64) data for multiple signals over a time range.
    ///
    /// Returns a list of numpy arrays, one per signal. Signals with different
    /// sample rates produce arrays of different lengths.
    ///
    /// If `signal_indices` is None, reads all ordinary (non-annotation) signals.
    ///
    /// When `use_time` is false (default), time params are converted to flat
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

    /// Read a page of digital (raw int32) data for multiple signals over a time range.
    ///
    /// Same parameters as `read_page` but returns int32 arrays of the raw digital values.
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

    #[pyo3(signature = (idx_or_label, cache_capacity=0))]
    fn signal(&self, idx_or_label: &Bound<'_, PyAny>, cache_capacity: usize) -> PyResult<PyAsyncSignal> {
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
        let proxy = if cache_capacity > 0 { proxy.with_cache(cache_capacity) } else { proxy };
        Ok(PyAsyncSignal::new(proxy))
    }

    #[pyo3(signature = (signal_indices=None))]
    fn array_proxy(&self, signal_indices: Option<Vec<usize>>) -> PyResult<PyAsyncArrayProxy> {
        let inner = self.get()?;
        let proxy = inner.array_proxy(signal_indices.as_deref()).map_err(to_py_err)?;
        Ok(PyAsyncArrayProxy::new(proxy))
    }
}

#[pyclass(name = "Signal", module = "edfarray._core.aio")]
pub struct PyAsyncSignal {
    inner: Arc<edfarray_core::proxy::SignalProxy>,
}

impl PyAsyncSignal {
    pub fn new(proxy: edfarray_core::proxy::SignalProxy) -> Self {
        PyAsyncSignal { inner: Arc::new(proxy) }
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

    fn read_physical<'py>(&self, py: Python<'py>, start: usize, stop: usize) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let buf: Vec<f64> = tokio::task::spawn_blocking(move || -> Result<Vec<f64>, edfarray_core::error::EdfError> {
                let count = stop.saturating_sub(start);
                let mut buf = vec![0.0; count];
                if count > 0 {
                    proxy.read_physical(start, stop, &mut buf)?;
                }
                Ok(buf)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            Python::attach(|py| -> PyResult<Py<PyArray1<f64>>> {
                Ok(PyArray1::from_vec(py, buf).unbind())
            })
        })
    }

    fn read_digital<'py>(&self, py: Python<'py>, start: usize, stop: usize) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let buf: Vec<i32> = tokio::task::spawn_blocking(move || -> Result<Vec<i32>, edfarray_core::error::EdfError> {
                let count = stop.saturating_sub(start);
                let mut buf = vec![0i32; count];
                if count > 0 {
                    proxy.read_digital(start, stop, &mut buf)?;
                }
                Ok(buf)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            Python::attach(|py| -> PyResult<Py<PyArray1<i32>>> {
                Ok(PyArray1::from_vec(py, buf).unbind())
            })
        })
    }

    fn read_at<'py>(&self, py: Python<'py>, start_sec: f64, end_sec: f64) -> PyResult<Bound<'py, PyAny>> {
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
            let buf = tokio::task::spawn_blocking(move || -> Result<Vec<f64>, edfarray_core::error::EdfError> {
                let mut buf = vec![0.0; len];
                if len > 0 {
                    proxy.read_physical(0, len, &mut buf)?;
                }
                Ok(buf)
            })
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
            let buf = tokio::task::spawn_blocking(move || -> Result<Vec<i32>, edfarray_core::error::EdfError> {
                let mut buf = vec![0i32; len];
                if len > 0 {
                    proxy.read_digital(0, len, &mut buf)?;
                }
                Ok(buf)
            })
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
            let buf = tokio::task::spawn_blocking(move || -> Result<Vec<f64>, edfarray_core::error::EdfError> {
                let mut buf = vec![0.0; len];
                if len > 0 {
                    proxy.read_times(0, len, &mut buf)?;
                }
                Ok(buf)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            Python::attach(|py| -> PyResult<Py<PyArray1<f64>>> {
                Ok(PyArray1::from_vec(py, buf).unbind())
            })
        })
    }
}

#[pyclass(name = "ArrayProxy", module = "edfarray._core.aio")]
pub struct PyAsyncArrayProxy {
    inner: Arc<edfarray_core::array_proxy::ArrayProxy>,
}

impl PyAsyncArrayProxy {
    pub fn new(proxy: edfarray_core::array_proxy::ArrayProxy) -> Self {
        PyAsyncArrayProxy { inner: Arc::new(proxy) }
    }
}

#[pymethods]
impl PyAsyncArrayProxy {
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.inner.shape()
    }

    #[getter]
    fn sample_rate(&self) -> f64 {
        self.inner.sample_rate()
    }

    #[getter]
    fn signal_indices(&self) -> Vec<usize> {
        self.inner.signal_indices().to_vec()
    }

    fn __repr__(&self) -> String {
        let (r, c) = self.inner.shape();
        format!("<edfarray.aio.ArrayProxy shape=({}, {}) rate={}Hz>", r, c, self.inner.sample_rate())
    }

    #[pyo3(signature = (sample_start, sample_stop, signal_indices=None))]
    fn read_physical<'py>(
        &self,
        py: Python<'py>,
        sample_start: usize,
        sample_stop: usize,
        signal_indices: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.inner.clone();
        let indices = signal_indices.unwrap_or_else(|| proxy.signal_indices().to_vec());
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let data = tokio::task::spawn_blocking(move || {
                proxy.read_physical(&indices, sample_start..sample_stop)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            let n_sig = data.len();
            let n_samp = if n_sig > 0 { data[0].len() } else { 0 };
            Python::attach(|py| -> PyResult<Py<PyArray2<f64>>> {
                let array = PyArray2::<f64>::zeros(py, (n_sig, n_samp), false);
                unsafe {
                    let slice = array.as_slice_mut()?;
                    for (i, row) in data.iter().enumerate() {
                        slice[i * n_samp..i * n_samp + row.len()].copy_from_slice(row);
                    }
                }
                Ok(array.unbind())
            })
        })
    }

    #[pyo3(signature = (sample_start, sample_stop, signal_indices=None))]
    fn read_digital<'py>(
        &self,
        py: Python<'py>,
        sample_start: usize,
        sample_stop: usize,
        signal_indices: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.inner.clone();
        let indices = signal_indices.unwrap_or_else(|| proxy.signal_indices().to_vec());
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let data = tokio::task::spawn_blocking(move || {
                proxy.read_digital(&indices, sample_start..sample_stop)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            let n_sig = data.len();
            let n_samp = if n_sig > 0 { data[0].len() } else { 0 };
            Python::attach(|py| -> PyResult<Py<PyArray2<i32>>> {
                let array = PyArray2::<i32>::zeros(py, (n_sig, n_samp), false);
                unsafe {
                    let slice = array.as_slice_mut()?;
                    for (i, row) in data.iter().enumerate() {
                        slice[i * n_samp..i * n_samp + row.len()].copy_from_slice(row);
                    }
                }
                Ok(array.unbind())
            })
        })
    }

    fn get<'py>(&self, py: Python<'py>, signal_idx: usize, sample_idx: usize) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let val = tokio::task::spawn_blocking(move || proxy.get(signal_idx, sample_idx))
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
                .map_err(to_py_err)?;
            Ok(val)
        })
    }

    #[pyo3(signature = (sample_idx, signal_indices=None))]
    fn read_signals_at_sample<'py>(
        &self,
        py: Python<'py>,
        sample_idx: usize,
        signal_indices: Option<Vec<usize>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let proxy = self.inner.clone();
        let indices = signal_indices.unwrap_or_else(|| proxy.signal_indices().to_vec());
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let data = tokio::task::spawn_blocking(move || {
                proxy.read_signals_at_sample(&indices, sample_idx)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
            Python::attach(|py| -> PyResult<Py<PyArray1<f64>>> {
                Ok(PyArray1::from_vec(py, data).unbind())
            })
        })
    }
}

#[pyfunction]
#[pyo3(name = "open")]
fn open_async<'py>(py: Python<'py>, path: String) -> PyResult<Bound<'py, PyAny>> {
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        let edf = tokio::task::spawn_blocking(move || EdfFile::open(&path))
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("task join: {e}")))?
            .map_err(to_py_err)?;
        Python::attach(|py| Py::new(py, PyAsyncEdfFile { inner: Some(Arc::new(edf)) }))
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
    aio.add_class::<PyAsyncEdfFile>()?;
    aio.add_class::<PyAsyncSignal>()?;
    aio.add_class::<PyAsyncArrayProxy>()?;
    aio.add_function(wrap_pyfunction!(open_async, &aio)?)?;
    aio.add_function(wrap_pyfunction!(inspect_async, &aio)?)?;
    parent.add_submodule(&aio)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("edfarray._core.aio", &aio)?;
    Ok(())
}
