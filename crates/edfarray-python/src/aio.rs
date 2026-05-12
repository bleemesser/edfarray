use std::sync::Arc;

use chrono::{Datelike, Timelike};
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyDict;

use edfarray_core::file::EdfFile;
use edfarray_core::header::Sex;

use crate::annotations::PyAnnotation;
use crate::errors::to_py_err;

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
    aio.add_function(wrap_pyfunction!(open_async, &aio)?)?;
    aio.add_function(wrap_pyfunction!(inspect_async, &aio)?)?;
    parent.add_submodule(&aio)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("edfarray._core.aio", &aio)?;
    Ok(())
}
