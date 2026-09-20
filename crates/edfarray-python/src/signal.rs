use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PySlice;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use edfarray_core::error::EdfError;
use edfarray_core::proxy::SignalProxy;

use crate::errors::to_py_err;
use crate::indexing::{extract_index, normalize_index, unsupported_index_err};
use crate::numpy_util::numpy_dtype;

/// Proxy view of a single signal, supporting numpy-style indexing.
#[gen_stub_pyclass]
#[pyclass(name = "Signal", module = "edfarray._core")]
pub struct PySignal {
    proxy: SignalProxy,
}

impl PySignal {
    pub fn new(proxy: SignalProxy) -> Self {
        PySignal { proxy }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PySignal {
    /// Signal label.
    #[getter]
    fn label(&self) -> &str {
        &self.proxy.header().label
    }

    /// Transducer type.
    #[getter]
    fn transducer(&self) -> &str {
        &self.proxy.header().transducer
    }

    /// Physical units.
    #[getter]
    fn physical_dimension(&self) -> &str {
        &self.proxy.header().physical_dimension
    }

    /// Prefiltering description.
    #[getter]
    fn prefiltering(&self) -> &str {
        &self.proxy.header().prefiltering
    }

    /// Sample frequency in Hz.
    #[getter]
    fn sample_rate(&self) -> f64 {
        self.proxy.sample_rate()
    }

    /// Number of samples per data record.
    #[getter]
    fn samples_per_record(&self) -> usize {
        self.proxy.header().num_samples
    }

    /// Physical minimum value.
    #[getter]
    fn physical_min(&self) -> f64 {
        self.proxy.header().physical_min
    }

    /// Physical maximum value.
    #[getter]
    fn physical_max(&self) -> f64 {
        self.proxy.header().physical_max
    }

    /// Digital minimum value.
    #[getter]
    fn digital_min(&self) -> i32 {
        self.proxy.header().digital_min
    }

    /// Digital maximum value.
    #[getter]
    fn digital_max(&self) -> i32 {
        self.proxy.header().digital_max
    }

    /// Total number of samples.
    #[getter]
    fn num_samples(&self) -> usize {
        self.proxy.len()
    }

    fn __len__(&self) -> usize {
        self.proxy.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Signal(label={:?}, samples={}, rate={}Hz)",
            self.proxy.header().label,
            self.proxy.len(),
            self.proxy.sample_rate()
        )
    }

    /// Index with an integer or a slice.
    ///
    /// Integers accept negative values and return a Python float; slices return a float64
    /// array and support any step. Boolean masks, fancy indexing, `None`, and `Ellipsis` are
    /// not supported and raise `TypeError`.
    #[gen_stub(override_return_type(type_repr = "builtins.float | numpy.typing.NDArray[numpy.float64]", imports = ("builtins", "numpy", "numpy.typing")))]
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        if let Some(idx) = extract_index(key) {
            let idx = self.normalize_index(idx)?;
            let val = self.proxy.get_physical(idx).map_err(to_py_err)?;
            Ok(val.into_pyobject(py)?.into_any().unbind())
        } else if let Ok(slice) = key.cast::<PySlice>() {
            let len = self.proxy.len() as isize;
            let indices = slice.indices(len)?;
            let start = indices.start;
            let stop = indices.stop;
            let step = indices.step;

            if step == 1 {
                let start = start as usize;
                let stop = stop as usize;
                let count = stop.saturating_sub(start);
                let array = PyArray1::<f64>::zeros(py, count, false);
                if count > 0 {
                    let slice = unsafe { array.as_slice_mut()? };
                    py.detach(|| self.proxy.read_physical(start, stop, slice))
                        .map_err(to_py_err)?;
                }
                Ok(array.into_any().unbind())
            } else {
                let indices: Vec<usize> = StridedRange::new(start, stop, step).collect();
                let array = PyArray1::<f64>::zeros(py, indices.len(), false);
                let slice = unsafe { array.as_slice_mut()? };
                py.detach(|| -> Result<(), EdfError> {
                    for (dst, &idx) in slice.iter_mut().zip(indices.iter()) {
                        *dst = self.proxy.get_physical(idx)?;
                    }
                    Ok(())
                })
                .map_err(to_py_err)?;
                Ok(array.into_any().unbind())
            }
        } else {
            Err(unsupported_index_err(key, "an integer or a slice"))
        }
    }

    /// Number of samples, as a one-element tuple. Mirrors `numpy.ndarray.shape`.
    #[getter]
    fn shape(&self) -> (usize,) {
        (self.proxy.len(),)
    }

    /// Always 1: a signal is one-dimensional.
    #[getter]
    fn ndim(&self) -> usize {
        1
    }

    /// dtype of the physical values this signal decodes to.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        numpy_dtype(py, "float64")
    }

    /// Support `numpy.asarray(signal)`.
    ///
    /// Without this, numpy falls back to the sequence protocol and decodes one sample per
    /// `__getitem__` call, which is correct but thousands of times slower.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let array = self.to_physical(py)?.into_any();
        if copy == Some(false) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot avoid a copy: samples are decoded from the file on access",
            ));
        }
        match dtype {
            None => Ok(array),
            Some(dt) => array.call_method1("astype", (dt,)),
        }
    }

    /// Return the entire signal as a float64 numpy array of physical values.
    fn to_physical<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let len = self.proxy.len();
        let array = PyArray1::<f64>::zeros(py, len, false);
        if len > 0 {
            let slice = unsafe { array.as_slice_mut()? };
            py.detach(|| self.proxy.read_physical(0, len, slice))
                .map_err(to_py_err)?;
        }
        Ok(array)
    }

    /// Return the entire signal as a raw int32 numpy array.
    fn to_digital<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let len = self.proxy.len();
        let array = PyArray1::<i32>::zeros(py, len, false);
        if len > 0 {
            let slice = unsafe { array.as_slice_mut()? };
            py.detach(|| self.proxy.read_digital(0, len, slice))
                .map_err(to_py_err)?;
        }
        Ok(array)
    }

    /// Return physical values for samples `[start, stop)`, indexed by sample number.
    ///
    /// Equivalent to `signal[start:stop]`. Use `read_time_range` to index by seconds.
    fn read_range<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        stop: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let stop = stop.min(self.proxy.len());
        let count = stop.saturating_sub(start);
        let array = PyArray1::<f64>::zeros(py, count, false);
        if count > 0 {
            let slice = unsafe { array.as_slice_mut()? };
            py.detach(|| self.proxy.read_physical(start, stop, slice))
                .map_err(to_py_err)?;
        }
        Ok(array)
    }

    /// Return raw digital values for samples `[start, stop)`, indexed by sample number.
    fn read_range_digital<'py>(
        &self,
        py: Python<'py>,
        start: usize,
        stop: usize,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let stop = stop.min(self.proxy.len());
        let count = stop.saturating_sub(start);
        let array = PyArray1::<i32>::zeros(py, count, false);
        if count > 0 {
            let slice = unsafe { array.as_slice_mut()? };
            py.detach(|| self.proxy.read_digital(start, stop, slice))
                .map_err(to_py_err)?;
        }
        Ok(array)
    }

    /// Return timestamps (in seconds) for each sample.
    fn times<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let len = self.proxy.len();
        let array = PyArray1::<f64>::zeros(py, len, false);
        if len > 0 {
            let slice = unsafe { array.as_slice_mut()? };
            py.detach(|| self.proxy.read_times(0, len, slice))
                .map_err(to_py_err)?;
        }
        Ok(array)
    }

    /// Return physical values for samples whose time falls in `[start_sec, end_sec)`.
    ///
    /// Arguments are seconds. Use `read_range` to index by sample number instead.
    ///
    /// For EDF+D files this accounts for gaps between records using the record onset times
    /// from the annotation index (blocks until the scan completes). For EDF and EDF+C it is
    /// equivalent to indexing by flat sample number. Each time maps to the first sample at
    /// or after it.
    fn read_time_range<'py>(
        &self,
        py: Python<'py>,
        start_sec: f64,
        end_sec: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let buf = py
            .detach(|| self.proxy.read_at(start_sec, end_sec))
            .map_err(to_py_err)?;
        let len = buf.len();
        let array = PyArray1::<f64>::zeros(py, len, false);
        if len > 0 {
            unsafe {
                array.as_slice_mut()?.copy_from_slice(&buf);
            }
        }
        Ok(array)
    }
}

impl PySignal {
    fn normalize_index(&self, idx: isize) -> PyResult<usize> {
        normalize_index(idx, self.proxy.len(), "sample")
    }
}

/// Iterator over indices produced by a Python slice with arbitrary step.
struct StridedRange {
    current: isize,
    stop: isize,
    step: isize,
}

impl StridedRange {
    fn new(start: isize, stop: isize, step: isize) -> Self {
        StridedRange {
            current: start,
            stop,
            step,
        }
    }
}

impl Iterator for StridedRange {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let in_range = if self.step > 0 {
            self.current < self.stop
        } else {
            self.step < 0 && self.current > self.stop
        };
        if !in_range {
            return None;
        }
        let val = self.current as usize;
        self.current += self.step;
        Some(val)
    }
}
