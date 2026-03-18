use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use pyo3::types::PySlice;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use edfarray_core::proxy::SignalProxy;

use crate::errors::to_py_err;

/// Proxy view of a single signal, supporting numpy-style indexing.
#[gen_stub_pyclass]
#[pyclass(name = "Signal")]
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
    fn digital_min(&self) -> i16 {
        self.proxy.header().digital_min
    }

    /// Digital maximum value.
    #[getter]
    fn digital_max(&self) -> i16 {
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

    /// Supports `s[i]`, `s[start:stop]`, and `s[start:stop:step]`.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        if let Ok(idx) = key.extract::<isize>() {
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
                    unsafe {
                        let slice = array.as_slice_mut()?;
                        self.proxy.read_physical(start, stop, slice).map_err(to_py_err)?;
                    }
                }
                Ok(array.into_any().unbind())
            } else {
                let indices: Vec<usize> = StridedRange::new(start, stop, step).collect();
                let array = PyArray1::<f64>::zeros(py, indices.len(), false);
                unsafe {
                    let slice = array.as_slice_mut()?;
                    for (i, &idx) in indices.iter().enumerate() {
                        slice[i] = self.proxy.get_physical(idx).map_err(to_py_err)?;
                    }
                }
                Ok(array.into_any().unbind())
            }
        } else {
            Err(PyIndexError::new_err("index must be int or slice"))
        }
    }

    /// Return the entire signal as a float64 numpy array.
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let len = self.proxy.len();
        let array = PyArray1::<f64>::zeros(py, len, false);
        if len > 0 {
            unsafe {
                let slice = array.as_slice_mut()?;
                self.proxy.read_physical(0, len, slice).map_err(to_py_err)?;
            }
        }
        Ok(array)
    }

    /// Return the entire signal as a raw int16 numpy array.
    fn to_digital<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i16>>> {
        let len = self.proxy.len();
        let array = PyArray1::<i16>::zeros(py, len, false);
        if len > 0 {
            unsafe {
                let slice = array.as_slice_mut()?;
                self.proxy.read_digital(0, len, slice).map_err(to_py_err)?;
            }
        }
        Ok(array)
    }

    /// Return timestamps (in seconds) for each sample.
    fn times<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let len = self.proxy.len();
        let array = PyArray1::<f64>::zeros(py, len, false);
        if len > 0 {
            let mut buf = vec![0.0f64; len];
            self.proxy.read_times(0, len, &mut buf).map_err(to_py_err)?;
            unsafe {
                array.as_slice_mut()?.copy_from_slice(&buf);
            }
        }
        Ok(array)
    }
}

impl PySignal {
    fn normalize_index(&self, idx: isize) -> PyResult<usize> {
        let len = self.proxy.len() as isize;
        let normalized = if idx < 0 { len + idx } else { idx };
        if normalized < 0 || normalized >= len {
            return Err(PyIndexError::new_err(format!(
                "index {idx} out of range for signal with {len} samples"
            )));
        }
        Ok(normalized as usize)
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
        if self.step > 0 && self.current < self.stop {
            let val = self.current as usize;
            self.current += self.step;
            Some(val)
        } else if self.step < 0 && self.current > self.stop {
            let val = self.current as usize;
            self.current += self.step;
            Some(val)
        } else {
            None
        }
    }
}
