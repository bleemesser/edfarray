use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use edfarray_core::group::PadMode;
use edfarray_core::proxy_2d::Proxy2D;

use crate::errors::to_py_err;

/// 2D array proxy for numpy-style multi-channel signal access.
///
/// Supports indexing with `proxy[signal, sample]` where each axis accepts
/// int, slice, or list (signal axis only). All signals must share the same
/// sample rate.
#[gen_stub_pyclass]
#[pyclass(name = "Proxy2D")]
pub struct PyProxy2D {
    proxy: Proxy2D,
}

impl PyProxy2D {
    pub fn new(proxy: Proxy2D) -> Self {
        PyProxy2D { proxy }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProxy2D {
    /// Shape of the proxy: (num_signals, total_samples).
    #[getter]
    fn shape(&self) -> (usize, usize) {
        self.proxy.shape()
    }

    /// Common sample rate (Hz), or `None` if the underlying group has mixed rates.
    #[getter]
    fn sample_rate(&self) -> Option<f64> {
        self.proxy.sample_rate()
    }

    /// Per-channel valid sample counts, in proxy-coordinate order.
    #[getter]
    fn valid_lengths(&self) -> Vec<usize> {
        self.proxy.valid_lengths().to_vec()
    }

    /// Pad-mode policy as a string: "raise", "nan", "zero", "value", or "edge".
    #[getter]
    fn pad_mode(&self) -> &'static str {
        pad_mode_name(self.proxy.pad_mode())
    }

    fn __repr__(&self) -> String {
        let (rows, cols) = self.proxy.shape();
        let rate = match self.proxy.sample_rate() {
            Some(r) => format!("{}Hz", r),
            None => "mixed".into(),
        };
        format!("Proxy2D(shape=({}, {}), rate={}, pad={})", rows, cols, rate, pad_mode_name(self.proxy.pad_mode()))
    }

    /// Numpy-style 2D indexing: `proxy[signal_spec, sample_spec]`.
    ///
    /// | signal_spec | sample_spec | Return type |
    /// |---|---|---|
    /// | int | int | float |
    /// | int | slice | 1D ndarray |
    /// | slice/list | int | 1D ndarray |
    /// | slice/list | slice | 2D ndarray |
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let tuple = if let Ok(t) = key.cast::<PyTuple>() {
            if t.len() != 2 {
                return Err(PyIndexError::new_err(
                    "Proxy2D requires exactly 2 indices: [signal, sample]",
                ));
            }
            t.clone()
        } else {
            return Err(PyIndexError::new_err(
                "Proxy2D requires exactly 2 indices: [signal, sample]",
            ));
        };

        let sig_spec = tuple.get_item(0)?;
        let samp_spec = tuple.get_item(1)?;

        let (num_signals, num_samples) = self.proxy.shape();

        let sig_int = sig_spec.extract::<isize>().ok();
        let samp_int = samp_spec.extract::<isize>().ok();

        if let (Some(si), Some(sa)) = (sig_int, samp_int) {
            let si = normalize_index(si, num_signals)?;
            let sa = normalize_index(sa, num_samples)?;
            let val = self.proxy.get(si, sa).map_err(to_py_err)?;
            return Ok(val.into_pyobject(py)?.into_any().unbind());
        }

        if let Some(si) = sig_int {
            let si = normalize_index(si, num_signals)?;
            let (samp_start, samp_end) = parse_sample_spec(&samp_spec, num_samples)?;
            let count = samp_end.saturating_sub(samp_start);
            let data = self.proxy.read_slice(si..si + 1, samp_start..samp_end).map_err(to_py_err)?;
            let array = PyArray1::<f64>::zeros(py, count, false);
            if count > 0 {
                unsafe {
                    array.as_slice_mut()?.copy_from_slice(&data[0]);
                }
            }
            return Ok(array.into_any().unbind());
        }

        let signal_indices = parse_signal_spec(&sig_spec, num_signals)?;
        let (samp_start, samp_end) = parse_sample_spec(&samp_spec, num_samples)?;
        let count = samp_end.saturating_sub(samp_start);

        if let Some(sa) = samp_int {
            let sa = normalize_index(sa, num_samples)?;
            let vals = self.proxy.read_signals_at_sample(&signal_indices, sa).map_err(to_py_err)?;
            let array = PyArray1::<f64>::from_vec(py, vals);
            return Ok(array.into_any().unbind());
        }

        let data = self.proxy.read_physical(&signal_indices, samp_start..samp_end).map_err(to_py_err)?;
        let n_sig = signal_indices.len();
        let array = PyArray2::<f64>::zeros(py, (n_sig, count), false);
        unsafe {
            let slice = array.as_slice_mut()?;
            for (i, row) in data.iter().enumerate() {
                let row_start = i * count;
                slice[row_start..row_start + row.len()].copy_from_slice(row);
            }
        }
        Ok(array.into_any().unbind())
    }
}

fn normalize_index(idx: isize, len: usize) -> PyResult<usize> {
    let len_i = len as isize;
    let normalized = if idx < 0 { len_i + idx } else { idx };
    if normalized < 0 || normalized >= len_i {
        return Err(PyIndexError::new_err(format!(
            "index {idx} out of range for axis with size {len}"
        )));
    }
    Ok(normalized as usize)
}

fn parse_sample_spec(spec: &Bound<'_, PyAny>, length: usize) -> PyResult<(usize, usize)> {
    if let Ok(slice) = spec.cast::<PySlice>() {
        let indices = slice.indices(length as isize)?;
        if indices.step != 1 {
            return Err(PyValueError::new_err("step != 1 not supported in Proxy2D"));
        }
        Ok((indices.start as usize, indices.stop as usize))
    } else if let Ok(idx) = spec.extract::<isize>() {
        let idx = normalize_index(idx, length)?;
        Ok((idx, idx + 1))
    } else {
        Err(PyTypeError::new_err("sample index must be int or slice"))
    }
}

fn parse_signal_spec(spec: &Bound<'_, PyAny>, length: usize) -> PyResult<Vec<usize>> {
    if let Ok(slice) = spec.cast::<PySlice>() {
        let indices = slice.indices(length as isize)?;
        if indices.step != 1 {
            return Err(PyValueError::new_err("step != 1 not supported for signal axis"));
        }
        Ok((indices.start as usize..indices.stop as usize).collect())
    } else if let Ok(list) = spec.cast::<PyList>() {
        let mut result = Vec::with_capacity(list.len());
        for item in list {
            let idx: isize = item.extract()?;
            result.push(normalize_index(idx, length)?);
        }
        Ok(result)
    } else if let Ok(idx) = spec.extract::<isize>() {
        Ok(vec![normalize_index(idx, length)?])
    } else {
        Err(PyTypeError::new_err("signal index must be int, slice, or list"))
    }
}

pub(crate) fn parse_pad_mode(spec: Option<&Bound<'_, PyAny>>) -> PyResult<PadMode> {
    let Some(spec) = spec else {
        return Ok(PadMode::Raise);
    };
    if spec.is_none() {
        return Ok(PadMode::Raise);
    }
    if let Ok(name) = spec.extract::<String>() {
        return match name.to_ascii_lowercase().as_str() {
            "raise" => Ok(PadMode::Raise),
            "nan" => Ok(PadMode::Nan),
            "zero" => Ok(PadMode::Zero),
            "edge" => Ok(PadMode::Edge),
            other => Err(PyValueError::new_err(format!(
                "unknown pad_mode '{other}' (use 'raise', 'nan', 'zero', 'edge', a number, or None)"
            ))),
        };
    }
    if let Ok(v) = spec.extract::<f64>() {
        return Ok(PadMode::Value(v));
    }
    Err(PyTypeError::new_err(
        "pad_mode must be a string ('raise'/'nan'/'zero'/'edge'), a number, or None",
    ))
}

pub(crate) fn pad_mode_name(mode: PadMode) -> &'static str {
    match mode {
        PadMode::Raise => "raise",
        PadMode::Nan => "nan",
        PadMode::Zero => "zero",
        PadMode::Value(_) => "value",
        PadMode::Edge => "edge",
    }
}
