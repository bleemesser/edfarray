use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use edfarray_core::group::PadMode;
use edfarray_core::proxy_2d::Proxy2D;

use crate::errors::{invalid_argument_err, to_py_err};
use crate::indexing::{extract_index, normalize_index, unsupported_index_err};
use crate::numpy_util::numpy_dtype;

/// 2D array proxy for numpy-style multi-channel signal access.
///
/// Supports indexing with `proxy[signal, sample]` where each axis accepts
/// int, slice, or list (signal axis only). All signals must share the same
/// sample rate.
#[gen_stub_pyclass]
#[pyclass(name = "Proxy2D", module = "edfarray._core")]
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

    /// Read raw digital values for `signals` over samples `[start, stop)`.
    ///
    /// The counterpart to physical indexing, which `__getitem__` provides. Returns a 2D int32
    /// array. `pad_mode="nan"` has no int32 representation and raises here.
    fn read_digital<'py>(
        &self,
        py: Python<'py>,
        signals: Vec<usize>,
        start: usize,
        stop: usize,
    ) -> PyResult<Bound<'py, numpy::PyArray2<i32>>> {
        let count = stop.saturating_sub(start);
        let data = py
            .detach(|| self.proxy.read_digital(&signals, start..stop))
            .map_err(to_py_err)?;
        let array = numpy::PyArray2::<i32>::zeros(py, (signals.len(), count), false);
        if count > 0 {
            let slice = unsafe { array.as_slice_mut()? };
            for (i, row) in data.iter().enumerate() {
                slice[i * count..i * count + count].copy_from_slice(row);
            }
        }
        Ok(array)
    }

    /// Always 2.
    #[getter]
    fn ndim(&self) -> usize {
        2
    }

    /// dtype of the physical values this proxy decodes to.
    #[getter]
    fn dtype<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        numpy_dtype(py, "float64")
    }

    /// Support `numpy.asarray(proxy)` by materializing every channel.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if copy == Some(false) {
            return Err(PyValueError::new_err(
                "cannot avoid a copy: samples are decoded from the file on access",
            ));
        }
        let key = PyTuple::new(
            py,
            [PySlice::full(py).into_any(), PySlice::full(py).into_any()],
        )?;
        let array = self.__getitem__(py, key.as_any())?.into_bound(py);
        match dtype {
            None => Ok(array),
            Some(dt) => array.call_method1("astype", (dt,)),
        }
    }

    fn __repr__(&self) -> String {
        let (rows, cols) = self.proxy.shape();
        let rate = match self.proxy.sample_rate() {
            Some(r) => format!("{}Hz", r),
            None => "mixed".into(),
        };
        format!(
            "Proxy2D(shape=({}, {}), rate={}, pad={})",
            rows,
            cols,
            rate,
            pad_mode_name(self.proxy.pad_mode())
        )
    }

    /// Numpy-style 2D indexing: `proxy[signal_spec, sample_spec]`.
    ///
    /// | signal_spec | sample_spec | Return type |
    /// |---|---|---|
    /// | int | int | float |
    /// | int | slice | 1D ndarray |
    /// | slice/list | int | 1D ndarray |
    /// | slice/list | slice | 2D ndarray |
    ///
    /// The sample (time) axis accepts a step (e.g. `p[:, ::4]` to downsample);
    /// the signal axis does not. A strided sample read still reads the full
    /// enclosing span and then subsamples, so it costs about the same as the
    /// unstrided read of that span — it shrinks the result, not the I/O.
    #[gen_stub(override_return_type(type_repr = "builtins.float | numpy.typing.NDArray[numpy.float64]", imports = ("builtins", "numpy", "numpy.typing")))]
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let tuple = match key.cast::<PyTuple>() {
            Ok(t) if t.len() == 2 => t.clone(),
            Ok(t) => {
                return Err(PyTypeError::new_err(format!(
                    "Proxy2D requires exactly 2 indices [signal, sample], got {}",
                    t.len()
                )));
            }
            Err(_) => {
                return Err(unsupported_index_err(
                    key,
                    "a 2-tuple of [signal, sample] indices",
                ));
            }
        };

        let sig_spec = tuple.get_item(0)?;
        let samp_spec = tuple.get_item(1)?;

        let (num_signals, num_samples) = self.proxy.shape();

        let sig_int = extract_index(&sig_spec);
        let samp_int = extract_index(&samp_spec);

        if let (Some(si), Some(sa)) = (sig_int, samp_int) {
            let si = normalize_index(si, num_signals, "signal")?;
            let sa = normalize_index(sa, num_samples, "sample")?;
            let val = self.proxy.get(si, sa).map_err(to_py_err)?;
            return Ok(val.into_pyobject(py)?.into_any().unbind());
        }

        if let Some(si) = sig_int {
            let si = normalize_index(si, num_signals, "signal")?;
            let samp = parse_sample_spec(&samp_spec, num_samples)?;
            let array = PyArray1::<f64>::zeros(py, samp.count, false);
            if samp.count > 0 {
                let (lo, hi) = samp.window();
                let data = py
                    .detach(|| self.proxy.read_slice(si..si + 1, lo..hi))
                    .map_err(to_py_err)?;
                unsafe {
                    let out = array.as_slice_mut()?;
                    if samp.is_contiguous() {
                        out.copy_from_slice(&data[0]);
                    } else {
                        samp.gather(&data[0], lo, out);
                    }
                }
            }
            return Ok(array.into_any().unbind());
        }

        let signal_indices = parse_signal_spec(&sig_spec, num_signals)?;

        if let Some(sa) = samp_int {
            let sa = normalize_index(sa, num_samples, "sample")?;
            let vals = py
                .detach(|| self.proxy.read_signals_at_sample(&signal_indices, sa))
                .map_err(to_py_err)?;
            let array = PyArray1::<f64>::from_vec(py, vals);
            return Ok(array.into_any().unbind());
        }

        let samp = parse_sample_spec(&samp_spec, num_samples)?;
        let (lo, hi) = samp.window();
        let data = py
            .detach(|| self.proxy.read_physical(&signal_indices, lo..hi))
            .map_err(to_py_err)?;
        let n_sig = signal_indices.len();
        let array = PyArray2::<f64>::zeros(py, (n_sig, samp.count), false);
        unsafe {
            let slice = array.as_slice_mut()?;
            for (i, row) in data.iter().enumerate() {
                let dst = &mut slice[i * samp.count..i * samp.count + samp.count];
                if samp.is_contiguous() {
                    dst.copy_from_slice(row);
                } else {
                    samp.gather(row, lo, dst);
                }
            }
        }
        Ok(array.into_any().unbind())
    }
}

/// A parsed sample-axis slice that may carry a step. `count` is the number of
/// emitted elements; `window()` gives the contiguous span to read.
struct SampleSlice {
    start: isize,
    step: isize,
    count: usize,
}

impl SampleSlice {
    fn is_contiguous(&self) -> bool {
        self.step == 1
    }

    /// Minimal contiguous `[lo, hi)` sample window covering every emitted index.
    fn window(&self) -> (usize, usize) {
        if self.count == 0 {
            return (0, 0);
        }
        let last = self.start + (self.count as isize - 1) * self.step;
        let (lo, hi) = if self.step >= 0 {
            (self.start, last)
        } else {
            (last, self.start)
        };
        (lo as usize, hi as usize + 1)
    }

    /// Gather strided samples from `buf` (which covers the window starting at
    /// `lo`) into `out` (length `count`).
    fn gather(&self, buf: &[f64], lo: usize, out: &mut [f64]) {
        for (k, slot) in out.iter_mut().enumerate() {
            let abs = (self.start + k as isize * self.step) as usize;
            *slot = buf[abs - lo];
        }
    }
}

/// Number of elements in `range(start, stop, step)`, matching Python semantics.
fn range_len(start: isize, stop: isize, step: isize) -> usize {
    if step > 0 {
        if stop > start {
            (((stop - start) + step - 1) / step) as usize
        } else {
            0
        }
    } else if start > stop {
        (((start - stop) + (-step) - 1) / (-step)) as usize
    } else {
        0
    }
}

fn parse_sample_spec(spec: &Bound<'_, PyAny>, length: usize) -> PyResult<SampleSlice> {
    if let Ok(slice) = spec.cast::<PySlice>() {
        let indices = slice.indices(length as isize)?;
        Ok(SampleSlice {
            start: indices.start,
            step: indices.step,
            count: range_len(indices.start, indices.stop, indices.step),
        })
    } else if let Some(idx) = extract_index(spec) {
        let idx = normalize_index(idx, length, "sample")?;
        Ok(SampleSlice {
            start: idx as isize,
            step: 1,
            count: 1,
        })
    } else {
        Err(unsupported_index_err(spec, "an integer or a slice"))
    }
}

fn parse_signal_spec(spec: &Bound<'_, PyAny>, length: usize) -> PyResult<Vec<usize>> {
    if let Ok(slice) = spec.cast::<PySlice>() {
        let indices = slice.indices(length as isize)?;
        if indices.step != 1 {
            return Err(PyValueError::new_err(
                "step != 1 not supported for signal axis",
            ));
        }
        Ok((indices.start as usize..indices.stop as usize).collect())
    } else if let Ok(list) = spec.cast::<PyList>() {
        let mut result = Vec::with_capacity(list.len());
        for item in list {
            let idx = extract_index(&item)
                .ok_or_else(|| unsupported_index_err(&item, "a list of integers"))?;
            result.push(normalize_index(idx, length, "signal")?);
        }
        Ok(result)
    } else if let Some(idx) = extract_index(spec) {
        Ok(vec![normalize_index(idx, length, "signal")?])
    } else {
        Err(PyTypeError::new_err(
            "signal index must be int, slice, or list",
        ))
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
            other => Err(invalid_argument_err(format!(
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
