use numpy::{PyArray1, PyArray3, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PySlice, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use edfarray_core::proxy_3d::Proxy3D;

use crate::errors::to_py_err;

/// 3D view over a `Rectangular` signal group, shape
/// `(num_records, num_channels, samples_per_record)`.
///
/// Indexing semantics match NumPy 3D: `proxy[rec, ch, samp]` returns a scalar
/// when all three are ints, a 2D ndarray when two are slices, etc. Step != 1
/// is not supported.
#[gen_stub_pyclass]
#[pyclass(name = "Proxy3D")]
pub struct PyProxy3D {
    proxy: Proxy3D,
}

impl PyProxy3D {
    pub fn new(proxy: Proxy3D) -> Self {
        PyProxy3D { proxy }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyProxy3D {
    /// Shape of the proxy: `(num_records, num_channels, samples_per_record)`.
    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.proxy.shape()
    }

    /// Common sample rate (Hz) of the group's channels.
    #[getter]
    fn sample_rate(&self) -> f64 {
        self.proxy.sample_rate()
    }

    /// `True` if the file/group support a zero-copy stride view via
    /// [`as_strided`].
    #[getter]
    fn supports_strided_view(&self) -> bool {
        self.proxy.stride_info().is_some()
    }

    fn __repr__(&self) -> String {
        let (r, c, s) = self.proxy.shape();
        format!(
            "Proxy3D(shape=({}, {}, {}), rate={}Hz)",
            r,
            c,
            s,
            self.proxy.sample_rate()
        )
    }

    /// NumPy-style 3D indexing: `proxy[rec, channel, sample]`.
    ///
    /// Each axis accepts an int or a slice; the sample axis additionally
    /// accepts a step (e.g. `p[:, :, ::4]` to downsample), while the record
    /// and channel axes require step 1. Returns a scalar (all three ints), a
    /// 1D array (one non-int axis), a 2D array (two), or a 3D array (all).
    /// The full enclosing record block is materialized regardless of the
    /// sample step, so striding shrinks the result, not the work.
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
        let tuple = key.cast::<PyTuple>().map_err(|_| {
            PyIndexError::new_err("Proxy3D requires exactly 3 indices: [record, channel, sample]")
        })?;
        if tuple.len() != 3 {
            return Err(PyIndexError::new_err(
                "Proxy3D requires exactly 3 indices: [record, channel, sample]",
            ));
        }

        let (n_rec, n_ch, spr) = self.proxy.shape();
        let (rec_spec, ch_spec, samp_spec) =
            (tuple.get_item(0)?, tuple.get_item(1)?, tuple.get_item(2)?);

        let rec = AxisSpec::parse(&rec_spec, n_rec, "record")?;
        let ch = AxisSpec::parse(&ch_spec, n_ch, "channel")?;
        let samp = AxisSpec::parse_sample(&samp_spec, spr)?;

        // Fast path: all ints -> scalar.
        if let (AxisSpec::Int(ri), AxisSpec::Int(ci), AxisSpec::Int(si)) = (rec, ch, samp) {
            let v = self.proxy.get(ri, ci, si).map_err(to_py_err)?;
            return Ok(v.into_pyobject(py)?.into_any().unbind());
        }

        // Otherwise materialize the smallest enclosing block and squeeze ints.
        let rec_range = rec.to_range();
        let ch_range = ch.to_range();
        let block = self
            .proxy
            .read_physical_block(rec_range.clone(), ch_range.clone())
            .map_err(to_py_err)?;

        let nr = rec_range.len();
        let nc = ch_range.len();
        let ns_full = spr;
        let ns = samp.count();

        // Slice (and possibly stride) the sample axis out of the record-major
        // block. The block already holds every sample per record, so striding
        // is an in-memory gather with no extra reads.
        let mut out = Vec::with_capacity(nr * nc * ns);
        for ri in 0..nr {
            for ci in 0..nc {
                let row_base = (ri * nc + ci) * ns_full;
                match samp {
                    AxisSpec::Strided { .. } => {
                        for k in 0..ns {
                            out.push(block[row_base + samp.nth(k)]);
                        }
                    }
                    _ => {
                        let s_start = samp.to_range().start;
                        out.extend_from_slice(&block[row_base + s_start..row_base + s_start + ns]);
                    }
                }
            }
        }

        // Build the result with axes squeezed where the caller passed an int.
        let dims: Vec<usize> = [rec, ch, samp]
            .into_iter()
            .zip([nr, nc, ns])
            .filter_map(|(spec, n)| if spec.is_int() { None } else { Some(n) })
            .collect();

        match dims.len() {
            1 => {
                let arr = PyArray1::<f64>::from_vec(py, out);
                Ok(arr.into_any().unbind())
            }
            2 => {
                let (a, b) = (dims[0], dims[1]);
                let arr = numpy::PyArray2::<f64>::zeros(py, (a, b), false);
                unsafe {
                    arr.as_slice_mut()?.copy_from_slice(&out);
                }
                Ok(arr.into_any().unbind())
            }
            3 => {
                let arr = PyArray3::<f64>::zeros(py, (dims[0], dims[1], dims[2]), false);
                unsafe {
                    arr.as_slice_mut()?.copy_from_slice(&out);
                }
                Ok(arr.into_any().unbind())
            }
            _ => unreachable!(),
        }
    }

    /// Zero-copy stride-view metadata as a dict, or `None` when not eligible.
    ///
    /// Keys: `base_offset`, `record_stride_bytes`, `channel_stride_bytes`,
    /// `sample_stride_bytes`, `shape`. Eligibility requires 2-byte samples,
    /// a contiguous channel-index range, and no annotation channel inside
    /// that span.
    fn stride_info<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let info = self.proxy.stride_info();
        let Some(info) = info else {
            return Ok(py.None());
        };
        let d = pyo3::types::PyDict::new(py);
        d.set_item("base_offset", info.base_offset)?;
        d.set_item("record_stride_bytes", info.record_stride_bytes)?;
        d.set_item("channel_stride_bytes", info.channel_stride_bytes)?;
        d.set_item("sample_stride_bytes", info.sample_stride_bytes)?;
        d.set_item("shape", info.shape)?;
        Ok(d.into_any().unbind())
    }
}

#[derive(Clone, Copy, Debug)]
enum AxisSpec {
    Int(usize),
    Range(usize, usize),
    /// Strided slice, only produced for the sample axis. `start`/`step` follow
    /// Python slice semantics; `count` is the number of emitted elements.
    Strided {
        start: isize,
        step: isize,
        count: usize,
    },
}

impl AxisSpec {
    /// Parse a record/channel axis: int or step-1 slice only.
    fn parse(spec: &Bound<'_, PyAny>, length: usize, axis: &str) -> PyResult<Self> {
        if let Ok(idx) = spec.extract::<isize>() {
            let n = normalize(idx, length, axis)?;
            return Ok(AxisSpec::Int(n));
        }
        if let Ok(s) = spec.cast::<PySlice>() {
            let i = s.indices(length as isize)?;
            if i.step != 1 {
                return Err(PyValueError::new_err(format!(
                    "step != 1 not supported on the {axis} axis"
                )));
            }
            return Ok(AxisSpec::Range(i.start as usize, i.stop as usize));
        }
        Err(PyTypeError::new_err(format!(
            "{axis} index must be int or slice"
        )))
    }

    /// Parse the sample axis: int or slice, with an arbitrary step allowed.
    fn parse_sample(spec: &Bound<'_, PyAny>, length: usize) -> PyResult<Self> {
        if let Ok(idx) = spec.extract::<isize>() {
            let n = normalize(idx, length, "sample")?;
            return Ok(AxisSpec::Int(n));
        }
        if let Ok(s) = spec.cast::<PySlice>() {
            let i = s.indices(length as isize)?;
            if i.step == 1 {
                return Ok(AxisSpec::Range(i.start as usize, i.stop as usize));
            }
            return Ok(AxisSpec::Strided {
                start: i.start,
                step: i.step,
                count: range_len(i.start, i.stop, i.step),
            });
        }
        Err(PyTypeError::new_err("sample index must be int or slice"))
    }

    fn to_range(self) -> std::ops::Range<usize> {
        match self {
            AxisSpec::Int(i) => i..i + 1,
            AxisSpec::Range(s, e) => s..e,
            AxisSpec::Strided { .. } => unreachable!("strided axis has no contiguous range"),
        }
    }

    fn count(self) -> usize {
        match self {
            AxisSpec::Int(_) => 1,
            AxisSpec::Range(s, e) => e - s,
            AxisSpec::Strided { count, .. } => count,
        }
    }

    /// Absolute index of the k-th emitted element along a strided sample axis.
    fn nth(self, k: usize) -> usize {
        match self {
            AxisSpec::Strided { start, step, .. } => (start + k as isize * step) as usize,
            AxisSpec::Int(i) => i,
            AxisSpec::Range(s, _) => s + k,
        }
    }

    fn is_int(self) -> bool {
        matches!(self, AxisSpec::Int(_))
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

fn normalize(idx: isize, len: usize, axis: &str) -> PyResult<usize> {
    let len_i = len as isize;
    let n = if idx < 0 { len_i + idx } else { idx };
    if n < 0 || n >= len_i {
        return Err(PyIndexError::new_err(format!(
            "{axis} index {idx} out of range for size {len}"
        )));
    }
    Ok(n as usize)
}
