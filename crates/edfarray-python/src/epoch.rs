use numpy::PyArrayMethods;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

use edfarray_core::epoch::EpochPad;
use edfarray_core::file::EdfFile;
use edfarray_core::group::SignalGroup;

use crate::annotations::PyAnnotation;
use crate::errors::{invalid_argument_err, to_py_err};
use crate::group::PySignalGroup;
use crate::proxy_2d::parse_pad_mode;

/// Parse the `pad` argument: `"drop"`, any proxy pad mode, or `None` (= `"drop"`).
pub(crate) fn parse_epoch_pad(spec: Option<&Bound<'_, PyAny>>) -> PyResult<EpochPad> {
    match spec {
        None => Ok(EpochPad::Drop),
        Some(s) if s.is_none() => Ok(EpochPad::Drop),
        Some(s) => {
            if let Ok(name) = s.extract::<String>()
                && name.eq_ignore_ascii_case("drop")
            {
                return Ok(EpochPad::Drop);
            }
            Ok(EpochPad::Fill(parse_pad_mode(spec)?))
        }
    }
}

/// Coerce the `events` argument to onset times, preserving order.
///
/// Accepts a float, a `Sequence[float]`, an `Annotation` or `Sequence[Annotation]`, and any
/// object exposing `.onset` (mixed lists of floats and `Annotation`s are fine).
pub(crate) fn event_onsets(events: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(a) = events.extract::<PyRef<'_, PyAnnotation>>() {
        return Ok(vec![a.onset]);
    }
    if let Ok(t) = events.extract::<f64>() {
        return Ok(vec![t]);
    }
    // numpy arrays (and anything array-like) expose `.tolist()`, which yields a plain list we
    // can coerce element by element.
    if let Ok(item) = events.getattr("tolist") {
        let as_list = item.call0()?;
        return event_onsets(&as_list);
    }
    let mut out = Vec::new();
    let iter = events.try_iter().map_err(|_| {
        PyTypeError::new_err("events must be a float, Annotation, or a sequence of either")
    })?;
    for element in iter {
        let element = element?;
        if let Ok(a) = element.extract::<PyRef<'_, PyAnnotation>>() {
            out.push(a.onset);
        } else if let Ok(t) = element.extract::<f64>() {
            out.push(t);
        } else if let Ok(any) = element.getattr("onset") {
            out.push(any.extract::<f64>()?);
        } else {
            return Err(PyTypeError::new_err(
                "events must contain floats or Annotations, got an unsupported element type",
            ));
        }
    }
    Ok(out)
}

/// Resolve the `group` argument: a `SignalGroup`, a sequence of indices, or `None` for the
/// largest rectangular group.
pub(crate) fn resolve_group(
    f: &EdfFile,
    group: Option<&Bound<'_, PyAny>>,
) -> PyResult<SignalGroup> {
    let Some(g) = group else {
        return largest_rectangular_group(f);
    };
    if let Ok(pg) = g.extract::<PyRef<'_, PySignalGroup>>() {
        return Ok(pg.inner().clone());
    }
    if let Ok(items) = g.extract::<Vec<usize>>() {
        if items.is_empty() {
            return Err(invalid_argument_err(
                "group must be a SignalGroup, a sequence of signal indices, or None",
            ));
        }
        return SignalGroup::from_indices(f.header(), &items).map_err(to_py_err);
    }
    Err(invalid_argument_err(
        "group must be a SignalGroup, a sequence of signal indices, or None",
    ))
}

/// The rectangular group with the most channels; ties keep `signal_groups()` order.
pub(crate) fn largest_rectangular_group(f: &EdfFile) -> PyResult<SignalGroup> {
    let mut best: Option<SignalGroup> = None;
    for g in f.signal_groups() {
        if best
            .as_ref()
            .is_none_or(|b| g.indices().len() > b.indices().len())
        {
            best = Some(g);
        }
    }
    best.ok_or_else(|| {
        invalid_argument_err("file has no ordinary signals to extract; pass an explicit group")
    })
}

/// Plan + decode in one call. The caller releases the GIL.
///
/// Returns `(onsets, valid, data, dropped, n_samples)` where `onsets` are the kept event times
/// in caller order (filtered by the planner flags under `EpochPad::Drop`), `data` is the flat
/// row-major block, and `n_samples` is the planner's nominal row width.
pub(crate) type PlanExtract = (Vec<f64>, Vec<bool>, Vec<f64>, Vec<usize>, usize);

pub(crate) fn plan_and_extract(
    f: &EdfFile,
    group: &SignalGroup,
    events: &[f64],
    pre: f64,
    post: f64,
    pad: EpochPad,
) -> PyResult<PlanExtract> {
    use edfarray_core::epoch::{extract_epochs, plan_epochs};
    let plan = plan_epochs(f, group, events, pre, post).map_err(to_py_err)?;
    let (data, valid, dropped) = extract_epochs(f, group, &plan, pad).map_err(to_py_err)?;
    let onsets_out: Vec<f64> = match pad {
        EpochPad::Drop => plan
            .windows
            .iter()
            .zip(&plan.valid)
            .filter(|(_, v)| **v)
            .map(|(w, _)| w.onset)
            .collect(),
        _ => plan.windows.iter().map(|w| w.onset).collect(),
    };
    Ok((onsets_out, valid, data, dropped, plan.n_samples))
}

/// Extracted epochs: a dense `(n_epochs, n_channels, n_samples)` float64 block plus metadata.
///
/// `np.asarray(ep)` returns `ep.data`. `valid` is False for rows that contain padded samples.
/// Under `pad="drop"` it is all True, because padded epochs were removed (see `dropped`).
#[gen_stub_pyclass]
#[pyclass(name = "Epochs", module = "edfarray._core")]
pub(crate) struct PyEpochs {
    pub(crate) data: Vec<f64>,
    pub(crate) shape: (usize, usize, usize),
    pub(crate) onsets: Vec<f64>,
    pub(crate) labels: Vec<String>,
    pub(crate) sample_rate: f64,
    pub(crate) valid: Vec<bool>,
    pub(crate) dropped: Vec<usize>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEpochs {
    /// The decoded block, shape `(n_epochs, n_channels, n_samples)`, float64.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float64]", imports = ("numpy",)))]
    fn data<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let (a, b, c) = self.shape;
        let array = numpy::PyArray3::<f64>::zeros(py, (a, b, c), false);
        if !self.data.is_empty() {
            unsafe { array.as_slice_mut()? }.copy_from_slice(&self.data);
        }
        Ok(array.into_any().to_owned())
    }

    /// Event onset times in caller order, float64.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.float64]", imports = ("numpy",)))]
    fn onsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        numpy::PyArray1::from_vec(py, self.onsets.clone())
            .into_any()
            .to_owned()
    }

    /// Channel labels, in group order.
    #[getter]
    fn labels(&self) -> Vec<String> {
        self.labels.clone()
    }

    /// Common sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Per output epoch: false where samples were padded (fill modes) or, with `"drop"`,
    /// always true.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "numpy.typing.NDArray[numpy.bool_]", imports = ("numpy",)))]
    fn valid<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        numpy::PyArray1::from_iter(py, self.valid.iter().copied())
            .into_any()
            .to_owned()
    }

    /// Indices into the original events argument of epochs removed by `pad="drop"`.
    #[getter]
    fn dropped(&self) -> Vec<usize> {
        self.dropped.clone()
    }

    fn __len__(&self) -> usize {
        self.shape.0
    }

    /// Support `numpy.asarray(ep)`.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if copy == Some(false) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cannot avoid a copy: epochs are decoded from the file on access",
            ));
        }
        let array = self.data(py)?;
        match dtype {
            None => Ok(array),
            Some(dt) => array.call_method1("astype", (dt,)),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Epochs(n_epochs={}, n_channels={}, n_samples={}, sample_rate={}Hz, dropped={})",
            self.shape.0,
            self.shape.1,
            self.shape.2,
            self.sample_rate,
            self.dropped.len(),
        )
    }
}
