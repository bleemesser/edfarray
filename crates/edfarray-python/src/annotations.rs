use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};

/// A single annotation from the EDF+ file.
#[gen_stub_pyclass]
#[pyclass(frozen, name = "Annotation", module = "edfarray._core", from_py_object)]
#[derive(Clone)]
pub struct PyAnnotation {
    #[pyo3(get)]
    pub onset: f64,
    #[pyo3(get)]
    pub duration: Option<f64>,
    #[pyo3(get)]
    pub text: String,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAnnotation {
    #[new]
    #[pyo3(signature = (onset, text, duration=None))]
    fn new(onset: f64, text: String, duration: Option<f64>) -> Self {
        PyAnnotation {
            onset,
            duration,
            text,
        }
    }

    fn __repr__(&self) -> String {
        match self.duration {
            Some(d) => format!(
                "Annotation(onset={}, duration={}, text={:?})",
                self.onset, d, self.text
            ),
            None => format!("Annotation(onset={}, text={:?})", self.onset, self.text),
        }
    }

    /// Compare by value, so annotations work with `in`, `set`, and `==` on lists.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        let Ok(other) = other.extract::<PyRef<'_, PyAnnotation>>() else {
            return false;
        };
        self.onset == other.onset && self.duration == other.duration && self.text == other.text
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.onset.to_bits().hash(&mut hasher);
        self.duration.map(|d| d.to_bits()).hash(&mut hasher);
        self.text.hash(&mut hasher);
        hasher.finish()
    }

    /// Ordering follows onset, then text, matching the order annotations are returned in.
    fn __lt__(&self, other: PyRef<'_, PyAnnotation>) -> bool {
        (self.onset, &self.text) < (other.onset, &other.text)
    }

    /// Support `copy` and `pickle`, so annotations survive multiprocessing.
    fn __reduce__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let args = (slf.onset, slf.text.clone(), slf.duration)
            .into_pyobject(py)?
            .into_any()
            .unbind();
        let cls = slf.into_pyobject(py)?.get_type().into_any().unbind();
        Ok((cls, args))
    }
}

impl From<&edfarray_core::annotation::Annotation> for PyAnnotation {
    fn from(ann: &edfarray_core::annotation::Annotation) -> Self {
        PyAnnotation {
            onset: ann.onset,
            duration: ann.duration,
            text: ann.text.clone(),
        }
    }
}
