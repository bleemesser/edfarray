use pyo3::prelude::*;

/// A single annotation from the EDF+ file.
#[pyclass(frozen, name = "Annotation", from_py_object)]
#[derive(Clone)]
pub struct PyAnnotation {
    #[pyo3(get)]
    pub onset: f64,
    #[pyo3(get)]
    pub duration: Option<f64>,
    #[pyo3(get)]
    pub text: String,
}

#[pymethods]
impl PyAnnotation {
    fn __repr__(&self) -> String {
        match self.duration {
            Some(d) => format!(
                "Annotation(onset={}, duration={}, text={:?})",
                self.onset, d, self.text
            ),
            None => format!(
                "Annotation(onset={}, text={:?})",
                self.onset, self.text
            ),
        }
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
