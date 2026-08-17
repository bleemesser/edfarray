use chrono::{Datelike, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use edfarray_core::annotation::Annotation;
use edfarray_core::header::EdfVariant;
use edfarray_core::writer::{EdfWriter, WriterSignal, WriterSpec, write_edf};

use crate::annotations::PyAnnotation;
use crate::errors::to_py_err;

/// Per-signal description used by [`EdfWriter`] and [`write_edf`].
///
/// `physical_min`/`physical_max` define the unit range. `digital_min`/`digital_max`
/// define the integer range used in the binary file (16-bit for EDF, 24-bit for BDF).
#[gen_stub_pyclass]
#[pyclass(name = "WriterSignal", from_py_object)]
#[derive(Clone)]
pub struct PyWriterSignal {
    inner: WriterSignal,
}

impl PyWriterSignal {
    pub(crate) fn into_inner(self) -> WriterSignal {
        self.inner
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyWriterSignal {
    #[new]
    #[pyo3(signature = (
        label,
        physical_dimension,
        physical_min,
        physical_max,
        digital_min,
        digital_max,
        samples_per_record,
        transducer = String::new(),
        prefiltering = String::new(),
        reserved = String::new(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        label: String,
        physical_dimension: String,
        physical_min: f64,
        physical_max: f64,
        digital_min: i32,
        digital_max: i32,
        samples_per_record: usize,
        transducer: String,
        prefiltering: String,
        reserved: String,
    ) -> Self {
        PyWriterSignal {
            inner: WriterSignal {
                label,
                transducer,
                physical_dimension,
                physical_min,
                physical_max,
                digital_min,
                digital_max,
                prefiltering,
                samples_per_record,
                reserved,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "WriterSignal(label={:?}, samples_per_record={}, range_phys=[{}, {}], range_dig=[{}, {}])",
            self.inner.label,
            self.inner.samples_per_record,
            self.inner.physical_min,
            self.inner.physical_max,
            self.inner.digital_min,
            self.inner.digital_max,
        )
    }
}

pub(crate) fn parse_variant(s: &str) -> PyResult<EdfVariant> {
    match s {
        "EDF" => Ok(EdfVariant::Edf),
        "EDF+C" => Ok(EdfVariant::EdfPlusC),
        "EDF+D" => Ok(EdfVariant::EdfPlusD),
        "BDF" => Ok(EdfVariant::Bdf),
        "BDF+C" => Ok(EdfVariant::BdfPlusC),
        "BDF+D" => Ok(EdfVariant::BdfPlusD),
        other => Err(PyValueError::new_err(format!(
            "unknown variant {:?}; expected one of EDF, EDF+C, EDF+D, BDF, BDF+C, BDF+D",
            other
        ))),
    }
}

pub(crate) fn parse_start_datetime(obj: Option<&Bound<'_, PyAny>>) -> PyResult<NaiveDateTime> {
    let Some(obj) = obj else {
        // Default: epoch start (the writer encodes integer seconds; sub-second
        // precision is not preserved in the header anyway).
        return Ok(NaiveDateTime::new(
            NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
            NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
        ));
    };
    if obj.is_none() {
        return Ok(NaiveDateTime::new(
            NaiveDate::from_ymd_opt(2000, 1, 1).unwrap(),
            NaiveTime::from_hms_opt(0, 0, 0).unwrap(),
        ));
    }
    let year: i32 = obj.getattr("year")?.extract()?;
    let month: u32 = obj.getattr("month")?.extract()?;
    let day: u32 = obj.getattr("day")?.extract()?;
    let hour: u32 = obj
        .getattr("hour")
        .and_then(|v| v.extract())
        .unwrap_or(0u32);
    let minute: u32 = obj
        .getattr("minute")
        .and_then(|v| v.extract())
        .unwrap_or(0u32);
    let second: u32 = obj
        .getattr("second")
        .and_then(|v| v.extract())
        .unwrap_or(0u32);
    let date = NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| PyValueError::new_err("invalid start_datetime: bad date"))?;
    let time = NaiveTime::from_hms_opt(hour, minute, second)
        .ok_or_else(|| PyValueError::new_err("invalid start_datetime: bad time"))?;
    Ok(NaiveDateTime::new(date, time))
}

pub(crate) fn build_spec(
    variant: &str,
    record_duration: f64,
    signals: Vec<WriterSignal>,
    start_datetime: Option<&Bound<'_, PyAny>>,
    patient_id: Option<String>,
    recording_id: Option<String>,
    annotation_bytes_per_record: Option<usize>,
) -> PyResult<WriterSpec> {
    Ok(WriterSpec {
        variant: parse_variant(variant)?,
        patient_id: patient_id.unwrap_or_else(|| "X X X X".into()),
        recording_id: recording_id.unwrap_or_else(|| "Startdate X X X X".into()),
        start_datetime: parse_start_datetime(start_datetime)?,
        record_duration_secs: record_duration,
        signals,
        annotation_bytes_per_record,
    })
}

pub(crate) fn anns_to_core(anns: &[PyAnnotation]) -> Vec<Annotation> {
    anns.iter()
        .map(|a| Annotation {
            onset: a.onset,
            duration: a.duration,
            text: a.text.clone(),
        })
        .collect()
}

/// Streaming EDF/BDF writer.
///
/// Use as a context manager (`with edfarray.EdfWriter(...) as w:`) or call
/// `.finish()` explicitly. `__exit__` calls `finish()` automatically.
#[gen_stub_pyclass]
#[pyclass(name = "EdfWriter", unsendable)]
pub struct PyEdfWriter {
    inner: Option<EdfWriter>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEdfWriter {
    #[new]
    #[pyo3(signature = (
        path,
        *,
        variant,
        record_duration,
        signals,
        start_datetime = None,
        patient_id = None,
        recording_id = None,
        annotation_bytes_per_record = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        path: String,
        variant: &str,
        record_duration: f64,
        signals: Vec<PyWriterSignal>,
        start_datetime: Option<&Bound<'_, PyAny>>,
        patient_id: Option<String>,
        recording_id: Option<String>,
        annotation_bytes_per_record: Option<usize>,
    ) -> PyResult<Self> {
        let spec = build_spec(
            variant,
            record_duration,
            signals
                .into_iter()
                .map(PyWriterSignal::into_inner)
                .collect(),
            start_datetime,
            patient_id,
            recording_id,
            annotation_bytes_per_record,
        )?;
        let writer = EdfWriter::create(&path, spec).map_err(to_py_err)?;
        Ok(PyEdfWriter {
            inner: Some(writer),
        })
    }

    fn __enter__<'py>(slf: PyRef<'py, Self>) -> PyRef<'py, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, _args: Bound<'_, pyo3::types::PyTuple>) -> PyResult<()> {
        if let Some(w) = self.inner.take() {
            w.finish().map_err(to_py_err)?;
        }
        Ok(())
    }

    /// Queue an annotation to be embedded in the next written record.
    fn add_annotation(&mut self, annotation: PyAnnotation) -> PyResult<()> {
        let w = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("EdfWriter has been finished"))?;
        w.add_annotation(Annotation {
            onset: annotation.onset,
            duration: annotation.duration,
            text: annotation.text,
        });
        Ok(())
    }

    /// Write one data record from physical (float) values.
    ///
    /// `physical` is a list of 1D float64 numpy arrays (one per user signal),
    /// each of length `samples_per_record`. `annotations` is an optional list
    /// of annotations to embed in this record's annotation channel (only valid
    /// for `+C`/`+D` variants).
    #[pyo3(signature = (physical, annotations=None))]
    fn write_record(
        &mut self,
        py: Python<'_>,
        physical: Vec<PyReadonlyArray1<f64>>,
        annotations: Option<Vec<PyAnnotation>>,
    ) -> PyResult<()> {
        let w = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("EdfWriter has been finished"))?;
        let slices: Vec<&[f64]> = physical
            .iter()
            .map(|arr| arr.as_slice())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| {
                PyTypeError::new_err(format!("physical arrays must be contiguous: {e}"))
            })?;
        let anns_owned = annotations.map(|a| anns_to_core(&a)).unwrap_or_default();
        py.detach(|| w.write_record_with_annotations(&slices, &anns_owned))
            .map_err(to_py_err)
    }

    /// Finalize the file: flush buffers and patch `num_records` in the header.
    /// After calling, the writer is no longer usable.
    fn finish(&mut self) -> PyResult<()> {
        if let Some(w) = self.inner.take() {
            w.finish().map_err(to_py_err)?;
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        if self.inner.is_some() {
            "EdfWriter(open)".to_string()
        } else {
            "EdfWriter(finished)".to_string()
        }
    }
}

/// Write a complete EDF/BDF file in one call.
///
/// `data` is a list of 1D float64 numpy arrays, one per user signal. Each
/// array's length must be `num_records * samples_per_record` and consistent
/// across signals.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(name = "write_edf", signature = (
    path,
    *,
    variant,
    record_duration,
    signals,
    data,
    annotations = None,
    start_datetime = None,
    patient_id = None,
    recording_id = None,
    annotation_bytes_per_record = None,
))]
#[allow(clippy::too_many_arguments)]
pub fn write_edf_py(
    py: Python<'_>,
    path: String,
    variant: &str,
    record_duration: f64,
    signals: Vec<PyWriterSignal>,
    data: Vec<PyReadonlyArray1<f64>>,
    annotations: Option<Vec<PyAnnotation>>,
    start_datetime: Option<&Bound<'_, PyAny>>,
    patient_id: Option<String>,
    recording_id: Option<String>,
    annotation_bytes_per_record: Option<usize>,
) -> PyResult<()> {
    let spec = build_spec(
        variant,
        record_duration,
        signals
            .into_iter()
            .map(PyWriterSignal::into_inner)
            .collect(),
        start_datetime,
        patient_id,
        recording_id,
        annotation_bytes_per_record,
    )?;
    let slices: Vec<&[f64]> = data
        .iter()
        .map(|arr| arr.as_slice())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| PyTypeError::new_err(format!("data arrays must be contiguous: {e}")))?;
    let anns_owned = annotations.map(|a| anns_to_core(&a)).unwrap_or_default();
    py.detach(|| write_edf(&path, spec, &slices, &anns_owned))
        .map_err(to_py_err)
}

// suppress unused-imports lint when used only via gen_stub macros
#[allow(dead_code)]
fn _datetime_traits_used(dt: &NaiveDateTime) -> u32 {
    dt.year() as u32 + dt.hour()
}
