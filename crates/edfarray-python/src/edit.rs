use chrono::{NaiveDate, NaiveDateTime, NaiveTime};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

use edfarray_core::edit::{self, AnonymizeOptions, HeaderEdit};

use crate::errors::{invalid_argument_err, to_py_err};

/// Replace header identification fields in place.
///
/// The function rewrites only the fixed-width header fields. It never changes the record
/// data, so an edit of a multi-gigabyte recording costs a few hundred bytes of I/O. To leave
/// a field unchanged, pass `None` for it. The function validates every value before it
/// writes anything, so a rejected edit leaves the file unchanged.
///
/// The function returns a dict that maps each changed field name (`"patient_id"`,
/// `"recording_id"`, `"start_datetime"`) to `{"before": str, "after": str}`. The dict
/// does not include fields whose value did not change.
///
/// `start_datetime` is a `datetime.datetime` or a `datetime.date` (midnight). The function
/// writes its wall-clock fields as given and ignores any `tzinfo`. The header stores only
/// whole seconds.
///
/// If an `EdfFile`, or a signal or proxy taken from one, has `path` open, the function
/// raises `EdfFileError`. Before you edit the file, close it and drop those objects.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (path, patient_id=None, recording_id=None, start_datetime=None))]
pub fn edit_header<'py>(
    py: Python<'py>,
    path: &str,
    patient_id: Option<&str>,
    recording_id: Option<&str>,
    start_datetime: Option<&Bound<'_, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let edit = HeaderEdit {
        patient_id: patient_id.map(|s| s.to_string()),
        recording_id: recording_id.map(|s| s.to_string()),
        start_datetime: parse_datetime_opt(start_datetime)?,
    };
    let diff = py
        .detach(|| edit::edit_header(path, &edit))
        .map_err(to_py_err)?;

    let out = PyDict::new(py);
    for (key, change) in [
        ("patient_id", diff.patient_id),
        ("recording_id", diff.recording_id),
        ("start_datetime", diff.start_datetime),
    ] {
        if let Some(change) = change {
            let entry = PyDict::new(py);
            entry.set_item("before", change.before)?;
            entry.set_item("after", change.after)?;
            out.set_item(key, entry)?;
        }
    }
    Ok(out)
}

/// Anonymize the patient and recording identification of a file in place.
///
/// The function replaces the patient name with a pseudonym. The admin code always becomes `X`.
/// The patient code, technician, and free-text subfields become `X` unless you keep them. All dates
/// (birthdate, recording start, header startdate) move by the same number of days. Thus the
/// age stays the same, but the calendar dates change.
///
/// - `seed`: makes the pseudonym and the date shift reproducible, so that the recordings of
///   one subject stay linkable across a corpus. The function computes the pseudonym from the
///   patient name, code, and birthdate. Thus per-session notes in the free-text subfield do
///   not split a subject into several pseudonyms. Without a seed, each call uses a new random
///   seed.
///
///   A reused seed is the re-identification key. A person who has the seed can compute the
///   pseudonym for a guessed name and find the date shift. Keep the seed secret. Never
///   publish it with the files that it anonymized.
/// - `pseudonym`: an explicit replacement for the patient name. The default is
///   `Subject-XXXXXXXX`.
/// - `date_shift_days`: an explicit shift. The default shift comes from the seed (less than
///   10 years).
/// - `keep_sex`/`keep_code`/`keep_technician`/`keep_equipment`/`keep_additional`: control
///   which subfields stay. By default, the function keeps sex and equipment, and clears all
///   other subfields.
/// - `dry_run`: compute, validate, and report everything, but write nothing. A dry run
///   rejects exactly the same input as a real run.
///
/// The function returns a dict with `pseudonym`, `date_shift_days`, `patient_id_before`,
/// `patient_id_after`, `recording_id_before`, `recording_id_after`,
/// `start_datetime_before`, `start_datetime_after`, `scrubbed_terms`, and `dry_run`.
///
/// The function never rewrites signal labels or annotation text. This text can repeat the
/// original identity. After you anonymize a file and before you share it, run `audit()` with
/// the returned `scrubbed_terms`.
///
/// If `dry_run` is not set and an `EdfFile`, or a signal or proxy taken from one, has `path`
/// open, the function raises `EdfFileError`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (
    path,
    *,
    seed = None,
    pseudonym = None,
    date_shift_days = None,
    keep_sex = true,
    keep_code = false,
    keep_technician = false,
    keep_equipment = true,
    keep_additional = false,
    dry_run = false,
))]
#[allow(clippy::too_many_arguments)]
pub fn anonymize<'py>(
    py: Python<'py>,
    path: &str,
    seed: Option<&str>,
    pseudonym: Option<&str>,
    date_shift_days: Option<i32>,
    keep_sex: bool,
    keep_code: bool,
    keep_technician: bool,
    keep_equipment: bool,
    keep_additional: bool,
    dry_run: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = AnonymizeOptions {
        seed: seed.map(|s| s.to_string()),
        pseudonym: pseudonym.map(|s| s.to_string()),
        date_shift_days,
        keep_sex,
        keep_code,
        keep_technician,
        keep_equipment,
        keep_additional,
        dry_run,
    };
    let result = py
        .detach(|| edit::anonymize(path, &opts))
        .map_err(to_py_err)?;

    let out = PyDict::new(py);
    out.set_item("pseudonym", result.pseudonym)?;
    out.set_item("date_shift_days", result.date_shift_days)?;
    out.set_item("patient_id_before", result.patient_id_before)?;
    out.set_item("patient_id_after", result.patient_id_after)?;
    out.set_item("recording_id_before", result.recording_id_before)?;
    out.set_item("recording_id_after", result.recording_id_after)?;
    out.set_item("start_datetime_before", result.start_datetime_before)?;
    out.set_item("start_datetime_after", result.start_datetime_after)?;
    out.set_item("scrubbed_terms", result.scrubbed_terms)?;
    out.set_item("dry_run", dry_run)?;
    Ok(out)
}

/// Scan a file for strings that repeat the patient identity.
///
/// The function examines signal labels, transducers, prefiltering text, and annotation text.
/// By default, the function takes the search terms from the patient name/code and the
/// recording technician/admin code in the current header. If you use the default terms, run
/// `audit()` before `anonymize()`. To examine a file again after anonymization, pass `terms`
/// (the `scrubbed_terms` from the earlier anonymization).
///
/// The function returns a dict with `terms`, `clean`, and `hits`. Each hit has `location`,
/// `signal_index` (None for annotation text), `term`, and `excerpt`.
#[gen_stub_pyfunction]
#[pyfunction]
#[pyo3(signature = (path, terms=None))]
pub fn audit<'py>(
    py: Python<'py>,
    path: &str,
    terms: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let report = py
        .detach(|| match &terms {
            None => edit::audit(path),
            Some(terms) => edit::audit_terms(path, terms),
        })
        .map_err(to_py_err)?;

    let out = PyDict::new(py);
    let clean = report.is_clean();
    out.set_item("terms", report.terms)?;
    out.set_item("clean", clean)?;
    let hits = pyo3::types::PyList::empty(py);
    for hit in report.hits {
        let entry = PyDict::new(py);
        entry.set_item("location", hit.location)?;
        entry.set_item("signal_index", hit.signal_index)?;
        entry.set_item("term", hit.term)?;
        entry.set_item("excerpt", hit.excerpt)?;
        hits.append(entry)?;
    }
    out.set_item("hits", hits)?;
    Ok(out)
}

/// `None` stays `None` (meaning "leave this field alone"), unlike the writer's parser
/// which defaults to a concrete datetime.
fn parse_datetime_opt(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Option<NaiveDateTime>> {
    let Some(obj) = obj else { return Ok(None) };
    if obj.is_none() {
        return Ok(None);
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
        .ok_or_else(|| invalid_argument_err("invalid start_datetime: bad date"))?;
    let time = NaiveTime::from_hms_opt(hour, minute, second)
        .ok_or_else(|| invalid_argument_err("invalid start_datetime: bad time"))?;
    Ok(Some(NaiveDateTime::new(date, time)))
}
