use chrono::{Datelike, Timelike};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use edfarray_core::file::EdfFile;
use edfarray_core::header::Sex;
use edfarray_core::mmap::ScanMode;
use edfarray_core::proxy::ReadStrategy;

use crate::annotations::PyAnnotation;
use crate::errors::{closed_file_err, invalid_argument_err, to_py_err};
use crate::group::PySignalGroup;
use crate::proxy_2d::{PyProxy2D, parse_pad_mode};
use crate::proxy_3d::PyProxy3D;
use crate::signal::PySignal;
use crate::writer::parse_variant;

/// An open EDF/EDF+ file.
#[gen_stub_pyclass]
#[pyclass(name = "EdfFile", module = "edfarray._core")]
pub struct PyEdfFile {
    inner: Option<EdfFile>,
}

impl PyEdfFile {
    /// The open file, or `ClosedFileError` if `close()` has been called.
    fn get(&self) -> PyResult<&EdfFile> {
        self.inner.as_ref().ok_or_else(closed_file_err)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEdfFile {
    /// Open an EDF/EDF+/BDF file.
    ///
    /// `variant` forces the file variant instead of trusting the auto-detected
    /// one, for files that omit or misreport the EDF+ "+C"/"+D" marker. It only
    /// controls the plain/"+C"/"+D" distinction; an override that changes the
    /// EDF-vs-BDF sample size (set by the version field) raises `ValueError`.
    ///
    /// By default the annotation index is built by a background scan started at open. That
    /// scan reads every data record, so for very large files it competes with your own reads
    /// for page cache. Pass `scan_annotations=False` to defer it until annotations are first
    /// accessed, at which point it runs on the calling thread.
    #[new]
    #[pyo3(signature = (path, variant=None, scan_annotations=true))]
    fn new(
        py: Python<'_>,
        path: &str,
        variant: Option<&str>,
        scan_annotations: bool,
    ) -> PyResult<Self> {
        let variant = variant.map(parse_variant).transpose()?;
        let scan_mode = if scan_annotations {
            ScanMode::Eager
        } else {
            ScanMode::Lazy
        };
        let inner = py
            .detach(|| EdfFile::open_with_options(path, variant, scan_mode))
            .map_err(to_py_err)?;
        Ok(PyEdfFile { inner: Some(inner) })
    }

    fn __enter__<'py>(slf: PyRef<'py, Self>) -> PyRef<'py, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&mut self, _args: Bound<'_, pyo3::types::PyTuple>) {
        self.inner = None;
    }

    /// Release the underlying memory-mapped file. Idempotent.
    fn close(&mut self) {
        self.inner = None;
    }

    /// Whether `close()` has been called.
    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_none()
    }

    fn __repr__(&self) -> String {
        let Some(inner) = self.inner.as_ref() else {
            return "EdfFile(closed)".to_string();
        };
        format!(
            "EdfFile(variant={:?}, signals={}, records={}, duration={}s)",
            inner.variant().to_string(),
            inner.num_signals(),
            inner.num_records(),
            inner.duration()
        )
    }

    /// Total number of signals, including annotation channels.
    #[getter]
    fn num_signals(&self) -> PyResult<usize> {
        Ok(self.get()?.num_signals())
    }

    /// Number of data records.
    #[getter]
    fn num_records(&self) -> PyResult<usize> {
        Ok(self.get()?.num_records())
    }

    /// Duration of each data record in seconds.
    #[getter]
    fn record_duration(&self) -> PyResult<f64> {
        Ok(self.get()?.record_duration())
    }

    /// Total recording duration in seconds.
    #[getter]
    fn duration(&self) -> PyResult<f64> {
        Ok(self.get()?.duration())
    }

    /// File variant: "EDF", "EDF+C", or "EDF+D".
    #[getter]
    fn variant(&self) -> PyResult<String> {
        Ok(self.get()?.variant().to_string())
    }

    /// Raw 80-byte patient identification field.
    #[getter]
    fn patient_id(&self) -> PyResult<&str> {
        Ok(&self.get()?.header().patient_id)
    }

    /// Raw 80-byte recording identification field.
    #[getter]
    fn recording_id(&self) -> PyResult<&str> {
        Ok(&self.get()?.header().recording_id)
    }

    /// Recording start time as `datetime.datetime`, or raw string if anonymized.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "datetime.datetime | builtins.str", imports = ("builtins", "datetime")))]
    fn start_datetime<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let mdt = &self.get()?.header().start_datetime;
        match mdt.as_datetime() {
            Some(dt) => {
                let datetime_mod = py.import("datetime")?;
                let datetime_cls = datetime_mod.getattr("datetime")?;
                let result = datetime_cls.call1((
                    dt.year(),
                    dt.month(),
                    dt.day(),
                    dt.hour(),
                    dt.minute(),
                    dt.second(),
                ))?;
                Ok(result.unbind())
            }
            None => {
                let s = format!("{} {}", mdt.raw_date(), mdt.raw_time());
                Ok(s.into_pyobject(py)?.into_any().unbind())
            }
        }
    }

    /// Patient name parsed from the identification field, or None.
    #[getter]
    fn patient_name(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.patient().name.as_deref())
    }

    /// Hospital patient code, or None.
    #[getter]
    fn patient_code(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.patient().code.as_deref())
    }

    /// "M" or "F", or None if unknown.
    #[getter]
    fn patient_sex(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.patient().sex.map(|s| match s {
            Sex::Male => "M",
            Sex::Female => "F",
        }))
    }

    /// Returns `datetime.date` if parseable, a raw string if anonymized, or `None` if absent.
    #[getter]
    fn patient_birthdate<'py>(&self, py: Python<'py>) -> PyResult<Option<Py<PyAny>>> {
        use edfarray_core::header::MaybeDate;
        match &self.get()?.patient().birthdate {
            Some(MaybeDate::Parsed(date)) => {
                let datetime_mod = py.import("datetime")?;
                let date_cls = datetime_mod.getattr("date")?;
                let result = date_cls.call1((date.year(), date.month(), date.day()))?;
                Ok(Some(result.unbind()))
            }
            Some(MaybeDate::Raw(s)) => Ok(Some(s.clone().into_pyobject(py)?.into_any().unbind())),
            None => Ok(None),
        }
    }

    /// Additional patient information, or None.
    #[getter]
    fn patient_additional(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.patient().additional.as_deref())
    }

    /// Hospital administration code, or None.
    #[getter]
    fn admin_code(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.recording().admin_code.as_deref())
    }

    /// Technician or investigator code, or None.
    #[getter]
    fn technician(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.recording().technician.as_deref())
    }

    /// Equipment code, or None.
    #[getter]
    fn equipment(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.recording().equipment.as_deref())
    }

    /// Additional recording information, or None.
    #[getter]
    fn recording_additional(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.recording().additional.as_deref())
    }

    /// All non-timekeeping annotations, sorted by onset.
    #[getter]
    fn annotations(&self) -> PyResult<Vec<PyAnnotation>> {
        Ok(self
            .get()?
            .annotations()
            .iter()
            .map(PyAnnotation::from)
            .collect())
    }

    /// Annotations with onset strictly before `t`.
    /// Uses binary search for efficiency.
    pub fn annotations_before(&self, t: f64) -> PyResult<Vec<PyAnnotation>> {
        Ok(self
            .get()?
            .annotations_before(t)
            .iter()
            .map(PyAnnotation::from)
            .collect())
    }

    /// Annotations with onset >= `t`.
    /// Uses binary search for efficiency.
    pub fn annotations_after(&self, t: f64) -> PyResult<Vec<PyAnnotation>> {
        Ok(self
            .get()?
            .annotations_after(t)
            .iter()
            .map(PyAnnotation::from)
            .collect())
    }

    /// Annotations with onset in [start, end).
    /// Uses binary search for efficiency.
    pub fn annotations_in_range(&self, start: f64, end: f64) -> PyResult<Vec<PyAnnotation>> {
        Ok(self
            .get()?
            .annotations_in_range(start, end)
            .iter()
            .map(PyAnnotation::from)
            .collect())
    }

    /// Filter annotations by text content.
    ///
    /// If `regex` is False, returns annotations whose text contains the query
    /// as a case-insensitive substring.
    ///
    /// If `regex` is True, returns annotations whose text matches the query
    /// as a case-insensitive regex pattern.
    ///
    /// Raises `ValueError` if the regex pattern is invalid.
    #[pyo3(signature = (query, regex=false))]
    fn filter_annotations(&self, query: &str, regex: bool) -> PyResult<Vec<PyAnnotation>> {
        let anns = self
            .get()?
            .filter_annotations(query, regex)
            .map_err(to_py_err)?;
        Ok(anns.iter().map(PyAnnotation::from).collect())
    }

    /// Annotations whose text matches `query`, as a named alias of `filter_annotations`.
    ///
    /// Provided so the epoch-extraction path reads clearly: `f.events("Spindle")` feeds
    /// straight into `f.extract_epochs(...)`. Same matching rules: case-insensitive substring,
    /// or case-insensitive regex when `regex=True`.
    #[pyo3(signature = (query, regex=false))]
    fn events(&self, query: &str, regex: bool) -> PyResult<Vec<PyAnnotation>> {
        self.filter_annotations(query, regex)
    }

    /// Annotations whose text exactly matches `text` (case-sensitive).
    pub fn annotations_by_text(&self, text: &str) -> PyResult<Vec<PyAnnotation>> {
        Ok(self
            .get()?
            .annotations_by_text(text)
            .iter()
            .map(PyAnnotation::from)
            .collect())
    }

    /// Return all signals whose label matches `label`.
    ///
    /// If `exact` is `False` (default), performs a case-insensitive substring match.
    /// If `exact` is `True`, performs a case-sensitive exact equality match.
    ///
    /// Searches all signals including annotation signals.
    #[pyo3(signature = (label, exact=false))]
    fn find_all_signals(&self, label: &str, exact: bool) -> PyResult<Vec<usize>> {
        Ok(self.get()?.find_all_signals(label, exact))
    }

    /// Parse warnings accumulated during file open.
    #[getter]
    fn warnings(&self) -> PyResult<Vec<String>> {
        Ok(self.get()?.warnings())
    }

    /// Raw header fields as a dict.
    ///
    /// Builds a fresh dict on each call, so it is a method rather than a property.
    #[gen_stub(override_return_type(type_repr = "dict[builtins.str, typing.Any]", imports = ("builtins", "typing")))]
    fn header<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("version", &self.get()?.header().version)?;
        dict.set_item("patient_id", &self.get()?.header().patient_id)?;
        dict.set_item("recording_id", &self.get()?.header().recording_id)?;
        dict.set_item("num_signals", self.get()?.num_signals())?;
        dict.set_item("num_records", self.get()?.num_records())?;
        dict.set_item("record_duration", self.get()?.record_duration())?;
        dict.set_item("duration", self.get()?.duration())?;
        dict.set_item("variant", self.get()?.variant().to_string())?;
        Ok(dict)
    }

    /// Get a signal by index or label.
    ///
    /// `cache_capacity` enables an LRU cache of decoded physical records for
    /// this signal. The unit is a count of EDF data records (not samples or
    /// bytes); one cached record holds `samples_per_record` float64 values, so
    /// the cache costs roughly `cache_capacity * samples_per_record * 8` bytes.
    /// 0 (the default) disables it.
    ///
    /// Leave it at 0 for one-pass or strictly forward reads -- the OS page cache
    /// already serves the raw bytes, so a cache only pays off when you re-decode
    /// the *same* records (overlapping windows, back-and-forth seeks, repeated
    /// slices). A good starting capacity is a few records more than your largest
    /// repeated window spans, i.e. `ceil(window_samples / samples_per_record) + 2`.
    /// The cache only accelerates physical reads -- `to_digital()` always
    /// re-decodes from the memory map. Caching is per-`Signal`: re-fetching from
    /// `signal()` starts fresh.
    ///
    /// `strategy` overrides how bytes are fetched: `"auto"` (default) streams large reads that
    /// are not already cached and uses the memory map otherwise, `"mmap"` always maps, and
    /// `"stream"` always reads sequentially through a bounded buffer.
    #[pyo3(signature = (idx_or_label, cache_capacity=0, strategy=None))]
    fn signal(
        &self,
        #[gen_stub(override_type(type_repr = "builtins.int | builtins.str"))] idx_or_label: &Bound<
            '_,
            PyAny,
        >,
        cache_capacity: usize,
        strategy: Option<&str>,
    ) -> PyResult<PySignal> {
        let proxy = if let Ok(idx) = idx_or_label.extract::<usize>() {
            self.get()?.signal(idx).map_err(to_py_err)?
        } else if let Ok(label) = idx_or_label.extract::<String>() {
            self.get()?.signal_by_label(&label).map_err(to_py_err)?
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "signal() argument must be int or str",
            ));
        };
        let proxy = match strategy {
            None => proxy,
            Some(s) => proxy.with_strategy(parse_strategy(s)?),
        };
        let proxy = if cache_capacity > 0 {
            proxy.with_cache(cache_capacity)
        } else {
            proxy
        };
        Ok(PySignal::new(proxy))
    }

    /// Labels of all signals in the file.
    fn signal_labels(&self) -> PyResult<Vec<&str>> {
        Ok(self.get()?.signal_labels())
    }

    /// Indices of all non-annotation (ordinary) signals.
    fn ordinary_signal_indices(&self) -> PyResult<Vec<usize>> {
        Ok(self.get()?.ordinary_signal_indices())
    }

    /// Read a page of physical data for multiple signals over a time range.
    ///
    /// Returns a list of numpy arrays, one per signal. Signals with different
    /// sample rates will produce arrays of different lengths.
    ///
    /// If `signal_indices` is None, reads all ordinary (non-annotation) signals.
    ///
    /// When `use_time` is false (default), time parameters are converted to flat
    /// sample indices. For EDF+D files with gaps, set `use_time=true` to resolve
    /// the time range using actual record onset times.
    #[pyo3(signature = (start_sec, end_sec, signal_indices=None, use_time=false))]
    fn read_page<'py>(
        &self,
        py: Python<'py>,
        start_sec: f64,
        end_sec: f64,
        signal_indices: Option<Vec<usize>>,
        use_time: bool,
    ) -> PyResult<Vec<Bound<'py, numpy::PyArray1<f64>>>> {
        let inner = self.get()?;
        let indices = signal_indices.unwrap_or_else(|| inner.ordinary_signal_indices());
        let buffers = py
            .detach(|| inner.read_page(&indices, start_sec, end_sec, use_time))
            .map_err(to_py_err)?;
        let mut arrays = Vec::with_capacity(buffers.len());
        for buf in buffers {
            let array = numpy::PyArray1::from_vec(py, buf);
            arrays.push(array);
        }
        Ok(arrays)
    }

    /// Whether the background annotation scan has completed.
    #[getter]
    fn annotations_ready(&self) -> PyResult<bool> {
        Ok(self.get()?.annotations_ready())
    }

    /// Progress of the background annotation scan: (records_scanned, total_records).
    #[getter]
    fn scan_progress(&self) -> PyResult<(usize, usize)> {
        Ok(self.get()?.scan_progress())
    }

    /// Build a 2D proxy from a `SignalGroup`.
    ///
    /// `pad_mode` controls reads past a channel's valid length when the group
    /// is `"open"` (mixed sample rates). Accepts the string `"raise"` (default),
    /// `"nan"`, `"zero"`, `"edge"`, or a numeric scalar (interpreted as
    /// `Value(x)`). `None` is treated as `"raise"`.
    #[pyo3(signature = (group, pad_mode=None))]
    fn proxy_2d(
        &self,
        group: &PySignalGroup,
        pad_mode: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyProxy2D> {
        let mode = parse_pad_mode(pad_mode.as_ref())?;
        let proxy = self
            .get()?
            .proxy_2d(group.inner().clone(), mode)
            .map_err(to_py_err)?;
        Ok(PyProxy2D::new(proxy))
    }

    /// Build a 3D proxy from a rectangular `SignalGroup`.
    ///
    /// Requires `group.kind == "rectangular"` (all channels share a sample
    /// rate). Use `signal_groups()` to discover eligible groups, or
    /// `signal_group(...)` to construct one from specific indices.
    fn proxy_3d(&self, group: &PySignalGroup) -> PyResult<PyProxy3D> {
        let proxy = self
            .get()?
            .proxy_3d(group.inner().clone())
            .map_err(to_py_err)?;
        Ok(PyProxy3D::new(proxy))
    }

    /// Classify an arbitrary list of file-level signal indices into a
    /// `SignalGroup`. Use this when you want a group that's a subset of (or
    /// crosses) the file's natural rate-based groupings.
    fn signal_group(&self, indices: Vec<usize>) -> PyResult<PySignalGroup> {
        let g = edfarray_core::group::SignalGroup::from_indices(self.get()?.header(), &indices)
            .map_err(to_py_err)?;
        Ok(PySignalGroup::new(g))
    }

    /// Partition all ordinary signals into groups by sample rate.
    ///
    /// Returns a list of `SignalGroup` objects, each carrying its sample rate,
    /// structural kind, sample-count range, and whether it covers every
    /// ordinary signal in the file. Sub-Hz precision is preserved.
    fn signal_groups(&self) -> PyResult<Vec<PySignalGroup>> {
        Ok(self
            .get()?
            .signal_groups()
            .into_iter()
            .map(PySignalGroup::new)
            .collect())
    }

    /// Write this file to `path`, optionally transcoding to a different variant.
    ///
    /// Annotations and ordinary signals are copied; the destination's annotation
    /// channel is rebuilt from parsed annotations rather than copied verbatim.
    /// `variant` may be one of "EDF", "EDF+C", "EDF+D", "BDF", "BDF+C", "BDF+D";
    /// if omitted, uses the source variant.
    ///
    /// Transcoding caveats:
    /// - EDF+D to EDF+D preserves the source record onsets, so gaps survive the
    ///   copy. Transcoding to any non-`+D` variant flattens timing: per-record
    ///   onsets/gaps are replaced by uniform `record_idx * record_duration` timing.
    /// - Because the annotation channel is rebuilt from parsed annotations,
    ///   transcoding to a plain (non-"+") EDF/BDF variant drops all annotations,
    ///   since plain variants have no annotation channel.
    /// - Downconverting sample size (e.g. BDF 24-bit to EDF 16-bit) clamps the
    ///   digital range and re-encodes from physical values, losing precision.
    #[pyo3(signature = (path, variant=None))]
    fn write_to(&self, path: &str, variant: Option<&str>) -> PyResult<()> {
        use edfarray_core::header::EdfVariant;
        let target = match variant {
            None => None,
            Some("EDF") => Some(EdfVariant::Edf),
            Some("EDF+C") => Some(EdfVariant::EdfPlusC),
            Some("EDF+D") => Some(EdfVariant::EdfPlusD),
            Some("BDF") => Some(EdfVariant::Bdf),
            Some("BDF+C") => Some(EdfVariant::BdfPlusC),
            Some("BDF+D") => Some(EdfVariant::BdfPlusD),
            Some(other) => {
                return Err(invalid_argument_err(format!("unknown variant {:?}", other)));
            }
        };
        self.get()?.write_to(path, target).map_err(to_py_err)
    }

    /// Read a page of digital (raw int32) data for multiple signals over a time range.
    ///
    /// If `signal_indices` is None, reads all ordinary (non-annotation) signals.
    ///
    /// When `use_time` is false (default), time parameters are converted to flat
    /// sample indices. For EDF+D files with gaps, set `use_time=true` to resolve
    /// the time range using actual record onset times.
    #[pyo3(signature = (start_sec, end_sec, signal_indices=None, use_time=false))]
    fn read_page_digital<'py>(
        &self,
        py: Python<'py>,
        start_sec: f64,
        end_sec: f64,
        signal_indices: Option<Vec<usize>>,
        use_time: bool,
    ) -> PyResult<Vec<Bound<'py, numpy::PyArray1<i32>>>> {
        let inner = self.get()?;
        let indices = signal_indices.unwrap_or_else(|| inner.ordinary_signal_indices());
        let buffers = py
            .detach(|| inner.read_page_digital(&indices, start_sec, end_sec, use_time))
            .map_err(to_py_err)?;
        let mut arrays = Vec::with_capacity(buffers.len());
        for buf in buffers {
            let array = numpy::PyArray1::from_vec(py, buf);
            arrays.push(array);
        }
        Ok(arrays)
    }

    /// Extract fixed windows around events as a dense `(n_epochs, n_channels, n_samples)`
    /// block, decoded in parallel and gap-aware (EDF+D onsets are honored).
    ///
    /// `events` accepts floats, `Annotation`s, a mixed sequence, a numpy float64 array, or a
    /// single value. Pass `events=None` together with `query` to extract around annotation
    /// text (same matching as `filter_annotations`).
    ///
    /// `group` selects channels: a `SignalGroup` or a sequence of signal indices. The group
    /// must be rectangular; the default is the largest rectangular group.
    ///
    /// `pad` governs epochs whose window runs off the file or straddles an EDF+D gap: `"drop"`
    /// (default) omits them, `"nan"`/`"zero"`/a number/`"edge"` keep and fill them (marked
    /// `valid=False`), `"raise"` errors on the first offender.
    #[pyo3(signature = (events, *, pre, post, group=None, pad=None, query=None, regex=false))]
    #[allow(clippy::too_many_arguments)]
    fn extract_epochs(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(
            type_repr = "builtins.float | builtins.Sequence[builtins.float] | Annotation | builtins.Sequence[Annotation] | builtins.NoneType"
        ))]
        events: &Bound<'_, PyAny>,
        pre: f64,
        post: f64,
        group: Option<&Bound<'_, PyAny>>,
        #[gen_stub(override_type(
            type_repr = "builtins.str | builtins.float | builtins.NoneType"
        ))]
        pad: Option<&Bound<'_, PyAny>>,
        query: Option<&str>,
        regex: bool,
    ) -> PyResult<crate::epoch::PyEpochs> {
        let inner = self.get()?;
        if query.is_some() && !events.is_none() {
            return Err(invalid_argument_err("pass events or query, not both"));
        }
        let group = crate::epoch::resolve_group(inner, group)?;
        let events_owned = if let Some(q) = query {
            inner
                .filter_annotations(q, regex)
                .map_err(to_py_err)?
                .iter()
                .map(|a| a.onset)
                .collect()
        } else {
            crate::epoch::event_onsets(events)?
        };
        let pad_policy = crate::epoch::parse_epoch_pad(pad)?;
        let (onsets, valid, data, dropped, n) = py.detach(|| {
            crate::epoch::plan_and_extract(inner, &group, &events_owned, pre, post, pad_policy)
        })?;
        let labels = group
            .indices()
            .iter()
            .map(|&i| inner.header().signals[i].label.clone())
            .collect();
        Ok(crate::epoch::PyEpochs {
            data,
            shape: (valid.len(), group.indices().len(), n),
            onsets,
            labels,
            sample_rate: group.sample_rate().unwrap_or(0.0),
            valid,
            dropped,
        })
    }

    /// Planned epoch windows without reading data: `([(onset, s_start, s_end)], valid)`.
    ///
    /// The window table is the same one `extract_epochs` decodes, so callers can inspect or
    /// clip what would be extracted before paying for the reads.
    #[pyo3(signature = (events, *, pre, post, group=None))]
    #[gen_stub(override_return_type(type_repr = "tuple[builtins.list[builtins.tuple[builtins.float, builtins.int, builtins.int]], numpy.typing.NDArray[numpy.bool_]]", imports = ("builtins", "numpy")))]
    fn epoch_windows<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(
            type_repr = "builtins.float | builtins.Sequence[builtins.float] | Annotation | builtins.Sequence[Annotation]"
        ))]
        events: &Bound<'_, PyAny>,
        pre: f64,
        post: f64,
        group: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<EpochWindows<'py>> {
        let inner = self.get()?;
        let group = crate::epoch::resolve_group(inner, group)?;
        let events_owned = crate::epoch::event_onsets(events)?;
        let plan = py.detach(|| {
            edfarray_core::epoch::plan_epochs(inner, &group, &events_owned, pre, post)
                .map_err(to_py_err)
        })?;
        let table = plan
            .windows
            .iter()
            .map(|w| (w.onset, w.s_start, w.s_end))
            .collect();
        Ok((
            table,
            numpy::PyArray1::from_iter(py, plan.valid.iter().copied()),
        ))
    }
}

/// Planned-window table returned by `epoch_windows`: one `(onset, s_start, s_end)` per event,
/// plus the per-window validity mask.
type EpochWindows<'py> = (Vec<(f64, usize, usize)>, Bound<'py, numpy::PyArray1<bool>>);

/// Lightweight metadata extracted from an EDF/EDF+ file header without
/// scanning data records or building an annotation index.
///
/// Returns a dict with keys:
/// `variant`, `num_signals`, `num_records`, `record_duration`, `duration`,
/// `patient_id`, `recording_id`, `signal_labels`, `sample_rates`.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn inspect<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let meta = EdfFile::inspect(path).map_err(to_py_err)?;

    let dict = PyDict::new(py);
    dict.set_item("variant", meta.variant.to_string())?;
    dict.set_item("num_signals", meta.num_signals)?;
    dict.set_item("num_records", meta.num_records)?;
    dict.set_item("record_duration", meta.record_duration)?;
    dict.set_item("duration", meta.duration)?;
    dict.set_item("patient_id", meta.patient_id)?;
    dict.set_item("recording_id", meta.recording_id)?;
    dict.set_item("signal_labels", meta.signal_labels)?;
    dict.set_item("sample_rates", meta.sample_rates)?;
    Ok(dict)
}

/// Parse the `strategy` argument accepted by `EdfFile.signal`.
fn parse_strategy(value: &str) -> PyResult<ReadStrategy> {
    match value {
        "auto" => Ok(ReadStrategy::Auto),
        "mmap" => Ok(ReadStrategy::Mmap),
        "stream" => Ok(ReadStrategy::Stream),
        other => Err(invalid_argument_err(format!(
            "strategy must be 'auto', 'mmap', or 'stream', got {other:?}"
        ))),
    }
}
