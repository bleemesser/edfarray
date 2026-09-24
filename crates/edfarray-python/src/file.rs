use chrono::{Datelike, Timelike};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

use edfarray_core::file::EdfFile;
use edfarray_core::header::Sex;
use edfarray_core::mmap::ScanMode;
use edfarray_core::proxy::ReadStrategy;

use crate::annotations::PyAnnotation;
use crate::errors::{closed_file_err, invalid_argument_err, out_of_range_err, to_py_err};
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
    /// The open file, or `ClosedFileError` if `close()` ran.
    fn get(&self) -> PyResult<&EdfFile> {
        self.inner.as_ref().ok_or_else(closed_file_err)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEdfFile {
    /// Open an EDF/EDF+/BDF file.
    ///
    /// `variant` sets the file variant and replaces the variant that edfarray detects. Use it
    /// for files that omit or misreport the EDF+ "+C"/"+D" marker. It controls only the
    /// plain/"+C"/"+D" distinction. The version field sets the EDF-vs-BDF sample size. If an
    /// override changes that sample size, the constructor raises `ValueError`.
    ///
    /// By default, a background scan starts at open and builds the annotation index. The scan
    /// reads every data record. For very large files, the scan competes with your own reads for
    /// the page cache. If you pass `scan_annotations=False`, the scan waits until the first
    /// access to the annotations. The scan then runs on the calling thread.
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

    /// Release the memory-mapped file. A second call has no effect.
    fn close(&mut self) {
        self.inner = None;
    }

    /// `True` if `close()` ran on this file.
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

    /// File variant: "EDF", "EDF+C", "EDF+D", "BDF", "BDF+C", or "BDF+D".
    #[getter]
    fn variant(&self) -> PyResult<String> {
        Ok(self.get()?.variant().to_string())
    }

    /// The raw 80-byte patient identification field.
    #[getter]
    fn patient_id(&self) -> PyResult<&str> {
        Ok(&self.get()?.header().patient_id)
    }

    /// The raw 80-byte recording identification field.
    #[getter]
    fn recording_id(&self) -> PyResult<&str> {
        Ok(&self.get()?.header().recording_id)
    }

    /// The recording start time as a `datetime.datetime`, or the raw string if edfarray cannot
    /// parse it.
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

    /// The patient name from the identification field, or None.
    #[getter]
    fn patient_name(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.patient().name.as_deref())
    }

    /// The hospital patient code, or None.
    #[getter]
    fn patient_code(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.patient().code.as_deref())
    }

    /// "M" or "F", or None if the sex is unknown.
    #[getter]
    fn patient_sex(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.patient().sex.map(|s| match s {
            Sex::Male => "M",
            Sex::Female => "F",
        }))
    }

    /// The patient birthdate.
    ///
    /// The value is a `datetime.date` if edfarray can parse the field. It is the raw string if
    /// edfarray cannot parse it, and `None` if the field is absent.
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

    /// The additional patient information, or None.
    #[getter]
    fn patient_additional(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.patient().additional.as_deref())
    }

    /// The hospital administration code, or None.
    #[getter]
    fn admin_code(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.recording().admin_code.as_deref())
    }

    /// The technician or investigator code, or None.
    #[getter]
    fn technician(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.recording().technician.as_deref())
    }

    /// The equipment code, or None.
    #[getter]
    fn equipment(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.recording().equipment.as_deref())
    }

    /// The additional recording information, or None.
    #[getter]
    fn recording_additional(&self) -> PyResult<Option<&str>> {
        Ok(self.get()?.recording().additional.as_deref())
    }

    /// All annotations, sorted by onset.
    ///
    /// The list does not include the timekeeping annotations that give the start time of each
    /// data record.
    #[getter]
    fn annotations(&self) -> PyResult<Vec<PyAnnotation>> {
        Ok(self
            .get()?
            .annotations()
            .iter()
            .map(PyAnnotation::from)
            .collect())
    }

    /// The annotations with an onset strictly before `t`.
    /// This method uses a binary search for speed.
    pub fn annotations_before(&self, t: f64) -> PyResult<Vec<PyAnnotation>> {
        Ok(self
            .get()?
            .annotations_before(t)
            .iter()
            .map(PyAnnotation::from)
            .collect())
    }

    /// The annotations with an onset >= `t`.
    /// This method uses a binary search for speed.
    pub fn annotations_after(&self, t: f64) -> PyResult<Vec<PyAnnotation>> {
        Ok(self
            .get()?
            .annotations_after(t)
            .iter()
            .map(PyAnnotation::from)
            .collect())
    }

    /// The annotations with an onset in [start, end).
    /// This method uses a binary search for speed.
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
    /// If `regex` is False, the method returns the annotations whose text contains the query
    /// as a case-insensitive substring.
    ///
    /// If `regex` is True, the method returns the annotations whose text matches the query
    /// as a case-insensitive regex pattern.
    ///
    /// If the regex pattern is invalid, the method raises `ValueError`.
    #[pyo3(signature = (query, regex=false))]
    fn filter_annotations(&self, query: &str, regex: bool) -> PyResult<Vec<PyAnnotation>> {
        let anns = self
            .get()?
            .filter_annotations(query, regex)
            .map_err(to_py_err)?;
        Ok(anns.iter().map(PyAnnotation::from).collect())
    }

    /// The annotations whose text matches `query`. This method is an alias of `filter_annotations`.
    ///
    /// The alias makes epoch-extraction code easy to read: the result of `f.events("Spindle")`
    /// goes directly into `f.extract_epochs(...)`. The matching rules are the same: a
    /// case-insensitive substring, or a case-insensitive regex when `regex=True`.
    #[pyo3(signature = (query, regex=false))]
    fn events(&self, query: &str, regex: bool) -> PyResult<Vec<PyAnnotation>> {
        self.filter_annotations(query, regex)
    }

    /// The annotations whose text exactly matches `text` (case-sensitive).
    pub fn annotations_by_text(&self, text: &str) -> PyResult<Vec<PyAnnotation>> {
        Ok(self
            .get()?
            .annotations_by_text(text)
            .iter()
            .map(PyAnnotation::from)
            .collect())
    }

    /// Return the indices of all signals whose label matches `label`.
    ///
    /// If `exact` is `False` (default), the method does a case-insensitive substring match.
    /// If `exact` is `True`, the method does a case-sensitive exact match.
    ///
    /// The search includes the annotation channels.
    #[pyo3(signature = (label, exact=false))]
    fn find_all_signals(&self, label: &str, exact: bool) -> PyResult<Vec<usize>> {
        Ok(self.get()?.find_all_signals(label, exact))
    }

    /// Warnings from the header parse and the annotation scan. Waits until the scan is complete.
    #[getter]
    fn warnings(&self) -> PyResult<Vec<String>> {
        Ok(self.get()?.warnings())
    }

    /// The raw header fields as a dict.
    ///
    /// Each call builds a new dict, so `header` is a method and not a property.
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
    /// `cache_capacity` turns on an LRU cache of decoded physical records for this signal.
    /// The unit is a count of EDF data records, not samples or bytes. One cached record holds
    /// `samples_per_record` float64 values. Thus the cache uses approximately
    /// `cache_capacity * samples_per_record * 8` bytes. The default, 0, turns the cache off.
    ///
    /// For one-pass or strictly forward reads, leave `cache_capacity` at 0. The OS page cache
    /// already holds the raw bytes. The cache helps only when you decode the same records again,
    /// for example with overlapping windows, back-and-forth seeks, or repeated slices. A good
    /// start value is `ceil(window_samples / samples_per_record) + 2`. That is a few records
    /// more than your largest repeated window spans.
    /// The cache makes only physical reads faster. `to_digital()` always decodes again from the
    /// memory map. Each `Signal` has its own cache. A new call to `signal()` starts with an
    /// empty cache.
    ///
    /// `strategy` sets how edfarray reads the bytes:
    /// - `"auto"` (default) streams large reads that are not already cached. It uses the memory
    ///   map for all other reads.
    /// - `"mmap"` always uses the memory map.
    /// - `"stream"` always reads sequentially through a bounded buffer.
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
        } else if let Ok(idx) = idx_or_label.extract::<i64>() {
            return Err(negative_index_err(idx, self.get()?.num_signals()));
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

    /// The labels of all signals in the file.
    fn signal_labels(&self) -> PyResult<Vec<&str>> {
        Ok(self.get()?.signal_labels())
    }

    /// The indices of all ordinary signals.
    ///
    /// An ordinary signal is a signal that is not an annotation channel.
    fn ordinary_signal_indices(&self) -> PyResult<Vec<usize>> {
        Ok(self.get()?.ordinary_signal_indices())
    }

    /// Read a page of physical data for multiple signals over a time range.
    ///
    /// The method returns a list of numpy arrays, one per signal. If the signals have different
    /// sample rates, the arrays have different lengths.
    ///
    /// If `signal_indices` is None, the method reads all ordinary signals. An ordinary signal is
    /// a signal that is not an annotation channel.
    ///
    /// When `use_time` is false (default), the method converts the time parameters to flat
    /// sample indices. For EDF+D files with gaps, set `use_time=True`. The method then finds
    /// the time range from the actual record onset times.
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

    /// `True` if the background annotation scan is complete.
    #[getter]
    fn annotations_ready(&self) -> PyResult<bool> {
        Ok(self.get()?.annotations_ready())
    }

    /// The progress of the background annotation scan, as (records_scanned, total_records).
    #[getter]
    fn scan_progress(&self) -> PyResult<(usize, usize)> {
        Ok(self.get()?.scan_progress())
    }

    /// Build a 2D proxy from a `SignalGroup`.
    ///
    /// A proxy is an array-like object that reads samples from the file only when you index it.
    ///
    /// `pad_mode` controls reads past the valid length of a signal when the group is `"open"`
    /// (mixed sample rates). It accepts the string `"raise"` (default), `"nan"`, `"zero"`,
    /// `"edge"`, or a numeric scalar. edfarray reads a numeric scalar as `Value(x)`. `None` has
    /// the same effect as `"raise"`.
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
    /// A proxy is an array-like object that reads samples from the file only when you index it.
    /// The group must have `group.kind == "rectangular"`, so all its signals share a sample rate.
    /// To find eligible groups, use `signal_groups()`. To make a group from specific indices,
    /// use `signal_group(...)`.
    fn proxy_3d(&self, group: &PySignalGroup) -> PyResult<PyProxy3D> {
        let proxy = self
            .get()?
            .proxy_3d(group.inner().clone())
            .map_err(to_py_err)?;
        Ok(PyProxy3D::new(proxy))
    }

    /// Classify a list of file-level signal indices into a `SignalGroup`.
    ///
    /// Use this method for a group that is a subset of the natural rate-based groups of the
    /// file, or for a group that crosses them.
    fn signal_group(&self, indices: Vec<usize>) -> PyResult<PySignalGroup> {
        let g = edfarray_core::group::SignalGroup::from_indices(self.get()?.header(), &indices)
            .map_err(to_py_err)?;
        Ok(PySignalGroup::new(g))
    }

    /// Partition all ordinary signals into groups by sample rate.
    ///
    /// An ordinary signal is a signal that is not an annotation channel. The method returns a
    /// list of `SignalGroup` objects. Each group has its sample rate, structural kind, and
    /// sample-count range. It also shows if it covers every ordinary signal in the file. The
    /// sample rates keep their sub-Hz precision.
    fn signal_groups(&self) -> PyResult<Vec<PySignalGroup>> {
        Ok(self
            .get()?
            .signal_groups()
            .into_iter()
            .map(PySignalGroup::new)
            .collect())
    }

    /// Write this file to `path`, and optionally transcode it to a different variant.
    ///
    /// The method copies the annotations and the ordinary signals. An ordinary signal is a
    /// signal that is not an annotation channel. The method does not copy the annotation channel
    /// byte for byte. It builds a new annotation channel in the destination from the parsed
    /// annotations. `variant` can be one of "EDF", "EDF+C", "EDF+D", "BDF", "BDF+C", "BDF+D".
    /// If you omit `variant`, the method uses the source variant.
    ///
    /// `signals` selects the ordinary signals to write. It accepts a `SignalGroup`, a signal
    /// index, a label, or a sequence that mixes indices and labels. Labels match exactly, as in
    /// `signal()`. The destination signals are in the given order, so the method rejects sets
    /// and dicts. `None` (the default) writes every ordinary signal. You cannot select the
    /// annotation channel. The method always builds it again, and copies all annotations for
    /// any selection.
    ///
    /// If another edfarray handle has `path` open, the method raises `EdfFileError`. This
    /// includes this file. Before you write to a path, close every `EdfFile` on that path.
    /// Also drop every signal and proxy taken from such a file.
    ///
    /// Transcoding caveats:
    /// - EDF+D to EDF+D keeps the source record onsets, so the gaps stay in the copy.
    ///   Transcoding to any variant without `+D` removes the per-record onsets and gaps. The
    ///   output uses uniform `record_idx * record_duration` timing.
    /// - The method builds the annotation channel from the parsed annotations. Plain (non-"+")
    ///   EDF/BDF variants have no annotation channel. Thus transcoding to a plain variant drops
    ///   all annotations.
    /// - A smaller sample size (for example, BDF 24-bit to EDF 16-bit) clamps the digital range.
    ///   The method encodes the samples again from the physical values, and precision decreases.
    #[pyo3(signature = (path, variant=None, signals=None))]
    fn write_to(
        &self,
        path: &str,
        variant: Option<&str>,
        #[gen_stub(override_type(
            type_repr = "SignalGroup | builtins.int | builtins.str | typing.Sequence[builtins.int | builtins.str] | builtins.NoneType"
        ))]
        signals: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
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
        let f = self.get()?;
        match resolve_signal_selection(f, signals)? {
            Some(selected) => f.write_subset_to(path, target, &selected),
            None => f.write_to(path, target),
        }
        .map_err(to_py_err)
    }

    /// Read a page of digital (raw int32) data for multiple signals over a time range.
    ///
    /// If `signal_indices` is None, the method reads all ordinary signals. An ordinary signal is
    /// a signal that is not an annotation channel.
    ///
    /// When `use_time` is false (default), the method converts the time parameters to flat
    /// sample indices. For EDF+D files with gaps, set `use_time=True`. The method then finds
    /// the time range from the actual record onset times.
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

    /// Extract fixed windows around events as a dense `(n_epochs, n_channels, n_samples)` block.
    ///
    /// The method decodes the windows in parallel. It uses the EDF+D record onsets, so the
    /// windows account for gaps.
    ///
    /// `events` accepts floats, `Annotation`s, a mixed sequence, a numpy float64 array, or a
    /// single value. To extract around annotation text, pass `events=None` together with
    /// `query`. The matching is the same as in `filter_annotations`.
    ///
    /// `group` selects the signals: a `SignalGroup` or a sequence of signal indices. The group
    /// must be rectangular. The default is the largest rectangular group.
    ///
    /// `pad` controls epochs whose window goes past the file or crosses an EDF+D gap:
    /// - `"drop"` (default) omits them.
    /// - `"nan"`, `"zero"`, a number, or `"edge"` keeps and fills them, and marks them `valid=False`.
    /// - `"raise"` raises an error at the first such epoch.
    #[pyo3(signature = (events, *, pre, post, group=None, pad=None, query=None, regex=false))]
    #[allow(clippy::too_many_arguments)]
    fn extract_epochs(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(
            type_repr = "builtins.float | typing.Sequence[builtins.float] | Annotation | typing.Sequence[Annotation] | builtins.NoneType"
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
    /// `extract_epochs` decodes the same window table. Thus callers can inspect or clip the
    /// windows before they pay for the reads.
    #[pyo3(signature = (events, *, pre, post, group=None))]
    #[gen_stub(override_return_type(type_repr = "tuple[builtins.list[builtins.tuple[builtins.float, builtins.int, builtins.int]], numpy.typing.NDArray[numpy.bool_]]", imports = ("builtins", "numpy")))]
    fn epoch_windows<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(
            type_repr = "builtins.float | typing.Sequence[builtins.float] | Annotation | typing.Sequence[Annotation]"
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

/// Read metadata from the header of an EDF/EDF+ file.
///
/// The function does not scan data records or build an annotation index. It returns a dict
/// with these keys:
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

/// Resolve the `signals` argument of `write_to` into file-level signal indices.
///
/// `None` means "copy every ordinary signal". Otherwise the argument must be a
/// `SignalGroup`, a signal index, a label (exact match, as in `signal()`), or an
/// ordered iterable mixing indices and labels. Returned indices are in destination
/// order. The core does the bounds, annotation-channel, and duplicate checks.
pub(crate) fn resolve_signal_selection(
    f: &EdfFile,
    signals: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Vec<usize>>> {
    use pyo3::exceptions::PyTypeError;
    use pyo3::types::{PyByteArray, PyBytes, PyDict, PyFrozenSet, PyMemoryView, PySet};

    const EXPECTED: &str = "signals must be a SignalGroup, a signal index, a label, \
                            or a sequence mixing indices and labels";

    let Some(value) = signals else {
        return Ok(None);
    };
    if let Ok(group) = value.extract::<PyRef<'_, PySignalGroup>>() {
        return Ok(Some(group.inner().indices().to_vec()));
    }
    if let Some(idx) = selection_item(f, value)? {
        return Ok(Some(vec![idx]));
    }
    // Bytes would iterate as small ints, and unordered containers would give an arbitrary
    // channel order.
    if value.cast::<PyBytes>().is_ok()
        || value.cast::<PyByteArray>().is_ok()
        || value.cast::<PyMemoryView>().is_ok()
    {
        return Err(PyTypeError::new_err(format!(
            "{EXPECTED}; bytes are not accepted"
        )));
    }
    if value.cast::<PySet>().is_ok()
        || value.cast::<PyFrozenSet>().is_ok()
        || value.cast::<PyDict>().is_ok()
    {
        return Err(PyTypeError::new_err(format!(
            "{EXPECTED}; sets and dicts are unordered, so the channel order would be arbitrary"
        )));
    }

    let iter = value
        .try_iter()
        .map_err(|_| PyTypeError::new_err(EXPECTED))?;
    let mut selected = Vec::new();
    for item in iter {
        let item = item?;
        match selection_item(f, &item)? {
            Some(idx) => selected.push(idx),
            None => {
                return Err(PyTypeError::new_err(
                    "each element of signals must be an int index or a str label",
                ));
            }
        }
    }
    Ok(Some(selected))
}

/// Resolve one index or label. `Ok(None)` means `value` is neither.
fn selection_item(f: &EdfFile, value: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    // bool subclasses int, but `True` as a channel index is almost certainly a mistake.
    if value.cast::<pyo3::types::PyBool>().is_ok() {
        return Ok(None);
    }
    if let Ok(idx) = value.extract::<usize>() {
        return Ok(Some(idx));
    }
    if let Ok(idx) = value.extract::<i64>() {
        return Err(negative_index_err(idx, f.num_signals()));
    }
    if let Ok(label) = value.extract::<String>() {
        let idx = f
            .find_all_signals(&label, true)
            .first()
            .copied()
            .ok_or_else(|| to_py_err(edfarray_core::error::EdfError::SignalNotFound { label }))?;
        return Ok(Some(idx));
    }
    Ok(None)
}

/// Error for an int index that does not fit `usize`. Negative indices do not wrap around.
pub(crate) fn negative_index_err(idx: i64, count: usize) -> PyErr {
    out_of_range_err(format!(
        "signal index {idx} out of range (file has {count} signals)"
    ))
}
