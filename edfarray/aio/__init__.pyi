"""Async API for edfarray. These stubs are hand-written, not generated."""

from __future__ import annotations

from typing import Any, Awaitable, Sequence
import datetime

import numpy as np
import numpy.typing as npt

from edfarray._core import Annotation, Epochs, Proxy2D, Proxy3D, SignalGroup, WriterSignal


class EdfFile:
    @property
    def num_signals(self) -> int: ...
    @property
    def num_records(self) -> int: ...
    @property
    def record_duration(self) -> float: ...
    @property
    def duration(self) -> float: ...
    @property
    def variant(self) -> str: ...
    @property
    def patient_id(self) -> str: ...
    @property
    def recording_id(self) -> str: ...
    @property
    def start_datetime(self) -> datetime.datetime | str: ...
    @property
    def patient_name(self) -> str | None: ...
    @property
    def patient_code(self) -> str | None: ...
    @property
    def patient_sex(self) -> str | None: ...
    @property
    def patient_birthdate(self) -> datetime.date | str | None: ...
    @property
    def patient_additional(self) -> str | None: ...
    @property
    def admin_code(self) -> str | None: ...
    @property
    def technician(self) -> str | None: ...
    @property
    def equipment(self) -> str | None: ...
    @property
    def recording_additional(self) -> str | None: ...
    @property
    def annotations(self) -> list[Annotation]: ...
    @property
    def warnings(self) -> list[str]: ...
    @property
    def annotations_ready(self) -> bool: ...
    @property
    def scan_progress(self) -> tuple[int, int]: ...
    @property
    def closed(self) -> bool: ...

    def annotations_before(self, t: float) -> list[Annotation]: ...
    def annotations_after(self, t: float) -> list[Annotation]: ...
    def annotations_in_range(self, start: float, end: float) -> list[Annotation]: ...
    def filter_annotations(self, query: str, regex: bool = False) -> list[Annotation]: ...
    def events(self, query: str, regex: bool = False) -> list[Annotation]: ...
    def annotations_by_text(self, text: str) -> list[Annotation]: ...
    def find_all_signals(self, label: str, exact: bool = False) -> list[int]: ...
    def proxy_2d(self, group: SignalGroup, pad_mode: Any | None = None) -> Proxy2D: ...
    def proxy_3d(self, group: SignalGroup) -> Proxy3D: ...
    def signal_labels(self) -> list[str]: ...
    def ordinary_signal_indices(self) -> list[int]: ...
    def signal_groups(self) -> list[SignalGroup]: ...
    def header(self) -> dict[str, Any]: ...
    def close(self) -> None: ...

    def wait_for_annotations(self) -> Awaitable[None]: ...
    def read_page(
        self,
        start_sec: float,
        end_sec: float,
        signal_indices: list[int] | None = None,
        use_time: bool = False,
    ) -> Awaitable[list[npt.NDArray[np.float64]]]: ...
    def read_page_digital(
        self,
        start_sec: float,
        end_sec: float,
        signal_indices: list[int] | None = None,
        use_time: bool = False,
    ) -> Awaitable[list[npt.NDArray[np.int32]]]: ...
    def extract_epochs(
        self,
        events: float | list[float] | Annotation | list[Annotation] | None,
        *,
        pre: float,
        post: float,
        group: Any | None = None,
        pad: str | float | None = None,
        query: str | None = None,
        regex: bool = False,
    ) -> Awaitable[Epochs]:
        r"""
        Extract a rectangular epoch array in a blocking task.

        The behavior is the same as in the synchronous `EdfFile.extract_epochs`.
        """
        ...
    def epoch_windows(
        self,
        events: float | list[float] | Annotation | list[Annotation],
        *,
        pre: float,
        post: float,
        group: Any | None = None,
    ) -> Awaitable[tuple[list[tuple[float, int, int]], npt.NDArray[np.bool_]]]:
        r"""
        Planned epoch windows without reading data. See `EdfFile.epoch_windows`.
        """
        ...
    def signal(self, idx_or_label: int | str, cache_capacity: int = 0) -> Signal:
        r"""
        Get a signal by index or label.

        ``cache_capacity`` turns on an LRU cache of decoded physical records for this signal.
        The unit is a count of EDF data records, not samples or bytes. One cached record holds
        ``samples_per_record`` float64 values. Thus the cache uses approximately
        ``cache_capacity * samples_per_record * 8`` bytes. The default, 0, turns the cache off.

        For one-pass or strictly forward reads, leave ``cache_capacity`` at 0. The OS page cache
        already holds the raw bytes. The cache helps only when you decode the same records again,
        for example with overlapping windows, back-and-forth seeks, or repeated slices. A good
        start value is ``ceil(window_samples / samples_per_record) + 2``. That is a few records
        more than your largest repeated window spans.
        The cache makes only physical reads faster. ``read_range_digital()`` always decodes again
        from the memory map. Each ``Signal`` has its own cache. A new call to ``signal()`` starts
        with an empty cache.
        """
        ...
    def signal_group(self, indices: list[int]) -> SignalGroup: ...
    def write_to(
        self,
        path: str,
        variant: str | None = None,
        signals: SignalGroup | int | str | Sequence[int | str] | None = None,
    ) -> Awaitable[None]:
        r"""
        Write to `path`, and optionally transcode the file to a different variant.

        `signals` selects the ordinary signals to write. An ordinary signal is a signal that is
        not an annotation channel. `signals` accepts a `SignalGroup`, a signal index, a label, or
        a sequence that mixes indices and labels. Labels match exactly, as in `signal()`. The
        destination signals are in the given order, so the method rejects sets and dicts. `None`
        (the default) writes every ordinary signal. You cannot select the annotation channel.
        The method always builds it again, and copies all annotations for any selection.

        If another edfarray handle has `path` open, the method raises `EdfFileError`. This
        includes this file. Before you write to a path, close every `EdfFile` on that path.
        Also drop every signal and proxy taken from such a file.

        Transcoding caveats:
        - EDF+D to EDF+D keeps the source record onsets, so the gaps stay in the copy.
          Transcoding to any variant without `+D` removes the per-record onsets and gaps. The
          output uses uniform `record_idx * record_duration` timing.
        - The method builds the annotation channel from the parsed annotations. Plain (non-"+")
          EDF/BDF variants have no annotation channel. Thus transcoding to a plain variant drops
          all annotations.
        - A smaller sample size (for example, BDF 24-bit to EDF 16-bit) clamps the digital range.
          The method encodes the samples again from the physical values, and precision decreases.
        """
        ...

    def __repr__(self) -> str: ...
    async def __aenter__(self) -> EdfFile: ...
    async def __aexit__(self, *args: Any) -> None: ...


class Signal:
    @property
    def label(self) -> str: ...
    @property
    def transducer(self) -> str: ...
    @property
    def physical_dimension(self) -> str: ...
    @property
    def prefiltering(self) -> str: ...
    @property
    def sample_rate(self) -> float: ...
    @property
    def samples_per_record(self) -> int: ...
    @property
    def physical_min(self) -> float: ...
    @property
    def physical_max(self) -> float: ...
    @property
    def digital_min(self) -> int: ...
    @property
    def digital_max(self) -> int: ...
    @property
    def num_samples(self) -> int: ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...

    def read_range(self, start: int, stop: int) -> Awaitable[npt.NDArray[np.float64]]: ...
    def read_range_digital(self, start: int, stop: int) -> Awaitable[npt.NDArray[np.int32]]: ...
    def read_time_range(self, start_sec: float, end_sec: float) -> Awaitable[npt.NDArray[np.float64]]: ...
    def to_physical(self) -> Awaitable[npt.NDArray[np.float64]]: ...
    def to_digital(self) -> Awaitable[npt.NDArray[np.int32]]: ...
    def times(self) -> Awaitable[npt.NDArray[np.float64]]: ...


def open(path: str, variant: str | None = None) -> Awaitable[EdfFile]:
    r"""
    Open an EDF/EDF+/BDF file.

    `variant` sets the file variant and replaces the variant that edfarray detects. Use it
    for files that omit or misreport the EDF+ "+C"/"+D" marker. It controls only the
    plain/"+C"/"+D" distinction. The version field sets the EDF-vs-BDF sample size. If an
    override changes that sample size, the function raises `ValueError`.
    """
    ...
def inspect(path: str) -> Awaitable[dict[str, Any]]: ...


class EdfWriter:
    @classmethod
    def create(
        cls,
        path: str,
        *,
        variant: str,
        record_duration: float,
        signals: list[WriterSignal],
        start_datetime: datetime.datetime | None = None,
        patient_id: str | None = None,
        recording_id: str | None = None,
        annotation_bytes_per_record: int | None = None,
    ) -> Awaitable[EdfWriter]: ...

    def add_annotation(self, annotation: Annotation) -> None: ...
    def write_record(
        self,
        physical: list[npt.NDArray[np.float64]],
        annotations: list[Annotation] | None = None,
    ) -> Awaitable[None]: ...
    def finish(self) -> Awaitable[None]: ...
    def __repr__(self) -> str: ...
    async def __aenter__(self) -> EdfWriter: ...
    async def __aexit__(self, *args: Any) -> None: ...


def write_edf(
    path: str,
    *,
    variant: str,
    record_duration: float,
    signals: list[WriterSignal],
    data: list[npt.NDArray[np.float64]],
    annotations: list[Annotation] | None = None,
    start_datetime: datetime.datetime | None = None,
    patient_id: str | None = None,
    recording_id: str | None = None,
    annotation_bytes_per_record: int | None = None,
) -> Awaitable[None]: ...
