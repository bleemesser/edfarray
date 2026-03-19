"""Validate annotation parsing against pyedflib reference values."""

import numpy as np
import pytest

from conftest import FIXTURES, FIXTURE_NAMES, load_fixture

import edfarray


@pytest.fixture(params=FIXTURE_NAMES)
def fixture(request):
    return load_fixture(request.param)


class TestAnnotations:
    def test_annotation_count(self, fixture):
        edf, ref = fixture
        ref_count = len(ref.get("annotations", []))
        assert len(edf.annotations) == ref_count

    def test_annotation_onsets(self, fixture):
        edf, ref = fixture
        for ref_ann, ann in zip(ref.get("annotations", []), edf.annotations):
            assert abs(ann.onset - ref_ann["onset"]) < 1e-4, (
                f"onset mismatch: ours={ann.onset}, ref={ref_ann['onset']}, "
                f"text={ann.text!r}"
            )

    def test_annotation_text(self, fixture):
        edf, ref = fixture
        for ref_ann, ann in zip(ref.get("annotations", []), edf.annotations):
            assert ann.text == ref_ann["text"]

    def test_annotations_sorted_by_onset(self, fixture):
        edf, _ = fixture
        onsets = [a.onset for a in edf.annotations]
        assert onsets == sorted(onsets)

    def test_annotation_fields(self):
        edf, _ = load_fixture("test_generator_2")
        assert len(edf.annotations) > 0
        ann = edf.annotations[0]
        assert isinstance(ann.onset, float)
        assert isinstance(ann.text, str)
        assert ann.duration is None or isinstance(ann.duration, float)

    def test_annotation_repr(self):
        edf, _ = load_fixture("test_generator_2")
        ann = edf.annotations[0]
        r = repr(ann)
        assert "Annotation" in r
        assert "onset=" in r


class TestEdgeCases:
    def test_edf_plus_d_opens(self):
        """EDF+D files should open without error (pyedflib can't even do this)."""
        edf = edfarray.EdfFile(str(FIXTURES / "edfPlusD.edf"))
        assert edf.variant == "EDF+D"
        assert edf.num_signals > 0
        assert len(edf.annotations) > 0


class TestEdfPlusDTimeMapping:
    """Verify that EDF+D time mapping works correctly and document the
    read_page flat-index behavior."""

    def test_times_reflect_gaps(self):
        """Signal.times() must reflect actual record onsets, not flat indices."""
        edf = edfarray.EdfFile(str(FIXTURES / "edfPlusD.edf"))
        sig = edf.signal(0)
        times = sig.times()
        dt = np.diff(times)
        expected_dt = 1.0 / sig.sample_rate

        gaps = np.where(dt > expected_dt * 1.5)[0]
        assert len(gaps) > 0, "EDF+D fixture should have time gaps"

        for idx in gaps:
            assert dt[idx] > expected_dt * 1.5

    def test_times_span_exceeds_record_count(self):
        """Physical time span should be larger than num_records * record_duration
        because of gaps."""
        edf = edfarray.EdfFile(str(FIXTURES / "edfPlusD.edf"))
        sig = edf.signal(0)
        times = sig.times()
        physical_span = times[-1] - times[0]
        flat_span = (len(sig) - 1) / sig.sample_rate
        assert physical_span > flat_span

    def test_read_page_uses_flat_indices(self):
        """read_page maps time to flat sample indices, which is incorrect for
        EDF+D. This test documents the behavior: requesting 0-10s of data
        returns samples whose physical timestamps extend beyond 10s."""
        edf = edfarray.EdfFile(str(FIXTURES / "edfPlusD.edf"))
        sig = edf.signal(0)
        sr = sig.sample_rate

        pages = edf.read_page(0.0, 10.0)
        n_samples = len(pages[0])
        assert n_samples == int(10.0 * sr)

        times = sig.times()
        last_time = times[n_samples - 1]
        assert last_time > 10.0, (
            f"Expected flat-index read to span beyond 10s, got {last_time:.3f}s"
        )

    def test_correct_time_window_workflow(self):
        """The correct way to get samples within a physical time window for
        EDF+D: use times() to mask the data."""
        edf = edfarray.EdfFile(str(FIXTURES / "edfPlusD.edf"))
        sig = edf.signal(0)
        sr = sig.sample_rate

        all_data = sig.to_numpy()
        all_times = sig.times()

        t_start, t_end = 0.0, 10.0
        mask = (all_times >= t_start) & (all_times < t_end)
        data_in_window = all_data[mask]
        times_in_window = all_times[mask]

        assert len(data_in_window) < int((t_end - t_start) * sr), (
            "Should have fewer samples than a contiguous 10s window due to gaps"
        )
        assert times_in_window[0] >= t_start
        assert times_in_window[-1] < t_end


class TestWarnings:
    def test_valid_files_no_warnings(self, fixture):
        edf, _ = fixture
        assert edf.warnings == [], f"unexpected warnings: {edf.warnings}"
