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


class TestWarnings:
    def test_valid_files_no_warnings(self, fixture):
        edf, _ = fixture
        assert edf.warnings == [], f"unexpected warnings: {edf.warnings}"
