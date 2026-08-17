"""Annotation lookup methods.

These use binary search over the sorted index, so the boundary conditions (a query exactly on
an onset, an empty result, a range covering everything) are what matter.
"""

import edfarray
import pytest
from conftest import FIXTURES
from edfbuilder import build_edf_plus

ANNOTATED = str(FIXTURES / "edfPlusC.edf")


@pytest.fixture
def many(tmp_path):
    """A file with annotations at 1s..5s, so ranges have unambiguous answers."""
    path = build_edf_plus(
        tmp_path / "many.edf",
        record_onsets=list(range(8)),
        annotations=[(i, float(i), f"E{i}", 0) for i in range(1, 6)],
    )
    return edfarray.EdfFile(path)


def texts(anns):
    return [a.text for a in anns]


class TestBefore:
    def test_strictly_before(self, many):
        assert texts(many.annotations_before(3.0)) == ["E1", "E2"]

    def test_before_everything_is_empty(self, many):
        assert many.annotations_before(0.0) == []

    def test_before_everything_past_the_end(self, many):
        assert texts(many.annotations_before(1e9)) == ["E1", "E2", "E3", "E4", "E5"]


class TestAfter:
    def test_inclusive_of_the_boundary(self, many):
        # `after` is >= t, unlike `before` which is strictly <.
        assert texts(many.annotations_after(3.0)) == ["E3", "E4", "E5"]

    def test_after_everything_is_empty(self, many):
        assert many.annotations_after(1e9) == []

    def test_before_and_after_partition_the_set(self, many):
        for t in (0.0, 2.5, 3.0, 99.0):
            assert texts(many.annotations_before(t)) + texts(many.annotations_after(t)) == [
                "E1",
                "E2",
                "E3",
                "E4",
                "E5",
            ]


class TestInRange:
    def test_half_open_range(self, many):
        assert texts(many.annotations_in_range(2.0, 4.0)) == ["E2", "E3"]

    def test_empty_range(self, many):
        assert many.annotations_in_range(2.0, 2.0) == []

    def test_range_covering_all(self, many):
        assert len(many.annotations_in_range(-1.0, 1e9)) == 5

    def test_range_between_annotations(self, many):
        assert many.annotations_in_range(2.25, 2.75) == []


class TestByText:
    def test_exact_match_is_case_sensitive(self, many):
        assert texts(many.annotations_by_text("E3")) == ["E3"]
        assert many.annotations_by_text("e3") == []

    def test_no_match(self, many):
        assert many.annotations_by_text("nope") == []


class TestFilter:
    def test_substring_is_case_insensitive(self):
        f = edfarray.EdfFile(ANNOTATED)
        assert texts(f.filter_annotations("record")) == ["RECORD START"]

    def test_regex_mode(self):
        f = edfarray.EdfFile(ANNOTATED)
        assert len(f.filter_annotations(r"^REC\s", regex=True)) == 1
        assert len(f.filter_annotations(r"^(RECORD|REC)\b", regex=True)) == 2

    def test_regex_is_case_insensitive(self):
        f = edfarray.EdfFile(ANNOTATED)
        assert len(f.filter_annotations(r"record start", regex=True)) == 1

    def test_regex_metacharacters_are_literal_without_the_flag(self):
        f = edfarray.EdfFile(ANNOTATED)
        assert f.filter_annotations(r"^REC\s") == []

    def test_invalid_regex_raises(self):
        f = edfarray.EdfFile(ANNOTATED)
        with pytest.raises(edfarray.InvalidArgumentError):
            f.filter_annotations("(unclosed", regex=True)


def test_queries_agree_with_the_full_list(many):
    everything = many.annotations
    assert texts(many.annotations_in_range(-1.0, 1e9)) == texts(everything)
    for ann in everything:
        assert ann in many.annotations_by_text(ann.text)
