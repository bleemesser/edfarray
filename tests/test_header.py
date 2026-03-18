"""Validate header parsing against pyedflib reference values."""

import pytest

from conftest import FIXTURES, FIXTURE_NAMES, load_fixture


@pytest.fixture(params=FIXTURE_NAMES)
def fixture(request):
    return load_fixture(request.param)


class TestBasicProperties:
    def test_num_signals(self, fixture):
        edf, ref = fixture
        # pyedflib hides annotation signals from its count. Our num_signals
        # is the raw header count including annotation signals.
        num_annotation_signals = sum(
            1 for i in range(edf.num_signals)
            if edf.signal(i).label == "EDF Annotations"
        )
        assert edf.num_signals - num_annotation_signals == ref["header"]["num_signals"]

    def test_num_records(self, fixture):
        edf, ref = fixture
        assert edf.num_records == ref["header"]["num_records"]

    def test_duration(self, fixture):
        edf, ref = fixture
        assert abs(edf.duration - ref["header"]["file_duration_seconds"]) < 0.01


class TestStartDateTime:
    def test_start_date_components(self, fixture):
        edf, ref = fixture
        dt = edf.start_datetime
        ref_year = ref["header"].get("startdate_year")
        if ref_year is None or not hasattr(dt, "year"):
            pytest.skip("no parseable start date")
        assert dt.year == ref_year
        assert dt.month == ref["header"]["startdate_month"]
        assert dt.day == ref["header"]["startdate_day"]

    def test_start_time_components(self, fixture):
        edf, ref = fixture
        dt = edf.start_datetime
        ref_hour = ref["header"].get("starttime_hour")
        if ref_hour is None or not hasattr(dt, "hour"):
            pytest.skip("no parseable start time")
        assert dt.hour == ref_hour
        assert dt.minute == ref["header"]["starttime_minute"]
        assert dt.second == ref["header"]["starttime_second"]
