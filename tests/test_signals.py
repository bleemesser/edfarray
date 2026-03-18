"""Validate signal metadata and data access against pyedflib reference values."""

import numpy as np
import pytest

from conftest import FIXTURES, FIXTURE_NAMES, load_fixture

import edfarray


@pytest.fixture(params=FIXTURE_NAMES)
def fixture(request):
    return load_fixture(request.param)


class TestSignalMetadata:
    def test_signal_count(self, fixture):
        edf, ref = fixture
        # pyedflib hides annotation signals. Our count includes them.
        num_annotation_signals = sum(
            1 for i in range(edf.num_signals)
            if edf.signal(i).label == "EDF Annotations"
        )
        assert edf.num_signals - num_annotation_signals == len(ref["signals"])

    def test_labels(self, fixture):
        edf, ref = fixture
        for ref_sig in ref["signals"]:
            sig = edf.signal(ref_sig["index"])
            assert sig.label == ref_sig["label"]

    def test_physical_range(self, fixture):
        edf, ref = fixture
        for ref_sig in ref["signals"]:
            sig = edf.signal(ref_sig["index"])
            assert abs(sig.physical_min - ref_sig["physical_min"]) < 1e-6
            assert abs(sig.physical_max - ref_sig["physical_max"]) < 1e-6

    def test_digital_range(self, fixture):
        edf, ref = fixture
        for ref_sig in ref["signals"]:
            sig = edf.signal(ref_sig["index"])
            assert sig.digital_min == ref_sig["digital_min"]
            assert sig.digital_max == ref_sig["digital_max"]

    def test_samples_per_record(self, fixture):
        edf, ref = fixture
        for ref_sig in ref["signals"]:
            sig = edf.signal(ref_sig["index"])
            assert sig.samples_per_record == int(ref_sig["samples_per_data_record"])

    def test_sample_rate(self, fixture):
        edf, ref = fixture
        for ref_sig in ref["signals"]:
            sig = edf.signal(ref_sig["index"])
            assert abs(sig.sample_rate - ref_sig["sample_frequency"]) < 1e-6

    def test_transducer(self, fixture):
        edf, ref = fixture
        for ref_sig in ref["signals"]:
            sig = edf.signal(ref_sig["index"])
            assert sig.transducer == ref_sig["transducer"]

    def test_prefiltering(self, fixture):
        edf, ref = fixture
        for ref_sig in ref["signals"]:
            sig = edf.signal(ref_sig["index"])
            assert sig.prefiltering == ref_sig["prefiltering"]

    def test_physical_dimension(self, fixture):
        edf, ref = fixture
        for ref_sig in ref["signals"]:
            sig = edf.signal(ref_sig["index"])
            assert sig.physical_dimension == ref_sig["physical_dimension"]

    def test_signal_by_label(self, fixture):
        edf, ref = fixture
        for ref_sig in ref["signals"]:
            sig = edf.signal(ref_sig["label"])
            assert sig.samples_per_record == int(ref_sig["samples_per_data_record"])


class TestSignalData:
    def test_physical_samples_match_reference(self, fixture):
        """First 10 physical samples should match pyedflib output."""
        edf, ref = fixture
        for snippet in ref.get("sample_snippets", []):
            sig = edf.signal(snippet["signal_index"])
            actual = sig[0:len(snippet["physical_first_10"])]
            expected = np.array(snippet["physical_first_10"])
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-10)

    def test_digital_samples_match_reference(self, fixture):
        """First 10 digital samples should match pyedflib output."""
        edf, ref = fixture
        for snippet in ref.get("sample_snippets", []):
            sig = edf.signal(snippet["signal_index"])
            actual = sig.to_digital()[:len(snippet["digital_first_10"])]
            expected = np.array(snippet["digital_first_10"], dtype=np.int16)
            np.testing.assert_array_equal(actual, expected)

    def test_single_sample_returns_float(self):
        edf, _ = load_fixture("short_psg")
        assert isinstance(edf.signal(0)[0], float)

    def test_slice_returns_numpy(self):
        edf, _ = load_fixture("short_psg")
        arr = edf.signal(0)[0:10]
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64
        assert len(arr) == 10

    def test_strided_slice(self):
        edf, _ = load_fixture("short_psg")
        sig = edf.signal(0)
        arr = sig[0:10:2]
        assert len(arr) == 5
        for i, idx in enumerate(range(0, 10, 2)):
            assert abs(arr[i] - sig[idx]) < 1e-10

    def test_negative_indexing(self):
        edf, _ = load_fixture("short_psg")
        sig = edf.signal(0)
        assert sig[-1] == sig[len(sig) - 1]

    def test_to_numpy_length(self, fixture):
        edf, ref = fixture
        if not ref["signals"]:
            pytest.skip("no signals")
        sig = edf.signal(0)
        arr = sig.to_numpy()
        assert len(arr) == len(sig)

    def test_to_digital_dtype(self):
        edf, _ = load_fixture("short_psg")
        arr = edf.signal(0).to_digital()
        assert arr.dtype == np.int16

    def test_times_monotonic(self):
        edf, _ = load_fixture("short_psg")
        times = edf.signal(0).times()
        assert np.all(np.diff(times) >= 0)

    def test_times_start_at_zero(self):
        edf, _ = load_fixture("short_psg")
        times = edf.signal(0).times()
        assert abs(times[0]) < 1e-10


class TestErrorHandling:
    def test_signal_not_found(self):
        edf, _ = load_fixture("short_psg")
        with pytest.raises(KeyError):
            edf.signal("NONEXISTENT")

    def test_signal_index_out_of_range(self):
        edf, _ = load_fixture("short_psg")
        with pytest.raises(IndexError):
            edf.signal(9999)

    def test_sample_index_out_of_range(self):
        edf, _ = load_fixture("short_psg")
        sig = edf.signal(0)
        with pytest.raises(IndexError):
            sig[len(sig)]


class TestContextManager:
    def test_with_statement(self):
        path = str(FIXTURES / "short_psg.edf")
        with edfarray.EdfFile(path) as f:
            assert f.num_signals == 7
            assert len(f.signal(0)) > 0
