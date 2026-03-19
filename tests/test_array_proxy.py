import numpy as np
import pytest

from edfarray._core import EdfFile
from conftest import FIXTURES


def open_edf(name):
    return EdfFile(str(FIXTURES / f"{name}.edf"))


def get_same_rate_proxy(f):
    """Get an array proxy using the largest group of same-rate signals."""
    by_rate = f.signal_indices_by_rate()
    if not by_rate:
        pytest.skip("no ordinary signals")
    largest_group = max(by_rate.values(), key=len)
    return f.array_proxy(largest_group), largest_group


class TestArrayProxy:
    def test_basic_shape(self):
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        sig = f.signal(indices[0])
        assert proxy.shape == (len(indices), len(sig))

    def test_sample_rate(self):
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        sig = f.signal(indices[0])
        assert proxy.sample_rate == sig.sample_rate

    def test_single_element(self):
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        sig = f.signal(indices[0])
        assert abs(proxy[0, 0] - sig[0]) < 1e-10

    def test_single_signal_slice(self):
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        sig = f.signal(indices[0])
        arr = proxy[0, 0:100]
        expected = sig[0:100]
        np.testing.assert_allclose(arr, expected, atol=1e-10)

    def test_multi_signal_slice(self):
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        n = min(100, proxy.shape[1])
        arr = proxy[:, 0:n]
        assert arr.shape == (len(indices), n)
        for i, idx in enumerate(indices):
            sig = f.signal(idx)
            np.testing.assert_allclose(arr[i], sig[0:n], atol=1e-10)

    def test_negative_index(self):
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        last_sig = f.signal(indices[-1])
        assert abs(proxy[-1, -1] - last_sig[-1]) < 1e-10

    def test_fancy_indexing(self):
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        if len(indices) < 3:
            pytest.skip("need at least 3 signals")
        arr = proxy[[0, 2], 0:50]
        assert arr.shape == (2, 50)
        for i, si in enumerate([0, 2]):
            sig = f.signal(indices[si])
            np.testing.assert_allclose(arr[i], sig[0:50], atol=1e-10)

    def test_out_of_range(self):
        f = open_edf("test_generator")
        proxy, _ = get_same_rate_proxy(f)
        with pytest.raises(IndexError):
            proxy[proxy.shape[0], 0]

    def test_repr(self):
        f = open_edf("test_generator")
        proxy, _ = get_same_rate_proxy(f)
        assert "ArrayProxy" in repr(proxy)

    def test_specific_signal_indices(self):
        f = open_edf("test_generator")
        indices = f.ordinary_signal_indices()
        proxy = f.array_proxy([indices[0]])
        assert proxy.shape[0] == 1

    def test_edf_plus_c(self):
        f = open_edf("test_generator_2")
        proxy, indices = get_same_rate_proxy(f)
        sig = f.signal(indices[0])
        n = min(50, proxy.shape[1])
        arr = proxy[0, 0:n]
        np.testing.assert_allclose(arr, sig[0:n], atol=1e-10)

    def test_column_vector(self):
        """proxy[slice, int] returns 1D array of one sample per signal."""
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        arr = proxy[:, 0]
        assert arr.shape == (len(indices),)
        for i, idx in enumerate(indices):
            sig = f.signal(idx)
            assert abs(arr[i] - sig[0]) < 1e-10

    def test_single_signal_single_sample(self):
        """proxy[int, int] returns a scalar float."""
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        val = proxy[0, 0]
        assert isinstance(val, float)

    def test_2d_result_is_ndarray(self):
        """proxy[:, slice] returns a 2D numpy array."""
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        arr = proxy[:, 0:10]
        assert arr.ndim == 2
        assert arr.shape == (len(indices), 10)

    def test_empty_proxy(self):
        f = open_edf("test_generator")
        proxy = f.array_proxy([])
        assert proxy.shape == (0, 0)

    def test_mixed_rates_error(self):
        f = open_edf("test_generator")
        with pytest.raises(ValueError, match="mixed sample rates"):
            f.array_proxy()

    def test_requires_two_indices(self):
        f = open_edf("test_generator")
        proxy, _ = get_same_rate_proxy(f)
        with pytest.raises(IndexError, match="2 indices"):
            proxy[0]

    def test_step_not_supported(self):
        f = open_edf("test_generator")
        proxy, _ = get_same_rate_proxy(f)
        with pytest.raises(ValueError, match="step"):
            proxy[0, 0:100:2]

    def test_negative_sample_index(self):
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        val_neg = proxy[0, -1]
        sig = f.signal(indices[0])
        val_pos = sig[-1]
        assert abs(val_neg - val_pos) < 1e-10


class TestSignalIndicesByRate:
    def test_returns_dict(self):
        f = open_edf("test_generator")
        result = f.signal_indices_by_rate()
        assert isinstance(result, dict)
        all_indices = []
        for indices in result.values():
            all_indices.extend(indices)
        assert sorted(all_indices) == sorted(f.ordinary_signal_indices())

    def test_groups_correctly(self):
        f = open_edf("test_generator")
        result = f.signal_indices_by_rate()
        for rate, indices in result.items():
            rates = set()
            for idx in indices:
                sig = f.signal(idx)
                rates.add(sig.sample_rate)
            assert len(rates) == 1


class TestScanProgress:
    def test_scan_complete(self):
        f = open_edf("test_generator")
        done, total = f.scan_progress
        assert done == total

    def test_scan_complete_edf_plus(self):
        f = open_edf("test_generator_2")
        _ = f.annotations
        done, total = f.scan_progress
        assert done == total

    def test_annotations_ready_plain_edf(self):
        f = open_edf("test_generator")
        assert f.annotations_ready is True

    def test_annotations_ready_after_access(self):
        f = open_edf("test_generator_2")
        _ = f.annotations
        assert f.annotations_ready is True

    def test_scan_progress_matches_ready(self):
        f = open_edf("edfPlusC")
        _ = f.annotations
        done, total = f.scan_progress
        assert done == total
        assert f.annotations_ready is True
