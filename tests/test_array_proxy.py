import numpy as np
import pytest

from edfarray import EdfFile, Proxy2D
from conftest import FIXTURES


def open_edf(name: str) -> EdfFile:
    return EdfFile(str(FIXTURES / f"{name}.edf"))


def get_same_rate_proxy(f: EdfFile) -> tuple[Proxy2D, list[int]]:
    """Return a 2D proxy for the largest same-rate signal group."""
    groups = f.signal_groups()
    if not groups:
        pytest.skip("no ordinary signals")
    largest = max(groups, key=len)
    return f.proxy_2d(largest), largest.indices


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
        assert "Proxy2D" in repr(proxy)

    def test_specific_signal_indices(self):
        f = open_edf("test_generator")
        indices = f.ordinary_signal_indices()
        group = f.signal_group([indices[0]])
        proxy = f.proxy_2d(group)
        assert proxy.shape[0] == 1

    def test_edf_plus_c(self):
        f = open_edf("test_generator_2")
        proxy, indices = get_same_rate_proxy(f)
        sig = f.signal(indices[0])
        n = min(50, proxy.shape[1])
        arr = proxy[0, 0:n]
        np.testing.assert_allclose(arr, sig[0:n], atol=1e-10)

    def test_column_vector(self):
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        arr = proxy[:, 0]
        assert arr.shape == (len(indices),)
        for i, idx in enumerate(indices):
            sig = f.signal(idx)
            assert abs(arr[i] - sig[0]) < 1e-10

    def test_single_signal_single_sample(self):
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        val = proxy[0, 0]
        assert isinstance(val, float)

    def test_2d_result_is_ndarray(self):
        f = open_edf("test_generator")
        proxy, indices = get_same_rate_proxy(f)
        arr = proxy[:, 0:10]
        assert arr.ndim == 2
        assert arr.shape == (len(indices), 10)

    def test_empty_group_rejected(self):
        f = open_edf("test_generator")
        with pytest.raises(ValueError):
            f.signal_group([])

    def test_open_group_accepted_with_pad(self):
        f = open_edf("test_generator")
        group = f.signal_group(f.ordinary_signal_indices())
        assert group.kind == "open"
        proxy = f.proxy_2d(group, pad_mode="nan")
        assert proxy.sample_rate is None

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


class TestSignalGroups:
    def test_returns_list(self):
        f = open_edf("test_generator")
        groups = f.signal_groups()
        assert isinstance(groups, list)
        all_indices = []
        for g in groups:
            all_indices.extend(g.indices)
        assert sorted(all_indices) == sorted(f.ordinary_signal_indices())

    def test_groups_share_sample_rate(self):
        f = open_edf("test_generator")
        for g in f.signal_groups():
            rates = {f.signal(i).sample_rate for i in g.indices}
            assert len(rates) == 1
            assert next(iter(rates)) == g.sample_rate

    def test_covers_all_ordinary_flag(self):
        f = open_edf("test_generator")
        groups = f.signal_groups()
        expected = len(groups) == 1
        for g in groups:
            assert g.covers_all_ordinary == expected

    def test_kind_is_rectangular_in_file(self):
        f = open_edf("test_generator")
        for g in f.signal_groups():
            assert g.kind == "rectangular"
            assert g.min_samples == g.max_samples
            assert g.is_rectangular

    def test_singleton_flag(self):
        f = open_edf("test_generator")
        for g in f.signal_groups():
            assert g.is_singleton == (len(g) == 1)


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
