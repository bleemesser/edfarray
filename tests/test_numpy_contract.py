"""The supported-indexing contract, frozen for 1.0.

edfarray accepts a deliberate subset of numpy indexing. Anything outside it must raise, not
quietly do something different -- a wrong answer is worse than an unsupported one.
"""

import edfarray
import numpy as np
import pytest
from conftest import FIXTURES

PATH = str(FIXTURES / "test_generator.edf")


@pytest.fixture
def file():
    return edfarray.EdfFile(PATH)


@pytest.fixture
def sig(file):
    return file.signal(0)


@pytest.fixture
def proxies(file):
    group = max(file.signal_groups(), key=len)
    return file.proxy_2d(group), file.proxy_3d(group)


class TestSupported:
    def test_integer_and_negative_indexing(self, sig):
        assert sig[-1] == sig[len(sig) - 1]
        assert sig[0] == sig[-len(sig)]

    def test_slices_including_steps(self, sig):
        full = sig.to_physical()
        np.testing.assert_array_equal(sig[10:20], full[10:20])
        np.testing.assert_array_equal(sig[::4][:50], full[::4][:50])
        np.testing.assert_array_equal(sig[100:10:-3], full[100:10:-3])

    def test_empty_slices_match_numpy_shapes(self, sig, proxies):
        p2, p3 = proxies
        assert sig[5:5].shape == (0,)
        assert p2[0:2, 5:5].shape == (2, 0)
        assert p2[0:0, 0:10].shape == (0, 10)
        assert p3[0:0, :, :].shape[0] == 0
        assert p3[0:2, :, 5:5].shape[2] == 0

    def test_list_indexing_on_signal_axis(self, proxies):
        p2, _ = proxies
        picked = p2[[2, 0], 0:10]
        assert picked.shape == (2, 10)
        np.testing.assert_array_equal(picked[0], p2[2, 0:10])
        np.testing.assert_array_equal(picked[1], p2[0, 0:10])

    def test_scalar_results_are_floats(self, sig, proxies):
        p2, p3 = proxies
        assert isinstance(sig[0], float)
        assert isinstance(p2[0, 0], float)
        assert isinstance(p3[0, 0, 0], float)


class TestArrayProtocol:
    def test_signal_asarray_matches_to_physical(self, sig):
        np.testing.assert_array_equal(np.asarray(sig), sig.to_physical())

    def test_proxy_asarray(self, proxies):
        p2, p3 = proxies
        np.testing.assert_array_equal(np.asarray(p2), p2[:, :])
        assert np.asarray(p3).shape == p3.shape

    def test_dtype_shape_ndim(self, sig, proxies):
        p2, p3 = proxies
        assert (sig.dtype, sig.ndim, sig.shape) == (np.dtype("float64"), 1, (len(sig),))
        assert (p2.ndim, p3.ndim) == (2, 3)
        assert p2.dtype == np.dtype("float64")

    def test_asarray_honors_dtype(self, sig):
        assert np.asarray(sig, dtype=np.float32).dtype == np.float32

    def test_copy_false_is_refused(self, sig):
        # The data is decoded on access, so a no-copy view is impossible.
        with pytest.raises(ValueError):
            np.asarray(sig, copy=False)


class TestRejected:
    def test_bool_is_not_treated_as_an_integer(self, sig):
        # bool subclasses int, so a naive extract reads True as index 1 and returns the wrong
        # sample instead of numpy's mask semantics.
        with pytest.raises(TypeError, match="boolean"):
            sig[True]

    def test_bool_rejected_on_proxy_axes(self, proxies):
        p2, p3 = proxies
        with pytest.raises(TypeError):
            p2[True, 0]
        with pytest.raises(TypeError):
            p3[True, 0, 0]

    @pytest.mark.parametrize("key", [None, Ellipsis, "label", 1.5, {0}])
    def test_unsupported_key_types_raise_type_error(self, sig, key):
        with pytest.raises(TypeError):
            sig[key]

    def test_ndarray_fancy_indexing_raises(self, sig):
        with pytest.raises(TypeError):
            sig[np.array([0, 2, 4])]

    def test_boolean_mask_raises(self, sig):
        mask = np.zeros(len(sig), dtype=bool)
        with pytest.raises(TypeError):
            sig[mask]

    def test_wrong_arity_raises_type_error(self, proxies):
        p2, p3 = proxies
        with pytest.raises(TypeError, match="2 indices"):
            p2[0, 0, 0]
        with pytest.raises(TypeError, match="3 indices"):
            p3[0, 0]
        with pytest.raises(TypeError, match="2-tuple"):
            p2[0]

    def test_out_of_range_raises_out_of_range_error(self, sig, proxies):
        p2, _ = proxies
        for call in (lambda: sig[len(sig)], lambda: p2[999, 0], lambda: p2[0, 10**9]):
            with pytest.raises(edfarray.OutOfRangeError):
                call()

    def test_out_of_range_is_also_an_index_error(self, sig):
        # Existing code that catches IndexError must keep working.
        with pytest.raises(IndexError):
            sig[len(sig)]
