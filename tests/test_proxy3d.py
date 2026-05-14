import numpy as np
import pytest

from edfarray import EdfFile, Proxy3D
from conftest import FIXTURES


def open_edf(name: str) -> EdfFile:
    return EdfFile(str(FIXTURES / f"{name}.edf"))


def largest_rect_group(f: EdfFile):
    rects = [g for g in f.signal_groups() if g.kind == "rectangular"]
    if not rects:
        pytest.skip("no rectangular groups")
    return max(rects, key=len)


class TestProxy3D:
    def test_shape_is_records_channels_spr(self):
        f = open_edf("test_generator")
        g = largest_rect_group(f)
        p = f.proxy_3d(g)
        n_records, n_channels, spr = p.shape
        assert n_channels == len(g)
        assert spr == g.samples_per_record
        assert n_records * spr == g.max_samples
        assert isinstance(p, Proxy3D)

    def test_sample_rate(self):
        f = open_edf("test_generator")
        g = largest_rect_group(f)
        p = f.proxy_3d(g)
        assert p.sample_rate == g.sample_rate

    def test_scalar_index(self):
        f = open_edf("test_generator")
        g = largest_rect_group(f)
        p = f.proxy_3d(g)
        val = p[0, 0, 0]
        assert isinstance(val, float)

    def test_axis_slices_match_2d(self):
        f = open_edf("test_generator")
        g = largest_rect_group(f)
        p3 = f.proxy_3d(g)
        p2 = f.proxy_2d(g)
        _, _, spr = p3.shape
        block = p3[0:3, :, :]
        assert block.shape == (3, len(g), spr)
        expected = p2[:, 0 : 3 * spr]
        for rec in range(3):
            for ch in range(len(g)):
                np.testing.assert_allclose(
                    block[rec, ch, :], expected[ch, rec * spr : (rec + 1) * spr],
                    atol=1e-10,
                )

    def test_rejects_open_group(self):
        f = open_edf("test_generator")
        group = f.signal_group(f.ordinary_signal_indices())
        if group.kind != "open":
            pytest.skip("test_generator should have mixed rates")
        with pytest.raises(ValueError):
            f.proxy_3d(group)

    def test_step_not_supported(self):
        f = open_edf("test_generator")
        g = largest_rect_group(f)
        p = f.proxy_3d(g)
        with pytest.raises(ValueError):
            p[0:4:2, :, :]

    def test_stride_info_none_when_annotations_interleaved(self):
        f = open_edf("test_generator")
        g = largest_rect_group(f)
        p = f.proxy_3d(g)
        assert p.stride_info() is None
        assert p.supports_strided_view is False

    def test_repr_contains_shape(self):
        f = open_edf("test_generator")
        g = largest_rect_group(f)
        p = f.proxy_3d(g)
        s = repr(p)
        assert "Proxy3D" in s
        assert "shape=" in s

    def test_wrong_index_count(self):
        f = open_edf("test_generator")
        g = largest_rect_group(f)
        p = f.proxy_3d(g)
        with pytest.raises(IndexError, match="3 indices"):
            p[0, 0]
