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

    def test_record_axis_step_not_supported(self):
        f = open_edf("test_generator")
        g = largest_rect_group(f)
        p = f.proxy_3d(g)
        with pytest.raises(ValueError):
            p[0:4:2, :, :]
        with pytest.raises(ValueError):
            p[:, 0:4:2, :]

    def test_sample_axis_step(self):
        f = open_edf("test_generator")
        g = largest_rect_group(f)
        p = f.proxy_3d(g)
        full = p[0:5, :, :]
        assert np.array_equal(p[0:5, :, ::4], full[:, :, ::4])
        assert np.array_equal(p[0:5, :, ::-1], full[:, :, ::-1])

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
        # A wrong-arity key is a type error, matching numpy, not an out-of-range error.
        with pytest.raises(TypeError, match="3 indices"):
            p[0, 0]


def _strided_group(f):
    """A rectangular group whose channels form a contiguous index range, which is what makes a
    zero-copy stride view possible."""
    for g in f.signal_groups():
        if g.kind == "rectangular" and f.proxy_3d(g).stride_info() is not None:
            return g
    pytest.skip("no stride-eligible group in this fixture")


class TestStrideInfoArithmetic:
    """The stride numbers describe a zero-copy view of the raw file, so they must be checked
    against the actual bytes, not just for presence."""

    def test_strides_locate_real_samples(self, tmp_path):
        import numpy as np

        f = open_edf("test_generator_2")
        g = _strided_group(f)
        p = f.proxy_3d(g)
        info = p.stride_info()
        assert info is not None

        raw = np.memmap(
            str(FIXTURES / "test_generator_2.edf"), dtype=np.uint8, mode="r"
        )
        n_rec, n_ch, spr = info["shape"]
        assert (n_rec, n_ch, spr) == p.shape

        view = np.lib.stride_tricks.as_strided(
            raw[info["base_offset"] :].view(np.int16),
            shape=(n_rec, n_ch, spr),
            strides=(
                info["record_stride_bytes"],
                info["channel_stride_bytes"],
                info["sample_stride_bytes"],
            ),
        )
        # Compare against decoded digital values for a few records.
        expected = f.proxy_3d(g).read_digital(0, 3, 0, n_ch)
        np.testing.assert_array_equal(view[0:3].astype(np.int32), expected)

    def test_stride_view_stays_inside_the_file(self):
        f = open_edf("test_generator_2")
        g = _strided_group(f)
        info = f.proxy_3d(g).stride_info()
        n_rec, n_ch, spr = info["shape"]
        last_byte = (
            info["base_offset"]
            + (n_rec - 1) * info["record_stride_bytes"]
            + (n_ch - 1) * info["channel_stride_bytes"]
            + spr * info["sample_stride_bytes"]
        )
        assert last_byte <= (FIXTURES / "test_generator_2.edf").stat().st_size
