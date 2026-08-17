"""`EdfFile.write_to` and its documented lossy conversions.

Each caveat in the docstring is a real behavior change, so each is pinned here. A silent
change to any of them would lose data without an error.
"""

import edfarray
import numpy as np
import pytest
from conftest import FIXTURES

PLAIN = str(FIXTURES / "test_generator.edf")
PLUS_C = str(FIXTURES / "edfPlusC.edf")
PLUS_D = str(FIXTURES / "edfPlusD.edf")


def ordinary_signals(f):
    return [f.signal(i).to_physical() for i in f.ordinary_signal_indices()]


def test_same_variant_roundtrip_preserves_samples(tmp_path):
    src = edfarray.EdfFile(PLAIN)
    out = str(tmp_path / "copy.edf")
    src.write_to(out)

    copy = edfarray.EdfFile(out)
    assert copy.variant == src.variant
    assert copy.num_records == src.num_records
    for a, b in zip(ordinary_signals(src), ordinary_signals(copy)):
        np.testing.assert_allclose(b, a, atol=0.2)


def test_plus_c_roundtrip_preserves_annotations(tmp_path):
    src = edfarray.EdfFile(PLUS_C)
    out = str(tmp_path / "plus.edf")
    src.write_to(out, variant="EDF+C")

    copy = edfarray.EdfFile(out)
    assert [(a.onset, a.text) for a in copy.annotations] == [
        (a.onset, a.text) for a in src.annotations
    ]


def test_transcoding_to_plain_edf_drops_annotations(tmp_path):
    # Documented: a plain variant has no annotation channel, so annotations cannot survive.
    src = edfarray.EdfFile(PLUS_C)
    assert len(src.annotations) > 0
    out = str(tmp_path / "plain.edf")
    src.write_to(out, variant="EDF")

    copy = edfarray.EdfFile(out)
    assert copy.variant == "EDF"
    assert copy.annotations == []


def test_transcoding_plus_d_to_plus_c_discards_gaps(tmp_path):
    # Documented: records are streamed contiguously, so the discontinuity is lost and timing
    # becomes uniform.
    src = edfarray.EdfFile(PLUS_D)
    src_steps = np.diff(src.signal(0).times())
    assert src_steps.max() > src_steps.min() * 2, "source must actually be discontinuous"

    out = str(tmp_path / "flat.edf")
    src.write_to(out, variant="EDF+C")

    copy = edfarray.EdfFile(out)
    assert copy.variant == "EDF+C"
    steps = np.diff(copy.signal(0).times())
    assert np.allclose(steps, steps[0]), "transcoded timing should be uniform"


def test_transcoding_preserves_sample_values_across_variants(tmp_path):
    src = edfarray.EdfFile(PLUS_D)
    out = str(tmp_path / "asc.edf")
    src.write_to(out, variant="EDF+C")
    copy = edfarray.EdfFile(out)
    # Timing changes, but the samples themselves must not.
    for a, b in zip(ordinary_signals(src), ordinary_signals(copy)):
        np.testing.assert_allclose(b, a, atol=0.5)


def test_unknown_target_variant_rejected(tmp_path):
    src = edfarray.EdfFile(PLAIN)
    with pytest.raises(edfarray.InvalidArgumentError):
        src.write_to(str(tmp_path / "x.edf"), variant="EDF++")


def test_write_to_after_close_raises(tmp_path):
    src = edfarray.EdfFile(PLAIN)
    src.close()
    with pytest.raises(edfarray.ClosedFileError):
        src.write_to(str(tmp_path / "x.edf"))
