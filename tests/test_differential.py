"""Differential tests against pyedflib.

pyedflib wraps the reference edflib C library, so it is the authority for sample values and
annotation onsets. It cannot read EDF+D, so those fixtures are skipped here.
"""

import edfarray
import numpy as np
import pytest
from conftest import FIXTURES

pyedflib = pytest.importorskip("pyedflib")

READABLE = ["test_generator", "test_generator_2", "S001R01", "edfPlusC"]


def _pyedflib_open(name):
    return pyedflib.EdfReader(str(FIXTURES / f"{name}.edf"))


@pytest.mark.parametrize("name", READABLE)
def test_signal_values_match_pyedflib(name):
    reader = _pyedflib_open(name)
    try:
        labels = reader.getSignalLabels()
        expected = {i: reader.readSignal(i) for i in range(len(labels))}
    finally:
        reader._close()

    f = edfarray.EdfFile(str(FIXTURES / f"{name}.edf"))
    for idx, ref in expected.items():
        if f.signal_labels()[idx].startswith("EDF Annotations"):
            continue
        got = f.signal(idx).to_physical()
        assert got.shape == ref.shape, f"{name}[{idx}] length differs"
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", READABLE)
def test_annotation_onsets_match_pyedflib(name):
    reader = _pyedflib_open(name)
    try:
        onsets, durations, texts = reader.readAnnotations()
    finally:
        reader._close()

    f = edfarray.EdfFile(str(FIXTURES / f"{name}.edf"))
    got = sorted((a.onset, a.text) for a in f.annotations)
    ref = sorted((float(o), str(t)) for o, t in zip(onsets, texts))
    assert len(got) == len(ref), f"{name}: annotation count differs"
    for (g_on, g_txt), (r_on, r_txt) in zip(got, ref):
        assert g_txt == r_txt
        assert g_on == pytest.approx(r_on, abs=1e-6)


@pytest.mark.parametrize("name", READABLE)
def test_sample_rates_and_counts_match_pyedflib(name):
    reader = _pyedflib_open(name)
    try:
        labels = reader.getSignalLabels()
        rates = [reader.getSampleFrequency(i) for i in range(len(labels))]
        counts = [reader.getNSamples()[i] for i in range(len(labels))]
    finally:
        reader._close()

    f = edfarray.EdfFile(str(FIXTURES / f"{name}.edf"))
    for i, label in enumerate(labels):
        if label.startswith("EDF Annotations"):
            continue
        sig = f.signal(i)
        assert sig.sample_rate == pytest.approx(rates[i])
        assert len(sig) == counts[i]


def test_subsecond_start_matches_pyedflib(tmp_path):
    # Built here rather than shipped: every fixture starts at onset 0, which is exactly the
    # case that cannot detect a mis-normalized subsecond start.
    from edfbuilder import build_edf_plus

    onsets = [0.25 + i for i in range(4)]
    path = build_edf_plus(
        tmp_path / "sub.edf",
        record_onsets=onsets,
        annotations=[(2, onsets[2], "EVENT", 0)],
    )

    reader = pyedflib.EdfReader(path)
    try:
        ref_onsets, _, ref_texts = reader.readAnnotations()
    finally:
        reader._close()

    f = edfarray.EdfFile(path)
    got = [(a.onset, a.text) for a in f.annotations]
    ref = [(float(o), str(t)) for o, t in zip(ref_onsets, ref_texts)]
    assert got == ref
