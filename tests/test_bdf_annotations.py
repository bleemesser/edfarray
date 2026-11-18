"""Annotation channels labelled "BDF Annotations", which the BDF+ specification requires."""

import edfarray
import numpy as np
import pytest
from edfbuilder import build_edf_plus


def _label_for(variant):
    return "BDF Annotations" if variant.startswith("BDF") else "EDF Annotations"


def _signal(label="EEG Fpz", bdf=False):
    limit = 8_388_607 if bdf else 32_767
    return edfarray.WriterSignal(
        label=label,
        physical_dimension="uV",
        physical_min=-3200.0,
        physical_max=3200.0,
        digital_min=-limit - 1,
        digital_max=limit,
        samples_per_record=256,
    )


def _ramp(n):
    return np.linspace(-100.0, 100.0, n, dtype=np.float64)


@pytest.mark.parametrize("variant", ["BDF+C", "EDF+C"])
def test_annotation_channel_is_not_data(tmp_path, variant):
    path = tmp_path / "annotated.edf"
    build_edf_plus(
        str(path),
        record_onsets=[0.0, 1.0, 2.0],
        annotations=[(0, 0.5, "start", 0), (2, 2.5, "stop", 0)],
        variant=variant,
        annotation_label=_label_for(variant),
    )
    with edfarray.EdfFile(str(path)) as f:
        assert f.variant == variant
        assert f.ordinary_signal_indices() == [0]
        assert len(f.signal_groups()) == 1, "the TAL channel was treated as data"
        assert [a.text for a in f.annotations] == ["start", "stop"]

        dst = tmp_path / "copy.edf"
        f.write_to(str(dst))

    with edfarray.EdfFile(str(dst)) as g:
        assert g.signal_labels() == ["Sig1", _label_for(variant)]
        assert g.ordinary_signal_indices() == [0]
        assert [a.text for a in g.annotations] == ["start", "stop"]
        assert g.warnings == []


@pytest.mark.parametrize(
    "variant,label",
    [
        ("BDF+C", "BDF Annotations"),
        ("BDF+D", "BDF Annotations"),
        ("EDF+C", "EDF Annotations"),
    ],
)
def test_writer_names_the_annotation_channel(tmp_path, variant, label):
    path = tmp_path / "written.edf"
    bdf = variant.startswith("BDF")
    kwargs = {}
    if variant.endswith("D"):
        kwargs["record_onsets"] = [0.0, 1.0]
    edfarray.write_edf(
        str(path),
        variant=variant,
        record_duration=1.0,
        signals=[_signal(bdf=bdf)],
        data=[_ramp(256 * 2)],
        annotations=[edfarray.Annotation(0.5, "event")],
        **kwargs,
    )
    with edfarray.EdfFile(str(path)) as f:
        assert f.signal_labels() == ["EEG Fpz", label]
        assert f.ordinary_signal_indices() == [0]
        assert [a.text for a in f.annotations] == ["event"]


@pytest.mark.parametrize("variant", ["EDF+C", "BDF+C"])
@pytest.mark.parametrize("label", ["EDF Annotations", "BDF Annotations"])
def test_writer_rejects_a_reserved_label(tmp_path, variant, label):
    path = tmp_path / "reserved.edf"
    with pytest.raises(edfarray.InvalidFileError):
        edfarray.write_edf(
            str(path),
            variant=variant,
            record_duration=1.0,
            signals=[_signal(label=label, bdf=variant.startswith("BDF"))],
            data=[_ramp(256)],
        )
