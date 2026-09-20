import datetime
from pathlib import Path

import numpy as np
import pytest

import edfarray


def _signal(samples_per_record=256, label="EEG Fpz"):
    return edfarray.WriterSignal(
        label=label,
        physical_dimension="uV",
        physical_min=-3200.0,
        physical_max=3200.0,
        digital_min=-32768,
        digital_max=32767,
        samples_per_record=samples_per_record,
    )


def _ramp(n):
    return np.linspace(-100.0, 100.0, n, dtype=np.float64)


def test_write_edf_plain_roundtrip(tmp_path: Path):
    p = tmp_path / "plain.edf"
    sig = _signal()
    data = _ramp(256 * 4)
    edfarray.write_edf(
        str(p), variant="EDF", record_duration=1.0,
        signals=[sig], data=[data],
    )
    f = edfarray.EdfFile(str(p))
    assert f.variant == "EDF"
    assert f.num_records == 4
    assert len(f.signal_labels()) == 1
    s = f.signal(0)
    got = s[0:len(data)]
    np.testing.assert_allclose(got, data, atol=0.2)


def test_write_edf_plus_c_with_annotations(tmp_path: Path):
    p = tmp_path / "plus.edf"
    sig = _signal()
    data = _ramp(256 * 5)
    anns = [
        edfarray.Annotation(onset=0.5, text="start"),
        edfarray.Annotation(onset=2.25, text="spike", duration=1.5),
        edfarray.Annotation(onset=4.9, text="end"),
    ]
    edfarray.write_edf(
        str(p), variant="EDF+C", record_duration=1.0,
        signals=[sig], data=[data], annotations=anns,
        start_datetime=datetime.datetime(2026, 1, 2, 12, 30, 0),
    )
    f = edfarray.EdfFile(str(p))
    assert f.variant == "EDF+C"
    assert f.num_records == 5
    out = list(f.annotations)
    assert len(out) == 3
    assert out[0].text == "start"
    assert out[1].duration == pytest.approx(1.5)
    assert out[2].text == "end"
    assert f.start_datetime.year == 2026


def test_streaming_writer_context_manager(tmp_path: Path):
    p = tmp_path / "stream.edf"
    sig = _signal()
    with edfarray.EdfWriter(
        str(p), variant="EDF+C", record_duration=1.0, signals=[sig]
    ) as w:
        for r in range(3):
            chunk = np.full(256, float(r * 10), dtype=np.float64)
            w.add_annotation(edfarray.Annotation(onset=r + 0.1, text=f"r{r}"))
            w.write_record([chunk])

    f = edfarray.EdfFile(str(p))
    assert f.num_records == 3
    assert [a.text for a in f.annotations] == ["r0", "r1", "r2"]


def test_streaming_writer_explicit_finish(tmp_path: Path):
    p = tmp_path / "stream2.edf"
    sig = _signal()
    w = edfarray.EdfWriter(
        str(p), variant="EDF", record_duration=0.5, signals=[sig]
    )
    chunk = np.zeros(256, dtype=np.float64)
    w.write_record([chunk])
    w.write_record([chunk])
    w.finish()
    w.finish()
    f = edfarray.EdfFile(str(p))
    assert f.num_records == 2
    assert f.record_duration == 0.5


def test_writer_rejects_annotations_in_plain_edf(tmp_path: Path):
    p = tmp_path / "bad.edf"
    sig = _signal()
    with pytest.raises(Exception):
        edfarray.write_edf(
            str(p), variant="EDF", record_duration=1.0,
            signals=[sig], data=[_ramp(256)],
            annotations=[edfarray.Annotation(onset=0.0, text="x")],
        )


def test_writer_validates_record_length(tmp_path: Path):
    p = tmp_path / "bad.edf"
    sig = _signal(samples_per_record=256)
    w = edfarray.EdfWriter(
        str(p), variant="EDF", record_duration=1.0, signals=[sig]
    )
    with pytest.raises(Exception):
        w.write_record([np.zeros(100, dtype=np.float64)])


def test_writer_unknown_variant_rejected(tmp_path: Path):
    p = tmp_path / "bad.edf"
    sig = _signal()
    with pytest.raises(ValueError):
        edfarray.write_edf(
            str(p), variant="EDF+X", record_duration=1.0,
            signals=[sig], data=[_ramp(256)],
        )


def test_write_bdf_roundtrip(tmp_path: Path):
    p = tmp_path / "test.bdf"
    sig = edfarray.WriterSignal(
        label="ECG",
        physical_dimension="mV",
        physical_min=-5.0,
        physical_max=5.0,
        digital_min=-(1 << 23),
        digital_max=(1 << 23) - 1,
        samples_per_record=128,
    )
    data = np.linspace(-2.0, 2.0, 128 * 3, dtype=np.float64)
    edfarray.write_edf(
        str(p), variant="BDF", record_duration=1.0,
        signals=[sig], data=[data],
    )
    f = edfarray.EdfFile(str(p))
    assert f.variant == "BDF"
    s = f.signal(0)
    np.testing.assert_allclose(s[0:len(data)], data, atol=1e-4)


def test_edffile_write_to(tmp_path: Path):
    src = tmp_path / "src.edf"
    dst = tmp_path / "dst.edf"
    sig = _signal()
    edfarray.write_edf(
        str(src), variant="EDF+C", record_duration=1.0,
        signals=[sig], data=[_ramp(256 * 3)],
        annotations=[edfarray.Annotation(onset=1.5, text="mid")],
    )
    f = edfarray.EdfFile(str(src))
    f.write_to(str(dst))
    g = edfarray.EdfFile(str(dst))
    assert g.num_records == 3
    assert [a.text for a in g.annotations] == ["mid"]


def test_write_to_inherits_large_annotation_channel(tmp_path: Path):
    src = tmp_path / "wide.edf"
    dst = tmp_path / "copy.edf"
    from edfbuilder import build_edf_plus

    text = "N" * 200
    anns = [(r, r + 0.5, text, 0) for r in range(4)]
    build_edf_plus(
        str(src),
        record_onsets=[0.0, 1.0, 2.0, 3.0],
        annotations=anns,
        variant="EDF+C",
        annotation_bytes=512,
    )
    f = edfarray.EdfFile(str(src))
    assert len(f.annotations) == 4
    f.write_to(str(dst))
    g = edfarray.EdfFile(str(dst))
    got = list(g.annotations)
    assert len(got) == 4
    for i, ann in enumerate(got):
        assert ann.text == text
        assert ann.onset == pytest.approx(i + 0.5)


def test_write_with_patient_recording_ids(tmp_path: Path):
    p = tmp_path / "meta.edf"
    sig = _signal()
    edfarray.write_edf(
        str(p), variant="EDF", record_duration=1.0,
        signals=[sig], data=[_ramp(256)],
        patient_id="MCH-1234 M 02-MAR-1980 Doe_J",
        recording_id="Startdate 03-MAR-2026 PSG-1 Tech1 Foo",
    )
    f = edfarray.EdfFile(str(p))
    assert "MCH-1234" in f.patient_id
    assert "PSG-1" in f.recording_id
