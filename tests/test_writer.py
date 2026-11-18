import datetime
from pathlib import Path

import numpy as np
import pytest

import edfarray
from edfbuilder import build_edf_plus


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


def test_write_to_preserves_edf_plus_d_gaps(tmp_path: Path):
    src = tmp_path / "gap.edf"
    dst = tmp_path / "copy.edf"
    flat = tmp_path / "flat.edf"
    onsets = [0.0, 1.0, 3.0, 6.0]
    build_edf_plus(
        str(src),
        record_onsets=onsets,
        annotations=[(2, 3.5, "gap-event", 0)],
        variant="EDF+D",
    )
    f = edfarray.EdfFile(str(src))
    src_times = f.signal(0).times()

    f.write_to(str(dst))
    g = edfarray.EdfFile(str(dst))
    assert g.variant == "EDF+D"
    np.testing.assert_allclose(g.signal(0).times(), src_times, atol=1e-9)
    anns = list(g.annotations)
    assert len(anns) == 1
    assert anns[0].onset == pytest.approx(3.5)

    # A non-+D target flattens timing: onsets become record_idx * record_duration, so the
    # gap vanishes and sample times run contiguously from 0.
    f.write_to(str(flat), variant="EDF+C")
    h = edfarray.EdfFile(str(flat))
    flat_times = h.signal(0).times()
    rate = len(flat_times) / (h.num_records * h.record_duration)
    np.testing.assert_allclose(flat_times, np.arange(len(flat_times)) / rate, atol=1e-9)


def test_write_edf_edf_plus_d_onsets(tmp_path: Path):
    p = tmp_path / "gap.edf"
    sig = _signal(samples_per_record=100)
    # Three records: onsets 0, 1, 3 leave a 1-second gap before the last record.
    onsets = [0.0, 1.0, 3.0]
    data = np.arange(300, dtype=np.float64)
    edfarray.write_edf(
        str(p), variant="EDF+D", record_duration=1.0,
        signals=[sig], data=[data],
        annotations=[edfarray.Annotation(onset=3.5, text="after-gap")],
        record_onsets=onsets,
    )
    f = edfarray.EdfFile(str(p))
    assert f.variant == "EDF+D"
    s = f.signal(0)
    # One onset per record: sample i sits at onsets[i // 100] + i % 100 / rate.
    times = s.times()
    assert times[0] == pytest.approx(0.0)
    assert times[100] == pytest.approx(1.0)
    assert times[200] == pytest.approx(3.0)
    anns = [a.text for a in f.annotations if a.text]
    assert "after-gap" in anns


def test_streaming_writer_edf_plus_d_onsets(tmp_path: Path):
    p = tmp_path / "gap_stream.edf"
    sig = _signal(samples_per_record=100)
    onsets = [0.0, 1.0, 3.0]
    with edfarray.EdfWriter(
        str(p), variant="EDF+D", record_duration=1.0,
        signals=[sig], record_onsets=onsets,
    ) as w:
        for i in range(3):
            w.write_record([np.full(100, float(i), dtype=np.float64)])
    f = edfarray.EdfFile(str(p))
    assert f.variant == "EDF+D"
    s = f.signal(0)
    # The gap shows up in the sample times and the decoded values are flat per record.
    times = s.times()
    assert times[200] == pytest.approx(3.0)
    # 16-bit quantization of the +/-3200 uV range leaves about 0.1 uV of slack.
    assert s[200] == pytest.approx(2.0, abs=0.1)
    assert s[250] == pytest.approx(2.0, abs=0.1)


def test_record_onsets_rejected_on_non_plus_d(tmp_path: Path):
    p = tmp_path / "c.edf"
    sig = _signal()
    data = _ramp(256 * 2)
    with pytest.raises(edfarray.InvalidArgumentError, match="record_onsets"):
        edfarray.write_edf(
            str(p), variant="EDF+C", record_duration=1.0,
            signals=[sig], data=[data],
            record_onsets=[0.0, 2.0],
        )


def test_record_onsets_must_match_record_count(tmp_path: Path):
    sig = _signal(samples_per_record=100)
    short = tmp_path / "short.edf"
    # Two onsets for three records: writing the third record fails.
    w = edfarray.EdfWriter(
        str(short), variant="EDF+D", record_duration=1.0,
        signals=[sig], record_onsets=[0.0, 1.0],
    )
    w.write_record([np.full(100, 0.0, dtype=np.float64)])
    w.write_record([np.full(100, 1.0, dtype=np.float64)])
    with pytest.raises(edfarray.InvalidArgumentError, match="record_onsets"):
        w.write_record([np.full(100, 2.0, dtype=np.float64)])
    # A rejected record writes nothing, so the writer still matches its table.
    w.finish()

    # The reverse mismatch (more onsets than records) is caught at finish().
    long = tmp_path / "long.edf"
    w2 = edfarray.EdfWriter(
        str(long), variant="EDF+D", record_duration=1.0,
        signals=[sig], record_onsets=[0.0, 1.0, 3.0, 4.0],
    )
    for i in range(3):
        w2.write_record([np.full(100, float(i), dtype=np.float64)])
    with pytest.raises(edfarray.InvalidArgumentError, match="record_onsets"):
        w2.finish()


def test_writer_rejects_non_finite_values(tmp_path: Path):
    p = tmp_path / "nan.edf"
    sig = _signal(samples_per_record=8)

    data = np.zeros(8)
    data[3] = np.nan
    with pytest.raises(edfarray.InvalidArgumentError) as ei:
        edfarray.write_edf(
            str(p), variant="EDF", record_duration=1.0, signals=[sig], data=[data]
        )
    msg = str(ei.value)
    assert "signal 0" in msg and "sample 3" in msg and "NaN" in msg

    for bad in (np.inf, -np.inf):
        d = np.zeros(8)
        d[5] = bad
        with pytest.raises(edfarray.InvalidArgumentError):
            edfarray.write_edf(
                str(p), variant="EDF", record_duration=1.0, signals=[sig], data=[d]
            )


def test_writer_rejects_non_finite_incremental(tmp_path: Path):
    p = tmp_path / "nan_stream.edf"
    sig = _signal(samples_per_record=8)
    w = edfarray.EdfWriter(str(p), variant="EDF", record_duration=1.0, signals=[sig])
    data = np.zeros(8)
    data[7] = np.nan
    with pytest.raises(edfarray.InvalidArgumentError):
        w.write_record([data])


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


def test_writer_rejects_physical_range_too_wide_for_header(tmp_path: Path):
    # 123456789.1 has no 8-character form, and truncating it would change the value
    sig = edfarray.WriterSignal(
        label="EEG", physical_dimension="uV", physical_min=-3200.0,
        physical_max=123456789.1, digital_min=-32768, digital_max=32767,
        samples_per_record=256,
    )
    with pytest.raises(edfarray.InvalidFileError, match="physical_max"):
        edfarray.write_edf(
            str(tmp_path / "wide.edf"), variant="EDF", record_duration=1.0,
            signals=[sig], data=[_ramp(256)],
        )


def test_bdf_plus_writer_labels_annotation_channel_bdf(tmp_path: Path):
    p = tmp_path / "plus.bdf"
    sig = edfarray.WriterSignal(
        label="EEG",
        physical_dimension="uV",
        physical_min=-3200.0,
        physical_max=3200.0,
        digital_min=-(1 << 23),
        digital_max=(1 << 23) - 1,
        samples_per_record=256,
    )
    edfarray.write_edf(
        str(p),
        variant="BDF+C",
        record_duration=1.0,
        signals=[sig],
        data=[_ramp(256 * 2)],
        annotations=[edfarray.Annotation(onset=0.5, text="blink")],
    )
    f = edfarray.EdfFile(str(p))
    assert f.signal_labels() == ["EEG", "BDF Annotations"]
    assert f.ordinary_signal_indices() == [0]
    anns = f.annotations
    assert len(anns) == 1 and anns[0].text == "blink"
    assert f.warnings == []

    dst = tmp_path / "copy.bdf"
    f.write_to(str(dst))
    g = edfarray.EdfFile(str(dst))
    assert g.signal_labels() == ["EEG", "BDF Annotations"]
    assert len(g.annotations) == 1


def test_writer_rejects_user_bdf_annotation_label(tmp_path: Path):
    sig = edfarray.WriterSignal(
        label="BDF Annotations",
        physical_dimension="uV",
        physical_min=-1.0,
        physical_max=1.0,
        digital_min=-32768,
        digital_max=32767,
        samples_per_record=4,
    )
    with pytest.raises(edfarray.InvalidFileError, match="BDF Annotations"):
        edfarray.write_edf(
            str(tmp_path / "bad.bdf"),
            variant="BDF+C",
            record_duration=1.0,
            signals=[sig],
            data=[np.zeros(4)],
        )


def test_writer_rejects_non_finite_annotation_onset_and_duration(tmp_path: Path):
    sig = _signal()

    w = edfarray.EdfWriter(
        str(tmp_path / "nan.edf"), variant="EDF+C", record_duration=1.0, signals=[sig]
    )
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(edfarray.InvalidArgumentError):
            w.add_annotation(edfarray.Annotation(onset=bad, text="x"))
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(edfarray.InvalidArgumentError):
            w.add_annotation(edfarray.Annotation(onset=0.5, text="x", duration=bad))
    w.write_record([np.zeros(256)])
    w.finish()
    f = edfarray.EdfFile(str(tmp_path / "nan.edf"))
    assert f.annotations == []

    p = tmp_path / "nan2.edf"
    with pytest.raises(edfarray.InvalidArgumentError):
        edfarray.write_edf(
            str(p),
            variant="EDF+C",
            record_duration=1.0,
            signals=[sig],
            data=[_ramp(256 * 2)],
            annotations=[
                edfarray.Annotation(onset=0.5, text="ok"),
                edfarray.Annotation(onset=float("nan"), text="bad"),
            ],
        )
    assert not p.exists()


def test_writer_rejects_overlong_and_non_ascii_text_fields(tmp_path: Path):
    sig = _signal()
    cases = [
        (
            edfarray.WriterSignal(
                label="EEG channel with a long name",
                physical_dimension="uV",
                physical_min=-3200.0,
                physical_max=3200.0,
                digital_min=-32768,
                digital_max=32767,
                samples_per_record=256,
            ),
            "label",
        ),
        (
            edfarray.WriterSignal(
                label="EEG",
                physical_dimension="microvolt",
                physical_min=-3200.0,
                physical_max=3200.0,
                digital_min=-32768,
                digital_max=32767,
                samples_per_record=256,
            ),
            "physical_dimension",
        ),
    ]
    for bad_sig, field in cases:
        with pytest.raises(edfarray.InvalidArgumentError, match=field):
            edfarray.write_edf(
                str(tmp_path / "wide.edf"),
                variant="EDF",
                record_duration=1.0,
                signals=[bad_sig],
                data=[_ramp(256)],
            )
    with pytest.raises(edfarray.InvalidArgumentError, match="patient_id"):
        edfarray.write_edf(
            str(tmp_path / "wide.edf"),
            variant="EDF",
            record_duration=1.0,
            signals=[sig],
            data=[_ramp(256)],
            patient_id="x" * 200,
        )

    sig_nonascii = edfarray.WriterSignal(
        label="Jos\u00e9",
        physical_dimension="uV",
        physical_min=-3200.0,
        physical_max=3200.0,
        digital_min=-32768,
        digital_max=32767,
        samples_per_record=256,
    )
    with pytest.raises(edfarray.InvalidArgumentError, match="label"):
        edfarray.write_edf(
            str(tmp_path / "accents.edf"),
            variant="EDF",
            record_duration=1.0,
            signals=[sig_nonascii],
            data=[_ramp(256)],
        )


def test_write_to_copies_files_with_non_utf8_header_text(tmp_path: Path):
    # invalid UTF-8 in the source patient field must still copy, written back as '?'
    src = tmp_path / "src.edf"
    edfarray.write_edf(
        str(src),
        variant="EDF+C",
        record_duration=1.0,
        signals=[_signal()],
        data=[_ramp(256)],
    )
    raw = bytearray(src.read_bytes())
    for i in range(12, 88):
        raw[i] = ord("x")
    for i, b in enumerate(b"Jos\xe9"):
        raw[8 + i] = b
    src.write_bytes(bytes(raw))

    f = edfarray.EdfFile(str(src))
    assert "\ufffd" in f.patient_id
    dst = tmp_path / "dst.edf"
    f.write_to(str(dst))
    g = edfarray.EdfFile(str(dst))
    assert g.patient_id == "Jos?" + "x" * 76


def test_tal_onsets_round_trip_to_nanoseconds(tmp_path: Path):
    p = tmp_path / "subsecond.edf"
    sig = _signal(samples_per_record=256)
    onsets = sorted([0.1234567, 1e-7, 3.0000001, 1.0000005])
    edfarray.write_edf(
        str(p),
        variant="EDF+C",
        record_duration=1.0,
        signals=[sig],
        data=[_ramp(256 * 4)],
        annotations=[edfarray.Annotation(onset=t, text="e") for t in onsets],
    )
    f = edfarray.EdfFile(str(p))
    assert f.warnings == []
    got = sorted(a.onset for a in f.annotations)
    for want, actual in zip(onsets, got):
        assert abs(actual - want) < 1e-9, f"{actual} vs {want}"

    dst = tmp_path / "copy.edf"
    f.write_to(str(dst))
    g = edfarray.EdfFile(str(dst))
    assert g.warnings == []
    got = sorted(a.onset for a in g.annotations)
    for want, actual in zip(onsets, got):
        assert abs(actual - want) < 1e-9


def test_writer_rejects_annotation_budget_below_timekeeping_tal(tmp_path: Path):
    with pytest.raises(
        edfarray.InvalidArgumentError, match="annotation_bytes_per_record"
    ):
        edfarray.EdfWriter(
            str(tmp_path / "tiny.edf"),
            variant="EDF+C",
            record_duration=1.0,
            signals=[_signal()],
            annotation_bytes_per_record=0,
        )
    assert not (tmp_path / "tiny.edf").exists()


def test_write_to_sizes_annotation_budget_from_reencoded_content(tmp_path: Path):
    texts = [f"t{i:03d}" for i in range(12)]
    src = build_edf_plus(
        tmp_path / "src.edf",
        record_onsets=[0.0],
        annotation_bytes=80,
        annotations=[(0, 0.5, texts, 0)],
    )
    f = edfarray.EdfFile(src)
    assert len(f.annotations) == 12

    dst = tmp_path / "dst.edf"
    f.write_to(str(dst))
    g = edfarray.EdfFile(str(dst))
    assert g.warnings == []
    texts = {a.text for a in g.annotations}
    assert texts == {f"t{i:03d}" for i in range(12)}


def test_write_to_copies_multiple_annotation_channels(tmp_path: Path):
    dst = tmp_path / "dst.edf"
    src = str(
        build_edf_plus(
            tmp_path / "src.edf",
            record_onsets=[0.0],
            annotation_channels=2,
            annotation_bytes=60,
            annotations=[(0, 0.5, "a" * 53, 1)],
        )
    )
    f = edfarray.EdfFile(src)
    assert len(f.annotations) == 1
    f.write_to(str(dst))
    g = edfarray.EdfFile(str(dst))
    assert g.warnings == []
    assert len(g.annotations) == 1
    assert g.annotations[0].text == "a" * 53


def test_rejected_annotation_does_not_write_the_record(tmp_path: Path):
    p = tmp_path / "partial.edf"
    w = edfarray.EdfWriter(
        str(p),
        variant="EDF+C",
        record_duration=1.0,
        signals=[_signal(samples_per_record=8)],
    )
    with pytest.raises(edfarray.InvalidArgumentError):
        w.write_record([np.zeros(8)], annotations=[edfarray.Annotation(0.5, "x", -2.0)])
    w.write_record([np.zeros(8)])
    w.finish()
    f = edfarray.EdfFile(str(p))
    assert f.num_records == 1
    assert list(f.annotations) == []


def test_write_to_moves_events_to_the_record_their_onset_falls_in(tmp_path: Path):
    # both events are stored in record 0 but belong to record 10, whose timekeeping TAL is longer
    anns = [(0, 10.5, "ev", 0), (0, 10.25, "ev", 0)]
    need = len(b"+0\x14\x14\x00") + sum(
        len(b"+%g\x14ev\x14\x00" % o) for _, o, _, _ in anns
    )
    src = build_edf_plus(
        str(tmp_path / "misplaced.edf"),
        record_onsets=[float(i) for i in range(12)],
        annotations=anns,
        annotation_bytes=need + need % 2,
    )
    dst = tmp_path / "misplaced_copy.edf"
    edfarray.EdfFile(str(src)).write_to(str(dst))
    g = edfarray.EdfFile(str(dst))
    assert [(a.onset, a.text) for a in g.annotations] == [(10.25, "ev"), (10.5, "ev")]
    assert g.warnings == []


def test_write_to_packs_gapped_onsets_within_budget(tmp_path: Path):
    # packing EDF+D into EDF+C recomputes onsets, and the budget must fit the written values
    anns = [(1, 10.6, "a", 0), (1, 10.9, "b", 0), (2, 20.8, "c", 0), (2, 21.1, "d", 0)]
    src = build_edf_plus(
        str(tmp_path / "gapped.edf"),
        record_onsets=[0.0, 10.3, 20.7],
        annotations=anns,
        variant="EDF+D",
        annotation_bytes=40,
    )
    dst = tmp_path / "packed.edf"
    edfarray.EdfFile(str(src)).write_to(str(dst), variant="EDF+C")
    g = edfarray.EdfFile(str(dst))
    assert [(round(a.onset, 9), a.text) for a in g.annotations] == [
        (1.3, "a"),
        (1.6, "b"),
        (2.1, "c"),
        (2.4, "d"),
    ]
    assert g.warnings == []
    assert b"1.299999" not in dst.read_bytes()
