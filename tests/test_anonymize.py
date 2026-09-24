"""Tests for header editing, anonymization, and the leak audit."""

import datetime

import edfarray
import pytest
from edfbuilder import build_edf_plus

PATIENT = "MCH-0234567 F 02-MAR-1951 Alice_Smith note_one"
RECORDING = "Startdate 10-DEC-2009 PSG-9999 Jane_Tech Nihon_Kohden ward_b"


def _build(tmp_path, name="f.edf"):
    return str(
        build_edf_plus(
            tmp_path / name,
            record_onsets=[0.0, 1.0, 2.0],
            annotations=[(1, 1.5, "Alice Smith was restless", 0)],
            patient_id=PATIENT,
            recording_id=RECORDING,
        )
    )


def _data_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()[256 + 256 * 2 :]


def test_edit_header_changes_only_header(tmp_path):
    path = _build(tmp_path)
    before = _data_bytes(path)

    diff = edfarray.edit_header(path, patient_id="X X X Subject-1")

    assert diff["patient_id"] == {"before": PATIENT, "after": "X X X Subject-1"}
    assert "recording_id" not in diff
    assert "start_datetime" not in diff

    with edfarray.EdfFile(path) as f:
        assert f.patient_id == "X X X Subject-1"
        assert f.patient_name == "Subject-1"
        assert f.recording_id == RECORDING
    assert _data_bytes(path) == before


def test_edit_header_start_datetime(tmp_path):
    path = _build(tmp_path)
    diff = edfarray.edit_header(
        path, start_datetime=datetime.datetime(2020, 7, 4, 23, 59, 58)
    )
    assert diff["start_datetime"]["after"] == "04.07.20 23.59.58"

    with edfarray.EdfFile(path) as f:
        assert f.start_datetime == datetime.datetime(2020, 7, 4, 23, 59, 58)


def test_edit_header_no_op_diff_is_empty(tmp_path):
    path = _build(tmp_path)
    assert edfarray.edit_header(path, patient_id=PATIENT) == {}


def test_edit_header_rejects_bad_values(tmp_path):
    path = _build(tmp_path)
    with pytest.raises(edfarray.InvalidArgumentError):
        edfarray.edit_header(path, patient_id="x" * 81)
    with pytest.raises(edfarray.InvalidArgumentError):
        edfarray.edit_header(path, patient_id="na\u00efve name")
    with pytest.raises(edfarray.InvalidArgumentError):
        edfarray.edit_header(path, start_datetime=datetime.datetime(1900, 1, 1))
    with edfarray.EdfFile(path) as f:
        assert f.patient_id == PATIENT


def test_anonymize_scrubs_and_preserves_age(tmp_path):
    path = _build(tmp_path)
    result = edfarray.anonymize(path, seed="s", date_shift_days=-400)

    assert result["pseudonym"].startswith("Subject-")
    assert result["date_shift_days"] == -400
    assert result["dry_run"] is False

    with edfarray.EdfFile(path) as f:
        assert f.patient_name == result["pseudonym"]
        assert f.patient_code is None
        assert f.patient_sex == "F"
        assert f.technician is None
        assert f.admin_code is None
        assert f.equipment == "Nihon Kohden"
        # Age preserved: birthdate and recording dates move by exactly the same shift.
        shift = datetime.timedelta(days=400)
        assert f.patient_birthdate == datetime.date(1951, 3, 2) - shift
        assert f.start_datetime.date() == datetime.date(2009, 12, 10) - shift
        assert (
            f.start_datetime.date() - f.patient_birthdate
            == datetime.date(2009, 12, 10) - datetime.date(1951, 3, 2)
        )
        assert "Alice" not in f.patient_id
        assert "Smith" not in f.patient_id


def test_anonymize_seed_is_reproducible(tmp_path):
    a = _build(tmp_path, "a.edf")
    b = _build(tmp_path, "b.edf")
    ra = edfarray.anonymize(a, seed="study")
    rb = edfarray.anonymize(b, seed="study")
    assert ra["pseudonym"] == rb["pseudonym"]
    assert ra["date_shift_days"] == rb["date_shift_days"]

    rc = edfarray.anonymize(_build(tmp_path, "c.edf"))
    rd = edfarray.anonymize(_build(tmp_path, "d.edf"))
    # Unseeded runs are random: two files must not collide.
    assert rc["pseudonym"] != rd["pseudonym"]


def test_anonymize_explicit_pseudonym_and_keep_flags(tmp_path):
    path = _build(tmp_path)
    result = edfarray.anonymize(
        path, pseudonym="CohortA-42", date_shift_days=10, keep_code=True
    )
    assert result["pseudonym"] == "CohortA-42"
    with edfarray.EdfFile(path) as f:
        assert f.patient_name == "CohortA-42"
        assert f.patient_code == "MCH-0234567"
        assert f.patient_sex is None or f.patient_sex == "F"


def test_anonymize_dry_run_writes_nothing(tmp_path):
    path = _build(tmp_path)
    before = open(path, "rb").read()

    result = edfarray.anonymize(path, seed="s", dry_run=True)
    assert result["dry_run"] is True
    assert result["patient_id_after"] != result["patient_id_before"]
    assert open(path, "rb").read() == before

    with edfarray.EdfFile(path) as f:
        assert f.patient_id == PATIENT


def test_audit_finds_leaks_before_and_after_anonymize(tmp_path):
    path = _build(tmp_path)

    report = edfarray.audit(path)
    assert report["clean"] is False
    assert "alice" in report["terms"] and "smith" in report["terms"]
    ann_hits = [h for h in report["hits"] if h["signal_index"] is None]
    assert {h["term"] for h in ann_hits} == {"alice", "smith"}

    result = edfarray.anonymize(path, seed="s")
    assert "alice" in result["scrubbed_terms"]

    # Header is scrubbed, so terms derived from the current header find nothing...
    assert edfarray.audit(path)["clean"] is True
    # ...but the annotation still repeats the original name until handled by hand.
    residual = edfarray.audit(path, terms=result["scrubbed_terms"])
    assert residual["clean"] is False
    assert all(h["signal_index"] is None for h in residual["hits"])


def test_audit_clean_file(tmp_path):
    path = str(
        build_edf_plus(
            tmp_path / "clean.edf",
            record_onsets=[0.0],
            annotations=[(0, 0.5, "sleep stage N2", 0)],
        )
    )
    report = edfarray.audit(path)
    assert report["clean"] is True
    assert report["hits"] == []


def test_failed_edit_leaves_file_unchanged(tmp_path):
    path = _build(tmp_path)
    with open(path, "rb") as fh:
        before = fh.read()
    with pytest.raises(edfarray.EdfError):
        edfarray.edit_header(path, patient_id="X X X Subject-OK", recording_id="R" * 81)
    with open(path, "rb") as fh:
        assert fh.read() == before


def test_pseudonym_is_stable_across_sessions(tmp_path):
    # Same subject, two sessions whose free-text notes differ. anonymize discards that note,
    # so it must not hand one subject two pseudonyms.
    base = "MCH-01 F 02-MAR-1951 Jane_Doe"
    one = str(
        build_edf_plus(
            tmp_path / "s1.edf",
            record_onsets=[0.0, 1.0],
            annotations=[(1, 1.5, "note", 0)],
            patient_id=f"{base} session_one",
            recording_id=RECORDING,
        )
    )
    two = str(
        build_edf_plus(
            tmp_path / "s2.edf",
            record_onsets=[0.0, 1.0],
            annotations=[(1, 1.5, "note", 0)],
            patient_id=f"{base} session_two",
            recording_id=RECORDING,
        )
    )
    a = edfarray.anonymize(one, seed="study-2026")
    b = edfarray.anonymize(two, seed="study-2026")
    assert a["pseudonym"] == b["pseudonym"]
    assert a["patient_id_after"] == b["patient_id_after"]


def test_dry_run_rejects_what_a_real_run_rejects(tmp_path):
    path = _build(tmp_path)
    with pytest.raises(edfarray.EdfError):
        edfarray.anonymize(path, seed="s", pseudonym="P" * 81, dry_run=True)


def test_dry_run_flag_is_reported(tmp_path):
    path = _build(tmp_path)
    assert edfarray.anonymize(path, seed="s", dry_run=True)["dry_run"] is True
    assert edfarray.anonymize(path, seed="s")["dry_run"] is False


def test_audit_location_has_no_sign_prefix(tmp_path):
    path = _build(tmp_path)
    hits = edfarray.audit(path)["hits"]
    ann = [h for h in hits if h["signal_index"] is None]
    assert ann and all(not h["location"].startswith("annotation +") for h in ann)


def test_edits_refuse_an_open_file(tmp_path):
    path = _build(tmp_path)
    before = open(path, "rb").read()

    f = edfarray.EdfFile(path)
    sig = f.signal(0)
    f.close()
    with pytest.raises(edfarray.EdfFileError, match="another edfarray handle"):
        edfarray.edit_header(path, patient_id="X X X Subject-001")
    with pytest.raises(edfarray.EdfFileError):
        edfarray.anonymize(path, seed="s")
    edfarray.anonymize(path, seed="s", dry_run=True)
    edfarray.audit(path)
    assert open(path, "rb").read() == before

    del sig
    edfarray.edit_header(path, patient_id="X X X Subject-001")
    with edfarray.EdfFile(path) as g:
        assert g.patient_id == "X X X Subject-001"
