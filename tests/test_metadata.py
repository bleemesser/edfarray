"""Header metadata: the EDF+ identification subfields, warnings, and `inspect`.

The shipped fixtures are anonymized (`X X X X`), so populated identification lines are built
here. The example values follow the EDF+ specification's own sample header.
"""

import datetime

import edfarray
import pytest
from conftest import FIXTURES
from edfbuilder import build_edf_plus

PATIENT = "MCH-0234567 F 02-MAY-1951 Haagse_Harry"
RECORDING = "Startdate 02-MAR-2002 EMG561 BK/JOP Sony._MNC"


@pytest.fixture
def populated(tmp_path):
    path = build_edf_plus(
        tmp_path / "meta.edf",
        record_onsets=[0, 1],
        patient_id=PATIENT,
        recording_id=RECORDING,
    )
    return edfarray.EdfFile(path)


class TestPatientSubfields:
    def test_code(self, populated):
        assert populated.patient_code == "MCH-0234567"

    def test_sex(self, populated):
        assert populated.patient_sex == "F"

    def test_birthdate_is_a_date(self, populated):
        assert populated.patient_birthdate == datetime.date(1951, 5, 2)

    def test_name_underscores_become_spaces(self, populated):
        # EDF+ forbids spaces inside a subfield, so names are written with underscores.
        assert populated.patient_name == "Haagse Harry"

    def test_raw_field_is_preserved(self, populated):
        assert populated.patient_id == PATIENT


class TestRecordingSubfields:
    def test_admin_code(self, populated):
        assert populated.admin_code == "EMG561"

    def test_technician(self, populated):
        assert populated.technician == "BK/JOP"

    def test_equipment(self, populated):
        assert populated.equipment == "Sony. MNC"

    def test_raw_field_is_preserved(self, populated):
        assert populated.recording_id == RECORDING


class TestAnonymized:
    """`X` means "not specified" in EDF+, and must read back as None rather than "X"."""

    @pytest.mark.parametrize(
        "attr",
        [
            "patient_name",
            "patient_code",
            "patient_sex",
            "patient_birthdate",
            "patient_additional",
            "admin_code",
            "technician",
            "recording_additional",
        ],
    )
    def test_x_subfields_are_none(self, attr):
        f = edfarray.EdfFile(str(FIXTURES / "edfPlusC.edf"))
        assert getattr(f, attr) is None


class TestPlainEdf:
    def test_unstructured_fields_do_not_parse_into_subfields(self):
        # A plain EDF header has no subfield structure to parse.
        f = edfarray.EdfFile(str(FIXTURES / "test_generator.edf"))
        assert f.patient_id == "test file"
        assert f.patient_name is None


class TestWarnings:
    def test_clean_file_has_no_warnings(self):
        assert edfarray.EdfFile(str(FIXTURES / "test_generator.edf")).warnings == []

    def test_warnings_is_a_list_of_strings(self):
        f = edfarray.EdfFile(str(FIXTURES / "edfPlusD.edf"))
        assert all(isinstance(w, str) for w in f.warnings)


class TestInspect:
    def test_matches_the_opened_file(self):
        path = str(FIXTURES / "edfPlusD.edf")
        info = edfarray.inspect(path)
        f = edfarray.EdfFile(path)
        assert info["variant"] == f.variant
        assert info["num_signals"] == f.num_signals
        assert info["num_records"] == f.num_records
        assert info["record_duration"] == f.record_duration
        assert info["duration"] == f.duration
        assert info["patient_id"] == f.patient_id
        assert info["recording_id"] == f.recording_id
        assert info["signal_labels"] == f.signal_labels()

    def test_sample_rates_match_signals(self):
        path = str(FIXTURES / "test_generator.edf")
        info = edfarray.inspect(path)
        f = edfarray.EdfFile(path)
        for i, rate in enumerate(info["sample_rates"]):
            assert rate == f.signal(i).sample_rate

    def test_missing_file_raises(self):
        with pytest.raises(edfarray.EdfFileError):
            edfarray.inspect("/definitely/not/here.edf")


class TestStartDatetime:
    def test_parsed_datetime(self):
        f = edfarray.EdfFile(str(FIXTURES / "edfPlusC.edf"))
        assert f.start_datetime == datetime.datetime(2009, 12, 10, 12, 44, 2)

    def test_repr_mentions_variant_and_counts(self):
        f = edfarray.EdfFile(str(FIXTURES / "edfPlusC.edf"))
        text = repr(f)
        assert "EDF+C" in text
        assert str(f.num_records) in text
