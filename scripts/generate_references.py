#!/usr/bin/env python3
"""Generate reference JSON files for all EDF test fixtures using pyedflib.

pyedflib wraps the original C edflib by Teunis van Beelen, making it the most
authoritative reference implementation available in Python.

Usage:
    uv run python scripts/generate_references.py
"""

import json
from pathlib import Path

import numpy as np
import pyedflib

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def extract_reference(edf_path: Path) -> dict:
    """Extract header, signal metadata, and sample snippets from an EDF file."""
    try:
        ef = pyedflib.EdfReader(str(edf_path))
    except OSError:
        ef = pyedflib.EdfReader(str(edf_path), pyedflib.DO_NOT_READ_ANNOTATIONS)

    try:
        num_signals = ef.signals_in_file
        num_records = ef.datarecords_in_file

        header = {
            "filename": edf_path.name,
            "num_signals": num_signals,
            "num_records": num_records,
            "file_duration_seconds": ef.file_duration,
            "filetype": ef.filetype,
        }

        # Patient / recording raw fields
        header["patient_id"] = ef.patient if hasattr(ef, "patient") else ""
        header["recording_id"] = ef.recording if hasattr(ef, "recording") else ""

        # Parsed patient fields
        header["patient_name"] = getattr(ef, "patient_name", None) or None
        header["patientcode"] = getattr(ef, "patientcode", None) or None
        header["sex"] = getattr(ef, "sex", None) or None
        header["birthdate"] = getattr(ef, "birthdate", None) or None

        # Date/time
        header["startdate_year"] = ef.startdate_year if hasattr(ef, "startdate_year") else None
        header["startdate_month"] = ef.startdate_month if hasattr(ef, "startdate_month") else None
        header["startdate_day"] = ef.startdate_day if hasattr(ef, "startdate_day") else None
        header["starttime_hour"] = ef.starttime_hour if hasattr(ef, "starttime_hour") else None
        header["starttime_minute"] = ef.starttime_minute if hasattr(ef, "starttime_minute") else None
        header["starttime_second"] = ef.starttime_second if hasattr(ef, "starttime_second") else None
        header["starttime_subsecond"] = (
            ef.starttime_subsecond if hasattr(ef, "starttime_subsecond") else None
        )

        # Parsed recording fields
        header["admincode"] = getattr(ef, "admincode", None) or None
        header["technician"] = getattr(ef, "technician", None) or None
        header["equipment"] = getattr(ef, "equipment", None) or None

        # Signal metadata
        signals = []
        for i in range(num_signals):
            sig = {
                "index": i,
                "label": ef.getLabel(i),
                "transducer": ef.getTransducer(i),
                "physical_dimension": ef.getPhysicalDimension(i),
                "physical_min": ef.getPhysicalMinimum(i),
                "physical_max": ef.getPhysicalMaximum(i),
                "digital_min": ef.getDigitalMinimum(i),
                "digital_max": ef.getDigitalMaximum(i),
                "prefiltering": ef.getPrefilter(i),
                "samples_per_data_record": ef.getSampleFrequency(i)
                * ef.datarecord_duration,
                "sample_frequency": ef.getSampleFrequency(i),
            }
            signals.append(sig)

        # Annotations
        annotations_raw = ef.readAnnotations()
        annotations = []
        if annotations_raw and len(annotations_raw) >= 3:
            onsets, durations, texts = annotations_raw
            for onset, duration, text in zip(onsets, durations, texts):
                ann = {
                    "onset": float(onset),
                    "duration": float(duration) if duration and duration != "" else None,
                    "text": text if isinstance(text, str) else text.decode("utf-8", errors="replace"),
                }
                annotations.append(ann)

        # Sample snippets for validation (first 10 samples of first non-annotation signal)
        sample_snippets = []
        for i in range(min(num_signals, 5)):
            label = ef.getLabel(i)
            if "Annotation" in label:
                continue
            try:
                physical = ef.readSignal(i, 0, min(10, ef.getNSamples()[i]))
                digital = ef.readSignal(i, 0, min(10, ef.getNSamples()[i]), digital=True)
                sample_snippets.append({
                    "signal_index": i,
                    "label": label,
                    "physical_first_10": physical.tolist(),
                    "digital_first_10": digital.tolist(),
                })
            except Exception as e:
                print(f"  Warning: could not read signal {i} ({label}): {e}")

        return {
            "header": header,
            "signals": signals,
            "annotations": annotations,
            "sample_snippets": sample_snippets,
        }
    finally:
        ef.close()


def main():
    edf_files = sorted(FIXTURES.glob("*.edf"))
    print(f"Found {len(edf_files)} EDF files in {FIXTURES}")

    for edf_path in edf_files:
        print(f"\nProcessing {edf_path.name}...")
        try:
            ref = extract_reference(edf_path)
            ref_path = edf_path.with_suffix(".reference.json")
            with open(ref_path, "w") as f:
                json.dump(ref, f, indent=2, default=str)
            print(f"  -> {ref_path.name}")
            print(f"     signals={ref['header']['num_signals']}, "
                  f"records={ref['header']['num_records']}, "
                  f"annotations={len(ref['annotations'])}, "
                  f"snippets={len(ref['sample_snippets'])}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
