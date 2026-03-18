import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"

FIXTURE_NAMES = sorted(
    p.stem.replace(".reference", "")
    for p in FIXTURES.glob("*.reference.json")
    if (FIXTURES / f"{p.stem.replace('.reference', '')}.edf").exists()
)


def load_fixture(name):
    """Load an EDF file and its pyedflib-generated reference data."""
    import edfarray

    edf_path = FIXTURES / f"{name}.edf"
    ref_path = FIXTURES / f"{name}.reference.json"
    with open(ref_path) as f:
        ref = json.load(f)
    return edfarray.EdfFile(str(edf_path)), ref
