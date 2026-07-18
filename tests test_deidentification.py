"""Smoke tests for the de-identification pipeline."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "privacy"))

import deidentification as di  # noqa: E402


def _sample_df():
    return pd.DataFrame(
        {
            "patient_id": [f"P{i:03d}" for i in range(10)],
            "age_months": [36, 37, 38, 39, 40, 36, 37, 38, 39, 40],
            "gender": ["F", "M"] * 5,
            "location": ["Nanjing, Jiangsu"] * 10,
            "diagnosis_date": ["2021-03-14"] * 10,
            "lab_value": [12.5, 13.0, 11.8, 12.2, 12.9, 12.5, 13.0, 11.8, 12.2, 12.9],
            "diagnosis": ["Gaucher disease"] * 10,
        }
    )


def test_generalisation_reduces_precision():
    df = _sample_df()
    generalised = di.apply_generalisation(df)
    assert (generalised["diagnosis_date"] == "2021-03").all()
    assert (generalised["location"] == "Nanjing").all()
    assert generalised["age_months_gen"].str.startswith("[").all()


def test_k_anonymity_bounds_risk():
    df = _sample_df()
    result = di.deidentify(df)
    # Under k=5 anonymity, average re-identification risk cannot exceed 1/k.
    risk = di.reidentification_risk(result)
    assert risk <= (1.0 / di.K) * 100.0 + 1e-6


def test_information_loss_in_range():
    df = _sample_df()
    generalised = di.apply_generalisation(df)
    anonymised = di.enforce_k_anonymity(generalised)
    il = di.information_loss(df, anonymised)
    assert 0.0 <= il <= 100.0
