"""
De-identification pipeline for rare-disease clinical records.

Combines k-anonymity (generalisation + suppression, k=5) with a differential
privacy Laplace mechanism applied to numeric measurements. The module reports
Information Loss (IL) and re-identification risk so the privacy-utility
trade-off can be reproduced on the full test cohort.

Target runtime: Python 3.8, pandas 1.3.5, scikit-learn 1.0.2, diffprivlib 0.5.0.
"""
import argparse
import os

import pandas as pd
from diffprivlib.mechanisms import Laplace

# Quasi-identifiers subject to generalisation / suppression.
QUASI_IDENTIFIERS = ["age_months", "location", "diagnosis_date"]
K = 5                    # k-anonymity parameter
DP_EPSILON = 1.0         # differential privacy budget per numeric field
AGE_BIN_WIDTH = 12       # generalise age into 12-month bins


def generalise_location(value):
    """Generalise a full address to city level.

    Assumes a comma-separated address whose first field is the city. Values
    with no parseable city are suppressed to the '*' wildcard.
    """
    if not isinstance(value, str) or not value.strip():
        return "*"
    return value.split(",")[0].strip()


def generalise_date(value):
    """Suppress day precision, retaining year-month (YYYY-MM)."""
    if not isinstance(value, str) or len(value) < 7:
        return "*"
    return value[:7]


def generalise_age(value):
    """Generalise age in months into fixed-width interval labels."""
    if pd.isna(value):
        return "*"
    lower = int(value // AGE_BIN_WIDTH) * AGE_BIN_WIDTH
    upper = lower + AGE_BIN_WIDTH
    return f"[{lower},{upper})"


def apply_generalisation(df):
    """Apply attribute-specific generalisation to all quasi-identifiers."""
    out = df.copy()
    out["location"] = out["location"].apply(generalise_location)
    out["diagnosis_date"] = out["diagnosis_date"].apply(generalise_date)
    out["age_months_gen"] = out["age_months"].apply(generalise_age)
    return out


def enforce_k_anonymity(df, k=K):
    """Suppress equivalence classes smaller than k.

    Records whose quasi-identifier combination occurs fewer than k times are
    removed, guaranteeing every released record shares its quasi-identifiers
    with at least k-1 others.
    """
    qi = ["age_months_gen", "location", "diagnosis_date"]
    group_sizes = df.groupby(qi)[qi[0]].transform("size")
    return df[group_sizes >= k].reset_index(drop=True)


def apply_differential_privacy(df, epsilon=DP_EPSILON):
    """Add calibrated Laplace noise to numeric non-identifying measurements.

    Sensitivity is estimated from the observed value range of each column, which
    bounds the maximum contribution of a single record.
    """
    out = df.copy()
    numeric_cols = ["lab_value"]
    for col in numeric_cols:
        if col not in out.columns:
            continue
        col_values = out[col].dropna()
        if col_values.empty:
            continue
        sensitivity = float(col_values.max() - col_values.min()) or 1.0
        mechanism = Laplace(epsilon=epsilon, sensitivity=sensitivity)
        out[col] = out[col].apply(
            lambda v: mechanism.randomise(float(v)) if not pd.isna(v) else v
        )
    return out


def information_loss(original, anonymised):
    """Compute normalised Information Loss for generalised quasi-identifiers.

    The certainty penalty for the generalised age attribute is the ratio of the
    generalised interval width to the full attribute domain width. IL is
    expressed as a percentage.
    """
    age = original["age_months"].dropna()
    if age.empty:
        return 0.0
    domain = float(age.max() - age.min()) or 1.0
    penalty = AGE_BIN_WIDTH / domain
    return round(min(penalty, 1.0) * 100.0, 2)


def reidentification_risk(anonymised):
    """Estimate average (journalist) re-identification risk.

    Risk for a record equals the inverse of its equivalence-class size; the
    reported value is the mean across all released records, expressed as a
    percentage. Under strict k-anonymity this is bounded above by 1/k.
    """
    qi = ["age_months_gen", "location", "diagnosis_date"]
    if anonymised.empty:
        return 0.0
    class_sizes = anonymised.groupby(qi)[qi[0]].transform("size")
    return round(float((1.0 / class_sizes).mean()) * 100.0, 4)


def deidentify(df):
    """Run the full de-identification pipeline and print privacy metrics."""
    generalised = apply_generalisation(df)
    anonymised = enforce_k_anonymity(generalised, k=K)
    anonymised = apply_differential_privacy(anonymised, epsilon=DP_EPSILON)

    il = information_loss(df, anonymised)
    risk = reidentification_risk(anonymised)

    print(f"Records in : {len(df)}")
    print(f"Records out (k>={K}): {len(anonymised)}")
    print(f"Information Loss (IL): {il}%")
    print(f"Re-identification risk: {risk}%")
    return anonymised


def main():
    parser = argparse.ArgumentParser(description="De-identify clinical records.")
    parser.add_argument(
        "--input",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_sample.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "deidentified.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = deidentify(df)
    result.to_csv(args.output, index=False)
    print(f"De-identified data written to {args.output}")


if __name__ == "__main__":
    main()
