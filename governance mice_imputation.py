"""MICE imputation for missing clinical and laboratory values.

Implements Multiple Imputation by Chained Equations using scikit-learn
IterativeImputer (v1.0.2). All numeric columns are used jointly as auxiliary
variables to maximise information retention before downstream NER processing.

Hardware requirements: CPU only;128 GB RAM is sufficient for the full dataset.
"""
import argparse
import pandas as pd
from sklearn.experimental import enable_iterative_imputer# noqa: F401
from sklearn.impute import IterativeImputer

NUMERIC_COLS = ["age_months", "lab_value"]


def impute(df: pd.DataFrame, max_iter: int = 10, seed: int = 42) -> pd.DataFrame:
    """Impute missing numeric fields via chained equations.

    Args:
        df: Input DataFrame containing NUMERIC_COLS.
        max_iter: Maximum number of imputation rounds.
        seed: Random state for reproducibility.

    Returns:
        DataFrame with all NUMERIC_COLS fully populated.
    """
    imputer = IterativeImputer(max_iter=max_iter, random_state=seed)
    out = df.copy()
    out[NUMERIC_COLS] = imputer.fit_transform(out[NUMERIC_COLS])
    return out


def main():
    parser = argparse.ArgumentParser(
        description="MICE imputation for rare-disease EHR records."
    )
    parser.add_argument("--input", default="data/synthetic_sample.csv")
    parser.add_argument("--output", default="data/imputed_sample.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    n_missing = int(df[NUMERIC_COLS].isna().sum().sum())
    imputed = impute(df)
    imputed.to_csv(args.output, index=False)
    print(f"Imputed {n_missing} missing value(s). Output: {args.output}")


if __name__ == "__main__":
    main()
