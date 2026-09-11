"""Data validation and target preparation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


@dataclass
class ValidationReport:
    """Summary of dataset quality checks."""

    n_rows: int
    n_cols: int
    missing_by_column: dict[str, int] = field(default_factory=dict)
    n_duplicates: int = 0
    dtypes: dict[str, str] = field(default_factory=dict)
    target_distribution: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "missing_by_column": self.missing_by_column,
            "n_duplicates": self.n_duplicates,
            "dtypes": self.dtypes,
            "target_distribution": self.target_distribution,
            "issues": self.issues,
        }


def validate_dataframe(df: pd.DataFrame, target_column: str = "Churn") -> ValidationReport:
    """Run basic data quality checks and return a structured report."""
    issues: list[str] = []

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing expected columns: {missing_cols}")

    missing_by_column = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    # TotalCharges is often stored as string with blanks for tenure==0
    if "TotalCharges" in df.columns:
        blank_total = (
            df["TotalCharges"].astype(str).str.strip().isin(["", "nan", "None"]).sum()
        )
        if blank_total and "TotalCharges" not in missing_by_column:
            missing_by_column["TotalCharges"] = int(blank_total)
            issues.append(
                f"TotalCharges has {blank_total} blank/non-numeric values "
                "(often new customers with tenure=0)."
            )

    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        issues.append(f"Found {n_duplicates} fully duplicate rows.")

    if "customerID" in df.columns:
        id_dups = int(df["customerID"].duplicated().sum())
        if id_dups:
            issues.append(f"Found {id_dups} duplicate customerID values.")

    target_distribution: dict[str, int] = {}
    if target_column in df.columns:
        target_distribution = {str(k): int(v) for k, v in df[target_column].value_counts().items()}
        churn_rate = target_distribution.get("Yes", 0) / max(len(df), 1)
        if churn_rate < 0.05 or churn_rate > 0.5:
            issues.append(f"Unusual churn rate: {churn_rate:.1%}. Check class imbalance carefully.")
    else:
        issues.append(f"Target column '{target_column}' not found.")

    return ValidationReport(
        n_rows=len(df),
        n_cols=df.shape[1],
        missing_by_column=missing_by_column,
        n_duplicates=n_duplicates,
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        target_distribution=target_distribution,
        issues=issues,
    )


def prepare_target(df: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    """Return a copy with cleaned TotalCharges and binary Churn (0/1)."""
    out = df.copy()

    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    if target_column in out.columns:
        mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0, "1": 1, "0": 0}
        out[target_column] = out[target_column].map(mapping)
        if out[target_column].isna().any():
            bad = out[target_column].isna().sum()
            raise ValueError(f"Could not map {bad} values in '{target_column}' to 0/1.")
        out[target_column] = out[target_column].astype(int)

    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)

    return out


def feature_columns(df: pd.DataFrame, target_column: str = "Churn", id_column: str = "customerID") -> list[str]:
    """Return modeling feature column names (exclude id and target)."""
    drop = {target_column, id_column}
    return [c for c in df.columns if c not in drop]


def numeric_and_categorical(
    df: pd.DataFrame,
    target_column: str = "Churn",
    id_column: str = "customerID",
) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric vs categorical lists."""
    feats = feature_columns(df, target_column, id_column)
    numeric: list[str] = []
    categorical: list[str] = []
    for col in feats:
        if col == "SeniorCitizen":
            # Binary indicator — treat as categorical for one-hot clarity in LR
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical
