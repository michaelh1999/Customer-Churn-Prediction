"""Sklearn preprocessing transformers for churn features."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.validate import numeric_and_categorical


def build_preprocessor(
    df: pd.DataFrame,
    target_column: str = "Churn",
    id_column: str = "customerID",
    numeric_features: Iterable[str] | None = None,
    categorical_features: Iterable[str] | None = None,
) -> ColumnTransformer:
    """Build a ColumnTransformer: impute+scale numerics, impute+one-hot categoricals."""
    if numeric_features is None or categorical_features is None:
        num, cat = numeric_and_categorical(df, target_column, id_column)
        numeric_features = list(numeric_features) if numeric_features is not None else num
        categorical_features = (
            list(categorical_features) if categorical_features is not None else cat
        )
    else:
        numeric_features = list(numeric_features)
        categorical_features = list(categorical_features)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def get_feature_lists(
    df: pd.DataFrame,
    target_column: str = "Churn",
    id_column: str = "customerID",
) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features)."""
    return numeric_and_categorical(df, target_column, id_column)
