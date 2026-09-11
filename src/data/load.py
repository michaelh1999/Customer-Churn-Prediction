"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration from ``configs/default.yaml`` by default."""
    root = project_root()
    path = Path(config_path) if config_path else root / "configs" / "default.yaml"
    if not path.is_absolute():
        path = root / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load the Telco customer churn CSV.

    Parameters
    ----------
    path:
        Path to the CSV. Defaults to the path in config.
    """
    root = project_root()
    if path is None:
        cfg = load_config()
        path = root / cfg["paths"]["raw_data"]
    else:
        path = Path(path)
        if not path.is_absolute():
            path = root / path

    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {path}. "
            "Run `python -m src.data.download` to fetch the IBM Telco dataset."
        )

    df = pd.read_csv(path)
    # Normalize whitespace in column names and object columns
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    return df
