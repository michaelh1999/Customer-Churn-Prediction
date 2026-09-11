"""Train/test splitting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load import load_config, load_raw_data, project_root
from src.data.validate import prepare_target


def make_stratified_split(
    df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified train/test split on the churn target."""
    cfg = config or load_config()
    if df is None:
        df = prepare_target(load_raw_data())

    target = cfg["data"]["target_column"]
    test_size = cfg["data"]["test_size"]
    seed = cfg["data"]["random_seed"]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    processed_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Persist train/test CSVs under data/processed/."""
    root = project_root()
    cfg = load_config()
    out_dir = Path(processed_dir) if processed_dir else root / cfg["paths"]["processed_dir"]
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def load_split(processed_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load previously saved train/test CSVs, or create them if missing."""
    root = project_root()
    cfg = load_config()
    out_dir = Path(processed_dir) if processed_dir else root / cfg["paths"]["processed_dir"]
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"
    if train_path.exists() and test_path.exists():
        return pd.read_csv(train_path), pd.read_csv(test_path)

    train_df, test_df = make_stratified_split()
    save_split(train_df, test_df, out_dir)
    return train_df, test_df
