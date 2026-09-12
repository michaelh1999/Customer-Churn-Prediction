"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from src.data.download import download_telco
from src.data.load import load_raw_data, load_config
from src.data.validate import prepare_target
from src.data.split import make_stratified_split
from src.models.train import train_and_evaluate, load_model


@pytest.fixture(scope="session")
def ensure_data():
    download_telco()
    return load_raw_data()


@pytest.fixture(scope="session")
def prepared_df(ensure_data):
    return prepare_target(ensure_data)


@pytest.fixture(scope="session")
def train_test(prepared_df):
    return make_stratified_split(prepared_df)


@pytest.fixture(scope="session")
def trained_lr(ensure_data):
    """Ensure logistic regression artifact exists for inference tests."""
    try:
        return load_model("logistic_regression")
    except FileNotFoundError:
        train_and_evaluate("logistic_regression")
        return load_model("logistic_regression")


@pytest.fixture
def config():
    return load_config()
