"""Tests for preprocessing."""

from src.features.preprocess import build_preprocessor, get_feature_lists
from src.models.train import prepare_xy


def test_feature_lists_exclude_id_and_target(prepared_df):
    num, cat = get_feature_lists(prepared_df)
    assert "customerID" not in num + cat
    assert "Churn" not in num + cat
    assert "tenure" in num
    assert "Contract" in cat


def test_preprocessor_output_shape(train_test):
    train_df, _ = train_test
    X, y = prepare_xy(train_df)
    pre = build_preprocessor(train_df)
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(X)
    assert Xt.shape[1] > len(X.columns)  # one-hot expands categoricals
    assert y.dtype.kind in "iu"
