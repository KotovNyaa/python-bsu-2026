"""Тесты предобработки."""

import numpy as np
import pandas as pd

from lab.prep import impute_median, load_markers, log_transform, normalize_tic


def test_normalize_tic() -> None:
    df = pd.DataFrame({"s1": [10.0, 30.0], "s2": [20.0, 80.0]})
    res = normalize_tic(df)
    assert np.allclose(res.sum(axis=0), [1.0, 1.0])


def test_log_transform() -> None:
    df = pd.DataFrame({"s1": [1.0, 4.0, 8.0]})
    res = log_transform(df)
    assert np.allclose(res["s1"], [0.0, 2.0, 3.0])


def test_impute_median() -> None:
    df = pd.DataFrame({"s1": [1.0, np.nan], "s2": [3.0, 5.0], "s3": [np.nan, 7.0]})
    res = impute_median(df)
    assert not res.isna().any().any()
    assert res.loc[0, "s3"] == 2.0


def test_load_markers() -> None:
    df = load_markers()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 426
    assert "Target" in df.columns
    assert set(df["Target"].unique()) == {0, 1}
    for m in ["AMBP", "FN1", "LUM", "TIMP1", "B2M"]:
        assert m in df.columns
