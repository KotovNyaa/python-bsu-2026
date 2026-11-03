"""Тесты статистического модуля."""

import numpy as np
import pandas as pd

from lab.stats import adjust_fdr, describe, diff


def test_describe() -> None:
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 10.0, 10.0, np.nan, 20.0]}
    )
    res = describe(df)
    assert res.loc["a", "mean"] == 3.0
    assert res.loc["a", "median"] == 3.0
    assert res.loc["b", "missing"] == 0.2


def test_adjust_fdr() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.001])
    adj = adjust_fdr(p)
    assert len(adj) == 4
    assert np.all(adj >= p)
    assert np.all(adj <= 1.0)


def test_diff() -> None:
    df = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 1.5, 1.8, 8.0, 9.0, 8.5, 8.8],
            "f2": [5.0, 5.1, 4.9, 5.0, 5.0, 5.2, 4.8, 5.1],
            "Target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    res = diff(df, ["f1", "f2"])
    assert len(res) == 2
    f1_row = res[res["feature"] == "f1"].iloc[0]
    assert f1_row["p_value"] < 0.05
    assert f1_row["log2_fc"] > 0
