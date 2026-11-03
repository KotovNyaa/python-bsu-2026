"""Тесты модуля машинного обучения и кросс-когортной валидации."""

import numpy as np
import pandas as pd

from lab.model import (
    evaluate_all_cohorts,
    evaluate_cohort,
    export_ml_artifacts,
    plot_roc_pr,
    train_ensemble_model,
    train_logistic_regression,
)


def _make_dummy_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "Dataset": ["Train"] * 30 + ["External validation1"] * 30,
            "Target": [0, 1] * 30,
            "AMBP": np.random.uniform(0.1, 2.0, n),
            "FN1": np.random.uniform(0.1, 2.0, n),
            "LUM": np.random.uniform(0.1, 2.0, n),
            "TIMP1": np.random.uniform(0.1, 2.0, n),
            "B2M": np.random.uniform(0.1, 2.0, n),
        }
    )


def test_train_and_eval_logistic() -> None:
    df = _make_dummy_df()
    train_df = df[df["Dataset"] == "Train"]
    model = train_logistic_regression(train_df)
    metrics = evaluate_cohort(model, train_df)
    assert "auroc" in metrics
    assert "aupr" in metrics
    assert 0.0 <= metrics["auroc"] <= 1.0


def test_train_and_eval_ensemble() -> None:
    df = _make_dummy_df()
    train_df = df[df["Dataset"] == "Train"]
    model = train_ensemble_model(train_df)
    metrics = evaluate_cohort(model, train_df)
    assert "f1" in metrics
    assert metrics["n_samples"] == 30


def test_evaluate_all_cohorts() -> None:
    df = _make_dummy_df()
    train_df = df[df["Dataset"] == "Train"]
    model = train_logistic_regression(train_df)
    res_df = evaluate_all_cohorts(model, df)
    assert "Train" in res_df.index
    assert "External validation1" in res_df.index


def test_plot_roc_pr(tmp_path) -> None:
    df = _make_dummy_df()
    train_df = df[df["Dataset"] == "Train"]
    model = train_logistic_regression(train_df)
    out = plot_roc_pr(model, df, path=tmp_path / "roc_pr_test.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_export_artifacts() -> None:
    res = export_ml_artifacts()
    assert res["features_parquet"].is_file()
    assert res["priors_parquet"].is_file()
    assert res["features_csv"].is_file()
    assert res["priors_csv"].is_file()
