"""Модульные тесты для Блока Б (Граф пациентов и протеомная импьютация LOD)."""

from pathlib import Path

import pandas as pd
import pytest

from lab.patient_graph import (
    build_patient_knn_graph,
    impute_proteomics_lod,
    run_block_b,
)


@pytest.fixture(scope="module")
def sample_data():
    """Тестовые данные для графа и импьютации."""
    features = pd.DataFrame(
        {
            "AMBP": [0.1, 0.2, 0.9, 0.8],
            "FN1": [0.05, 0.15, 0.85, 0.95],
            "LUM": [0.02, 0.08, 0.75, 0.80],
            "TIMP1": [0.1, 0.12, 0.60, 0.70],
            "B2M": [0.2, 0.25, 0.50, 0.55],
        },
        index=["P1", "P2", "P3", "P4"],
    )
    metadata = pd.DataFrame(
        {
            "Dataset": ["Train", "Train", "EV1", "EV1"],
            "Target": [0, 0, 1, 1],
            "Hospital_Center": ["MCH1", "MCH1", "MCH2", "MCH2"],
        },
        index=["P1", "P2", "P3", "P4"],
    )
    return features, metadata


def test_build_patient_knn_graph(sample_data):
    """Проверяет корректность построения графа соседей."""
    features, metadata = sample_data
    graph_df = build_patient_knn_graph(features, metadata, k_neighbors=2)

    assert not graph_df.empty
    assert "source" in graph_df.columns
    assert "target" in graph_df.columns
    assert "weight" in graph_df.columns
    assert "is_cross_center" in graph_df.columns

    # Проверяем диапазон весов RBF
    assert (graph_df["weight"] >= 0.0).all()
    assert (graph_df["weight"] <= 1.0).all()

    # Проверяем отсутствие петель (self-loops)
    assert (graph_df["source"] != graph_df["target"]).all()


def test_impute_proteomics_lod():
    """Проверяет LOD-импьютацию пропусков в протеомике."""
    raw_df = pd.DataFrame(
        {
            "ProtA": [10.0, 12.0, None, 11.0],
            "ProtB": [None, 5.0, 6.0, 5.5],
        }
    )
    imputed = impute_proteomics_lod(raw_df, shift=1.8, scale=0.3)

    assert not imputed.isna().any().any()
    assert imputed.shape == raw_df.shape
    # Импутированные значения должны быть ниже среднего исходных
    assert imputed.loc[2, "ProtA"] < raw_df["ProtA"].mean()


def test_run_block_b_full_pipeline(tmp_path: Path):
    """Проверяет выполнение Блока Б на реальных артефактах."""
    artifacts = run_block_b()
    assert len(artifacts.edge_index_df) > 0
    # 426 образцов по k=8 соседей = 426 * 8 = 3408 ребер
    assert len(artifacts.edge_index_df) == 426 * 8

    # Проверяем наличие межцентровых ребер (трансфер знаний)
    assert artifacts.edge_index_df["is_cross_center"].sum() > 0

    assert not artifacts.imputed_discovery.isna().any().any()
