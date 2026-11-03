"""Тесты модуля формирования когортных артефактов (Блок А)."""

from pathlib import Path

import pandas as pd
import pytest

from lab.cohorts import (
    CORE_MARKERS,
    export_cohort_artifacts,
    generate_cohort_artifacts,
)


@pytest.fixture(scope="module")
def artifacts():
    """Фикстура генерации артефактов."""
    return generate_cohort_artifacts()


def test_metadata_structure(artifacts):
    """Проверяет структуру и распределение метаданных пациентов."""
    meta = artifacts.metadata
    assert len(meta) == 426
    assert set(meta.columns) == {
        "Dataset",
        "Category",
        "Target",
        "Hospital_Center",
        "Center_Code",
    }
    assert meta["Target"].sum() == 89
    assert (meta["Target"] == 0).sum() == 337

    cohort_counts = meta["Dataset"].value_counts().to_dict()
    assert cohort_counts["Train"] == 82
    assert cohort_counts["Internal validation"] == 83
    assert cohort_counts["External validation1"] == 127
    assert cohort_counts["External validation2"] == 134


def test_features_5_integrity(artifacts):
    """Проверяет целостность матрицы 5 классических маркеров."""
    f5 = artifacts.features_5
    assert f5.shape == (426, 5)
    assert list(f5.columns) == CORE_MARKERS
    assert not f5.isna().any().any()
    assert (f5.index == artifacts.metadata.index).all()


def test_instrumental_evidence(artifacts):
    """Проверяет полноту приборных метрик достоверности."""
    inst = artifacts.instrumental_evidence
    assert not inst.empty
    assert "trans_count" in inst.columns
    assert "unique_peptides" in inst.columns

    for marker in ["LUM", "FN1", "AMBP", "TIMP1", "B2M"]:
        assert marker in inst.index


def test_discovery_wide(artifacts):
    """Проверяет корректность нормализации широкой матрицы белков."""
    disc = artifacts.discovery_wide
    assert not disc.empty
    assert disc.shape[0] == 35
    assert not disc.isna().any().any()


def test_export_artifacts(artifacts, tmp_path: Path):
    """Проверяет экспорт Parquet и CSV артефактов."""
    paths = export_cohort_artifacts(artifacts, output_dir=tmp_path)
    for p in paths.values():
        assert p.is_file()
        assert p.stat().st_size > 0

    read_meta = pd.read_parquet(paths["metadata_parquet"])
    assert len(read_meta) == 426

    read_f5 = pd.read_parquet(paths["features_5_parquet"])
    assert read_f5.shape == (426, 5)
