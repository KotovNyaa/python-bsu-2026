"""Тесты валидации логики Этапа 1 (предобработка, БД, статистика, визуализация)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lab.data import ROOT
from lab.db import enrich
from lab.prep import (
    CORE_MARKERS,
    encode_categorical_features,
    impute_hybrid_mar_mnar,
    load_elisa_paired,
    load_markers,
    prepare_maxlfq_matrix,
)
from lab.sql import decoy_stats, stratify_fdr_tiers
from lab.stats import (
    aggregate_by_groups,
    describe,
    describe_categorical,
    diff,
)
from lab.viz import categorical_bars, hist_distributions

# ==============================================================================
# 1. ТЕСТЫ ПРЕДОБРАБОТКИ И ГИБРИДНОЙ ИМПЬЮТАЦИИ (lab/prep.py)
# ==============================================================================


def test_log2_scale_and_skew_reduction():
    """Проверяет, что log2-шкала сохраняет неотрицательный таргет и снижает skewness."""
    excel_path = ROOT / "Extended Data Fig 5.xlsx"
    if not excel_path.is_file():
        pytest.skip("Extended Data Fig 5.xlsx не найден для интеграционного теста")

    df = load_markers(excel_path, log_scale=True)

    # 1. Таргет строго бинарный
    assert set(df["Target"].unique()).issubset({0, 1})
    assert len(df) == 426

    # 2. Все 5 маркеров присутствуют в log2 и raw
    for m in CORE_MARKERS:
        assert m in df.columns
        assert f"{m}_raw" in df.columns
        # Асимметрия log2 должна быть кардинально ниже сырой
        raw_skew = float(df[f"{m}_raw"].skew())
        log_skew = float(df[m].skew())
        assert abs(log_skew) < abs(raw_skew)
        assert abs(log_skew) < 1.0  # квазинормальное распределение


def test_genes_null_fallback_preserves_all_proteins():
    """Тест ограничения #3: 48 белков с Genes == NULL не теряются и получают Protein.Group."""
    sample_cols = [f"PTB_CQ_P1_{i}" for i in range(53, 58)]
    data = {
        "Protein.Group": ["P51884", "P02751", "P99999_NULL_GENE", "P88888_EMPTY"],
        "Genes": ["LUM", "FN1", np.nan, ""],
    }
    for col in sample_cols:
        data[col] = [1e6, 2e6, 3e5, 5e5]

    df_synth = pd.DataFrame(data)
    aligned, imputed, mask = prepare_maxlfq_matrix(df_synth)

    # Должны сохраниться все 4 белка в строках (или столбцах при транспонировании)
    assert aligned.shape == (5, 4)
    assert imputed.shape == (5, 4)
    assert mask.shape == (5, 4)

    # Идентификаторы в колонках не должны содержать nan или пустых строк
    cols = list(imputed.columns)
    assert "LUM" in cols
    assert "FN1" in cols
    assert "P99999_NULL_GENE" in cols
    assert "P88888_EMPTY" in cols
    assert not any(pd.isna(c) or c == "" for c in cols)


def test_hybrid_imputation_mar_vs_mnar():
    """Тест разделения MAR (восстановление через kNN) и MNAR (левый хвост LOD)."""
    rng = np.random.default_rng(42)
    # Матрица: 20 белков x 10 образцов
    # Белок 0 — высокоэкспрессированный (MAR), белок 1 — низкоэкспрессированный (MNAR)
    base_mat = rng.normal(loc=24.0, scale=1.5, size=(20, 10))
    base_mat[1, :] = rng.normal(loc=16.0, scale=0.5, size=10)  # низкий уровень

    df_log = pd.DataFrame(
        base_mat,
        index=[f"PROT_{i}" for i in range(20)],
        columns=[f"S_{j}" for j in range(10)],
    )

    # Создаем пропуск в образце 0 для обоих белков
    df_log.iloc[0, 0] = np.nan  # MAR кандидат
    df_log.iloc[1, 0] = np.nan  # MNAR кандидат

    imputed = impute_hybrid_mar_mnar(df_log, random_seed=42)

    assert not imputed.isna().any().any()
    # Пропуск у высокоэкспрессированного белка должен быть восстановлен близко к его уровню
    assert imputed.iloc[0, 0] > 20.0
    # Пропуск у низкоэкспрессированного белка должен быть импутирован из левого хвоста
    assert imputed.iloc[1, 0] < imputed.iloc[0, 0]


def test_elisa_paired_no_global_patient_drop():
    """Тест ограничения #2: неполные четверки в листе 5d фильтруются побелково, не удаляя пациентку."""
    excel_path = ROOT / "Extended Data Fig 5.xlsx"
    if not excel_path.is_file():
        pytest.skip("Extended Data Fig 5.xlsx не найден для интеграционного теста")

    df_elisa = load_elisa_paired(excel_path)
    # В аудите зафиксировано 1486 валидных строк парных измерений
    assert len(df_elisa) == 1486
    # Должно быть 386 уникальных пациенток, несмотря на то, что у части из них < 4 белков
    assert df_elisa["Sample ID"].nunique() == 386
    assert not df_elisa["log2(MS signal)"].isna().any()
    assert not df_elisa["log2(ELISA)"].isna().any()


def test_categorical_encoding():
    """Проверяет корректность порядкового и one-hot кодирования."""
    df_mock = pd.DataFrame(
        {
            "Dataset": ["Train", "Internal validation", "External validation1"],
            "Category": ["FTB", "PTB", "FTB"],
        }
    )
    encoded = encode_categorical_features(df_mock)

    assert "Category_Ordinal" in encoded.columns
    assert list(encoded["Category_Ordinal"]) == [0, 1, 0]
    assert "Dataset_Stage_Ordinal" in encoded.columns
    assert "Dataset_OHE_Train" in encoded.columns


# ==============================================================================
# 2. ТЕСТЫ БАЗЫ ДАННЫХ SQLite И SQL-ЗАПРОСОВ (lab/db.py, lab/sql.py)
# ==============================================================================


def test_sqlite_primary_key_and_enrich():
    """Проверяет создание строгой схемы со связкой PRIMARY KEY и закрытие NULL генов."""
    conn = sqlite3.connect(":memory:")

    # Создаем синтетическую proteins с 1 пропуском Genes
    conn.execute(
        """
        CREATE TABLE proteins (
            Protein_Group TEXT PRIMARY KEY,
            Genes TEXT,
            Protein_Names TEXT,
            PTB_CQ_P1_53 REAL
        )
        """
    )
    conn.execute("INSERT INTO proteins VALUES ('P51884', 'LUM', 'LUM_HUMAN', 1000.0)")
    conn.execute("INSERT INTO proteins VALUES ('P99999', NULL, NULL, 500.0)")

    # Создаем transitions
    conn.execute(
        """
        CREATE TABLE transitions (
            ProteinGroup TEXT,
            Genes TEXT,
            ProteinName TEXT,
            QValue REAL,
            decoy INTEGER
        )
        """
    )
    # P99999 нет в transitions — сработает второй этап закрытия COALESCE
    enrich(conn)

    res = pd.read_sql_query("SELECT Protein_Group, Genes FROM proteins", conn)
    genes_map = dict(zip(res["Protein_Group"], res["Genes"], strict=False))

    assert genes_map["P51884"] == "LUM"
    assert (
        genes_map["P99999"] == "P99999"
    )  # fallback на Protein_Group успешно отработал
    conn.close()


def test_decoy_stats_adaptive_to_zero_decoys():
    """Тест ограничения #1: адаптивное разделение по QValue > 0.01 при физическом decoy == 0."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE transitions (
            Genes TEXT,
            LibraryIntensity REAL,
            Tr_recalibrated REAL,
            decoy INTEGER,
            QValue REAL
        )
        """
    )
    # Вставляем данные, где decoy везде 0 (как в реальной DIA-NN выгрузке), но есть QValue
    conn.execute("INSERT INTO transitions VALUES ('LUM', 10000.0, 15.2, 0, 0.001)")
    conn.execute("INSERT INTO transitions VALUES ('FN1', 8000.0, 22.1, 0, 0.008)")
    conn.execute("INSERT INTO transitions VALUES ('NOISE', 500.0, 30.5, 0, 0.045)")
    conn.commit()

    stats_df = decoy_stats(conn)
    assert len(stats_df) == 2
    # Класс 0 — достоверные (Q <= 0.01), класс 1 — шумовой хвост (Q > 0.01)
    assert set(stats_df["decoy"]) == {0, 1}

    tier_df = stratify_fdr_tiers(conn)
    assert not tier_df.empty
    assert "transition_count" in tier_df.columns
    conn.close()


# ==============================================================================
# 3. ТЕСТЫ СТАТИСТИКИ И ВИЗУАЛИЗАЦИИ (lab/stats.py, lab/viz.py)
# ==============================================================================


def test_statistics_and_normality_functions(tmp_path: Path):
    """Проверяет расчет расширенных описательных статистик, Шапиро-Уилка и IQR."""
    rng = np.random.default_rng(42)
    # Экспоненциальное распределение (сильный skew) vs нормальное
    df_test = pd.DataFrame(
        {
            "skewed": rng.exponential(scale=2.0, size=150),
            "normal": rng.normal(loc=0.0, scale=1.0, size=150),
            "Category": ["FTB"] * 75 + ["PTB"] * 75,
            "Dataset": ["Train"] * 150,
        }
    )

    desc = describe(df_test)
    assert "iqr" in desc.columns
    assert "shapiro_p" in desc.columns
    assert "is_normal_005" in desc.columns
    assert desc.loc["skewed", "skew"] > desc.loc["normal", "skew"]

    cat_desc = describe_categorical(df_test, cols=["Category", "Dataset"])
    assert len(cat_desc) == 2
    assert cat_desc.loc["Category", "mode"] in ["FTB", "PTB"]

    grp = aggregate_by_groups(df_test, features=["normal"], group_cols=["Category"])
    assert grp.shape[0] == 2  # FTB и PTB

    diff_res = diff(df_test, features=["normal", "skewed"], target="Category")
    assert len(diff_res) == 2
    assert "log2_fc" in diff_res.columns
    assert "p_adj" in diff_res.columns


def test_visualizations_generate_files_without_gui(tmp_path: Path):
    """Проверяет генерацию 300 DPI графиков hist.png и countplot.png в безголовом режиме."""
    df_test = pd.DataFrame(
        {
            "LUM": [-2.5, -1.2, 0.5, 1.2, -0.4, 2.1],
            "FN1": [-3.1, -0.8, 1.1, 0.9, -1.0, 1.8],
            "Category": ["FTB", "FTB", "FTB", "PTB", "PTB", "PTB"],
            "Dataset": [
                "Train",
                "Train",
                "Internal validation",
                "Internal validation",
                "External validation1",
                "External validation1",
            ],
        }
    )

    hist_out = tmp_path / "hist.png"
    count_out = tmp_path / "countplot.png"

    hist_path = hist_distributions(df_test, features=["LUM", "FN1"], path=hist_out)
    count_path = categorical_bars(df_test, path=count_out)

    assert Path(hist_path).is_file()
    assert Path(count_path).is_file()
    assert Path(hist_path).stat().st_size > 0
    assert Path(count_out).stat().st_size > 0
