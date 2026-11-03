"""Ультимативный стресс-сьют готовности артефактов к GPU-кластеру и Colab."""

import numpy as np
import pandas as pd
import pytest

from lab.data import locate

CORE_MARKERS = ["AMBP", "FN1", "LUM", "TIMP1", "B2M"]
VALID_COHORTS = {
    "Train",
    "Internal validation",
    "External validation1",
    "External validation2",
}


@pytest.fixture(scope="module")
def artifacts():
    """Загрузка полного пакета артефактов кластера."""
    root = locate("metadata.parquet").parent
    return {
        "metadata": pd.read_parquet(root / "metadata.parquet"),
        "features": pd.read_parquet(root / "features_5.parquet"),
        "patient_edges": pd.read_parquet(root / "patient_graph_edges.parquet"),
        "protein_priors": pd.read_parquet(root / "protein_priors.parquet"),
        "ppi_edges": pd.read_parquet(root / "ppi_network_edges.parquet"),
        "discovery_lod": pd.read_parquet(root / "discovery_imputed_lod.parquet"),
    }


# =========================================================================
# 1. ТЕСТЫ ИНВАРИАНТОВ ПАЦИЕНТОВ И МЕТАДАННЫХ
# =========================================================================


def test_metadata_complete_invariants(artifacts):
    """Проверяет полноту и непротиворечивость метаданных пациентов."""
    meta = artifacts["metadata"]

    assert len(meta) == 426, f"Ожидалось 426 пациентов, получено {len(meta)}"
    assert meta.index.is_unique, "Обнаружены дубликаты Sample ID"
    assert not meta.index.hasnans, "Обнаружены пустые индексы Sample ID"

    # Строгое соответствие исходов
    assert set(meta["Category"].unique()) == {"FTB", "PTB"}
    assert set(meta["Target"].unique()) == {0, 1}
    assert ((meta["Category"] == "PTB") == (meta["Target"] == 1)).all(), (
        "Рассинхрон Category и бинарного Target"
    )

    # Когорты и центры
    assert set(meta["Dataset"].unique()) == VALID_COHORTS
    assert (meta["Hospital_Center"] != "Unknown").all(), "Найдены неизвестные клиники"
    assert (meta["Center_Code"] != "UNK").all()


def test_features_numerical_invariants(artifacts):
    """Проверяет числовые свойства матрицы биомаркеров."""
    f = artifacts["features"]
    meta = artifacts["metadata"]

    # Индексы строго совпадают с метаданными
    assert (f.index == meta.index).all(), "Рассинхрон порядка строк фичей и метаданных"
    assert list(f.columns) == CORE_MARKERS

    # Типы и отсутствие NaN / Inf
    assert (f.dtypes == np.float64).all(), "Все признаки обязаны быть float64"
    assert not f.isna().any().any(), "Обнаружены пропущенные значения (NaN)"
    assert np.isfinite(f.values).all(), "Обнаружены бесконечные значения (Inf)"

    # Физические ограничения (концентрации строго положительны)
    assert (f.values >= 0.0).all(), "Концентрации белков не могут быть отрицательными"
    assert (f.values.max(axis=0) > 0.0).all(), "Обнаружен константный нулевой признак"


# =========================================================================
# 2. ТЕСТЫ ТОПОЛОГИИ ГРАФА ПАЦИЕНТОВ (PATIENT-GNN)
# =========================================================================


def test_patient_graph_topology_invariants(artifacts):
    """Проверяет топологическую валидность графа пациентов."""
    edges = artifacts["patient_edges"]
    sample_ids = set(artifacts["metadata"].index)

    # Все узлы обязаны принадлежать исходной когорте
    assert set(edges["source"]).issubset(sample_ids), (
        "Source ребра вне списка пациентов"
    )
    assert set(edges["target"]).issubset(sample_ids), (
        "Target ребра вне списка пациентов"
    )

    # Отсутствие петель
    assert (edges["source"] != edges["target"]).all(), "Обнаружены петли (self-loops)"

    # Полнота покрытия (нет изолированных узлов)
    nodes_in_graph = set(edges["source"]).union(set(edges["target"]))
    assert nodes_in_graph == sample_ids, (
        f"Изолировано {len(sample_ids - nodes_in_graph)} узлов"
    )

    # Числовые ограничения весов RBF
    assert (edges["weight"] > 0.0).all(), "Вес ребра должен быть строго > 0"
    assert (edges["weight"] <= 1.0).all(), "Вес ребра RBF ядра не может превышать 1.0"
    assert (edges["distance"] >= 0.0).all(), "Расстояния не могут быть отрицательными"

    # Межцентровой трансфер
    cross_edges = edges[edges["is_cross_center"] == 1]
    assert len(cross_edges) > 500, "Слишком мало межцентровых связей для трансфера"


# =========================================================================
# 3. ТЕСТЫ БИОФИЗИЧЕСКИХ И ПРИБОРНЫХ ПРИОРОВ
# =========================================================================


def test_protein_priors_biophysical_invariants(artifacts):
    """Проверяет биофизические дескрипторы белков UniProt."""
    priors = artifacts["protein_priors"]

    for marker in CORE_MARKERS:
        assert marker in priors.index, f"Маркер {marker} отсутствует в таблице приоров"

    # Физические свойства молекул
    assert (priors["mass_kda"] > 5.0).all(), (
        "Молекулярная масса не может быть микроскопической"
    )
    assert (priors["mass_kda"] < 1000.0).all(), (
        "Молекулярная масса превышает предел белков"
    )
    assert (priors["length_aa"] > 50).all(), (
        "Длина цепи белка не может быть менее 50 аминокислот"
    )

    # Бинарные флаги
    for flag in ["is_secreted", "is_ecm", "signal_peptide"]:
        assert set(priors[flag].unique()).issubset({0, 1}), (
            f"Флаг {flag} обязан быть бинарным"
        )

    # Приорный вес доверия
    assert (priors["prior_confidence_weight"] >= 0.0).all(), (
        "Вес доверия не может быть отрицательным"
    )
    assert (priors["prior_confidence_weight"] <= 1.0).all(), (
        "Вес доверия не может быть > 1.0"
    )

    # Приборная надежность
    assert (priors["n_transitions"] >= 0).all()
    assert (priors["n_unique_peptides"] >= 0).all()
    assert (priors["mean_ms2_intensity"] >= 0.0).all()


# =========================================================================
# 4. ТЕСТЫ ИНТЕРАКТОМА STRING-DB
# =========================================================================


def test_string_ppi_network_invariants(artifacts):
    """Проверяет граф взаимодействий STRING-DB."""
    ppi = artifacts["ppi_edges"]

    assert len(ppi) > 0, "Сеть STRING пуста"
    assert (ppi["source_gene"] != ppi["target_gene"]).all(), (
        "Обнаружены петли в интерактоме"
    )

    # Скоры взаимодействий STRING
    assert (ppi["combined_score"] >= 0.40).all(), "Скоры STRING ниже порога доверия"
    assert (ppi["combined_score"] <= 1.00).all(), "Скор STRING превышает 1.0"

    for col in ["experimental_score", "database_score", "coexpression_score"]:
        assert (ppi[col] >= 0.0).all()
        assert (ppi[col] <= 1.0).all()

    # Ключевой биологический инвариант: связь FN1 и матриксных белков
    fn1_edges = ppi[(ppi["source_gene"] == "FN1") | (ppi["target_gene"] == "FN1")]
    assert len(fn1_edges) > 0, "FN1 обязан образовывать хаб во внеклеточном матриксе"


# =========================================================================
# 5. ТЕСТЫ ШИРОКОЙ МАТРИЦЫ И LOD ИМПЬЮТАЦИИ
# =========================================================================


def test_discovery_lod_invariants(artifacts):
    """Проверяет широкую матрицу с протеомной LOD-импьютацией."""
    disc = artifacts["discovery_lod"]

    assert disc.shape == (35, 2699), f"Некорректная размерность матрицы: {disc.shape}"
    assert not disc.isna().any().any(), "В матрице после импьютации остались пропуски"
    assert np.isfinite(disc.values).all(), "В матрице обнаружены значения Inf"


# =========================================================================
# 6. ТЕСТЫ СОВМЕСТИМОСТИ С PYTORCH И PYG (COLAB READY)
# =========================================================================


def test_pytorch_tensor_compatibility(artifacts):
    """Проверяет бесшовное создание тензоров для PyTorch Geometric и TabNet."""
    features = artifacts["features"].values
    target = artifacts["metadata"]["Target"].values
    edges = artifacts["patient_edges"]

    # Имитируем мэппинг ID в числовые индексы [0 .. N-1] для edge_index в PyG
    id_to_idx = {
        sample_id: idx for idx, sample_id in enumerate(artifacts["metadata"].index)
    }

    src_idx = np.array([id_to_idx[s] for s in edges["source"]], dtype=np.int64)
    tgt_idx = np.array([id_to_idx[t] for t in edges["target"]], dtype=np.int64)
    edge_index = np.vstack([src_idx, tgt_idx])
    edge_weight = edges["weight"].values.astype(np.float32)

    # Проверка размерностей для PyG Data(x=x, edge_index=edge_index, edge_attr=...)
    assert features.shape == (426, 5)
    assert target.shape == (426,)
    assert edge_index.shape == (2, len(edges))
    assert edge_index.min() == 0
    assert edge_index.max() == 425
    assert edge_weight.shape == (len(edges),)
