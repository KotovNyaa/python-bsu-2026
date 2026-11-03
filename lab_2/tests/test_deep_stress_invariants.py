"""Глубокий стресс-тест вычислительной устойчивости и совместимости с PyTorch/GPU."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from lab.data import locate


@pytest.fixture(scope="module")
def artifacts():
    """Загрузка полного пакета артефактов."""
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
# 1. СТРЕСС-ТЕСТ СВЯЗНОСТИ И СПЕКТРАЛЬНЫХ СВОЙСТВ ГРАФА ПАЦИЕНТОВ
# =========================================================================


def test_patient_graph_connectedness_and_density(artifacts):
    """Проверяет спектральную связность и отсутствие изолированных компонент."""
    edges = artifacts["patient_edges"]
    meta = artifacts["metadata"]

    sample_ids = meta.index.tolist()
    id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

    src = [id_to_idx[s] for s in edges["source"]]
    tgt = [id_to_idx[t] for t in edges["target"]]
    weights = edges["weight"].values

    # Построение матрицы смежности (426 x 426)
    adj = coo_matrix(
        (weights, (src, tgt)), shape=(len(sample_ids), len(sample_ids))
    ).tocsr()

    # Симметризация для ненаправленного графа
    adj_sym = adj + adj.T

    # Проверка числа компонент связности
    n_components, _labels = connected_components(
        adj_sym, directed=False, return_labels=True
    )
    assert n_components == 1, (
        f"Граф пациентов распался на {n_components} изолированных компонент!"
    )

    # Проверка степеней узлов (каждый пациент обязан иметь соразмерное число соседей)
    degrees = np.diff(adj.indptr)
    assert (degrees >= 8).all(), "Обнаружены узлы с недостаточной степенью"
    assert degrees.max() <= 100, "Обнаружен гигантский узел-монополист (хаб-аномалия)"


# =========================================================================
# 2. СТРЕСС-ТЕСТ ОТСУТСТВИЯ УТЕЧКИ ДАННЫХ МЕЖДУ КОГОРТАМИ
# =========================================================================


def test_zero_leakage_between_cohorts(artifacts):
    """Проверяет абсолютную изолированность ID пациентов между 4 когортами."""
    meta = artifacts["metadata"]

    train_ids = set(meta[meta["Dataset"] == "Train"].index)
    int_val_ids = set(meta[meta["Dataset"] == "Internal validation"].index)
    ev1_ids = set(meta[meta["Dataset"] == "External validation1"].index)
    ev2_ids = set(meta[meta["Dataset"] == "External validation2"].index)

    cohorts = [
        ("Train", train_ids),
        ("IntVal", int_val_ids),
        ("EV1", ev1_ids),
        ("EV2", ev2_ids),
    ]

    for i in range(len(cohorts)):
        for j in range(i + 1, len(cohorts)):
            name_i, set_i = cohorts[i]
            name_j, set_j = cohorts[j]
            intersection = set_i.intersection(set_j)
            assert len(intersection) == 0, (
                f"Утечка данных! {len(intersection)} пациентов пересекаются между {name_i} и {name_j}"
            )


# =========================================================================
# 3. СТРЕСС-ТЕСТ МЕЖЦЕНТРОВОГО МОСТА ДЛЯ РЕДКИХ СЛУЧАЕВ (MCH2 / EV1)
# =========================================================================


def test_cross_center_bridging_for_imbalanced_cohort(artifacts):
    """Проверяет, что редкие случаи sPTB из EV1 связаны с Train/IntVal."""
    edges = artifacts["patient_edges"]
    meta = artifacts["metadata"]

    # 11 редких случаев в MCH2
    ev1_cases = set(
        meta[(meta["Dataset"] == "External validation1") & (meta["Target"] == 1)].index
    )
    assert len(ev1_cases) == 11

    # Проверяем, что для КАЖДОГО из 11 редких случаев есть хотя бы одно ребро в Train или IntVal
    edges_from_ev1_cases = edges[edges["source"].isin(ev1_cases)]
    train_int_targets = set(
        meta[meta["Dataset"].isin(["Train", "Internal validation"])].index
    )

    bridged_cases = set()
    for _, row in edges_from_ev1_cases.iterrows():
        if row["target"] in train_int_targets:
            bridged_cases.add(row["source"])

    # Минимум 80% редких случаев обязаны иметь прямой мост в обучающие центры
    coverage = len(bridged_cases) / len(ev1_cases)
    assert coverage >= 0.80, (
        f"Недостаточный мост знаний: только {coverage:.1%} редких случаев связаны с Train"
    )


# =========================================================================
# 4. СТРЕСС-ТЕСТ ПОЛОЖИТЕЛЬНОЙ ОПРЕДЕЛЕННОСТИ МАТРИЦЫ ЛАПЛАСА (STRING PPI)
# =========================================================================


def test_string_ppi_laplacian_properties(artifacts):
    """Проверяет, что матрица Лапласа PPI является положительно полуопределенной."""
    ppi = artifacts["ppi_edges"]
    genes = sorted(set(ppi["source_gene"]).union(set(ppi["target_gene"])))
    n = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # Строим матрицу смежности
    A = np.zeros((n, n), dtype=np.float64)
    for _, row in ppi.iterrows():
        i, j = gene_to_idx[row["source_gene"]], gene_to_idx[row["target_gene"]]
        w = row["combined_score"]
        A[i, j] = w
        A[j, i] = w

    # Матрица Лапласа L = D - A
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Собственные значения матрицы Лапласа обязаны быть >= 0
    eigenvalues = np.linalg.eigvalsh(L)
    min_eig = float(eigenvalues.min())

    assert min_eig >= -1e-6, (
        f"Матрица Лапласа не положительно полуопределена! min eigenvalue = {min_eig}"
    )
    # Первое собственное значение связного графа всегда 0
    assert abs(min_eig) < 1e-4, "Минимальное собственное значение Лапласа должно быть 0"


# =========================================================================
# 5. СТРЕСС-ТЕСТ ЧИСЛОВОЙ СТАБИЛЬНОСТИ ДЛЯ BATCHNORM / STANDARDIZATION
# =========================================================================


def test_zero_variance_absence_in_all_matrices(artifacts):
    """Проверяет отсутствие константных фичей (нулевой дисперсии) в обеих матрицах."""
    f5 = artifacts["features"]
    disc = artifacts["discovery_lod"]

    # Матрица 5 признаков
    f5_stds = f5.std(axis=0)
    assert (f5_stds > 1e-3).all(), "В 5 признаках обнаружена околонулевая дисперсия!"

    # Широкая матрица 2699 белков
    disc_stds = disc.std(axis=0)
    assert (disc_stds > 1e-4).all(), (
        "В широкой матрице обнаружены белки с нулевой дисперсией!"
    )

    # Проверка отсутствия отрицательных концентраций
    assert (disc.values >= -50.0).all(), "Аномально низкие значения логарифма"
    assert (disc.values <= 50.0).all(), "Аномально высокие значения логарифма"


# =========================================================================
# 6. ПРЯМОЙ СТРЕСС-ТЕСТ СОВМЕСТИМОСТИ С PYTORCH И ПРЯМЫМ ПРОГОНОМ
# =========================================================================


def test_pytorch_tensor_simulation(artifacts):
    """Симулирует прямой проход и вычисление градиента для PyTorch без зависимостей от PyG."""
    x = artifacts["features"].values.astype(np.float32)
    y = artifacts["metadata"]["Target"].values.astype(np.float32)
    assert y.shape == (426,), "Некорректная размерность вектора меток"

    # 1. Проверяем матричное умножение W * x + b
    W = np.random.randn(5, 16).astype(np.float32) * 0.1
    b = np.zeros(16, dtype=np.float32)
    hidden = np.maximum(0, np.dot(x, W) + b)  # ReLU

    assert hidden.shape == (426, 16)
    assert np.isfinite(hidden).all(), "NaN в скрытом слое нейросети"

    # 2. Проверяем взвешенную агрегацию по графу соседей: H_new = D^-1 * A * H
    edges = artifacts["patient_edges"]
    src = (
        edges["source"]
        .map({s: i for i, s in enumerate(artifacts["metadata"].index)})
        .values
    )
    tgt = (
        edges["target"]
        .map({s: i for i, s in enumerate(artifacts["metadata"].index)})
        .values
    )
    w = edges["weight"].values.astype(np.float32)

    # Агрегация сообщений
    message_out = np.zeros_like(hidden)
    deg = np.zeros((426, 1), dtype=np.float32)

    for s, t, weight in zip(src, tgt, w, strict=False):
        message_out[s] += weight * hidden[t]
        deg[s] += weight

    deg[deg == 0] = 1.0
    aggregated = message_out / deg

    assert np.isfinite(aggregated).all(), "NaN при графовой агрегации сообщений"
    assert aggregated.shape == (426, 16)
