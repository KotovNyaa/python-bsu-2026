"""Тесты валидации Этапа 3 (индуктивный граф без утечек, кодировки GraphPFN, мультиплексный граф и маски BINN)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab.data import ROOT
from lab.external_knowledge import (
    MACRO_PATHWAYS_6,
    build_binn_hierarchy_masks,
    build_protein_multiplex_graph,
    compute_pathway_context_features,
    generate_deterministic_ppi_fallback,
    run_blocks_v_g,
)
from lab.patient_graph import (
    _compute_simplex,
    _pairwise_clr_cosine_distance,
    _pairwise_jsd_distance,
    build_graphpfn_node_encodings,
    build_patient_knn_graph,
    compute_inductive_tag_tabr_features,
    impute_proteomics_lod,
    run_block_b,
)
from lab.prep import CORE_MARKERS

# ==============================================================================
# 1. ТЕСТЫ ИНДУКТИВНОГО ГРАФА ПАЦИЕНТОК И ОТСУТСТВИЯ УТЕЧКИ (lab/patient_graph.py)
# ==============================================================================


def _make_mock_cohort_a(n_train: int = 24, n_val: int = 12, n_test: int = 14):
    """Создает синтетическую когорту А с Train, Internal validation и External validation1."""
    rng = np.random.default_rng(42)
    n_total = n_train + n_val + n_test
    sids = [f"PTB_{i:03d}" for i in range(n_total)]

    datasets = (
        ["Train"] * n_train
        + ["Internal validation"] * n_val
        + ["External validation1"] * n_test
    )
    centers = [
        "MCH1_Shenzhen_Longgang"
        if d != "External validation1"
        else "MCH2_Zhuhai_Dongguan"
        for d in datasets
    ]
    targets = ([0, 1] * (n_total // 2 + 1))[:n_total]

    # Генерируем информативные log2-признаки, где PTB выше FTB
    base = rng.normal(loc=-2.5, scale=1.1, size=(n_total, 5))
    base[np.array(targets) == 1] += 1.4

    feats_df = pd.DataFrame(base, index=sids, columns=CORE_MARKERS)
    meta_df = pd.DataFrame(
        {
            "Dataset": datasets,
            "Category": ["PTB" if t == 1 else "FTB" for t in targets],
            "Target": targets,
            "Hospital_Center": centers,
        },
        index=sids,
    )
    return feats_df, meta_df


def test_patient_knn_graph_strictly_inductive_zero_test_leakage():
    """Доказательство исправления Бага #2: 100% ребер ведут только в опорный пул Train \\ {i}."""
    feats_df, meta_df = _make_mock_cohort_a(n_train=20, n_val=10, n_test=10)
    k = 6
    edges_df = build_patient_knn_graph(feats_df, meta_df, k_neighbors=k)

    # 1. Ровно N * k направленных ребер (40 * 6 = 240)
    assert len(edges_df) == len(feats_df) * k

    # 2. 100% целевых узлов (target_dataset) принадлежат ИСКЛЮЧИТЕЛЬНО Train (0 ребер в Val/Test!)
    assert set(edges_df["target_dataset"].unique()) == {"Train"}

    # 3. Отсутствие самокопирования (i ∉ I_cand(i))
    assert (edges_df["source"] != edges_df["target"]).all()

    # 4. Проверка наличия весов всех 3 модальностей в диапазоне (0, 1]
    for w_col in ["weight", "view_jsd_weight", "view_clr_weight", "view_mahal_weight"]:
        assert w_col in edges_df.columns
        vals = edges_df[w_col].to_numpy()
        assert np.all((vals > 0.0) & (vals <= 1.0))


def test_jsd_and_clr_kernels_dilution_invariance():
    """Доказывает, что ядерные расстояния JSD и CLR-Cosine абсолютно инвариантны к разведению мазка."""
    rng = np.random.default_rng(101)
    x_orig = rng.normal(loc=-2.0, scale=1.3, size=(25, 5))
    # Добавляем случайный сдвиг разведения к каждой строке
    dilution_shift = rng.uniform(-3.0, 4.0, size=(25, 1))
    x_diluted = x_orig + dilution_shift

    p_orig = _compute_simplex(x_orig)
    p_dil = _compute_simplex(x_diluted)

    d_jsd_orig = _pairwise_jsd_distance(p_orig, p_orig[:10])
    d_jsd_dil = _pairwise_jsd_distance(p_dil, p_dil[:10])
    np.testing.assert_allclose(d_jsd_orig, d_jsd_dil, rtol=1e-12, atol=1e-12)

    d_clr_orig = _pairwise_clr_cosine_distance(x_orig, x_orig[:10])
    d_clr_dil = _pairwise_clr_cosine_distance(x_diluted, x_diluted[:10])
    np.testing.assert_allclose(d_clr_orig, d_clr_dil, rtol=1e-12, atol=1e-12)


def test_tag_tabr_features_single_patient_inference_isolation():
    """Доказывает, что возмущение тестовой пациентки (+100.0) НЕ влияет ни на Train, ни на других тестовых пациенток."""
    feats_df, meta_df = _make_mock_cohort_a(n_train=20, n_val=8, n_test=8)
    tag_clean = compute_inductive_tag_tabr_features(feats_df, meta_df, k_neighbors=5)

    assert not tag_clean.isna().any().any()
    assert "graph_knn_label_consensus" in tag_clean.columns
    assert "graph_tabr_delta1_LUM" in tag_clean.columns
    assert "graph_tag_hop2_FN1" in tag_clean.columns

    # Искажаем последнюю тестовую пациентку на +100 логарифмических единиц
    feats_perturbed = feats_df.copy()
    feats_perturbed.iloc[-1, :] += 100.0
    tag_perturbed = compute_inductive_tag_tabr_features(
        feats_perturbed, meta_df, k_neighbors=5
    )

    # Все остальные N-1 пациенток (и Train, и Val, и соседние Test!) обязаны сохранить идентичные признаки
    np.testing.assert_allclose(
        tag_clean.iloc[:-1].to_numpy(),
        tag_perturbed.iloc[:-1].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_graphpfn_node_encodings_unified_461_window():
    """Проверяет построение единого окна кодировок GraphPFN (Когорта А + Чунцин) с интерполяцией Нистрёма."""
    feats_a, meta_a = _make_mock_cohort_a(n_train=20, n_val=6, n_test=6)
    rng = np.random.default_rng(55)
    # Имитируем 10 образцов Чунцина в шкале log2(MaxLFQ) ~ 20.0 (без TIMP1)
    disc_b = pd.DataFrame(
        rng.normal(loc=21.0, scale=1.5, size=(10, 6)),
        index=[f"PTB_CQ_{i}" for i in range(10)],
        columns=["AMBP", "FN1", "LUM", "B2M", "C3", "ALB"],
    )

    enc_df = build_graphpfn_node_encodings(feats_a, meta_a, disc_b, k_neighbors=5)

    # Должно быть 32 (Когорта А) + 10 (Чунцин) = 42 строки
    assert len(enc_df) == len(feats_a) + len(disc_b)
    assert int(enc_df["Is_Query_Unlabeled"].sum()) == len(disc_b)

    # Проверяем наличие Laplacian PE, RWSE и PageRank без единого NaN
    num_cols = enc_df.select_dtypes(include=[np.number]).columns
    assert not enc_df[num_cols].isna().any().any()
    assert np.all(np.isfinite(enc_df[num_cols].to_numpy()))
    assert "lap_pe_dim01" in enc_df.columns
    assert "rwse_step1_max_prob" in enc_df.columns
    assert "ppr_risk_score" in enc_df.columns


def test_impute_proteomics_lod_left_tail():
    """Проверяет, что LOD-импьютация заполняет пропуски значениями из левого хвоста ниже среднего."""
    rng = np.random.default_rng(42)
    mat = rng.normal(loc=22.0, scale=1.5, size=(15, 8))
    df_nan = pd.DataFrame(mat, columns=[f"G_{i}" for i in range(8)])
    df_nan.iloc[0, 0] = np.nan
    df_nan.iloc[1, 2] = np.nan

    imputed = impute_proteomics_lod(df_nan, shift=1.8, scale=0.3, random_seed=42)
    assert not imputed.isna().any().any()
    # Импутированное значение обязано лежать в левом хвосте (< среднего 22.0)
    assert imputed.iloc[0, 0] < 20.5
    assert imputed.iloc[1, 2] < 20.5


# ==============================================================================
# 2. ТЕСТЫ МУЛЬТИПЛЕКСНОГО ГРАФА БЕЛКОВ, 6 ПУТЕЙ И МАСОК BINN (lab/external_knowledge.py)
# ==============================================================================


def test_protein_multiplex_graph_and_timp1_bridge():
    """Проверяет векторизованный бутстреп-граф коэкспрессии и интеграцию TIMP1 через STRING."""
    rng = np.random.default_rng(77)
    n_samples = 35
    genes = [
        "LUM",
        "FN1",
        "AMBP",
        "B2M",
        "COL1A2",
        "THBS1",
        "C3",
        "C4A",
        "CFH",
        "PLG",
        "HRG",
        "SERPINA1",
        "PRG2",
        "ZG16B",
        "ALB",
        "GC",
        "TF",
        "SOD1",
    ]
    # Создаем сильную коэкспрессию внутри пар (LUM, COL1A2) и (C3, C4A)
    latent_ecm = rng.normal(20.0, 1.5, size=n_samples)
    latent_comp = rng.normal(22.0, 1.8, size=n_samples)

    mat = rng.normal(19.0, 1.0, size=(n_samples, len(genes)))
    df_disc = pd.DataFrame(mat, columns=genes)
    df_disc["LUM"] = latent_ecm + rng.normal(0, 0.15, size=n_samples)
    df_disc["COL1A2"] = latent_ecm + rng.normal(0, 0.15, size=n_samples)
    df_disc["C3"] = latent_comp + rng.normal(0, 0.15, size=n_samples)
    df_disc["C4A"] = latent_comp + rng.normal(0, 0.15, size=n_samples)

    ppi_df = generate_deterministic_ppi_fallback([*genes, "TIMP1"])
    multiplex_df, tensors = build_protein_multiplex_graph(
        df_disc, pd.DataFrame(), ppi_df, n_bootstrap=200, r_thresh=0.50
    )

    assert not multiplex_df.empty
    # TIMP1 отсутствует в столбцах df_disc, но обязан войти в узлы мультиплексного графа через STRING!
    assert "TIMP1" in tensors["node_names"]
    timp1_edges = multiplex_df[
        (multiplex_df["source_gene"] == "TIMP1")
        | (multiplex_df["target_gene"] == "TIMP1")
    ]
    assert len(timp1_edges) >= 3
    assert tensors["edge_index"].shape[0] == 2
    assert tensors["edge_index"].shape[1] == tensors["edge_attr"].shape[0]


def test_pathway_context_features_full_rank_nonlinearity():
    """Математическое доказательство Нюанса 2 из answers_rewiew.md:
    нелинейные путевые операторы дают матрицу полного ранга (rank > 5), не вырождаясь в линейную комбинацию X*Z.
    """
    feats_a, meta_a = _make_mock_cohort_a(n_train=30, n_val=10, n_test=10)
    rng = np.random.default_rng(12)
    genes = [
        "LUM",
        "FN1",
        "AMBP",
        "B2M",
        "COL1A2",
        "C3",
        "C4A",
        "PLG",
        "SERPINA1",
        "PRG2",
        "ALB",
        "SOD1",
        "ATP5F1A",
    ]
    disc_b = pd.DataFrame(rng.normal(20.0, 1.2, size=(35, len(genes))), columns=genes)
    ppi_df = generate_deterministic_ppi_fallback([*genes, "TIMP1"])
    multiplex_df, _ = build_protein_multiplex_graph(
        disc_b, pd.DataFrame(), ppi_df, n_bootstrap=100
    )

    pw_feats, z_df = compute_pathway_context_features(
        feats_a, meta_a, disc_b, multiplex_df
    )

    # Матрица Z имеет размерность 5 ядерных маркеров x 24 признака 6 путей
    assert z_df.shape == (5, 24)
    # Проекционные признаки пациенток имеют размерность N x 24
    assert pw_feats.shape == (len(feats_a), 24)
    assert not pw_feats.isna().any().any()

    # Ранг объединенной матрицы (5 исходных + 24 путевых) обязан быть строго больше 5!
    combined = np.column_stack([feats_a.to_numpy(), pw_feats.to_numpy()])
    rank_val = np.linalg.matrix_rank(combined)
    assert rank_val >= 15, f"Ожидался нелинейный полный ранг >= 15, получен {rank_val}"


def test_binn_hierarchy_masks_sparsity_and_connectivity():
    """Проверяет размерности, разреженность (>95%) и сквозную связность 4 слоев масок BINN."""
    mock_proteins = [
        "LUM",
        "FN1",
        "C3",
        "PLG",
        "PRG2",
        "SOD1",
        "ATP5F1A",
        *[f"PROT_{i}" for i in range(100)],
    ]
    masks_dict, edge_table = build_binn_hierarchy_masks(
        mock_proteins, n_layer1=350, n_layer2=120, n_layer3=35
    )

    m1, m2, m3, m4 = (
        masks_dict["M1"],
        masks_dict["M2"],
        masks_dict["M3"],
        masks_dict["M4"],
    )
    assert m1.shape == (len(mock_proteins), 350)
    assert m2.shape == (350, 120)
    assert m3.shape == (120, 35)
    assert m4.shape == (35, len(MACRO_PATHWAYS_6))
    assert len(edge_table) == len(mock_proteins)

    # 1. Разреженность первого слоя M1 должна быть > 95% (защита от переобучения при P >> N)
    sparsity_m1 = 1.0 - float(np.mean(m1))
    assert sparsity_m1 > 0.95

    # 2. Каждый белок обязан иметь хотя бы один сквозной путь от входа до 6 макро-путей на выходе
    path_reachability = (
        m1.astype(np.int64)
        @ m2.astype(np.int64)
        @ m3.astype(np.int64)
        @ m4.astype(np.int64)
    )
    assert np.all(np.sum(path_reachability, axis=1) > 0)


# ==============================================================================
# 3. ИНТЕГРАЦИОННЫЙ ТЕСТ АРТЕФАКТОВ БЛОКОВ Б, В И Г НА РЕАЛЬНЫХ ДАННЫХ
# ==============================================================================


def test_real_stage3_artifacts_and_bug_fixes():
    """Проверяет на реальных данных устранение Бага #1 и Бага #2 и генерацию файлов Этапа 3."""
    if not (ROOT / "metadata.parquet").is_file() or not (ROOT / "lab.db").is_file():
        pytest.skip("Реальные файлы данных отсутствуют")

    b_art = run_block_b(ROOT)
    edges_df = b_art.edge_index_df
    lod_df = b_art.imputed_discovery
    disc_wide = pd.read_parquet(ROOT / "discovery_wide.parquet")

    # Проверка исправления Бага #2: все ребра ведут строго в Train
    assert set(edges_df["target_dataset"].unique()) == {"Train"}

    # Проверка исправления Бага #1: discovery_wide (QRILC+kNN) != discovery_imputed_lod (LOD)
    assert not disc_wide.equals(lod_df)

    # Проверка файла кодировок GraphPFN (426 + 35 = 461 строка)
    enc_path = ROOT / "graphpfn_node_encodings.parquet"
    assert enc_path.is_file()
    enc_df = pd.read_parquet(enc_path)
    assert len(enc_df) == 461

    vg_art = run_blocks_v_g(ROOT)
    assert not vg_art.priors_df.empty
    assert (ROOT / "protein_multiplex_graph.npz").is_file()
    assert (ROOT / "binn_hierarchy_masks.npz").is_file()
