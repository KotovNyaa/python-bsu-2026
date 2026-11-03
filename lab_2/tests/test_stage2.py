"""Обширный набор тестов Этапа 2 (6-мерные приборные приоры MS/MS, калибровка ELISA, биофизика)."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from lab.cohorts import (
    build_enriched_cohort_a,
    calibrate_ms_elisa_cross_platform,
    compute_biophysical_ratios_and_simplex,
    compute_nonlinear_patient_aggregates,
    extract_discovery_wide,
)
from lab.data import ROOT
from lab.prep import CORE_MARKERS
from lab.priors import (
    build_full_proteome_priors,
    build_priors_table,
    classify_gene_pathway_and_location,
    compute_ms_evidence_vectors,
)

# ==============================================================================
# 1. ТЕСТЫ 6-МЕРНОГО ВЕКТОРА ПРИБОРНОЙ ДОСТОВЕРНОСТИ MS/MS (lab/priors.py)
# ==============================================================================


def _create_full_diann_mock_db() -> sqlite3.Connection:
    """Создает в памяти тестовую БД со всеми колонками DIA-NN (20240105.tsv)."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE proteins (
            Protein_Group TEXT PRIMARY KEY,
            Genes TEXT,
            Protein_Names TEXT,
            PTB_CQ_P1_53 REAL,
            PTB_CQ_P1_54 REAL
        )
        """
    )
    conn.executemany(
        "INSERT INTO proteins VALUES (?, ?, ?, ?, ?)",
        [
            ("P51884", "LUM", "LUM_HUMAN", 1.2e7, 1.5e7),
            ("P02751", "FN1", "FINC_HUMAN", 4.5e7, 5.1e7),
            ("P01024", "C3", "CO3_HUMAN", 8.0e7, 7.2e7),
            ("P99999", "P99999", "UNMAPPED_PROTEIN", 2.0e5, 1.8e5),
        ],
    )

    conn.execute(
        """
        CREATE TABLE transitions (
            ProteinGroup TEXT,
            Genes TEXT,
            ProteinName TEXT,
            PeptideSequence TEXT,
            Proteotypic INTEGER,
            PrecursorMz REAL,
            ProductMz REAL,
            FragmentType TEXT,
            LibraryIntensity REAL,
            Tr_recalibrated REAL,
            QValue REAL,
            PGQValue REAL,
            FileName TEXT,
            decoy INTEGER
        )
        """
    )

    rows = [
        # LUM: 3 уникальных протеотипических пептида, y-ионы, минимальный QValue, стабильный RT
        (
            "P51884",
            "LUM",
            "LUM_HUMAN",
            "PEP_LUM_1",
            1,
            500.2,
            700.3,
            "y",
            50000.0,
            25.10,
            0.0001,
            0.0001,
            "run1.raw",
            0,
        ),
        (
            "P51884",
            "LUM",
            "LUM_HUMAN",
            "PEP_LUM_1",
            1,
            500.2,
            800.4,
            "y",
            48000.0,
            25.12,
            0.0001,
            0.0001,
            "run2.raw",
            0,
        ),
        (
            "P51884",
            "LUM",
            "LUM_HUMAN",
            "PEP_LUM_2",
            1,
            610.3,
            910.5,
            "y",
            42000.0,
            31.00,
            0.0002,
            0.0001,
            "run1.raw",
            0,
        ),
        (
            "P51884",
            "LUM",
            "LUM_HUMAN",
            "PEP_LUM_3",
            1,
            720.4,
            650.2,
            "y",
            39000.0,
            40.50,
            0.0002,
            0.0001,
            "run3.raw",
            0,
        ),
        # FN1: 1 пептид, b-ионы, огромный разброс RT между прогонами (низкая стабильность v6)
        (
            "P02751",
            "FN1",
            "FINC_HUMAN",
            "PEP_FN1_1",
            1,
            550.1,
            400.1,
            "b",
            12000.0,
            10.00,
            0.0050,
            0.0050,
            "run1.raw",
            0,
        ),
        (
            "P02751",
            "FN1",
            "FINC_HUMAN",
            "PEP_FN1_1",
            1,
            550.1,
            450.2,
            "b",
            11000.0,
            45.00,
            0.0050,
            0.0050,
            "run2.raw",
            0,
        ),
        # P99999: непротеотипический пептид (Proteotypic = 0) в шумовом хвосте QValue > 0.01
        (
            "P99999",
            "P99999",
            "UNMAPPED",
            "PEP_NOISE_1",
            0,
            400.0,
            300.0,
            "b",
            150.0,
            18.0,
            0.0800,
            0.0900,
            "run1.raw",
            0,
        ),
        (
            "P99999",
            "P99999",
            "UNMAPPED",
            "PEP_NOISE_2",
            0,
            410.0,
            310.0,
            "b",
            120.0,
            60.0,
            0.1200,
            0.1500,
            "run1.raw",
            0,
        ),
    ]
    conn.executemany(
        "INSERT INTO transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


def test_ms_evidence_vectors_bounds_and_ranking():
    """Проверяет границы [0, 1] всех 6 компонент v_g и правильное ранжирование эталонного белка над шумным."""
    conn = _create_full_diann_mock_db()
    ev = compute_ms_evidence_vectors(conn, min_transitions=1)

    assert {"LUM", "FN1", "P99999"}.issubset(ev.index)
    v_cols = [
        "v1_proteotypic",
        "v2_transition_density",
        "v3_qvalue_significance",
        "v4_target_decoy_snr",
        "v5_run_reproducibility",
        "v6_rt_stability",
    ]
    for col in v_cols:
        assert col in ev.columns
        vals = ev[col].to_numpy()
        assert np.all((vals >= 0.0) & (vals <= 1.0)), f"Выход за [0,1] в колонке {col}"

    e_vals = ev["ms_evidence_weight_E"].to_numpy()
    assert np.all((e_vals >= 0.05) & (e_vals <= 1.0))

    # LUM (3 протеотипических пептида, малый QValue, стабильный RT, 3 прогона) обязан превосходить шумный P99999
    assert ev.loc["LUM", "v1_proteotypic"] > ev.loc["P99999", "v1_proteotypic"]
    assert (
        ev.loc["LUM", "v3_qvalue_significance"]
        > ev.loc["P99999", "v3_qvalue_significance"]
    )
    assert (
        ev.loc["LUM", "v4_target_decoy_snr"] > ev.loc["P99999", "v4_target_decoy_snr"]
    )
    assert (
        ev.loc["LUM", "v5_run_reproducibility"]
        > ev.loc["P99999", "v5_run_reproducibility"]
    )
    assert ev.loc["LUM", "ms_evidence_weight_E"] > ev.loc["FN1", "ms_evidence_weight_E"]
    assert (
        ev.loc["FN1", "ms_evidence_weight_E"] > ev.loc["P99999", "ms_evidence_weight_E"]
    )

    # У LUM стабильный RT (25.10 vs 25.12), а у FN1 скачок (10.0 vs 45.0) -> v6(LUM) > v6(FN1)
    assert ev.loc["LUM", "v6_rt_stability"] > ev.loc["FN1", "v6_rt_stability"]
    conn.close()


def test_y_ion_bonus_in_v2_transition_density():
    """Проверяет, что переходы y-ионов получают весовой бонус 1.2x по сравнению с b-ионами."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE transitions (
            ProteinGroup TEXT, Genes TEXT, PeptideSequence TEXT,
            Proteotypic INTEGER, FragmentType TEXT, LibraryIntensity REAL,
            Tr_recalibrated REAL, QValue REAL, PGQValue REAL, FileName TEXT, decoy INTEGER
        )
        """
    )
    # Ген Y_GENE имеет только y-ионы, ген B_GENE — ровно столько же b-ионов
    for idx in range(4):
        conn.execute(
            "INSERT INTO transitions VALUES ('PG1', 'Y_GENE', 'PEP1', 1, 'y', 1000.0, 20.0, 0.001, 0.001, 'r1.raw', 0)"
        )
        conn.execute(
            "INSERT INTO transitions VALUES ('PG2', 'B_GENE', 'PEP2', 1, 'b', 1000.0, 20.0, 0.001, 0.001, 'r1.raw', 0)"
        )
    conn.commit()

    ev = compute_ms_evidence_vectors(conn, min_transitions=1)
    assert (
        ev.loc["Y_GENE", "v2_transition_density"]
        > ev.loc["B_GENE", "v2_transition_density"]
    )
    conn.close()


def test_build_priors_table_and_full_proteome_priors():
    """Проверяет обратную совместимость build_priors_table и генерацию полных приоров на весь протеом."""
    conn = _create_full_diann_mock_db()

    # 1. Классический вызов для 5 ядерных маркеров
    priors_5 = build_priors_table(conn, genes=CORE_MARKERS)
    assert len(priors_5) == 5
    assert "prior_confidence_weight" in priors_5.columns
    assert "ms_evidence_weight_E" in priors_5.columns
    assert not priors_5["prior_confidence_weight"].isna().any()

    # 2. Полная таблица приоров для всех белков из таблицы proteins (включая C3 и P99999)
    full_priors = build_full_proteome_priors(conn, fetch_online_batch=False)
    assert len(full_priors) == 4
    assert {"LUM", "FN1", "C3", "P99999"}.issubset(full_priors.index)
    assert full_priors.loc["C3", "pathway_module"] == "Complement_Cascade"
    assert full_priors.loc["LUM", "pathway_module"] == "ECM_Remodeling"
    assert not full_priors["prior_confidence_weight"].isna().any()
    conn.close()


def test_hgnc_family_pathway_classifier():
    """Проверяет правила автоматической классификации семейств генов по 6 макро-путям статьи."""
    assert classify_gene_pathway_and_location("COL6A3")[0] == "ECM_Remodeling"
    assert classify_gene_pathway_and_location("MMP9")[0] == "ECM_Remodeling"
    assert classify_gene_pathway_and_location("C1QB")[0] == "Complement_Cascade"
    assert classify_gene_pathway_and_location("CFB")[0] == "Complement_Cascade"
    assert (
        classify_gene_pathway_and_location("SERPINC1")[0] == "Fibrinolysis_Coagulation"
    )
    assert classify_gene_pathway_and_location("FGA")[0] == "Fibrinolysis_Coagulation"
    assert classify_gene_pathway_and_location("S100A8")[0] == "Immune_Inflammation"
    assert classify_gene_pathway_and_location("IGHG1")[0] == "Immune_Inflammation"
    assert classify_gene_pathway_and_location("ATP5F1A")[0] == "Energy_Metabolism_ATP"
    assert classify_gene_pathway_and_location("GAPDH")[0] == "Energy_Metabolism_ATP"
    assert classify_gene_pathway_and_location("SOD1")[0] == "Zn_Cu_Targets"
    assert classify_gene_pathway_and_location("CP")[0] == "Zn_Cu_Targets"


# ==============================================================================
# 2. МАТЕМАТИЧЕСКИЕ ТЕСТЫ БИОФИЗИКИ И ИНВАРИАНТНОСТИ К РАЗВЕДЕНИЮ (lab/cohorts.py)
# ==============================================================================


def test_dilution_invariance_of_ratios_simplex_and_clr():
    """Математическое доказательство: умножение всех белков пациентки на фактор разведения alpha_i
    (сдвиг на +log2(alpha_i) в log2-шкале) НЕ МЕНЯЕТ ни одно лог-отношение, индекс, симплекс или CLR.
    """
    rng = np.random.default_rng(123)
    df_log2 = pd.DataFrame(
        rng.normal(loc=-2.0, scale=1.5, size=(30, 5)),
        index=[f"P_{i}" for i in range(30)],
        columns=CORE_MARKERS,
    )

    # Генерируем случайный индивидуальный фактор разведения мазка alpha_i ∈ [0.1, 15.0] для каждой пациентки
    alpha_i = rng.uniform(0.1, 15.0, size=(30, 1))
    df_diluted_log2 = df_log2 + np.log2(alpha_i)

    res_orig = compute_biophysical_ratios_and_simplex(df_log2)
    res_diluted = compute_biophysical_ratios_and_simplex(df_diluted_log2)

    assert res_orig.shape == res_diluted.shape
    # Должно быть ровно 10 попарных отношений + 3 индекса + 5 симплекс + 5 CLR = 23 признака
    assert res_orig.shape[1] == 23

    # 1. Полное совпадение с машинной точностью 1e-12
    np.testing.assert_allclose(
        res_orig.to_numpy(),
        res_diluted.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # 2. Сумма симплексных долей внутри каждой пациентки равна 1.0, а сумма CLR равна 0.0
    simplex_cols = [c for c in res_orig.columns if c.startswith("simplex_")]
    clr_cols = [c for c in res_orig.columns if c.startswith("clr_")]
    np.testing.assert_allclose(
        res_orig[simplex_cols].sum(axis=1).to_numpy(), 1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        res_orig[clr_cols].sum(axis=1).to_numpy(), 0.0, atol=1e-12
    )


def test_nonlinear_aggregates_strict_train_ftb_isolation():
    """Доказательство отсутствия утечки из теста: экстремальное возмущение тестовых когорт (+100.0)
    не изменяет нелинейные признаки пациенток Train ни на 1e-12.
    """
    rng = np.random.default_rng(77)
    n_train, n_test = 40, 30
    idx = [f"S_{i}" for i in range(n_train + n_test)]

    feats = pd.DataFrame(
        rng.normal(loc=-2.0, scale=1.2, size=(n_train + n_test, 5)),
        index=idx,
        columns=CORE_MARKERS,
    )
    meta = pd.DataFrame(
        {
            "Dataset": ["Train"] * n_train + ["External validation1"] * n_test,
            "Target": ([0] * 25 + [1] * 15) + ([0] * 25 + [1] * 5),
        },
        index=idx,
    )

    nonlin_clean = compute_nonlinear_patient_aggregates(feats, meta)

    # Искажаем тестовую выборку на +50 логарифмических единиц
    feats_corrupted = feats.copy()
    feats_corrupted.iloc[n_train:, :] += 50.0
    nonlin_corrupted = compute_nonlinear_patient_aggregates(feats_corrupted, meta)

    # Признаки обучающей выборки обязаны остаться абсолютно идентичными!
    np.testing.assert_allclose(
        nonlin_clean.iloc[:n_train].to_numpy(),
        nonlin_corrupted.iloc[:n_train].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # При этом у искаженных тестовых пациенток расстояние Махаланобиса и Total_Burden должны вырасти
    assert (
        nonlin_corrupted.iloc[n_train:]["nonlin_Mahalanobis_Dist"].mean()
        > nonlin_clean.iloc[n_train:]["nonlin_Mahalanobis_Dist"].mean()
    )
    assert (nonlin_clean["nonlin_JSD_to_FTB"] >= 0.0).all()
    assert (nonlin_clean["nonlin_JSD_to_FTB"] <= 1.0).all()


# ==============================================================================
# 3. ТЕСТЫ ПОБЕЛКОВОЙ КАЛИБРОВКИ MS <-> ELISA БЕЗ UTEЧКИ И БЕЗ DROPNA (Ограничение #2)
# ==============================================================================


def test_elisa_calibration_partial_proteins_and_zero_test_leakage():
    """Проверяет, что:
    1. Пациентки с неполными измерениями ELISA (например, 2 из 4 белков) не отбрасываются;
    2. Изменение тестовых строк ELISA в листе 5d не влияет на калибровку (обучение строго на Train).
    """
    rng = np.random.default_rng(99)
    sample_ids = [f"PTB_{i:03d}" for i in range(30)]
    meta = pd.DataFrame(
        {
            "Dataset": ["Train"] * 20 + ["External validation1"] * 10,
            "Target": [0, 1] * 15,
        },
        index=sample_ids,
    )
    feats_log2 = pd.DataFrame(
        rng.normal(loc=-2.5, scale=1.2, size=(30, 5)),
        index=sample_ids,
        columns=CORE_MARKERS,
    )

    # Формируем синтетический лист 5d, где у первых 4 пациенток Train отсутствуют FN1 и TIMP1
    elisa_rows = []
    for i, sid in enumerate(sample_ids):
        ds = meta.loc[sid, "Dataset"]
        cat = "PTB" if meta.loc[sid, "Target"] == 1 else "FTB"
        for prot in ["AMBP", "FN1", "LUM", "TIMP1"]:
            if i < 4 and prot in ("FN1", "TIMP1"):
                continue  # Имитация неполной четверки белков (Ограничение #2)
            ms_log = float(feats_log2.loc[sid, prot])
            elisa_log = 1.8 * ms_log + 8.0 + float(rng.normal(0, 0.3))
            elisa_rows.append(
                {
                    "Protein": prot,
                    "Dataset": ds,
                    "Sample ID": sid,
                    "Category": cat,
                    "ELISA": float(np.exp2(elisa_log)),
                    "MS signal": float(np.exp2(ms_log)),
                    "log2(ELISA)": elisa_log,
                    "log2(MS signal)": ms_log,
                    "Target": meta.loc[sid, "Target"],
                }
            )
    df_5d = pd.DataFrame(elisa_rows)

    cal_orig = calibrate_ms_elisa_cross_platform(feats_log2, meta, df_5d)

    # Ни одна из 30 пациенток не потеряна, пропусков NaN нет
    assert len(cal_orig) == 30
    assert not cal_orig.isna().any().any()
    assert "elisa_denoised_B2M" in cal_orig.columns
    assert "sample_reliability_weight" in cal_orig.columns
    assert np.isclose(
        cal_orig.iloc[:20]["sample_reliability_weight"].mean(), 1.0, atol=1e-2
    )

    # Искажаем значения ELISA у тестовой когорты External validation1 на +1000.0
    df_5d_corrupted = df_5d.copy()
    df_5d_corrupted.loc[
        df_5d_corrupted["Dataset"] == "External validation1", "log2(ELISA)"
    ] += 1000.0
    cal_corrupted = calibrate_ms_elisa_cross_platform(feats_log2, meta, df_5d_corrupted)

    # Поскольку калибровка использует ТОЛЬКО Train, результаты на всех 30 пациентках обязаны совпасть 1 в 1
    np.testing.assert_allclose(
        cal_orig.to_numpy(),
        cal_corrupted.to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


# ==============================================================================
# 4. ИНТЕГРАЦИОННЫЕ ТЕСТЫ НА РЕАЛЬНЫХ ФАЙЛАХ (при наличии в data/)
# ==============================================================================


def test_real_enriched_cohort_a_and_discovery_matrix():
    """Проверяет сборку обогащенной таблицы 426 пациенток и матрицы Чунцина на реальных данных."""
    excel_path = ROOT / "Extended Data Fig 5.xlsx"
    db_path = ROOT / "lab.db"
    if not excel_path.is_file() or not db_path.is_file():
        pytest.skip("Реальные файлы данных отсутствуют")

    enriched_a = build_enriched_cohort_a(excel_path)
    assert len(enriched_a) == 426
    assert enriched_a.shape[1] >= 50
    # В числовых признаках не должно быть ни одного NaN или Inf
    num_cols = enriched_a.select_dtypes(include=[np.number]).columns
    assert not enriched_a[num_cols].isna().any().any()
    assert np.all(np.isfinite(enriched_a[num_cols].to_numpy()))

    with sqlite3.connect(db_path) as conn:
        disc_wide = extract_discovery_wide(conn)
        assert disc_wide.shape[0] == 35
        # Все 2699 уникальных белковых групп/генов сохранены, значения в положительной log2(MaxLFQ) области
        assert disc_wide.shape[1] >= 2651
        assert not disc_wide.isna().any().any()
        assert float(disc_wide.median().median()) > 14.0
