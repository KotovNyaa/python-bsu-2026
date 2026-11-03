"""Формирование когортных датасетов, побелковая калибровка MS+ELISA и биофизический инжиниринг."""

from __future__ import annotations

import contextlib
import sqlite3
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import HuberRegressor, RidgeCV

from lab.data import locate
from lab.errors import LoadError
from lab.prep import (
    CORE_MARKERS,
    FIG5_NAME,
    encode_categorical_features,
    load_elisa_paired,
    load_markers,
    prepare_maxlfq_matrix,
)
from lab.priors import compute_ms_evidence_vectors

ELISA_VALIDATED_MARKERS = ["AMBP", "FN1", "LUM", "TIMP1"]

# Маппинг когорт на клинические центры согласно дизайну исследования (Zhao et al.)
DATASET_TO_CENTER = {
    "Train": "MCH1_Shenzhen_Longgang",
    "Internal validation": "MCH1_Shenzhen_Longgang",
    "External validation1": "MCH2_Zhuhai_Dongguan",
    "External validation2": "MCH3_Shenzhen",
}

DATASET_TO_CODE = {
    "Train": "SZ_LG_TR",
    "Internal validation": "SZ_LG_IV",
    "External validation1": "ZH_DG_EV1",
    "External validation2": "SZ_MCH3_EV2",
}

DATASET_TO_FOLD = {
    "Train": 0,
    "Internal validation": 1,
    "External validation1": 2,
    "External validation2": 3,
}


class CohortArtifacts(NamedTuple):
    """Контейнер сформированных датасетов Блока А."""

    metadata: pd.DataFrame
    features_5: pd.DataFrame
    discovery_wide: pd.DataFrame
    instrumental_evidence: pd.DataFrame


def build_metadata(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Формирует очищенную таблицу метаданных и меток пациенток (5 канонических колонок)."""
    meta = pd.DataFrame(index=df_raw["Sample ID"].astype(str))
    meta["Dataset"] = df_raw["Dataset"].astype(str).values
    meta["Category"] = df_raw["Category"].astype(str).values
    meta["Target"] = (df_raw["Category"] == "PTB").astype(int).values
    meta["Hospital_Center"] = meta["Dataset"].map(DATASET_TO_CENTER).fillna("Unknown")
    meta["Center_Code"] = meta["Dataset"].map(DATASET_TO_CODE).fillna("UNK")
    return meta


def extract_features_5(
    df_raw: pd.DataFrame,
    log_scale: bool = False,
) -> pd.DataFrame:
    """Извлекает матрицу 5 ядерных биомаркеров статьи."""
    raw_cols = [f"{m}_raw" for m in CORE_MARKERS]
    if not log_scale and set(raw_cols).issubset(df_raw.columns):
        feats = df_raw[raw_cols].copy().astype(np.float64)
        feats.columns = CORE_MARKERS
    else:
        feats = df_raw[CORE_MARKERS].copy().astype(np.float64)

    feats.index = df_raw["Sample ID"].astype(str)
    if feats.isna().any().any():
        feats = feats.fillna(feats.median())

    if log_scale and bool((feats > 0).all().all()) and float(feats.max().max()) > 2.0:
        feats = np.log2(feats.clip(lower=1e-9))
    return feats


def compute_biophysical_ratios_and_simplex(
    features_log2: pd.DataFrame,
) -> pd.DataFrame:
    """Вычисляет 10 попарных лог-отношений, 3 композитных индекса, симплекс долей и CLR."""
    out = pd.DataFrame(index=features_log2.index)

    # 1. Все 10 попарных разностей логарифмов (инвариантных к разведению мазка alpha_i)
    ordered_pairs = [
        ("LUM", "TIMP1"),
        ("FN1", "TIMP1"),
        ("LUM", "AMBP"),
        ("FN1", "AMBP"),
        ("TIMP1", "AMBP"),
        ("B2M", "LUM"),
        ("B2M", "FN1"),
        ("LUM", "FN1"),
        ("B2M", "TIMP1"),
        ("B2M", "AMBP"),
    ]
    for prot_a, prot_b in ordered_pairs:
        if prot_a in features_log2.columns and prot_b in features_log2.columns:
            out[f"ratio_{prot_a}_{prot_b}"] = (
                features_log2[prot_a] - features_log2[prot_b]
            )

    # Если переданы нестандартные имена столбцов, дополняем комбинациями
    if out.empty:
        for c1, c2 in combinations(features_log2.columns, 2):
            out[f"ratio_{c1}_{c2}"] = features_log2[c1] - features_log2[c2]

    # 2. Три главных патофизиологических композитных индекса (answer1.md, п. 1)
    if set(CORE_MARKERS).issubset(features_log2.columns):
        ecm_damage = 0.5 * (features_log2["LUM"] + features_log2["FN1"])
        ecm_defense = 0.5 * (features_log2["TIMP1"] + features_log2["AMBP"])
        out["idx_ECM_Stress"] = ecm_damage - ecm_defense
        out["idx_Subtype_Polarization_SPI"] = features_log2["B2M"] - ecm_damage
        out["idx_Immuno_Proteolytic"] = features_log2["B2M"] - ecm_defense

    # 3. Композиционный профиль на симплексе P_{i,p} и центрированные лог-отношения (CLR)
    linear_signals = np.exp2(features_log2.to_numpy(dtype=np.float64))
    simplex_denom = np.sum(linear_signals, axis=1, keepdims=True)
    simplex_denom = np.where(simplex_denom <= 0, 1.0, simplex_denom)
    simplex_probs = linear_signals / simplex_denom

    row_mean_log2 = np.mean(
        features_log2.to_numpy(dtype=np.float64), axis=1, keepdims=True
    )
    clr_vals = features_log2.to_numpy(dtype=np.float64) - row_mean_log2

    for idx, col in enumerate(features_log2.columns):
        out[f"simplex_{col}"] = simplex_probs[:, idx]
        out[f"clr_{col}"] = clr_vals[:, idx]

    return out


def compute_nonlinear_patient_aggregates(
    features_log2: pd.DataFrame,
    metadata: pd.DataFrame,
    tau: float = 0.5,
) -> pd.DataFrame:
    """Вычисляет нелинейные внутрипациентские инварианты строго относительно эталона Train FTB (n=55)."""
    out = pd.DataFrame(index=features_log2.index)
    x_mat = features_log2.to_numpy(dtype=np.float64)

    train_ftb_mask = (
        (metadata["Dataset"] == "Train") & (metadata["Target"] == 0)
    ).to_numpy(dtype=bool)
    if not np.any(train_ftb_mask):
        train_ftb_mask = (metadata["Target"] == 0).to_numpy(dtype=bool)
    if not np.any(train_ftb_mask):
        train_ftb_mask = np.ones(len(features_log2), dtype=bool)

    x_ref = x_mat[train_ftb_mask]
    mu_ftb = np.mean(x_ref, axis=0)
    sd_ftb = np.std(x_ref, axis=0, ddof=1)
    sd_ftb = np.where(sd_ftb < 1e-6, 1.0, sd_ftb)

    # Стандартизованный относительно нормы Train FTB профиль u_{i,p}
    u_mat = (x_mat - mu_ftb[None, :]) / sd_ftb[None, :]

    # 1. Дивергенция Йенсена-Шеннона (Single-Sample JSD) и Кульбака-Лейблера до центроида Q_FTB
    lin_mat = np.exp2(x_mat)
    p_mat = lin_mat / np.maximum(np.sum(lin_mat, axis=1, keepdims=True), 1e-12)
    p_mat = np.clip(p_mat, 1e-12, 1.0)
    p_mat = p_mat / np.sum(p_mat, axis=1, keepdims=True)

    q_ftb = np.mean(p_mat[train_ftb_mask], axis=0)
    q_ftb = np.clip(q_ftb, 1e-12, 1.0)
    q_ftb = q_ftb / np.sum(q_ftb)

    m_mat = 0.5 * (p_mat + q_ftb[None, :])
    kl_p_m = np.sum(p_mat * np.log2(p_mat / m_mat), axis=1)
    kl_q_m = np.sum(q_ftb[None, :] * np.log2(q_ftb[None, :] / m_mat), axis=1)
    jsd_vals = np.clip(0.5 * (kl_p_m + kl_q_m), 0.0, 1.0)

    out["nonlin_JSD_to_FTB"] = jsd_vals
    out["nonlin_JSD_Dist_to_FTB"] = np.sqrt(jsd_vals)
    out["nonlin_KL_Div_to_FTB"] = np.sum(
        p_mat * np.log2(p_mat / q_ftb[None, :]), axis=1
    )

    # 2. Энтропия Шеннона, перплексия и индекс одиночного триггера
    shannon_h = -np.sum(p_mat * np.log2(p_mat), axis=1)
    out["nonlin_Shannon_Entropy"] = shannon_h
    out["nonlin_Perplexity"] = np.exp2(shannon_h)
    out["nonlin_Single_Trigger_Peak"] = np.max(u_mat, axis=1) - np.median(u_mat, axis=1)

    # 3. Расстояние Махаланобиса (Ledoit-Wolf) и квадратичная форма потери связности (Decoupling Score)
    lw = LedoitWolf().fit(x_ref)
    precision_mat = lw.precision_
    diag_prec = np.diag(np.diag(precision_mat))
    off_diag_prec = precision_mat - diag_prec

    diff_ref = x_mat - mu_ftb[None, :]
    mahal_sq = np.einsum("ij,jk,ik->i", diff_ref, precision_mat, diff_ref)
    decoupling = np.einsum("ij,jk,ik->i", diff_ref, off_diag_prec, diff_ref)

    out["nonlin_Mahalanobis_Sq"] = np.maximum(mahal_sq, 0.0)
    out["nonlin_Mahalanobis_Dist"] = np.sqrt(np.maximum(mahal_sq, 0.0))
    out["nonlin_Decoupling_Score"] = decoupling

    # 4. Суммарная протеомная нагрузка, гладкий максимум LogSumExp и внутрипациентская дисперсия
    out["nonlin_Total_Burden"] = np.mean(u_mat, axis=1)
    max_u = np.max(u_mat / tau, axis=1, keepdims=True)
    lse_vals = tau * (
        np.squeeze(max_u, axis=1)
        + np.log(np.sum(np.exp((u_mat / tau) - max_u), axis=1))
    )
    out["nonlin_LSE_Max_tau05"] = lse_vals
    out["nonlin_Profile_Variance"] = np.var(u_mat, axis=1, ddof=1)

    return out


def calibrate_ms_elisa_cross_platform(
    features_log2: pd.DataFrame,
    metadata: pd.DataFrame,
    df_elisa_paired: pd.DataFrame,
) -> pd.DataFrame:
    """Выполняет побелковую калибровку и денойзинг MS -> ELISA строго по Train без глобального dropna.

    Учитывает Ограничение #2: в листе 5d у 50 пациенток (включая 4 в Train) измерены не все 4 белка,
    поэтому регрессии обучаются независимо для каждого белка p ∈ {AMBP, FN1, LUM, TIMP1}.
    """
    out = pd.DataFrame(index=features_log2.index)

    # Расширенная матрица предикторов MS (5 маркеров + 10 попарных лог-отношений) для многомерного денойзинга
    ratios_df = compute_biophysical_ratios_and_simplex(features_log2)
    ratio_cols = [c for c in ratios_df.columns if c.startswith("ratio_")]
    x_full_ms = pd.concat([features_log2[CORE_MARKERS], ratios_df[ratio_cols]], axis=1)

    train_ids = set(metadata.index[metadata["Dataset"] == "Train"].astype(str))
    df_5d_train = df_elisa_paired[
        df_elisa_paired["Sample ID"].astype(str).isin(train_ids)
    ].copy()

    sigma_sq_cols: list[np.ndarray] = []
    shift_matrix_train = pd.DataFrame(
        np.nan,
        index=sorted(train_ids),
        columns=ELISA_VALIDATED_MARKERS,
        dtype=np.float64,
    )

    for prot in ELISA_VALIDATED_MARKERS:
        x_prot_all = features_log2[prot].to_numpy(dtype=np.float64)
        sub_p = df_5d_train[df_5d_train["Protein"] == prot].dropna(
            subset=["log2(MS signal)", "log2(ELISA)"]
        )
        # Убираем дубликаты Sample ID внутри одного белка, если есть
        sub_p = sub_p.drop_duplicates(subset=["Sample ID"], keep="first")
        valid_sample_ids = [
            sid for sid in sub_p["Sample ID"].astype(str) if sid in x_full_ms.index
        ]

        if len(valid_sample_ids) >= 8:
            sub_p = sub_p.set_index("Sample ID").loc[valid_sample_ids]
            x_tr_uni = sub_p["log2(MS signal)"].to_numpy(dtype=np.float64)
            z_tr_elisa = sub_p["log2(ELISA)"].to_numpy(dtype=np.float64)

            # 1. Робастная одномерная калибровочная кривая Huber: log2(ELISA) ~ log2(MS)
            huber = HuberRegressor(max_iter=500).fit(
                x_tr_uni.reshape(-1, 1), z_tr_elisa
            )
            z_hat_uni_tr = huber.predict(x_tr_uni.reshape(-1, 1))
            residuals_sq = (z_tr_elisa - z_hat_uni_tr) ** 2

            # 2. Полиномиальная оценка функции гетероскедастической дисперсии ошибки sigma_p^2(x)
            poly_tr = np.column_stack([x_tr_uni, x_tr_uni**2])
            poly_all = np.column_stack([x_prot_all, x_prot_all**2])
            var_reg = RidgeCV(alphas=np.logspace(-2, 3, 15)).fit(
                poly_tr, np.log(residuals_sq + 0.05)
            )
            sigma_sq_all = np.clip(np.exp(var_reg.predict(poly_all)), 0.05, 10.0)

            # 3. Многомерная Ridge-регрессия денойзинга MS -> ELISA со сжатием к монотонной кривой Хубера
            z_hat_uni_all = huber.predict(x_prot_all.reshape(-1, 1))
            x_tr_multi = x_full_ms.loc[valid_sample_ids].to_numpy(dtype=np.float64)
            denoiser = RidgeCV(alphas=np.logspace(0, 4, 20)).fit(
                x_tr_multi, z_tr_elisa - z_hat_uni_tr
            )
            z_multi_corr = denoiser.predict(x_full_ms.to_numpy(dtype=np.float64))
            z_denoised_all = z_hat_uni_all + 0.25 * z_multi_corr

            # 4. Сохранение стандартизованных невязок для оценки индивидуального сдвига пробоподготовки delta_i
            x_std_tr = (x_tr_uni - np.mean(x_tr_uni)) / max(
                float(np.std(x_tr_uni)), 1e-6
            )
            z_std_tr = (z_tr_elisa - np.mean(z_tr_elisa)) / max(
                float(np.std(z_tr_elisa)), 1e-6
            )
            shift_matrix_train.loc[valid_sample_ids, prot] = x_std_tr - z_std_tr
        else:
            sigma_sq_all = np.ones(len(features_log2), dtype=np.float64)
            z_denoised_all = x_prot_all.copy()

        out[f"elisa_denoised_{prot}"] = z_denoised_all
        out[f"elisa_reliability_{prot}"] = np.round(1.0 / sigma_sq_all, 4)
        sigma_sq_cols.append(sigma_sq_all)

    # 5. Латентная калибровка 5-го маркера B2M через среднюю техническую ошибку образца (без глобального dropna!)
    mean_shift_per_train = shift_matrix_train.mean(axis=1, skipna=True).dropna()
    b2m_train_vals = features_log2.loc[metadata["Dataset"] == "Train", "B2M"].to_numpy(
        dtype=np.float64
    )
    b2m_mu = float(np.mean(b2m_train_vals)) if len(b2m_train_vals) > 0 else 0.0
    b2m_sd = (
        max(float(np.std(b2m_train_vals)), 1e-6) if len(b2m_train_vals) > 1 else 1.0
    )
    b2m_z_all = (features_log2["B2M"].to_numpy(dtype=np.float64) - b2m_mu) / b2m_sd

    if len(mean_shift_per_train) >= 8:
        shift_sids = mean_shift_per_train.index.tolist()
        shift_reg = RidgeCV(alphas=np.logspace(-1, 3, 15)).fit(
            x_full_ms.loc[shift_sids].to_numpy(dtype=np.float64),
            mean_shift_per_train.to_numpy(dtype=np.float64),
        )
        delta_hat_all = shift_reg.predict(x_full_ms.to_numpy(dtype=np.float64))
    else:
        delta_hat_all = np.zeros(len(features_log2), dtype=np.float64)

    out["elisa_tech_shift_delta"] = delta_hat_all
    out["elisa_denoised_B2M"] = b2m_z_all - delta_hat_all

    # 6. Интегральный вес достоверности пациентки W_i (нормированный на среднее 1.0 в Train)
    mean_sigma_sq = np.mean(np.column_stack(sigma_sq_cols), axis=1)
    raw_sample_w = 1.0 / np.maximum(mean_sigma_sq, 0.05)
    train_mask = (metadata["Dataset"] == "Train").to_numpy(dtype=bool)
    norm_factor = (
        float(np.mean(raw_sample_w[train_mask]))
        if np.any(train_mask)
        else float(np.mean(raw_sample_w))
    )
    out["sample_reliability_weight"] = np.round(
        raw_sample_w / max(norm_factor, 1e-6), 4
    )

    return out


def build_enriched_cohort_a(
    excel_path: Path | str | None = None,
) -> pd.DataFrame:
    """Собирает обогащенную таблицу 426 пациенток (базовый слой cohort_a_426_enriched)."""
    target_excel = locate(FIG5_NAME) if excel_path is None else Path(excel_path)
    df_raw = load_markers(target_excel, log_scale=True, align_batches=True)
    df_elisa = load_elisa_paired(target_excel)

    meta = build_metadata(df_raw)
    meta_enc = encode_categorical_features(meta.copy())
    meta_enc["Fold_ID"] = (
        meta_enc["Dataset"].map(DATASET_TO_FOLD).fillna(-1).astype(int)
    )

    f5_log2 = extract_features_5(df_raw, log_scale=True)
    ratios_df = compute_biophysical_ratios_and_simplex(f5_log2)
    nonlin_df = compute_nonlinear_patient_aggregates(f5_log2, meta_enc)
    elisa_df = calibrate_ms_elisa_cross_platform(f5_log2, meta_enc, df_elisa)

    enriched = pd.concat(
        [
            meta_enc[
                [
                    "Dataset",
                    "Category",
                    "Target",
                    "Fold_ID",
                    "Hospital_Center",
                    "Center_Code",
                ]
            ],
            f5_log2,
            elisa_df,
            ratios_df,
            nonlin_df,
        ],
        axis=1,
    )
    return enriched


def extract_instrumental_evidence(conn: sqlite3.Connection) -> pd.DataFrame:
    """Извлекает инструментальные метрики и 6-мерный вектор достоверности MS/MS по всем генам."""
    return compute_ms_evidence_vectors(conn, genes=None, min_transitions=2)


def extract_discovery_wide(conn: sqlite3.Connection) -> pd.DataFrame:
    """Извлекает широкую матрицу экспрессии белков Чунцина (35 x 2699) по протоколу MaxLFQ + QRILC.

    Сохраняет все 48 белков с пустым Genes через Genes.fillna(Protein_Group) (Ограничение #3).
    """
    df_prot = pd.read_sql_query("SELECT * FROM proteins;", conn)
    if df_prot.empty:
        return pd.DataFrame()

    _, imputed_hybrid, _ = prepare_maxlfq_matrix(df_prot)
    return imputed_hybrid


def generate_cohort_artifacts(
    excel_path: Path | str | None = None,
    db_path: Path | str | None = None,
) -> CohortArtifacts:
    """Генерирует полный набор эталонных артефактов Блока А."""
    target_excel = locate(FIG5_NAME) if excel_path is None else Path(excel_path)
    if not target_excel.is_file():
        raise LoadError(f"Файл не найден: {target_excel}")

    df_raw = load_markers(target_excel, log_scale=True, align_batches=True)
    meta = build_metadata(df_raw)
    f5 = extract_features_5(df_raw, log_scale=False)

    target_db = locate("lab.db") if db_path is None else Path(db_path)
    with sqlite3.connect(target_db) as conn:
        inst = extract_instrumental_evidence(conn)
        discovery = extract_discovery_wide(conn)

    return CohortArtifacts(
        metadata=meta,
        features_5=f5,
        discovery_wide=discovery,
        instrumental_evidence=inst,
    )


def export_cohort_artifacts(
    artifacts: CohortArtifacts,
    output_dir: Path | str | None = None,
    db_path: Path | str | None = None,
    excel_path: Path | str | None = None,
) -> dict[str, Path]:
    """Экспортирует артефакты Блока А, маску пропусков Чунцина и обогащенную таблицу Когорты А."""
    out_dir = locate(FIG5_NAME).parent if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "metadata_parquet": out_dir / "metadata.parquet",
        "metadata_csv": out_dir / "metadata.csv",
        "features_5_parquet": out_dir / "features_5.parquet",
        "features_5_csv": out_dir / "features_5.csv",
        "discovery_wide_parquet": out_dir / "discovery_wide.parquet",
        "discovery_wide_csv": out_dir / "discovery_wide.csv",
        "instrumental_evidence_parquet": out_dir / "instrumental_evidence.parquet",
        "instrumental_evidence_csv": out_dir / "instrumental_evidence.csv",
        "discovery_unimputed_parquet": out_dir / "discovery_unimputed.parquet",
        "discovery_missing_mask_parquet": out_dir / "discovery_missing_mask.parquet",
        "cohort_a_enriched_parquet": out_dir / "cohort_a_426_enriched.parquet",
        "cohort_a_enriched_csv": out_dir / "cohort_a_426_enriched.csv",
    }

    artifacts.metadata.to_parquet(paths["metadata_parquet"])
    artifacts.metadata.to_csv(paths["metadata_csv"])

    artifacts.features_5.to_parquet(paths["features_5_parquet"])
    artifacts.features_5.to_csv(paths["features_5_csv"])

    artifacts.discovery_wide.to_parquet(paths["discovery_wide_parquet"])
    artifacts.discovery_wide.to_csv(paths["discovery_wide_csv"])

    artifacts.instrumental_evidence.to_parquet(paths["instrumental_evidence_parquet"])
    artifacts.instrumental_evidence.to_csv(paths["instrumental_evidence_csv"])

    with contextlib.suppress(OSError, sqlite3.Error, ValueError, KeyError):
        target_db = locate("lab.db") if db_path is None else Path(db_path)
        with sqlite3.connect(target_db) as conn:
            df_prot = pd.read_sql_query("SELECT * FROM proteins;", conn)
        polished_nan, _, mask_df = prepare_maxlfq_matrix(df_prot)
        if not polished_nan.empty:
            polished_nan.to_parquet(paths["discovery_unimputed_parquet"])
            mask_df.to_parquet(paths["discovery_missing_mask_parquet"])

    with contextlib.suppress(OSError, sqlite3.Error, ValueError, KeyError):
        enriched_a = build_enriched_cohort_a(excel_path)
        enriched_a.to_parquet(paths["cohort_a_enriched_parquet"])
        enriched_a.to_csv(paths["cohort_a_enriched_csv"])

    return paths
