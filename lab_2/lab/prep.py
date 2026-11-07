"""Предобработка, нормализация MaxLFQ, гибридная импьютация и межбольничная гармонизация."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from lab.data import locate
from lab.errors import LoadError

FIG5_NAME = "Extended Data Fig 5.xlsx"
SHEET_5A = "Extended Data Fig 5a"
SHEET_5B = "Extended Data Fig 5b"
SHEET_5C = "Extended Data Fig 5c"
SHEET_5D = "Extended Data Fig 5d"

CORE_MARKERS = ["AMBP", "FN1", "LUM", "TIMP1", "B2M"]
DATASET_ORDER = [
    "Train",
    "Internal validation",
    "External validation1",
    "External validation2",
]


def normalize_tic(df: pd.DataFrame) -> pd.DataFrame:
    """Нормализует интенсивности по общему току ионов (сохранено для совместимости)."""
    denom = df.sum(axis=0).replace(0, np.nan)
    return df.div(denom, axis=1)


def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Применяет логарифмирование log2 к строго положительным значениям."""
    clean_df = df.where(df > 0, np.nan)
    return np.log2(clean_df)


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски медианой по строкам (базовый метод)."""
    return df.T.fillna(df.T.median()).T.fillna(0.0)


def extract_missingness_mask(df: pd.DataFrame) -> pd.DataFrame:
    """Формирует бинарную маску наблюдений M (1 — сигнал детектирован, 0 — пропуск)."""
    return (df.notna() & (df > 0)).astype(np.int8)


def polish_anchor_medians(
    df_log2: pd.DataFrame,
    max_missing_rate: float = 0.05,
) -> pd.DataFrame:
    """Выравнивает медианы образцов по якорному инвариантному ядру белков (Anchor Median Polish)."""
    if df_log2.empty:
        return df_log2.copy()

    missing_rate = df_log2.isna().mean(axis=1)
    anchor_rows = df_log2.index[missing_rate <= max_missing_rate]
    if len(anchor_rows) < 5:
        anchor_rows = df_log2.index

    sample_medians = df_log2.loc[anchor_rows].median(axis=0)
    global_median = float(sample_medians.median())
    if np.isnan(global_median):
        return df_log2.copy()

    shifts = (sample_medians - global_median).fillna(0.0)
    return df_log2.sub(shifts, axis=1)


def impute_hybrid_mar_mnar(
    df_log2_norm: pd.DataFrame,
    mar_mean_quantile: float = 0.30,
    mar_max_missing: float = 0.30,
    knn_k: int = 5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Реализует двухэтапную гибридную импьютацию пропусков (MAR -> взвешенный kNN, MNAR -> QRILC)."""
    if df_log2_norm.empty:
        return df_log2_norm.copy()

    work = df_log2_norm.copy().astype(np.float64)
    n_proteins, n_samples = work.shape
    if n_samples == 0 or n_proteins == 0:
        return work

    observed_vals = work.values[~np.isnan(work.values)]
    if len(observed_vals) == 0:
        return work.fillna(0.0)

    global_med = float(np.median(observed_vals))
    lod_const = float(np.quantile(observed_vals, 0.01) - 1.0)

    prot_means = work.mean(axis=1)
    prot_missing_rate = work.isna().mean(axis=1)
    valid_means = prot_means.dropna()
    mean_thresh = (
        float(valid_means.quantile(mar_mean_quantile))
        if not valid_means.empty
        else global_med
    )

    is_mar_protein = (prot_means > mean_thresh) & (prot_missing_rate <= mar_max_missing)

    vals = work.to_numpy(dtype=np.float64, copy=True)
    dist_matrix = np.full((n_samples, n_samples), np.inf, dtype=np.float64)
    for j in range(n_samples):
        for k in range(j + 1, n_samples):
            both_obs = (~np.isnan(vals[:, j])) & (~np.isnan(vals[:, k]))
            if np.any(both_obs):
                diff_jk = vals[both_obs, j] - vals[both_obs, k]
                d_jk = float(np.sqrt(np.mean(diff_jk**2)))
            else:
                d_jk = 10.0
            dist_matrix[j, k] = d_jk
            dist_matrix[k, j] = d_jk

    finite_dists = dist_matrix[np.isfinite(dist_matrix)]
    tau = float(np.median(finite_dists)) if len(finite_dists) > 0 else 1.0
    tau = max(tau, 1e-3)

    mar_indices = np.where(is_mar_protein.values)[0]
    for p_idx in mar_indices:
        row = vals[p_idx, :]
        miss_cols = np.where(np.isnan(row))[0]
        obs_cols = np.where(~np.isnan(row))[0]
        if len(miss_cols) == 0 or len(obs_cols) == 0:
            continue
        for j in miss_cols:
            d_to_obs = dist_matrix[j, obs_cols]
            order = np.argsort(d_to_obs)[:knn_k]
            nearest_cols = obs_cols[order]
            nearest_dists = d_to_obs[order]
            weights = np.exp(-nearest_dists / tau)
            w_sum = float(np.sum(weights))
            if w_sum > 0:
                vals[p_idx, j] = float(np.sum(weights * row[nearest_cols]) / w_sum)
            else:
                vals[p_idx, j] = float(np.mean(row[nearest_cols]))

    rng = np.random.default_rng(random_seed)
    prot_shift = np.clip(
        np.nan_to_num(prot_means.values - global_med, nan=-1.0), -1.5, 0.0
    )

    for j in range(n_samples):
        col = vals[:, j]
        miss_mask = np.isnan(col)
        n_miss = int(np.sum(miss_mask))
        if n_miss == 0:
            continue

        obs_col = col[~miss_mask]
        if len(obs_col) < 5:
            vals[miss_mask, j] = lod_const
            continue

        q_j = float(np.clip(n_miss / n_proteins, 0.02, 0.50))
        probs_full = np.linspace(min(q_j + 0.05, 0.50), 0.95, 25)
        probs_obs = np.clip((probs_full - q_j) / max(1.0 - q_j, 1e-3), 0.01, 0.99)
        emp_quantiles = np.quantile(obs_col, probs_obs)
        theo_quantiles = stats.norm.ppf(probs_full)

        slope, intercept = np.polyfit(theo_quantiles, emp_quantiles, deg=1)
        sigma_qr = max(float(slope), 0.25)
        mu_qr = float(intercept)

        u_draws = rng.uniform(0.001, q_j, size=n_miss)
        z_draws = stats.norm.ppf(u_draws)
        imputed_mnar = mu_qr + sigma_qr * z_draws + prot_shift[miss_mask]
        vals[miss_mask, j] = imputed_mnar

    vals[np.isnan(vals)] = lod_const
    return pd.DataFrame(vals, index=work.index, columns=work.columns)


def prepare_maxlfq_matrix(
    df_prot: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Подготавливает матрицу pg_matrix.tsv без потери 48 белков с Genes == NULL."""
    sample_cols = [c for c in df_prot.columns if c.startswith("PTB_")]
    if not sample_cols:
        empty = pd.DataFrame()
        return empty, empty, empty

    gene_col = "Genes" if "Genes" in df_prot.columns else "Protein.Group"
    pg_col = "Protein_Group" if "Protein_Group" in df_prot.columns else "Protein.Group"

    raw_genes = df_prot[gene_col].replace("", np.nan)
    identifiers = raw_genes.fillna(df_prot[pg_col]).astype(str)

    wide = df_prot[sample_cols].copy().astype(np.float64)
    wide.index = identifiers

    if wide.index.duplicated().any():
        wide = wide.groupby(wide.index, sort=False).median()

    mask = extract_missingness_mask(wide)
    logged = log_transform(wide)
    polished = polish_anchor_medians(logged, max_missing_rate=0.05)
    imputed = impute_hybrid_mar_mnar(polished)

    return polished.T, imputed.T, mask.T


def align_hospital_baselines_unsupervised(
    df: pd.DataFrame,
    markers: list[str] | None = None,
) -> pd.DataFrame:
    """Устраняет межбольничный приборный сдвиг по нижнему квантильному ядру [Q_0.10, Q_0.55] без меток."""
    if "Dataset" not in df.columns:
        return df

    feats = CORE_MARKERS if markers is None else [m for m in markers if m in df.columns]
    if not feats:
        return df

    out = df.copy()
    train_mask = out["Dataset"] == "Train"
    if int(train_mask.sum()) < 10:
        return out

    for m in feats:
        tr_vals = out.loc[train_mask, m].dropna().to_numpy(dtype=np.float64)
        q10_tr, q32_tr, q55_tr = np.quantile(tr_vals, [0.10, 0.32, 0.55])
        scale_tr = max(float(q55_tr - q10_tr), 1e-4)

        for cohort_name, group_idx in out.groupby(
            "Dataset", observed=True
        ).groups.items():
            if cohort_name == "Train" or len(group_idx) < 10:
                continue
            c_vals = out.loc[group_idx, m].to_numpy(dtype=np.float64)
            q10_c, q32_c, q55_c = np.quantile(c_vals, [0.10, 0.32, 0.55])
            scale_c = max(float(q55_c - q10_c), 1e-4)

            aligned_vals = float(q32_tr) + (c_vals - float(q32_c)) * (
                scale_tr / scale_c
            )
            out.loc[group_idx, m] = aligned_vals

    return out


def load_markers(
    path: Path | str | None = None,
    log_scale: bool = True,
    align_batches: bool = False,
) -> pd.DataFrame:
    """Загружает эталонные маркеры из листа 5c, переводит в log2 и выравнивает приборный фон клиник."""
    target = locate(FIG5_NAME) if path is None else Path(path)
    if not target.is_file():
        raise LoadError(f"Файл не найден: {target}")
    df = pd.read_excel(target, sheet_name=SHEET_5C)
    df["Target"] = (df["Category"] == "PTB").astype(int)

    for marker in CORE_MARKERS:
        if marker in df.columns:
            raw_vals = df[marker].astype(np.float64)
            df[f"{marker}_raw"] = raw_vals
            if log_scale:
                log_vals = np.log2(raw_vals.clip(lower=1e-9))
                df[f"{marker}_log2_unaligned"] = log_vals
                df[marker] = log_vals

    if log_scale and align_batches:
        df = align_hospital_baselines_unsupervised(df, CORE_MARKERS)

    return df


def load_elisa_paired(path: Path | str | None = None) -> pd.DataFrame:
    """Загружает парные измерения MS + ELISA из листа 5d без глобального dropna по пациенткам."""
    target = locate(FIG5_NAME) if path is None else Path(path)
    if not target.is_file():
        raise LoadError(f"Файл не найден: {target}")

    df = pd.read_excel(target, sheet_name=SHEET_5D)
    df["Sample ID"] = df["Sample ID"].astype(str).str.strip()
    df["Protein"] = df["Protein"].astype(str).str.strip()
    df["Dataset"] = df["Dataset"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    df["Target"] = (df["Category"] == "PTB").astype(int)

    valid_row = df["log2(MS signal)"].notna() & df["log2(ELISA)"].notna()
    return df.loc[valid_row].reset_index(drop=True)


def load_shap_sheets(path: Path | str | None = None) -> dict[str, pd.DataFrame]:
    """Загружает эталонные матрицы SHAP-значений авторской модели из листов 5a и 5b."""
    target = locate(FIG5_NAME) if path is None else Path(path)
    if not target.is_file():
        raise LoadError(f"Файл не найден: {target}")

    return {
        "External validation1": pd.read_excel(target, sheet_name=SHEET_5A),
        "External validation2": pd.read_excel(target, sheet_name=SHEET_5B),
    }


def encode_categorical_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Выполняет порядковое (Ordinal) и номинальное (One-Hot) кодирование категориальных признаков."""
    out = df.copy()

    if "Category" in out.columns:
        ord_cat = OrdinalEncoder(
            categories=[["FTB", "PTB"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
        out["Category_Ordinal"] = ord_cat.fit_transform(
            out[["Category"]].astype(str)
        ).ravel()

    if "Dataset" in out.columns:
        ord_ds = OrdinalEncoder(
            categories=[DATASET_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.int64,
        )
        out["Dataset_Stage_Ordinal"] = ord_ds.fit_transform(
            out[["Dataset"]].astype(str)
        ).ravel()

        ohe = OneHotEncoder(sparse_output=False, dtype=np.int8)
        ohe_mat = ohe.fit_transform(out[["Dataset"]].astype(str))
        ohe_cols = [
            f"Dataset_OHE_{cat.replace(' ', '_')}" for cat in ohe.categories_[0]
        ]
        for idx, col_name in enumerate(ohe_cols):
            out[col_name] = ohe_mat[:, idx]

    return out
