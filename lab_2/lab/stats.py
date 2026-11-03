"""Описательная статистика, проверка нормальности, выбросы и непараметрические тесты."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет расширенные описательные статистики, тест нормальности и долю выбросов по IQR."""
    num = df.select_dtypes(include=[np.number])

    skew_vals: dict[str, float] = {}
    kurt_vals: dict[str, float] = {}
    iqr_vals: dict[str, float] = {}
    outlier_counts: dict[str, int] = {}
    outlier_rates: dict[str, float] = {}
    shapiro_pvals: dict[str, float] = {}

    for col in num.columns:
        s = num[col].dropna()
        n = len(s)
        if n > 2:
            skew_vals[col] = float(stats.skew(s))
            kurt_vals[col] = float(stats.kurtosis(s))
            q25, q75 = np.percentile(s, [25.0, 75.0])
            iqr = float(q75 - q25)
            iqr_vals[col] = iqr
            low_fence = q25 - 1.5 * iqr
            high_fence = q75 + 1.5 * iqr
            n_out = int(((s < low_fence) | (s > high_fence)).sum())
            outlier_counts[col] = n_out
            outlier_rates[col] = round(n_out / n, 4)

            sample_s = s.iloc[:5000] if n > 5000 else s
            if float(sample_s.std()) > 0:
                _, p_norm = stats.shapiro(sample_s)
                shapiro_pvals[col] = float(p_norm)
            else:
                shapiro_pvals[col] = 0.0
        else:
            skew_vals[col] = 0.0
            kurt_vals[col] = 0.0
            iqr_vals[col] = 0.0
            outlier_counts[col] = 0
            outlier_rates[col] = 0.0
            shapiro_pvals[col] = 1.0

    res = pd.DataFrame(
        {
            "mean": num.mean(),
            "std": num.std(),
            "median": num.median(),
            "min": num.min(),
            "max": num.max(),
            "iqr": pd.Series(iqr_vals),
            "skew": pd.Series(skew_vals),
            "kurtosis": pd.Series(kurt_vals),
            "outlier_count": pd.Series(outlier_counts),
            "outlier_rate": pd.Series(outlier_rates),
            "shapiro_p": pd.Series(shapiro_pvals),
            "missing": df[num.columns].isna().mean(),
        }
    )
    res["is_normal_005"] = res["shapiro_p"] > 0.05
    return res


def compare_normality_raw_vs_log2(
    df: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Сравнивает асимметрию, долю выбросов и p-value нормальности до и после log2-трансформации."""
    rows: list[dict[str, float | str]] = []
    for feat in features:
        raw_col = f"{feat}_raw" if f"{feat}_raw" in df.columns else feat
        s_raw = df[raw_col].dropna().astype(np.float64)
        if s_raw.empty:
            continue

        if f"{feat}_raw" in df.columns and feat in df.columns:
            s_lin = s_raw.clip(lower=1e-9)
            s_log2 = df[feat].dropna().astype(np.float64)
        elif (s_raw <= 0).any():
            s_log2 = s_raw
            s_lin = np.exp2(s_log2)
        else:
            s_lin = s_raw.clip(lower=1e-9)
            s_log2 = np.log2(s_lin)

        q25_r, q75_r = np.percentile(s_lin, [25.0, 75.0])
        iqr_r = q75_r - q25_r
        out_r = float(
            ((s_lin < q25_r - 1.5 * iqr_r) | (s_lin > q75_r + 1.5 * iqr_r)).mean()
        )

        q25_l, q75_l = np.percentile(s_log2, [25.0, 75.0])
        iqr_l = q75_l - q25_l
        out_l = float(
            ((s_log2 < q25_l - 1.5 * iqr_l) | (s_log2 > q75_l + 1.5 * iqr_l)).mean()
        )

        _, p_raw = stats.shapiro(s_lin.iloc[:5000])
        _, p_log = stats.shapiro(s_log2.iloc[:5000])

        rows.append(
            {
                "feature": feat,
                "skew_raw": round(float(stats.skew(s_lin)), 4),
                "skew_log2": round(float(stats.skew(s_log2)), 4),
                "outlier_rate_raw": round(out_r, 4),
                "outlier_rate_log2": round(out_l, 4),
                "shapiro_p_raw": float(p_raw),
                "shapiro_p_log2": float(p_log),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def describe_categorical(
    df: pd.DataFrame,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Рассчитывает частоты, количество уникальных значений и моду для категориальных признаков."""
    target_cols = (
        [c for c in ["Dataset", "Category", "Hospital_Center"] if c in df.columns]
        if cols is None
        else [c for c in cols if c in df.columns]
    )
    rows: list[dict[str, str | int | float]] = []
    for col in target_cols:
        s = df[col].dropna().astype(str)
        if s.empty:
            continue
        vc = s.value_counts()
        mode_val = str(vc.index[0])
        mode_cnt = int(vc.iloc[0])
        rows.append(
            {
                "column": col,
                "n_total": len(s),
                "n_unique": int(s.nunique()),
                "mode": mode_val,
                "mode_count": mode_cnt,
                "mode_share": round(mode_cnt / len(s), 4),
                "distribution": ", ".join(f"{k}: {v}" for k, v in vc.items()),
            }
        )
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def aggregate_by_groups(
    df: pd.DataFrame,
    features: list[str],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Проводит группировку и расчет агрегированных показателей (mean, median, std) по когортам и классам."""
    g_cols = (
        [c for c in ["Dataset", "Category"] if c in df.columns]
        if group_cols is None
        else [c for c in group_cols if c in df.columns]
    )
    valid_feats = [f for f in features if f in df.columns]
    if not g_cols or not valid_feats:
        return pd.DataFrame()

    grouped = (
        df.groupby(g_cols, observed=True)[valid_feats]
        .agg(["count", "mean", "median", "std"])
        .round(4)
    )
    return grouped


def diff(
    df: pd.DataFrame,
    features: list[str],
    target: str = "Target",
) -> pd.DataFrame:
    """Проводит двусторонний тест Манна-Уитни с поправкой Бенджамини-Хохберга (FDR)."""
    col_vals = df[target].dropna().unique()
    if set(col_vals).issubset({0, 1}):
        g0 = df[df[target] == 0]
        g1 = df[df[target] == 1]
    elif "FTB" in col_vals and "PTB" in col_vals:
        g0 = df[df[target] == "FTB"]
        g1 = df[df[target] == "PTB"]
    elif len(col_vals) >= 2:
        u0, u1 = sorted(col_vals)[:2]
        g0 = df[df[target] == u0]
        g1 = df[df[target] == u1]
    else:
        return pd.DataFrame(
            columns=["feature", "u_stat", "p_value", "log2_fc", "p_adj"]
        )

    rows = []
    for f in features:
        v0, v1 = g0[f].dropna(), g1[f].dropna()
        if len(v0) == 0 or len(v1) == 0:
            continue
        stat, pval = stats.mannwhitneyu(v1, v0, alternative="two-sided")

        is_logged = (
            f"{f}_raw" in df.columns or bool((v0 <= 0).any()) or bool((v1 <= 0).any())
        )
        if is_logged:
            log2_fc = float(v1.median() - v0.median())
        else:
            fc = (float(v1.median()) + 1e-9) / (float(v0.median()) + 1e-9)
            log2_fc = float(np.log2(max(fc, 1e-12)))

        rows.append(
            {
                "feature": f,
                "u_stat": float(stat),
                "p_value": float(pval),
                "log2_fc": log2_fc,
            }
        )
    res = pd.DataFrame(rows)
    if not res.empty:
        res["p_adj"] = adjust_fdr(res["p_value"])
    return res


def adjust_fdr(pvals: pd.Series | np.ndarray) -> np.ndarray:
    """Вычисляет скорректированные p-значения методом Бенджамини-Хохберга."""
    p = np.asanyarray(pvals, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    rank = np.empty(n, int)
    rank[order] = np.arange(1, n + 1)
    adj = p * n / rank
    return np.clip(np.minimum.accumulate(adj[order[::-1]])[::-1][rank - 1], 0.0, 1.0)
