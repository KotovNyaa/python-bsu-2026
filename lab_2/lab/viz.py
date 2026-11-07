"""Модуль визуализации данных (гистограммы, boxplot, тепловая карта, вулкан, барплот, FDR-хвост)."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"


def ensure_dir(path: Path) -> Path:
    """Создает родительскую директорию при её отсутствии."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def hist_distributions(
    df: pd.DataFrame,
    features: list[str],
    target: str = "Category",
    path: Path | str | None = None,
) -> Path:
    """Строит многопанельные гистограммы и KDE распределений числовых биомаркеров."""
    out = PLOTS_DIR / "hist.png" if path is None else Path(path)
    ensure_dir(out)

    n_feats = len(features)
    fig, axes = plt.subplots(1, n_feats, figsize=(3.2 * n_feats, 4.2), sharey=False)
    if n_feats == 1:
        axes = [axes]

    palette = {"FTB": "#2b83ba", "PTB": "#d7191c"}
    for ax, feat in zip(axes, features, strict=False):
        if target in df.columns:
            sns.histplot(
                data=df,
                x=feat,
                hue=target,
                kde=True,
                stat="density",
                common_norm=False,
                palette=palette,
                alpha=0.45,
                bins=22,
                ax=ax,
            )
        else:
            sns.histplot(data=df, x=feat, kde=True, stat="density", bins=22, ax=ax)

        s = df[feat].dropna()
        skew_val = float(s.skew()) if len(s) > 2 else 0.0
        ax.set_title(f"{feat} (skew={skew_val:.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("log2(MS signal)", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(
        "Biomarker Distributions Across Clinical Groups (log2 Scale)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def categorical_bars(
    df: pd.DataFrame,
    path: Path | str | None = None,
) -> Path:
    """Строит барплоты распределения частот категориальных признаков и сдвига превалентности sPTB."""
    out = PLOTS_DIR / "countplot.png" if path is None else Path(path)
    ensure_dir(out)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    if "Dataset" in df.columns and "Category" in df.columns:
        order = [
            c
            for c in [
                "Train",
                "Internal validation",
                "External validation1",
                "External validation2",
            ]
            if c in df["Dataset"].values
        ]
        if not order:
            order = sorted(df["Dataset"].astype(str).unique().tolist())

        sns.countplot(
            data=df,
            x="Dataset",
            hue="Category",
            order=order,
            palette={"FTB": "#66c2a5", "PTB": "#fc8d62"},
            ax=ax1,
        )
        ax1.set_title("Sample Counts by Cohort and Outcome", fontsize=11)
        ax1.set_xlabel("Clinical Cohort", fontsize=10)
        ax1.set_ylabel("Number of Patients", fontsize=10)
        ax1.tick_params(axis="x", rotation=20)
        ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

        prev_series = (
            df.assign(_is_ptb=(df["Category"] == "PTB").astype(float) * 100.0)
            .groupby("Dataset", observed=True)["_is_ptb"]
            .mean()
            .reindex(order)
        )
        bars = ax2.bar(
            prev_series.index,
            prev_series.values,
            color=["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"][: len(order)],
            alpha=0.85,
            edgecolor="black",
        )
        for bar in bars:
            height = float(bar.get_height())
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.8,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax2.set_ylim(0, max(float(prev_series.max()) * 1.25, 40.0))
        ax2.set_title("Prior Probability Shift: sPTB Prevalence (%)", fontsize=11)
        ax2.set_xlabel("Clinical Cohort", fontsize=10)
        ax2.set_ylabel("sPTB Rate (%)", fontsize=10)
        ax2.tick_params(axis="x", rotation=20)
        ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def box(
    df: pd.DataFrame,
    features: list[str],
    target: str = "Category",
    path: Path | str | None = None,
) -> Path:
    """Строит диаграммы размаха (Boxplot) для признаков в разрезе групп FTB/PTB."""
    out = PLOTS_DIR / "box.png" if path is None else Path(path)
    ensure_dir(out)
    melted = df.melt(
        id_vars=[target], value_vars=features, var_name="Marker", value_name="Level"
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.boxplot(data=melted, x="Marker", y="Level", hue=target, palette="Set2", ax=ax)
    ax.set_title("Biomarker Expression Levels by Group (log2 Scale)")
    ax.set_ylabel("log2(MS Intensity)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def corr(
    df: pd.DataFrame,
    features: list[str],
    path: Path | str | None = None,
) -> Path:
    """Строит тепловую карту ранговых корреляций Спирмена."""
    out = PLOTS_DIR / "corr.png" if path is None else Path(path)
    ensure_dir(out)
    matrix = df[features].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(matrix, annot=True, cmap="vlag", fmt=".2f", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Spearman Rank Correlation Matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def volcano(
    df_diff: pd.DataFrame,
    path: Path | str | None = None,
    p_thresh: float = 0.05,
    fc_thresh: float = 1.0,
) -> Path:
    """Строит вулканный график дифференциального анализа с подписями биомаркеров."""
    out = PLOTS_DIR / "volcano.png" if path is None else Path(path)
    ensure_dir(out)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    neg_log = -np.log10(df_diff["p_value"].clip(lower=1e-18))
    colors = [
        "#d7191c"
        if (p < p_thresh and fc > fc_thresh)
        else "#2b83ba"
        if (p < p_thresh and fc < -fc_thresh)
        else "gray"
        for p, fc in zip(df_diff["p_value"], df_diff["log2_fc"], strict=False)
    ]
    ax.scatter(
        df_diff["log2_fc"],
        neg_log,
        c=colors,
        alpha=0.85,
        s=55,
        edgecolors="black",
        linewidths=0.6,
    )

    if "feature" in df_diff.columns and len(df_diff) <= 35:
        for feat_name, x_val, y_val in zip(
            df_diff["feature"], df_diff["log2_fc"], neg_log, strict=False
        ):
            ax.annotate(
                str(feat_name),
                (float(x_val), float(y_val)),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.axhline(-np.log10(p_thresh), color="black", linestyle="--", lw=0.8)
    ax.axvline(-fc_thresh, color="black", linestyle="--", lw=0.8)
    ax.axvline(fc_thresh, color="black", linestyle="--", lw=0.8)
    ax.set_xlabel("log2 Fold Change (PTB vs FTB)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Differential Expression Volcano Plot")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def decoy(
    df_trans: pd.DataFrame,
    path: Path | str | None = None,
) -> Path:
    """Строит распределение интенсивности достоверных мишеней против шумового хвоста (Decoy / QValue > 0.01)."""
    out = PLOTS_DIR / "decoy.png" if path is None else Path(path)
    ensure_dir(out)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    has_explicit_decoy = "decoy" in df_trans.columns and bool(
        (df_trans["decoy"] == 1).any()
    )
    if has_explicit_decoy:
        t0 = df_trans[df_trans["decoy"] == 0]["LibraryIntensity"].dropna()
        t1 = df_trans[df_trans["decoy"] == 1]["LibraryIntensity"].dropna()
        label_0 = "Target (decoy = 0)"
        label_1 = "Decoy (decoy = 1)"
    elif "QValue" in df_trans.columns:
        t0 = df_trans[df_trans["QValue"] <= 0.01]["LibraryIntensity"].dropna()
        t1 = df_trans[df_trans["QValue"] > 0.01]["LibraryIntensity"].dropna()
        label_0 = "Confident Target (QValue <= 0.01)"
        label_1 = "FDR Noise Tail (QValue > 0.01)"
    else:
        t0 = df_trans["LibraryIntensity"].dropna()
        t1 = pd.Series(dtype=np.float64)
        label_0 = "Target (0)"
        label_1 = "Decoy (1)"

    t0_pos = t0[t0 > 0]
    if len(t0_pos) > 0:
        ax.hist(
            np.log10(t0_pos),
            bins=45,
            alpha=0.55,
            label=label_0,
            color="#2ca02c",
            density=True,
        )

    t1_pos = t1[t1 > 0]
    if len(t1_pos) > 0:
        ax.hist(
            np.log10(t1_pos),
            bins=45,
            alpha=0.55,
            label=label_1,
            color="#d62728",
            density=True,
        )

    ax.set_xlabel("log10(LibraryIntensity)")
    ax.set_ylabel("Density")
    ax.set_title("MS2 Intensity Distribution: Confident Targets vs Noise Tail")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out
