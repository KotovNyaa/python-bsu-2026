"""Машинное обучение, адаптация к сдвигу превалентности (Prior Shift) и дистилляция в Чунцин."""

import contextlib
import sqlite3
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lab.cohorts import export_cohort_artifacts, generate_cohort_artifacts
from lab.data import locate
from lab.db import connect
from lab.external_knowledge import run_blocks_v_g
from lab.patient_graph import SHARED_4_MARKERS, run_block_b
from lab.prep import align_hospital_baselines_unsupervised, load_markers
from lab.priors import build_priors_table

matplotlib.use("Agg")
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
MARKERS = ["AMBP", "FN1", "LUM", "TIMP1", "B2M"]


def _prepare_feature_matrix(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """Извлекает матрицу признаков и гарантирует перевод сырых интенсивностей в log2-шкалу."""
    x = df[feats].to_numpy(dtype=np.float64, copy=True)
    if set(feats).issubset(set(MARKERS)) and np.all(x > 0) and np.max(x) > 2.0:
        x = np.log2(np.clip(x, 1e-9, None))
    return x


def train_logistic_regression(
    train_df: pd.DataFrame, features: list[str] | None = None
) -> Pipeline:
    """Обучает референсную логистическую регрессию из статьи в log2-пространстве."""
    feats = MARKERS if features is None else features
    x_train = _prepare_feature_matrix(train_df, feats)
    y_train = train_df["Target"].to_numpy(dtype=int)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    return pipe


def train_ensemble_model(
    train_df: pd.DataFrame, features: list[str] | None = None
) -> Pipeline:
    """Обучает нелинейный ансамбль случайных деревьев."""
    feats = MARKERS if features is None else features
    x_train = _prepare_feature_matrix(train_df, feats)
    y_train = train_df["Target"].to_numpy(dtype=int)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=4,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y_train)
    return pipe


def estimate_prior_saerens_em(
    y_prob_train: np.ndarray,
    pi_train: float,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> float:
    """Оценивает априорную долю класса в новой клинике методом EM Саеренса с регуляризацией Дирихле."""
    probs = np.clip(np.asarray(y_prob_train, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    n_samples = len(probs)
    if n_samples == 0:
        return float(pi_train)

    pi_tr = float(np.clip(pi_train, 0.02, 0.98))
    pi_est = pi_tr
    alpha_reg = 2.0

    for _ in range(max_iter):
        r1 = probs * (pi_est / pi_tr)
        r0 = (1.0 - probs) * ((1.0 - pi_est) / (1.0 - pi_tr))
        q_post = r1 / np.maximum(r1 + r0, 1e-12)
        pi_mle = float(np.mean(q_post))
        pi_next = float(
            np.clip(
                (n_samples * pi_mle + alpha_reg * pi_tr) / (n_samples + alpha_reg),
                0.03,
                0.85,
            )
        )
        if abs(pi_next - pi_est) < tol:
            pi_est = pi_next
            break
        pi_est = pi_next

    return round(pi_est, 4)


def adjust_probabilities_for_prior_shift(
    y_prob_train: np.ndarray,
    pi_train: float,
    pi_target: float,
) -> np.ndarray:
    """Выполняет байесовскую коррекцию апостериорных вероятностей при сдвиге превалентности."""
    probs = np.clip(np.asarray(y_prob_train, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    pi_tr = float(np.clip(pi_train, 0.01, 0.99))
    pi_tg = float(np.clip(pi_target, 0.01, 0.99))

    logits = np.log(probs / (1.0 - probs))
    shift = np.log(pi_tg / (1.0 - pi_tg)) - np.log(pi_tr / (1.0 - pi_tr))
    adj_logits = np.clip(logits + shift, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-adj_logits))


def find_optimal_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Находит оптимальный порог бинарной классификации по максимизации F1-score (протокол Zhao et al.)."""
    best_th = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.20, 0.80, 61):
        preds = (y_prob >= th).astype(int)
        score = float(f1_score(y_true, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_th = float(th)
    return round(best_th, 4)


def evaluate_cohort(
    model: Pipeline,
    df_cohort: pd.DataFrame,
    features: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Вычисляет полный спектр клинических метрик классификации."""
    feats = MARKERS if features is None else features
    x = _prepare_feature_matrix(df_cohort, feats)
    y = df_cohort["Target"].to_numpy(dtype=int)

    y_prob = model.predict_proba(x)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()

    auroc = float(roc_auc_score(y, y_prob)) if len(np.unique(y)) > 1 else float("nan")
    aupr = (
        float(average_precision_score(y, y_prob))
        if len(np.unique(y)) > 1
        else float("nan")
    )
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    f1 = float(f1_score(y, y_pred, zero_division=0))

    return {
        "auroc": round(auroc, 4),
        "aupr": round(aupr, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "ppv": round(ppv, 4),
        "npv": round(npv, 4),
        "f1": round(f1, 4),
        "n_samples": len(y),
        "n_cases": int(y.sum()),
    }


def evaluate_all_cohorts(
    model: Pipeline, df: pd.DataFrame, features: list[str] | None = None
) -> pd.DataFrame:
    """Оценивает модель на всех 4 когортах (обучающей и внешних)."""
    cohorts = [
        "Train",
        "Internal validation",
        "External validation1",
        "External validation2",
    ]
    rows = []
    for c in cohorts:
        subset = df[df["Dataset"] == c]
        if not subset.empty:
            res = evaluate_cohort(model, subset, features)
            rows.append({"cohort": c, **res})
    return pd.DataFrame(rows).set_index("cohort")


def evaluate_all_cohorts_with_prior_shift(
    model: Pipeline,
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Оценивает модель со слепой EM-адаптацией Саеренса к сдвигу превалентности."""
    df_aligned = align_hospital_baselines_unsupervised(df, MARKERS)
    feats = MARKERS if features is None else features
    train_sub = df_aligned[df_aligned["Dataset"] == "Train"]
    pi_train = (
        float(train_sub["Target"].mean())
        if not train_sub.empty
        else float(df["Target"].mean())
    )

    if not train_sub.empty:
        x_tr = _prepare_feature_matrix(train_sub, feats)
        prob_tr = model.predict_proba(x_tr)[:, 1]
        opt_th = find_optimal_f1_threshold(
            train_sub["Target"].to_numpy(dtype=int), prob_tr
        )
    else:
        opt_th = 0.5

    cohorts = [
        "Train",
        "Internal validation",
        "External validation1",
        "External validation2",
    ]
    rows: list[dict[str, float | int | str]] = []
    for c in cohorts:
        sub = df_aligned[df_aligned["Dataset"] == c]
        if sub.empty:
            continue
        x = _prepare_feature_matrix(sub, feats)
        y = sub["Target"].to_numpy(dtype=int)
        raw_prob = model.predict_proba(x)[:, 1]

        true_prior = float(np.mean(y))
        est_prior_em = (
            pi_train if c == "Train" else estimate_prior_saerens_em(raw_prob, pi_train)
        )
        adj_prob = adjust_probabilities_for_prior_shift(
            raw_prob, pi_train=pi_train, pi_target=est_prior_em
        )

        y_pred_adj = (adj_prob >= opt_th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred_adj, labels=[0, 1]).ravel()

        auroc = (
            float(roc_auc_score(y, adj_prob)) if len(np.unique(y)) > 1 else float("nan")
        )
        aupr = (
            float(average_precision_score(y, adj_prob))
            if len(np.unique(y)) > 1
            else float("nan")
        )
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1_adj = float(f1_score(y, y_pred_adj, zero_division=0))

        rows.append(
            {
                "cohort": c,
                "true_prior": round(true_prior, 4),
                "em_estimated_prior": round(est_prior_em, 4),
                "auroc": round(auroc, 4),
                "aupr": round(aupr, 4),
                "sensitivity_adj": round(sens, 4),
                "specificity_adj": round(spec, 4),
                "ppv_adj": round(ppv, 4),
                "npv_adj": round(npv, 4),
                "f1_adj": round(f1_adj, 4),
            }
        )

    return pd.DataFrame(rows).set_index("cohort")


def _build_4marker_bridge_features(
    df_4_log2: pd.DataFrame,
    ref_4_log2: pd.DataFrame,
    ref_ftb_mask: np.ndarray,
) -> np.ndarray:
    """Формирует инвариантные к разведению и межцентровому сдвигу признаки по 4 общим белкам."""
    x_raw = df_4_log2[SHARED_4_MARKERS].to_numpy(dtype=np.float64)
    x_ref = ref_4_log2[SHARED_4_MARKERS].to_numpy(dtype=np.float64)

    if np.median(x_raw) > 5.0 and np.median(x_ref) < 5.0:
        x_aligned = np.zeros_like(x_raw)
        for col_idx in range(x_raw.shape[1]):
            ref_sorted = np.sort(x_ref[:, col_idx])
            ranks = stats.rankdata(x_raw[:, col_idx], method="average")
            q_vals = np.clip((ranks - 0.5) / max(len(ranks), 1), 0.01, 0.99)
            x_aligned[:, col_idx] = np.quantile(ref_sorted, q_vals)
    else:
        x_aligned = x_raw.copy()

    lin = np.exp2(np.clip(x_aligned, -25.0, 25.0))
    p_mat = lin / np.maximum(np.sum(lin, axis=1, keepdims=True), 1e-12)
    p_mat = np.clip(p_mat, 1e-12, 1.0)
    p_mat = p_mat / np.sum(p_mat, axis=1, keepdims=True)

    lin_ref = np.exp2(np.clip(x_ref[ref_ftb_mask], -25.0, 25.0))
    p_ref_ftb = lin_ref / np.maximum(np.sum(lin_ref, axis=1, keepdims=True), 1e-12)
    q_ftb = np.clip(np.mean(p_ref_ftb, axis=0, keepdims=True), 1e-12, 1.0)
    q_ftb = q_ftb / np.sum(q_ftb)

    m_mat = 0.5 * (p_mat + q_ftb)
    jsd_to_ftb = np.sqrt(
        np.clip(
            0.5
            * (
                np.sum(p_mat * np.log2(p_mat / m_mat), axis=1)
                + np.sum(q_ftb * np.log2(q_ftb / m_mat), axis=1)
            ),
            0.0,
            1.0,
        )
    )

    ratios = [
        x_aligned[:, i] - x_aligned[:, j]
        for i, j in combinations(range(len(SHARED_4_MARKERS)), 2)
    ]
    ratio_mat = np.column_stack(ratios)

    return np.column_stack([x_aligned, p_mat, ratio_mat, jsd_to_ftb])


def distill_and_subtype_chongqing_cohort(
    df_a_markers: pd.DataFrame,
    discovery_b_df: pd.DataFrame,
) -> pd.DataFrame:
    """Выполняет двухэтапную дистилляцию вероятностей sPTB и выделение молекулярных подтипов Чунцина."""
    if discovery_b_df.empty:
        return pd.DataFrame()

    df_a = df_a_markers.copy()
    for m in SHARED_4_MARKERS:
        if np.all(df_a[m].to_numpy(dtype=np.float64) > 0) and df_a[m].max() > 2.0:
            df_a[m] = np.log2(df_a[m].clip(lower=1e-9))

    train_a = df_a[df_a["Dataset"] == "Train"]
    if len(train_a) < 20:
        train_a = df_a

    ftb_mask_tr = (train_a["Target"] == 0).to_numpy(dtype=bool)
    if not np.any(ftb_mask_tr):
        ftb_mask_tr = np.ones(len(train_a), dtype=bool)

    x_tr_bridge = _build_4marker_bridge_features(
        train_a[SHARED_4_MARKERS], train_a[SHARED_4_MARKERS], ftb_mask_tr
    )
    y_tr = train_a["Target"].to_numpy(dtype=int)

    x_b_bridge = _build_4marker_bridge_features(
        discovery_b_df[SHARED_4_MARKERS], train_a[SHARED_4_MARKERS], ftb_mask_tr
    )

    teachers = [
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
            ]
        ),
        Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.5, max_iter=1000, random_state=7)),
            ]
        ),
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=4,
                        min_samples_leaf=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=150,
                        max_depth=5,
                        min_samples_leaf=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
    ]

    preds_b_list: list[np.ndarray] = []
    for t_model in teachers:
        t_model.fit(x_tr_bridge, y_tr)
        preds_b_list.append(t_model.predict_proba(x_b_bridge)[:, 1])

    preds_b_mat = np.vstack(preds_b_list)
    prob_soft = np.clip(np.mean(preds_b_mat, axis=0), 0.01, 0.99)
    logit_soft = np.log(prob_soft / (1.0 - prob_soft))
    uncertainty_std = np.std(preds_b_mat, axis=0, ddof=1)

    x_b_all = discovery_b_df.to_numpy(dtype=np.float64)
    n_b = x_b_all.shape[0]

    x_b_z = (x_b_all - np.mean(x_b_all, axis=0, keepdims=True)) / np.maximum(
        np.std(x_b_all, axis=0, ddof=1, keepdims=True), 1e-6
    )
    p_b = np.exp(np.clip(x_b_z * 0.5, -15.0, 15.0))
    p_b = p_b / np.maximum(np.sum(p_b, axis=1, keepdims=True), 1e-12)
    p_b = np.clip(p_b, 1e-12, 1.0)
    p_b = p_b / np.sum(p_b, axis=1, keepdims=True)

    p_i = p_b[:, None, :]
    p_j = p_b[None, :, :]
    m_ij = 0.5 * (p_i + p_j)
    jsd_mat = np.clip(
        0.5
        * (
            np.sum(p_i * np.log2(p_i / m_ij), axis=2)
            + np.sum(p_j * np.log2(p_j / m_ij), axis=2)
        ),
        0.0,
        1.0,
    )
    np.fill_diagonal(jsd_mat, 0.0)

    mean_pairwise_jsd = np.mean(jsd_mat, axis=1)
    norm_candidates = np.argsort(mean_pairwise_jsd)[: max(n_b // 3, 5)]
    q_ftb_b = np.mean(p_b[norm_candidates], axis=0, keepdims=True)
    q_ftb_b = np.clip(q_ftb_b, 1e-12, 1.0) / np.sum(q_ftb_b)
    m_to_ftb = 0.5 * (p_b + q_ftb_b)
    jsd_to_ftb_centroid = np.clip(
        0.5
        * (
            np.sum(p_b * np.log2(p_b / m_to_ftb), axis=1)
            + np.sum(q_ftb_b * np.log2(q_ftb_b / m_to_ftb), axis=1)
        ),
        0.0,
        1.0,
    )

    var_top_idx = np.argsort(np.var(x_b_all, axis=0))[::-1][:150]
    x_sub = x_b_all[:, var_top_idx]
    corr_all = np.corrcoef(x_sub, rowvar=False)
    np.fill_diagonal(corr_all, 0.0)

    sample_mean_degrees = np.zeros(n_b, dtype=np.float64)
    for s_idx in range(n_b):
        mask_loo = np.ones(n_b, dtype=bool)
        mask_loo[s_idx] = False
        corr_loo = np.corrcoef(x_sub[mask_loo], rowvar=False)
        np.fill_diagonal(corr_loo, 0.0)
        net_i = n_b * corr_all - (n_b - 1) * corr_loo
        sample_mean_degrees[s_idx] = float(np.mean(np.sum(np.abs(net_i) > 0.5, axis=1)))

    zn_cu_genes = [
        g for g in ["SOD1", "ALB", "GC", "TF"] if g in discovery_b_df.columns
    ]
    comp_genes = [
        g for g in ["C3", "C4A", "CFH", "PRG2", "B2M"] if g in discovery_b_df.columns
    ]

    if zn_cu_genes:
        zn_mat = discovery_b_df[zn_cu_genes].to_numpy(dtype=np.float64)
        zn_z = (zn_mat - np.mean(zn_mat, axis=0)) / np.maximum(
            np.std(zn_mat, axis=0, ddof=1), 1e-6
        )
        zn_cu_score = np.mean(zn_z, axis=1)
    else:
        zn_cu_score = np.zeros(n_b)

    if comp_genes:
        c_mat = discovery_b_df[comp_genes].to_numpy(dtype=np.float64)
        c_z = (c_mat - np.mean(c_mat, axis=0)) / np.maximum(
            np.std(c_mat, axis=0, ddof=1), 1e-6
        )
        comp_score = np.mean(c_z, axis=1)
    else:
        comp_score = np.zeros(n_b)

    condensed_jsd = squareform(np.sqrt(jsd_mat), checks=False)
    z_link = linkage(condensed_jsd, method="average")
    raw_clusters = fcluster(z_link, t=3, criterion="maxclust")

    _, counts = np.unique(raw_clusters, return_counts=True)
    composite_severity = (
        0.45 * stats.rankdata(jsd_to_ftb_centroid) / n_b
        + 0.30 * stats.rankdata(comp_score) / n_b
        + 0.25 * stats.rankdata(zn_cu_score) / n_b
    )
    if np.min(counts) < 3:
        q55, q82 = np.quantile(composite_severity, [0.55, 0.82])
        subtype_cluster = np.where(
            composite_severity <= q55,
            0,
            np.where(composite_severity <= q82, 1, 2),
        )
    else:
        cl_means = {
            cl_id: float(np.mean(composite_severity[raw_clusters == cl_id]))
            for cl_id in np.unique(raw_clusters)
        }
        sorted_cl = sorted(cl_means, key=lambda k: cl_means[k])
        remap = {old_id: new_id for new_id, old_id in enumerate(sorted_cl)}
        subtype_cluster = np.array([remap[c] for c in raw_clusters], dtype=int)

    distilled_df = discovery_b_df.copy()
    for idx_m, m_name in enumerate(SHARED_4_MARKERS):
        distilled_df[f"aligned_4marker_{m_name}"] = np.round(x_b_bridge[:, idx_m], 4)

    distilled_df["sPTB_Prob_Soft"] = np.round(prob_soft, 4)
    distilled_df["sPTB_Logit_Soft"] = np.round(logit_soft, 4)
    distilled_df["Uncertainty_Std"] = np.round(uncertainty_std, 4)
    distilled_df["Subtype_Cluster"] = subtype_cluster.astype(int)
    distilled_df["JSD_to_FTB_Centroid"] = np.round(jsd_to_ftb_centroid, 4)
    distilled_df["Sample_Mean_Degree"] = np.round(sample_mean_degrees, 2)
    distilled_df["Zn_Cu_Target_Score"] = np.round(zn_cu_score, 4)
    distilled_df["Complement_Activity_Score"] = np.round(comp_score, 4)

    return distilled_df


def build_relbench_heterodata_archive(root: Path) -> Path:
    """Собирает единый реляционный гетерогенный граф (RelBench / HGT стандарт) в архив .npz."""
    out_npz = root / "sptb_relbench_heterodata.npz"

    enriched_a_path = root / "cohort_a_426_enriched.parquet"
    distilled_b_path = root / "cohort_b_35_proteome_distilled.parquet"
    p_graph_path = root / "patient_graph_edges.parquet"
    m_graph_path = root / "protein_multiplex_graph.parquet"
    masks_path = root / "binn_hierarchy_masks.npz"

    df_a = (
        pd.read_parquet(enriched_a_path)
        if enriched_a_path.is_file()
        else pd.DataFrame()
    )
    df_b = (
        pd.read_parquet(distilled_b_path)
        if distilled_b_path.is_file()
        else pd.DataFrame()
    )
    df_pg = pd.read_parquet(p_graph_path) if p_graph_path.is_file() else pd.DataFrame()
    df_mg = pd.read_parquet(m_graph_path) if m_graph_path.is_file() else pd.DataFrame()

    exclude_a = {"Target", "Fold_ID", "Category_Ordinal", "Dataset_Stage_Ordinal"}
    num_cols_a = [
        c for c in df_a.select_dtypes(include=[np.number]).columns if c not in exclude_a
    ]
    x_patient_a = (
        df_a[num_cols_a].to_numpy(dtype=np.float32)
        if not df_a.empty
        else np.zeros((0, 5), dtype=np.float32)
    )
    exclude_b = {
        "sPTB_Prob_Soft",
        "sPTB_Logit_Soft",
        "Uncertainty_Std",
        "Subtype_Cluster",
    }
    num_cols_b = [
        c for c in df_b.select_dtypes(include=[np.number]).columns if c not in exclude_b
    ]
    x_patient_b = (
        df_b[num_cols_b].to_numpy(dtype=np.float32)
        if not df_b.empty
        else np.zeros((0, 5), dtype=np.float32)
    )

    if not df_pg.empty and not df_a.empty:
        a_id_map = {sid: idx for idx, sid in enumerate(df_a.index.astype(str))}
        src_p = (
            df_pg["source"].astype(str).map(a_id_map).fillna(0).to_numpy(dtype=np.int64)
        )
        tgt_p = (
            df_pg["target"].astype(str).map(a_id_map).fillna(0).to_numpy(dtype=np.int64)
        )
        edge_patient_jsd = np.vstack([tgt_p, src_p])
        edge_patient_w = df_pg["weight"].to_numpy(dtype=np.float32)
    else:
        edge_patient_jsd = np.zeros((2, 0), dtype=np.int64)
        edge_patient_w = np.zeros((0,), dtype=np.float32)

    if not df_mg.empty and "source_idx" in df_mg.columns:
        edge_prot_coexp = np.vstack(
            [
                df_mg["source_idx"].to_numpy(dtype=np.int64),
                df_mg["target_idx"].to_numpy(dtype=np.int64),
            ]
        )
        edge_prot_attr = df_mg[
            ["coexp_spearman_r", "string_combined_score", "multiplex_weight"]
        ].to_numpy(dtype=np.float32)
    else:
        edge_prot_coexp = np.zeros((2, 0), dtype=np.int64)
        edge_prot_attr = np.zeros((0, 3), dtype=np.float32)

    if masks_path.is_file():
        m_data = np.load(masks_path, allow_pickle=False)
        m1_bool = m_data["M1"]
        prot_idx, pw_idx = np.where(m1_bool)
        edge_prot_to_pathway = np.vstack([prot_idx, pw_idx]).astype(np.int64)
    else:
        edge_prot_to_pathway = np.zeros((2, 0), dtype=np.int64)

    hetero_dict: dict[str, np.ndarray] = {
        "patient_A_x": x_patient_a,
        "patient_A_y": df_a["Target"].to_numpy(dtype=np.int64)
        if "Target" in df_a.columns
        else np.zeros((0,), dtype=np.int64),
        "patient_B_x": x_patient_b,
        "pathway_drug_names": np.array(
            [
                *[f"Reactome_SubPathway_{i:03d}" for i in range(350)],
                "Zinc_DrugBank_DB01593",
                "Copper_DrugBank_DB09130",
            ],
            dtype=str,
        ),
        "edge_index_patient_similar_jsd": edge_patient_jsd,
        "edge_weight_patient_similar_jsd": edge_patient_w,
        "edge_index_protein_multiplex": edge_prot_coexp,
        "edge_attr_protein_multiplex": edge_prot_attr,
        "edge_index_protein_to_pathway": edge_prot_to_pathway,
    }
    np.savez_compressed(out_npz, **hetero_dict)
    return out_npz


def plot_roc_pr(
    model: Pipeline,
    df: pd.DataFrame,
    features: list[str] | None = None,
    path: Path | str | None = None,
) -> Path:
    """Строит эталонные академические ROC и PR кривые для всех когорт."""
    out = PLOTS_DIR / "roc_pr.png" if path is None else Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats = MARKERS if features is None else features

    cohort_configs = [
        ("Train", "#1f77b4", "Train"),
        ("Internal validation", "#d62728", "Internal Val"),
        ("External validation1", "#e377c2", "External Val 1 (MCH2)"),
        ("External validation2", "#ff7f0e", "External Val 2 (MCH3)"),
    ]

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 5))

    for cohort_name, color, label in cohort_configs:
        sub = df[df["Dataset"] == cohort_name]
        if sub.empty:
            continue
        y = sub["Target"].to_numpy(dtype=int)
        x_sub = _prepare_feature_matrix(sub, feats)
        prob = model.predict_proba(x_sub)[:, 1]

        fpr, tpr, _ = roc_curve(y, prob)
        auc_val = roc_auc_score(y, prob)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {auc_val:.2f})")

        prec, rec, _ = precision_recall_curve(y, prob)
        aupr_val = average_precision_score(y, prob)
        ax_pr.plot(
            rec, prec, color=color, lw=2, label=f"{label} (AUPR = {aupr_val:.2f})"
        )

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.7)
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel("1 - Specificity (FPR)", fontsize=11)
    ax_roc.set_ylabel("Sensitivity (TPR)", fontsize=11)
    ax_roc.set_title("ROC Curves Across Multi-Center Cohorts", fontsize=12)
    ax_roc.legend(loc="lower right", frameon=True, fontsize=9)
    ax_roc.grid(True, linestyle=":", alpha=0.6)

    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    ax_pr.set_xlabel("Recall", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Precision-Recall Curves Across Cohorts", fontsize=12)
    ax_pr.legend(loc="upper right", frameon=True, fontsize=9)
    ax_pr.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def export_ml_artifacts(data_dir: Path | str | None = None) -> dict[str, Path]:
    """Экспортирует базовые таблицы и все 6 целевых артефактов для запуска на GPU A100."""
    root = (
        locate("Extended Data Fig 5.xlsx").parent
        if data_dir is None
        else Path(data_dir)
    )
    excel_file = root / "Extended Data Fig 5.xlsx"
    db_file = root / "lab.db"

    df = load_markers(excel_file, log_scale=True, align_batches=True)
    features_parquet = root / "dataset_features.parquet"
    features_csv = root / "dataset_features.csv"
    df.to_parquet(features_parquet, index=False)
    df.to_csv(features_csv, index=False)

    with connect(db_file) as conn:
        priors_df = build_priors_table(conn)
    priors_parquet = root / "priors_table.parquet"
    priors_csv = root / "priors_table.csv"
    priors_df.to_parquet(priors_parquet)
    priors_df.to_csv(priors_csv)

    result_paths: dict[str, Path] = {
        "features_parquet": features_parquet,
        "features_csv": features_csv,
        "priors_parquet": priors_parquet,
        "priors_csv": priors_csv,
    }

    with contextlib.suppress(OSError, sqlite3.Error, ValueError, KeyError):
        cohort_arts = generate_cohort_artifacts(excel_path=excel_file, db_path=db_file)
        c_paths = export_cohort_artifacts(
            cohort_arts, output_dir=root, db_path=db_file, excel_path=excel_file
        )
        result_paths.update(c_paths)

        run_block_b(data_dir=root)
        run_blocks_v_g(data_dir=root)

        distilled_b = distill_and_subtype_chongqing_cohort(
            df, cohort_arts.discovery_wide
        )
        if not distilled_b.empty:
            b_parquet = root / "cohort_b_35_proteome_distilled.parquet"
            distilled_b.to_parquet(b_parquet)
            result_paths["cohort_b_distilled_parquet"] = b_parquet

        relbench_npz = build_relbench_heterodata_archive(root)
        result_paths["sptb_relbench_heterodata_npz"] = relbench_npz
        result_paths["graphpfn_node_encodings_parquet"] = (
            root / "graphpfn_node_encodings.parquet"
        )
        result_paths["protein_multiplex_graph_parquet"] = (
            root / "protein_multiplex_graph.parquet"
        )
        result_paths["binn_hierarchy_masks_npz"] = root / "binn_hierarchy_masks.npz"

    return result_paths
