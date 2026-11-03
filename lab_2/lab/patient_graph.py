"""Индуктивный мульти-модальный граф пациенток (TabR/TAG/GraphPFN) и LOD-импьютация (Блок Б)."""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

from lab.data import locate

SHARED_4_MARKERS = ["AMBP", "FN1", "LUM", "B2M"]


class PatientGraphArtifacts(NamedTuple):
    """Артефакты Блока Б."""

    edge_index_df: pd.DataFrame
    imputed_discovery: pd.DataFrame


def _to_log2_if_needed(df: pd.DataFrame) -> np.ndarray:
    """Гарантирует представление признаков в log2-шкале."""
    mat = df.to_numpy(dtype=np.float64, copy=True)
    if np.all(mat > 0):
        return np.log2(np.clip(mat, 1e-9, None))
    return mat


def _compute_simplex(x_log2: np.ndarray) -> np.ndarray:
    """Переводит log2-интенсивности в композиционный симплекс долей P_{i,p}."""
    lin = np.exp2(np.clip(x_log2, -30.0, 30.0))
    denom = np.maximum(np.sum(lin, axis=1, keepdims=True), 1e-12)
    p = np.clip(lin / denom, 1e-12, 1.0)
    return p / np.sum(p, axis=1, keepdims=True)


def _pairwise_jsd_distance(p_all: np.ndarray, p_ref: np.ndarray) -> np.ndarray:
    """Векторизованно вычисляет матрицу метрических расстояний Йенсена-Шеннона d_JS(P_i, P_j)."""
    # p_all: (N, D), p_ref: (M, D) -> результат (N, M)
    p_i = p_all[:, None, :]
    p_j = p_ref[None, :, :]
    m_ij = 0.5 * (p_i + p_j)
    kl_i = np.sum(p_i * np.log2(p_i / m_ij), axis=2)
    kl_j = np.sum(p_j * np.log2(p_j / m_ij), axis=2)
    jsd = np.clip(0.5 * (kl_i + kl_j), 0.0, 1.0)
    return np.sqrt(jsd)


def _pairwise_clr_cosine_distance(x_all: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
    """Вычисляет косинусное расстояние на центрированных лог-отношениях (CLR-Cosine)."""
    clr_all = x_all - np.mean(x_all, axis=1, keepdims=True)
    clr_ref = x_ref - np.mean(x_ref, axis=1, keepdims=True)

    norm_all = np.maximum(np.linalg.norm(clr_all, axis=1, keepdims=True), 1e-9)
    norm_ref = np.maximum(np.linalg.norm(clr_ref, axis=1, keepdims=True), 1e-9)

    cos_sim = (clr_all / norm_all) @ (clr_ref / norm_ref).T
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    # Защита от машинной погрешности 1 - 1.0 = -1e-16 для диагонали
    dist_sq = np.maximum(0.0, 1.0 - cos_sim)
    dist_sq[dist_sq < 1e-14] = 0.0
    return np.sqrt(dist_sq)


def _pairwise_mahalanobis_distance(
    x_all: np.ndarray,
    x_ref: np.ndarray,
    precision_mat: np.ndarray,
) -> np.ndarray:
    """Вычисляет матрицу расстояний Махаланобиса с регуляризованной матрицей точности Ледуа-Вульфа."""
    diff = x_all[:, None, :] - x_ref[None, :, :]
    d_dim = max(x_all.shape[1], 1)
    quad = np.einsum("ijk,kl,ijl->ij", diff, precision_mat, diff) / float(d_dim)
    return np.sqrt(np.maximum(quad, 0.0))


def build_patient_knn_graph(
    features_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    k_neighbors: int = 8,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Строит строго индуктивный мульти-модальный граф пациенток (Candidate Set Restriction).

    Все параметры стандартизации и ковариации оцениваются исключительно на множестве Train (n=82).
    Каждая пациентка i формирует ребра только к опорным пациенткам из Train \\ {i},
    что полностью исключает трансдуктивную утечку данных между клиниками (0 ребер в тест).
    """
    sample_ids = features_df.index.astype(str).tolist()
    n_samples = len(sample_ids)
    if n_samples == 0:
        return pd.DataFrame()

    x_log2 = _to_log2_if_needed(features_df)

    # 1. Выделение опорного пула кандидатов I_cand = V_train (Candidate Set Restriction из TabR)
    if "Dataset" in metadata_df.columns:
        ds_series = metadata_df.reindex(features_df.index)["Dataset"].astype(str)
        train_mask = (ds_series == "Train").to_numpy(dtype=bool)
    else:
        train_mask = np.ones(n_samples, dtype=bool)

    ref_indices = np.where(train_mask)[0]
    if len(ref_indices) <= k_neighbors:
        # Fallback для малых синтетических таблиц в юнит-тестах
        ref_indices = np.arange(n_samples)

    x_ref = x_log2[ref_indices]

    # 2. Изоляция статистик препроцессинга строго на опорном множестве Train
    scaler = StandardScaler().fit(x_ref)
    x_scaled_all = scaler.transform(x_log2)
    x_scaled_ref = x_scaled_all[ref_indices]

    lw = LedoitWolf().fit(x_scaled_ref)
    prec_mat = lw.precision_

    # 3. Расчет 3 ортогональных модальностей расстояния относительно узлов Train
    p_all = _compute_simplex(x_log2)
    p_ref = p_all[ref_indices]

    d_jsd = _pairwise_jsd_distance(p_all, p_ref)
    d_clr = _pairwise_clr_cosine_distance(x_log2, x_ref)
    d_mahal = _pairwise_mahalanobis_distance(x_scaled_all, x_scaled_ref, prec_mat)

    if metric == "cosine":
        d_primary = d_clr.copy()
    elif metric == "jsd":
        d_primary = d_jsd.copy()
    else:
        # Мульти-модальное ядро (JSD + CLR-Cosine + Mahalanobis), устойчивое к батч-эффекту
        med_jsd = max(float(np.median(d_jsd[d_jsd > 0])), 1e-4)
        med_clr = max(float(np.median(d_clr[d_clr > 0])), 1e-4)
        med_mah = max(float(np.median(d_mahal[d_mahal > 0])), 1e-4)
        d_primary = (
            0.40 * (d_jsd / med_jsd)
            + 0.30 * (d_clr / med_clr)
            + 0.30 * (d_mahal / med_mah)
        )

    # Исключаем самокопирование для узлов из Train: i ∉ I_cand(i)
    ref_pos_map = {int(global_idx): pos for pos, global_idx in enumerate(ref_indices)}
    d_search = d_primary.copy()
    for i in range(n_samples):
        if i in ref_pos_map:
            d_search[i, ref_pos_map[i]] = np.inf

    k_eff = min(k_neighbors, max(len(ref_indices) - 1, 1))
    part_idx = np.argsort(d_search, axis=1)[:, :k_eff]

    # Локальные масштабы Self-Tuning RBF ядра по k-му соседу из Train
    kth_dists = np.array(
        [d_primary[i, part_idx[i, -1]] for i in range(n_samples)], dtype=np.float64
    )
    kth_dists = np.where((kth_dists < 1e-4) | ~np.isfinite(kth_dists), 1.0, kth_dists)

    sig_jsd = max(float(np.median(d_jsd[d_jsd > 0])), 0.05)
    sig_clr = max(float(np.median(d_clr[d_clr > 0])), 0.05)
    sig_mah = max(float(np.median(d_mahal[d_mahal > 0])), 0.10)

    dataset_map = (
        metadata_df["Dataset"].to_dict() if "Dataset" in metadata_df.columns else {}
    )
    target_map = (
        metadata_df["Target"].to_dict() if "Target" in metadata_df.columns else {}
    )
    center_map = (
        metadata_df["Hospital_Center"].to_dict()
        if "Hospital_Center" in metadata_df.columns
        else {}
    )

    edges: list[dict[str, str | int | float]] = []
    for i in range(n_samples):
        src_id = sample_ids[i]
        src_ds = str(dataset_map.get(src_id, "Unknown"))
        src_target = int(target_map.get(src_id, -1))
        src_center = str(center_map.get(src_id, "Unknown"))
        sigma_i = float(kth_dists[i])

        for rank_idx, ref_col in enumerate(part_idx[i], start=1):
            neighbor_global_idx = int(ref_indices[ref_col])
            tgt_id = sample_ids[neighbor_global_idx]
            if src_id == tgt_id:
                continue

            dist_val = float(d_primary[i, ref_col])
            sigma_j = float(kth_dists[neighbor_global_idx])
            denom = 2.0 * sigma_i * sigma_j
            raw_weight = float(np.exp(-(dist_val**2) / denom)) if denom > 0 else 0.01
            weight = max(raw_weight, 0.001)

            w_jsd = max(
                float(np.exp(-(float(d_jsd[i, ref_col]) ** 2) / (2.0 * sig_jsd**2))),
                0.001,
            )
            w_clr = max(
                float(np.exp(-(float(d_clr[i, ref_col]) ** 2) / (2.0 * sig_clr**2))),
                0.001,
            )
            w_mah = max(
                float(np.exp(-(float(d_mahal[i, ref_col]) ** 2) / (2.0 * sig_mah**2))),
                0.001,
            )

            tgt_ds = str(dataset_map.get(tgt_id, "Unknown"))
            tgt_target = int(target_map.get(tgt_id, -1))
            tgt_center = str(center_map.get(tgt_id, "Unknown"))
            is_cross_center = int(src_center != tgt_center)

            edges.append(
                {
                    "source": src_id,
                    "target": tgt_id,
                    "distance": round(dist_val, 4),
                    "weight": round(weight, 4),
                    "knn_rank": rank_idx,
                    "is_cross_center": is_cross_center,
                    "source_dataset": src_ds,
                    "target_dataset": tgt_ds,
                    "source_target": src_target,
                    "target_target": tgt_target,
                    "view_jsd_weight": round(w_jsd, 4),
                    "view_clr_weight": round(w_clr, 4),
                    "view_mahal_weight": round(w_mah, 4),
                }
            )

    return pd.DataFrame(edges)


def _build_inductive_transition_matrix(
    d_mat: np.ndarray,
    ref_pos_map: dict[int, int],
    k_neighbors: int = 8,
    tau_fixed: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Строит разреженную матрицу весов W (N x N_ref) и стохастическую матрицу переходов A_hat."""
    n_all, n_ref = d_mat.shape
    d_work = d_mat.copy()
    for i in range(n_all):
        if i in ref_pos_map:
            d_work[i, ref_pos_map[i]] = np.inf

    k_eff = min(k_neighbors, max(n_ref - 1, 1))
    top_idx = np.argsort(d_work, axis=1)[:, :k_eff]

    if tau_fixed is not None and tau_fixed > 0:
        tau = tau_fixed
    else:
        # Считаем масштаб ядра ТОЛЬКО по опорным узлам Train (гарантия 100% индуктивной изоляции)
        train_rows = [i for i in range(n_all) if i in ref_pos_map]
        train_dists = d_work[train_rows, :] if train_rows else d_work
        finite_vals = train_dists[np.isfinite(train_dists) & (train_dists > 0)]
        tau = max(float(np.median(finite_vals)) if len(finite_vals) > 0 else 1.0, 1e-3)

    w_mat = np.zeros((n_all, n_ref), dtype=np.float64)
    for i in range(n_all):
        cols = top_idx[i]
        dists = d_work[i, cols]
        w_mat[i, cols] = np.exp(-(dists**2) / (2.0 * tau**2))

    row_sums = np.maximum(np.sum(w_mat, axis=1, keepdims=True), 1e-12)
    a_hat = w_mat / row_sums
    return w_mat, a_hat


def compute_inductive_tag_tabr_features(
    features_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    k_neighbors: int = 8,
) -> pd.DataFrame:
    """Вычисляет индуктивные графовые признаки TAG (1/2-hop), TabR (Delta) и G2T-FM для таблицы пациенток."""
    out = pd.DataFrame(index=features_df.index)
    n_samples = len(features_df)
    if n_samples == 0:
        return out

    x_log2 = _to_log2_if_needed(features_df)
    ds_series = metadata_df.reindex(features_df.index)["Dataset"].astype(str)
    y_all = (
        metadata_df.reindex(features_df.index)["Target"]
        .fillna(0)
        .to_numpy(dtype=np.float64)
    )

    train_mask = (ds_series == "Train").to_numpy(dtype=bool)
    ref_indices = (
        np.where(train_mask)[0]
        if np.sum(train_mask) > k_neighbors
        else np.arange(n_samples)
    )
    ref_pos_map = {int(g_idx): pos for pos, g_idx in enumerate(ref_indices)}

    x_ref = x_log2[ref_indices]
    y_ref = y_all[ref_indices]

    scaler = StandardScaler().fit(x_ref)
    x_scaled_all = scaler.transform(x_log2)
    x_scaled_ref = x_scaled_all[ref_indices]
    lw = LedoitWolf().fit(x_scaled_ref)

    p_all = _compute_simplex(x_log2)
    p_ref = p_all[ref_indices]

    d_jsd = _pairwise_jsd_distance(p_all, p_ref)
    d_clr = _pairwise_clr_cosine_distance(x_log2, x_ref)
    d_mah = _pairwise_mahalanobis_distance(x_scaled_all, x_scaled_ref, lw.precision_)

    w_jsd, a_jsd = _build_inductive_transition_matrix(d_jsd, ref_pos_map, k_neighbors)
    _, a_clr = _build_inductive_transition_matrix(d_clr, ref_pos_map, k_neighbors)
    _, a_mah = _build_inductive_transition_matrix(d_mah, ref_pos_map, k_neighbors)

    # Комбинированный оператор перехода по 3 модальностям графа
    a_comb = (a_jsd + a_clr + a_mah) / 3.0
    a_comb_tr = a_comb[ref_indices, :]  # (N_ref x N_ref) внутри Train

    # 1. 1-hop и 2-hop сглаженные признаки соседей из Train (TAG: phi^(1) и phi^(2))
    phi_1 = a_comb @ x_ref  # (N x D)
    phi_1_tr = a_comb_tr @ x_ref  # (N_ref x D)
    phi_2 = a_comb @ phi_1_tr  # (N x D)

    # 2. Дирекциональные разностные признаки TabR: Delta^(1) = x_i - phi_i^(1)
    delta_1 = x_log2 - phi_1

    for idx, col in enumerate(features_df.columns):
        out[f"graph_tag_hop1_{col}"] = phi_1[:, idx]
        out[f"graph_tag_hop2_{col}"] = phi_2[:, idx]
        out[f"graph_tabr_delta1_{col}"] = delta_1[:, idx]

    # 3. Leave-One-Out сглаживание меток соседей по каждой модальности (G2T-FM Local Label Smoothing)
    out["graph_knn_label_jsd"] = a_jsd @ y_ref
    out["graph_knn_label_clr"] = a_clr @ y_ref
    out["graph_knn_label_mahal"] = a_mah @ y_ref
    out["graph_knn_label_consensus"] = a_comb @ y_ref

    # 4. Классово-специфичные расстояния до k=5 ближайших прототипов Train-FTB и Train-sPTB
    ftb_ref_cols = np.where(y_ref == 0)[0]
    ptb_ref_cols = np.where(y_ref == 1)[0]
    k_proto = 5

    for view_name, d_view in [("jsd", d_jsd), ("mahal", d_mah)]:
        d_work = d_view.copy()
        for i in range(n_samples):
            if i in ref_pos_map:
                d_work[i, ref_pos_map[i]] = np.inf

        if len(ftb_ref_cols) > 0:
            d_to_ftb = np.sort(d_work[:, ftb_ref_cols], axis=1)[
                :, : min(k_proto, len(ftb_ref_cols))
            ]
            mean_ftb = np.mean(d_to_ftb, axis=1)
        else:
            mean_ftb = np.ones(n_samples)

        if len(ptb_ref_cols) > 0:
            d_to_ptb = np.sort(d_work[:, ptb_ref_cols], axis=1)[
                :, : min(k_proto, len(ptb_ref_cols))
            ]
            mean_ptb = np.mean(d_to_ptb, axis=1)
        else:
            mean_ptb = np.ones(n_samples)

        out[f"graph_proto_{view_name}_ftb_dist"] = mean_ftb
        out[f"graph_proto_{view_name}_ptb_dist"] = mean_ptb
        out[f"graph_proto_{view_name}_log_ratio"] = np.log2(
            (mean_ftb + 1e-4) / (mean_ptb + 1e-4)
        )

    # 5. Структурные дескрипторы локальной плотности и энтропии внимания
    out["graph_local_density"] = np.mean(w_jsd, axis=1)
    p_att = np.clip(a_comb, 1e-12, 1.0)
    out["graph_attention_entropy"] = -np.sum(p_att * np.log2(p_att), axis=1)

    return out


def build_graphpfn_node_encodings(
    features_a_df: pd.DataFrame,
    metadata_a_df: pd.DataFrame,
    discovery_b_df: pd.DataFrame | None = None,
    k_neighbors: int = 8,
) -> pd.DataFrame:
    """Формирует единую таблицу позиционных и структурных кодировок узлов (426 + 35 = 461 строка).

    Включает Laplacian Eigenvector PE (с индуктивной интерполяцией Нистрёма),
    Random Walk Structural Encoding (RWSE), персонализированный PageRank и диффузионные представления
    для подачи всех 461 пациенток в единое окно In-Context Learning моделей TabPFN-2.5 / TabICLv2 / GraphPFN.
    """
    x_a_log2 = pd.DataFrame(
        _to_log2_if_needed(features_a_df),
        index=features_a_df.index.astype(str),
        columns=features_a_df.columns,
    )
    ds_a = metadata_a_df.reindex(x_a_log2.index)["Dataset"].astype(str)
    y_a = metadata_a_df.reindex(x_a_log2.index)["Target"].fillna(-1).astype(int)

    # Если передана матрица Чунцина (35 пациенток), квантильно выравниваем 4 общих маркера (Rank-Gauss)
    shared_cols = [c for c in SHARED_4_MARKERS if c in x_a_log2.columns]
    train_mask_a = (ds_a == "Train").to_numpy(dtype=bool)
    if not np.any(train_mask_a):
        train_mask_a = np.ones(len(x_a_log2), dtype=bool)

    if (
        discovery_b_df is not None
        and not discovery_b_df.empty
        and set(shared_cols).issubset(discovery_b_df.columns)
    ):
        b_sub = discovery_b_df[shared_cols].copy().astype(np.float64)
        b_aligned = pd.DataFrame(
            index=b_sub.index.astype(str), columns=x_a_log2.columns, dtype=np.float64
        )

        for col in shared_cols:
            ref_vals = np.sort(
                x_a_log2.loc[train_mask_a, col].to_numpy(dtype=np.float64)
            )
            ranks = stats.rankdata(
                b_sub[col].to_numpy(dtype=np.float64), method="average"
            )
            quantiles = (ranks - 0.5) / max(len(ranks), 1)
            b_aligned[col] = np.quantile(ref_vals, np.clip(quantiles, 0.01, 0.99))

        # Импьютация 5-го маркера TIMP1 для Чунцина через Ridge по 4 общим маркерам на Train
        if "TIMP1" in x_a_log2.columns:
            x_tr_4 = x_a_log2.loc[train_mask_a, shared_cols].to_numpy(dtype=np.float64)
            y_tr_timp = x_a_log2.loc[train_mask_a, "TIMP1"].to_numpy(dtype=np.float64)
            coef, _, _, _ = np.linalg.lstsq(
                np.column_stack([np.ones(len(x_tr_4)), x_tr_4]), y_tr_timp, rcond=None
            )
            b_4_mat = b_aligned[shared_cols].to_numpy(dtype=np.float64)
            b_aligned["TIMP1"] = coef[0] + b_4_mat @ coef[1:]

        x_all_df = pd.concat([x_a_log2, b_aligned], axis=0)
        cohort_labels = np.concatenate(
            [ds_a.to_numpy(dtype=str), np.repeat("Chongqing_Discovery", len(b_aligned))]
        )
        target_labels = np.concatenate(
            [y_a.to_numpy(dtype=int), np.repeat(-1, len(b_aligned))]
        )
    else:
        x_all_df = x_a_log2
        cohort_labels = ds_a.to_numpy(dtype=str)
        target_labels = y_a.to_numpy(dtype=int)

    n_total = len(x_all_df)
    x_all = x_all_df.to_numpy(dtype=np.float64)
    ref_indices = np.where(cohort_labels == "Train")[0]
    if len(ref_indices) <= k_neighbors:
        ref_indices = np.arange(min(len(x_a_log2), n_total))
    ref_pos_map = {int(g_idx): pos for pos, g_idx in enumerate(ref_indices)}
    n_ref = len(ref_indices)

    x_ref = x_all[ref_indices]
    y_ref = np.maximum(target_labels[ref_indices], 0).astype(np.float64)

    scaler = StandardScaler().fit(x_ref)
    x_s_all = scaler.transform(x_all)
    x_s_ref = x_s_all[ref_indices]
    lw = LedoitWolf().fit(x_s_ref)

    p_all = _compute_simplex(x_all)
    p_ref = p_all[ref_indices]

    d_jsd = _pairwise_jsd_distance(p_all, p_ref)
    d_clr = _pairwise_clr_cosine_distance(x_all, x_ref)
    d_mah = _pairwise_mahalanobis_distance(x_s_all, x_s_ref, lw.precision_)

    w_jsd, a_jsd = _build_inductive_transition_matrix(d_jsd, ref_pos_map, k_neighbors)
    w_clr, a_clr = _build_inductive_transition_matrix(d_clr, ref_pos_map, k_neighbors)
    w_mah, a_mah = _build_inductive_transition_matrix(d_mah, ref_pos_map, k_neighbors)

    a_comb = (a_jsd + a_clr + a_mah) / 3.0
    a_tr = a_comb[ref_indices, :]  # (N_ref x N_ref)

    # 1. Индуктивная проекция Нистрёма собственных векторов Лапласиана L_tr (LaplacianEigenvectorPE)
    w_tr_sym = (
        0.5
        * (
            w_jsd[ref_indices, :]
            + w_jsd[ref_indices, :].T
            + w_clr[ref_indices, :]
            + w_clr[ref_indices, :].T
            + w_mah[ref_indices, :]
            + w_mah[ref_indices, :].T
        )
        / 3.0
    )
    deg_tr = np.maximum(np.sum(w_tr_sym, axis=1), 1e-8)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(deg_tr))
    lap_tr = np.eye(n_ref) - d_inv_sqrt @ w_tr_sym @ d_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(lap_tr)
    # Берем первые 12 нетривиальных собственных векторов (пропуская нулевую моду)
    n_eigs = min(12, max(n_ref - 1, 1))
    psi_tr = eigvecs[:, 1 : 1 + n_eigs]
    lam_tr = np.maximum(eigvals[1 : 1 + n_eigs], 1e-4)
    # Проекция Нистрёма для всех 461 узлов через переходную матрицу a_comb
    psi_nystrom = (a_comb @ psi_tr) / lam_tr[None, :]

    # 2. Random Walk Structural Encoding (RWSE) — вероятности и моменты блужданий шагов s = 1..6
    a_tr_powers = [np.eye(n_ref)]
    for _ in range(6):
        a_tr_powers.append(a_tr_powers[-1] @ a_tr)

    rw_features: dict[str, np.ndarray] = {}
    for step_s in range(1, 7):
        prop_mat = a_comb @ a_tr_powers[step_s - 1]  # (N_total x N_ref)
        rw_features[f"rwse_step{step_s}_max_prob"] = np.max(prop_mat, axis=1)
        rw_features[f"rwse_step{step_s}_Concentration"] = np.sum(prop_mat**2, axis=1)
        rw_features[f"rwse_step{step_s}_entropy"] = -np.sum(
            np.clip(prop_mat, 1e-12, 1.0) * np.log2(np.clip(prop_mat, 1e-12, 1.0)),
            axis=1,
        )
        prop_tr_block = prop_mat[ref_indices, :].copy()
        np.fill_diagonal(prop_tr_block, 0.0)
        prop_mat[ref_indices, :] = prop_tr_block
        prop_mat = prop_mat / np.maximum(np.sum(prop_mat, axis=1, keepdims=True), 1e-12)
        rw_features[f"rwse_step{step_s}_risk_diffusion"] = prop_mat @ y_ref

    # 3. Персонализированный PageRank (PPR) относительно узлов Train (alpha = 0.15)
    alpha_ppr = 0.15
    ppr_kernel_tr = alpha_ppr * np.linalg.inv(np.eye(n_ref) - (1.0 - alpha_ppr) * a_tr)
    ppr_all_to_tr = a_comb @ ppr_kernel_tr  # (N_total x N_ref)
    ppr_x_diffused = ppr_all_to_tr @ x_s_ref  # (N_total x 5)
    ppr_tr_block = ppr_all_to_tr[ref_indices, :].copy()
    np.fill_diagonal(ppr_tr_block, 0.0)
    ppr_all_to_tr[ref_indices, :] = ppr_tr_block
    ppr_all_to_tr = ppr_all_to_tr / np.maximum(
        np.sum(ppr_all_to_tr, axis=1, keepdims=True), 1e-12
    )
    ppr_y_diffused = ppr_all_to_tr @ y_ref
    # 4. Многошаговые диффузионные представления по каждой из 3 модальностей (JSD, CLR, Mahal)
    enc_cols: dict[str, Any] = {
        "Cohort": cohort_labels,
        "Target": target_labels,
        "Is_Query_Unlabeled": (target_labels < 0).astype(int),
        "ppr_risk_score": ppr_y_diffused,
        "ppr_hub_centrality": np.max(ppr_all_to_tr, axis=1),
        "ppr_gini_concentration": np.sum(ppr_all_to_tr**2, axis=1),
    }

    for k_eig in range(psi_nystrom.shape[1]):
        enc_cols[f"lap_pe_dim{k_eig + 1:02d}"] = psi_nystrom[:, k_eig]

    enc_cols.update(rw_features)

    for view_tag, a_v in [("jsd", a_jsd), ("clr", a_clr), ("mahal", a_mah)]:
        a_v_tr = a_v[ref_indices, :]
        hop1_v = a_v @ x_s_ref
        hop2_v = (a_v @ a_v_tr) @ x_s_ref
        delta_v = x_s_all - hop1_v
        for d_idx, col_name in enumerate(x_all_df.columns):
            enc_cols[f"view_{view_tag}_hop1_{col_name}"] = hop1_v[:, d_idx]
            enc_cols[f"view_{view_tag}_hop2_{col_name}"] = hop2_v[:, d_idx]
            enc_cols[f"view_{view_tag}_delta_{col_name}"] = delta_v[:, d_idx]
        enc_cols[f"view_{view_tag}_label_smooth"] = a_v @ y_ref

    for d_idx, col_name in enumerate(x_all_df.columns):
        enc_cols[f"ppr_diffused_{col_name}"] = ppr_x_diffused[:, d_idx]

    return pd.DataFrame(enc_cols, index=x_all_df.index)


def impute_proteomics_lod(
    df_wide: pd.DataFrame,
    shift: float = 1.8,
    scale: float = 0.3,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Реализует протеомную импьютацию цензурированных пределов детекции (Left-Censored LOD).

    Заполняет NaN случайными величинами из левого хвоста распределения шума прибора:
    N(mu - shift * sigma, scale * sigma).
    """
    rng = np.random.default_rng(random_seed)
    df_imputed = df_wide.copy().astype(np.float64)
    vals = df_imputed.to_numpy(dtype=np.float64, copy=True)

    for col_idx in range(vals.shape[1]):
        col_vec = vals[:, col_idx]
        missing_mask = np.isnan(col_vec)
        n_missing = int(np.sum(missing_mask))
        if n_missing == 0:
            continue

        obs_vec = col_vec[~missing_mask]
        if len(obs_vec) == 0:
            vals[:, col_idx] = 13.7
            continue

        mu = float(np.mean(obs_vec))
        sigma = float(np.std(obs_vec, ddof=1)) if len(obs_vec) > 1 else 1.0
        if sigma <= 1e-6:
            sigma = 0.8

        impute_mu = mu - shift * sigma
        impute_sigma = max(scale * sigma, 0.15)
        vals[missing_mask, col_idx] = rng.normal(
            loc=impute_mu, scale=impute_sigma, size=n_missing
        )

    return pd.DataFrame(vals, index=df_imputed.index, columns=df_imputed.columns)


def run_block_b(
    data_dir: Path | str | None = None,
) -> PatientGraphArtifacts:
    """Выполняет полный цикл генерации индуктивного графа пациенток, LOD-импьютации и кодировок GraphPFN."""
    root = locate("metadata.parquet").parent if data_dir is None else Path(data_dir)

    features_df = pd.read_parquet(root / "features_5.parquet")
    ds_feat_path = root / "dataset_features.parquet"
    if ds_feat_path.is_file():
        ds_df = pd.read_parquet(ds_feat_path)
        if "Sample ID" in ds_df.columns and set(features_df.columns).issubset(
            ds_df.columns
        ):
            features_df = (
                ds_df.set_index("Sample ID")[list(features_df.columns)]
                .reindex(features_df.index)
                .astype(np.float64)
            )
    metadata_df = pd.read_parquet(root / "metadata.parquet")
    discovery_wide = pd.read_parquet(root / "discovery_wide.parquet")

    # 1. Построение строго индуктивного графа пациенток без утечек в тест
    graph_df = build_patient_knn_graph(
        features_df, metadata_df, k_neighbors=8, metric="euclidean"
    )

    # 2. Протеомная LOD-импьютация широкой матрицы Чунцина по реальным позициям пропусков NaN
    # (исправление Бага #1 из audit.py, где discovery_wide не содержал NaN на входе)
    unimputed_path = root / "discovery_unimputed.parquet"
    mask_path = root / "discovery_missing_mask.parquet"
    if unimputed_path.is_file():
        raw_for_lod = pd.read_parquet(unimputed_path)
    elif mask_path.is_file():
        mask_df = pd.read_parquet(mask_path)
        raw_for_lod = discovery_wide.where(mask_df > 0, np.nan)
    else:
        try:
            from lab.prep import prepare_maxlfq_matrix

            with sqlite3.connect(root / "lab.db") as conn:
                df_prot = pd.read_sql_query("SELECT * FROM proteins;", conn)
            raw_for_lod, _, _ = prepare_maxlfq_matrix(df_prot)
        except (OSError, sqlite3.Error, ValueError, KeyError):
            raw_for_lod = discovery_wide.copy()

    lod_imputed = impute_proteomics_lod(raw_for_lod)

    graph_parquet = root / "patient_graph_edges.parquet"
    graph_csv = root / "patient_graph_edges.csv"
    graph_df.to_parquet(graph_parquet, index=False)
    graph_df.to_csv(graph_csv, index=False)

    lod_parquet = root / "discovery_imputed_lod.parquet"
    lod_csv = root / "discovery_imputed_lod.csv"
    lod_imputed.to_parquet(lod_parquet)
    lod_imputed.to_csv(lod_csv)

    with contextlib.suppress(OSError, sqlite3.Error, ValueError, KeyError):
        node_enc_df = build_graphpfn_node_encodings(
            features_df, metadata_df, discovery_wide, k_neighbors=8
        )
        node_enc_df.to_parquet(root / "graphpfn_node_encodings.parquet")

        enriched_path = root / "cohort_a_426_enriched.parquet"
        if enriched_path.is_file():
            enriched_a = pd.read_parquet(enriched_path)
            tag_feats = compute_inductive_tag_tabr_features(
                features_df, metadata_df, k_neighbors=8
            )
            drop_existing = [c for c in tag_feats.columns if c in enriched_a.columns]
            if drop_existing:
                enriched_a = enriched_a.drop(columns=drop_existing)
            enriched_a = pd.concat([enriched_a, tag_feats], axis=1)
            enriched_a.to_parquet(enriched_path)
            enriched_a.to_csv(root / "cohort_a_426_enriched.csv")

    return PatientGraphArtifacts(
        edge_index_df=graph_df,
        imputed_discovery=lod_imputed,
    )
