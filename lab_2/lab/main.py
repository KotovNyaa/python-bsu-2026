"""Точка входа полного аналитического, статистического и ML-пайплайна."""

import pandas as pd

from lab.db import connect, enrich, populate_clinical_cohort, populate_matrix
from lab.model import (
    evaluate_all_cohorts,
    evaluate_all_cohorts_with_prior_shift,
    export_ml_artifacts,
    plot_roc_pr,
    train_ensemble_model,
    train_logistic_regression,
)
from lab.prep import load_markers
from lab.priors import build_priors_table
from lab.sql import (
    decoy_stats,
    group_trans_by_gene,
    join_coverage,
    select_markers,
    stratify_fdr_tiers,
    top_transitions,
)
from lab.stats import (
    aggregate_by_groups,
    compare_normality_raw_vs_log2,
    describe,
    describe_categorical,
    diff,
)
from lab.viz import (
    box,
    categorical_bars,
    corr,
    decoy,
    hist_distributions,
    volcano,
)


def run() -> dict[str, pd.DataFrame]:
    """Выполняет полный цикл анализа, генерации графиков, ML-валидации и экспорта артефактов A100."""
    df_markers = load_markers(log_scale=True, align_batches=True)
    markers = ["AMBP", "FN1", "LUM", "TIMP1", "B2M"]

    stats_df = describe(df_markers[markers])
    normality_df = compare_normality_raw_vs_log2(df_markers, markers)
    cat_df = describe_categorical(df_markers)
    grouped_df = aggregate_by_groups(df_markers, markers)
    diff_df = diff(df_markers, markers)

    hist_distributions(df_markers, markers)
    categorical_bars(df_markers)
    box(df_markers, markers)
    corr(df_markers, markers)
    volcano(diff_df)

    with connect() as conn:
        prot_info = pd.read_sql_query("PRAGMA table_info(proteins)", conn)
        if prot_info.empty or not (prot_info["pk"] > 0).any():
            populate_matrix(conn)
            enrich(conn)
        populate_clinical_cohort(conn)

        info_df = pd.read_sql_query("PRAGMA table_info(transitions)", conn)

        t_cols = set(info_df["name"].astype(str).tolist())
        q_col_sql = ", QValue" if "QValue" in t_cols else ""
        df_trans = pd.read_sql_query(
            f"SELECT LibraryIntensity, decoy{q_col_sql} FROM transitions LIMIT 100000",
            conn,
        )
        decoy(df_trans)
        priors_df = build_priors_table(conn, markers)
        sql_res = {
            "markers": select_markers(conn),
            "top_genes": group_trans_by_gene(conn),
            "decoy": decoy_stats(conn),
            "top_trans": top_transitions(conn),
            "coverage": join_coverage(conn),
            "fdr_tiers": stratify_fdr_tiers(conn),
        }

    train_df = df_markers[df_markers["Dataset"] == "Train"]
    lr_model = train_logistic_regression(train_df, markers)
    rf_model = train_ensemble_model(train_df, markers)

    eval_lr = evaluate_all_cohorts(lr_model, df_markers, markers)
    eval_rf = evaluate_all_cohorts(rf_model, df_markers, markers)
    eval_prior_shift = evaluate_all_cohorts_with_prior_shift(
        lr_model, df_markers, markers
    )

    plot_roc_pr(lr_model, df_markers, markers)

    artifacts = export_ml_artifacts()

    return {
        "stats": stats_df,
        "normality_comparison": normality_df,
        "categorical_summary": cat_df,
        "group_aggregations": grouped_df,
        "diff": diff_df,
        "priors": priors_df,
        "eval_logistic_regression": eval_lr,
        "eval_random_forest": eval_rf,
        "eval_prior_shift_adjusted": eval_prior_shift,
        "artifacts": pd.DataFrame(
            [{"name": k, "path": str(v)} for k, v in artifacts.items()]
        ),
        **sql_res,
    }


if __name__ == "__main__":
    res = run()
    for k, v in res.items():
        print(f"=== {k.upper()} ===")
        print(v)
        print()
