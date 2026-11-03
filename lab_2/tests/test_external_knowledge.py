"""Тесты модулей внешних знаний UniProt и STRING-DB (Блоки В и Г)."""

import sqlite3
from pathlib import Path

from lab.data import locate
from lab.external_knowledge import (
    CURATED_UNIPROT_KNOWLEDGE,
    build_full_priors_table,
    fetch_string_ppi_network,
    run_blocks_v_g,
)


def test_fetch_string_ppi_network():
    """Проверяет получение сети взаимодействий STRING-DB."""
    genes = ["LUM", "FN1", "TIMP1", "AMBP", "B2M", "COL1A2"]
    edges_df = fetch_string_ppi_network(genes)

    assert not edges_df.empty
    assert "source_gene" in edges_df.columns
    assert "target_gene" in edges_df.columns
    assert "combined_score" in edges_df.columns
    assert (edges_df["combined_score"] > 0.0).all()
    assert (edges_df["combined_score"] <= 1.0).all()


def test_build_full_priors_table():
    """Проверяет слияние дескрипторов UniProt с масс-спектрометрией."""
    with sqlite3.connect(locate("lab.db")) as conn:
        priors = build_full_priors_table(conn, list(CURATED_UNIPROT_KNOWLEDGE.keys()))

    assert not priors.empty
    assert "mass_kda" in priors.columns
    assert "length_aa" in priors.columns
    assert "is_secreted" in priors.columns
    assert "is_ecm" in priors.columns
    assert "prior_confidence_weight" in priors.columns

    for gene in ["LUM", "FN1", "AMBP", "TIMP1", "B2M"]:
        assert gene in priors.index
        assert priors.loc[gene, "mass_kda"] > 0


def test_run_blocks_v_g_pipeline(tmp_path: Path):
    """Проверяет полный цикл генерации артефактов Блоков В и Г."""
    artifacts = run_blocks_v_g()
    assert not artifacts.priors_df.empty
    assert not artifacts.ppi_edges_df.empty

    # Проверяем сохранение таблиц в lab.db
    with sqlite3.connect(locate("lab.db")) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM uniprot_priors;")
        assert cur.fetchone()[0] > 0

        cur.execute("SELECT COUNT(*) FROM string_ppi_edges;")
        assert cur.fetchone()[0] > 0
