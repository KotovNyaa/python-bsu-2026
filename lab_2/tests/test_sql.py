"""Тесты аналитических SQL-запросов."""

from lab.db import connect
from lab.sql import (
    decoy_stats,
    group_trans_by_gene,
    join_coverage,
    select_markers,
    top_transitions,
)


def test_select_markers() -> None:
    with connect() as conn:
        df = select_markers(conn)
        assert len(df) == 4
        assert set(df["Genes"].tolist()) == {"AMBP", "B2M", "FN1", "LUM"}


def test_group_trans_by_gene() -> None:
    with connect() as conn:
        df = group_trans_by_gene(conn, limit=5)
        assert len(df) <= 5
        assert "trans_count" in df.columns
        assert df.iloc[0]["trans_count"] > 10


def test_decoy_stats() -> None:
    with connect() as conn:
        df = decoy_stats(conn)
        assert not df.empty
        assert "decoy" in df.columns


def test_top_transitions() -> None:
    with connect() as conn:
        df = top_transitions(conn, n=2)
        assert not df.empty
        assert set(df["rn"].unique()).issubset({1, 2})


def test_join_coverage() -> None:
    with connect() as conn:
        df = join_coverage(conn)
        assert len(df) == 4
        assert "unique_peptides" in df.columns
