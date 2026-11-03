"""Тесты базы данных."""

from lab.db import (
    build,
    clean,
    connect,
    enrich,
    populate_matrix,
    populate_transitions,
    query,
)


def test_clean() -> None:
    assert clean("Protein.Group") == "Protein_Group"
    assert clean("H:\\PTB.raw\\PTB_CQ_P1_53.raw") == "PTB_CQ_P1_53"


def test_populate_matrix(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    with connect(db_path) as conn:
        count = populate_matrix(conn)
        assert count == 2699
        df = query("SELECT COUNT(*) AS c FROM proteins", conn)
        assert df.iloc[0]["c"] == 2699


def test_populate_transitions_limit(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    with connect(db_path) as conn:
        count = populate_transitions(conn, limit=500, chunksize=250)
        assert count == 500
        df = query("SELECT COUNT(*) AS c FROM transitions", conn)
        assert df.iloc[0]["c"] == 500
        genes_df = query(
            "SELECT DISTINCT Genes FROM transitions WHERE Genes IS NOT NULL", conn
        )
        assert not genes_df.empty


def test_enrich(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    with connect(db_path) as conn:
        populate_matrix(conn)
        populate_transitions(conn, limit=1000)
        assert enrich(conn) >= 0
        df = query("SELECT COUNT(*) AS c FROM proteins WHERE Genes IS NOT NULL", conn)
        assert df.iloc[0]["c"] >= 0


def test_build_and_query_filter(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    n_prot, n_trans, _ = build(db_path, limit=200)
    assert n_prot == 2699
    assert n_trans == 200
    with connect(db_path) as conn:
        df = query("SELECT Protein_Group FROM proteins LIMIT 3", conn)
        assert len(df) == 3
