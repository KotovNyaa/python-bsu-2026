"""Тесты модуля априорных знаний (priors)."""

from lab.db import connect
from lab.priors import (
    BIOPHYSICAL_PRIORS,
    build_priors_table,
    fetch_uniprot_online,
    get_biophysical_priors,
    get_ms_instrument_priors,
)


def test_get_biophysical_priors() -> None:
    df = get_biophysical_priors(["LUM", "FN1", "UNKNOWN_GENE"])
    assert len(df) == 3
    assert df.loc["LUM", "uniprot_id"] == "P51884"
    assert df.loc["LUM", "is_secreted"] == 1
    assert df.loc["FN1", "mass_kda"] > 200.0
    assert df.loc["UNKNOWN_GENE", "is_secreted"] == 0


def test_get_ms_instrument_priors() -> None:
    with connect() as conn:
        df = get_ms_instrument_priors(conn, ["AMBP", "FN1", "LUM", "B2M"])
        assert not df.empty
        assert "n_unique_peptides" in df.columns
        assert df.loc["LUM", "n_unique_peptides"] > 0


def test_build_priors_table() -> None:
    with connect() as conn:
        df = build_priors_table(conn)
        assert len(df) == len(BIOPHYSICAL_PRIORS)
        assert "prior_confidence_weight" in df.columns
        assert df.loc["LUM", "prior_confidence_weight"] >= 0.7


def test_fetch_uniprot_online() -> None:
    res = fetch_uniprot_online("P61769", timeout=3)
    if res is not None:
        assert res["uniprot_id"] == "P61769"
        assert res["length_aa"] > 100
