"""Тесты загрузчика данных."""

import pandas as pd
import pytest

from lab.data import (
    MATRIX_NAME,
    TRANSITIONS_NAME,
    load_matrix,
    load_transitions,
    locate,
)
from lab.errors import LoadError, SchemaError


def test_locate_existing() -> None:
    assert locate(MATRIX_NAME).is_file()
    assert locate(TRANSITIONS_NAME).is_file()


def test_locate_missing() -> None:
    with pytest.raises(LoadError):
        locate("unknown.tsv")


def test_load_matrix_structure() -> None:
    df = load_matrix()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Protein.Group" in df.columns
    assert len(df) == 2699


def test_load_matrix_invalid(tmp_path) -> None:
    bad = tmp_path / "bad.tsv"
    bad.write_text("col1\tcol2\n1\t2\n")
    with pytest.raises(SchemaError):
        load_matrix(bad)


def test_load_transitions_slice() -> None:
    df = load_transitions(nrows=5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert "PeptideSequence" in df.columns
    assert "PrecursorMz" in df.columns


def test_load_transitions_chunked() -> None:
    reader = load_transitions(chunksize=10)
    chunk = next(iter(reader))
    assert len(chunk) == 10
