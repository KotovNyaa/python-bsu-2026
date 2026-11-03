"""Загрузка и валидация таблиц."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from lab.errors import LoadError, SchemaError

ROOT = Path(__file__).resolve().parent.parent / "data"
MATRIX_NAME = "PTB_CQ_P1_53_96report-first-pass.pg_matrix.tsv"
TRANSITIONS_NAME = "20240105.tsv"


def locate(name: str) -> Path:
    """Возвращает путь к файлу."""
    path = ROOT / name
    if not path.is_file():
        raise LoadError(f"Файл не найден: {path}")
    return path


def load_matrix(path: Path | str | None = None) -> pd.DataFrame:
    """Загружает матрицу экспрессии белков."""
    target = locate(MATRIX_NAME) if path is None else Path(path)
    if not target.is_file():
        raise LoadError(f"Файл не найден: {target}")
    df = pd.read_csv(target, sep="\t", low_memory=False)
    if "Protein.Group" not in df.columns:
        raise SchemaError("Отсутствует колонка Protein.Group")
    return df


def load_transitions(
    path: Path | str | None = None,
    nrows: int | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Загружает спектральные переходы."""
    target = locate(TRANSITIONS_NAME) if path is None else Path(path)
    if not target.is_file():
        raise LoadError(f"Файл не найден: {target}")
    return pd.read_csv(
        target,
        sep="\t",
        nrows=nrows,
        chunksize=chunksize,
        low_memory=False,
    )
