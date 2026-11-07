"""Управление реляционной базой данных SQLite со строгой схемой."""

import contextlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from lab.data import TRANSITIONS_NAME, load_matrix, load_transitions, locate
from lab.prep import CORE_MARKERS, load_markers

DB_NAME = "lab.db"


def clean(name: str) -> str:
    """Очищает имя колонки для совместимости с SQLite."""
    return (
        name.replace("\\", "/")
        .rsplit("/", 1)[-1]
        .removesuffix(".raw")
        .replace(".", "_")
    )


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Создает оптимизированное соединение с базой данных SQLite."""
    conn = sqlite3.connect(locate(DB_NAME) if path is None else Path(path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def populate_matrix(conn: sqlite3.Connection, path: Path | str | None = None) -> int:
    """Создает таблицу белков со строгой схемой (PRIMARY KEY) и заполняет её."""
    df = load_matrix(path)
    df.columns = [clean(c) for c in df.columns]

    df = df.drop_duplicates(subset=["Protein_Group"], keep="first").copy()

    sample_cols = [c for c in df.columns if c.startswith("PTB_")]
    meta_cols = [c for c in df.columns if c not in sample_cols]

    col_defs: list[str] = []
    for col in meta_cols:
        if col == "Protein_Group":
            col_defs.append(f'"{col}" TEXT PRIMARY KEY')
        else:
            col_defs.append(f'"{col}" TEXT')
    for col in sample_cols:
        col_defs.append(f'"{col}" REAL')

    conn.execute("DROP TABLE IF EXISTS proteins")
    conn.execute(f"CREATE TABLE proteins ({', '.join(col_defs)})")

    df.to_sql("proteins", conn, if_exists="append", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prot_group ON proteins(Protein_Group)")
    return len(df)


def populate_clinical_cohort(
    conn: sqlite3.Connection,
    excel_path: Path | str | None = None,
) -> int:
    """Создает и заполняет таблицу клинических когорт со строгим первичным ключом Sample_ID."""
    df = load_markers(excel_path, log_scale=True)
    df_sql = pd.DataFrame(
        {
            "Sample_ID": df["Sample ID"].astype(str),
            "Dataset": df["Dataset"].astype(str),
            "Category": df["Category"].astype(str),
            "Target": df["Target"].astype(int),
            **{m: df[m].astype(float) for m in CORE_MARKERS},
        }
    ).drop_duplicates(subset=["Sample_ID"], keep="first")

    conn.execute("DROP TABLE IF EXISTS clinical_cohort")
    conn.execute(
        """
        CREATE TABLE clinical_cohort (
            Sample_ID TEXT PRIMARY KEY,
            Dataset TEXT NOT NULL,
            Category TEXT NOT NULL,
            Target INTEGER NOT NULL,
            AMBP REAL NOT NULL,
            FN1 REAL NOT NULL,
            LUM REAL NOT NULL,
            TIMP1 REAL NOT NULL,
            B2M REAL NOT NULL
        )
        """
    )
    df_sql.to_sql("clinical_cohort", conn, if_exists="append", index=False)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_clin_dataset ON clinical_cohort(Dataset)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_clin_target ON clinical_cohort(Target)"
    )
    return len(df_sql)


def populate_transitions(
    conn: sqlite3.Connection,
    path: Path | str | None = None,
    chunksize: int = 100_000,
    limit: int | None = None,
) -> int:
    """Потоково заполняет таблицу спектральных переходов и строит поисковые индексы."""
    conn.execute("PRAGMA synchronous = OFF")
    total = 0
    step = min(chunksize, limit) if limit else chunksize
    reader = load_transitions(path=path, chunksize=step)
    assert isinstance(reader, Iterator)
    for i, chunk in enumerate(reader):
        if limit and total + len(chunk) > limit:
            chunk = chunk.iloc[: limit - total]
        chunk.to_sql(
            "transitions",
            conn,
            if_exists="replace" if i == 0 else "append",
            index=False,
        )
        total += len(chunk)
        if limit and total >= limit:
            break
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_trans_prot ON transitions(ProteinGroup)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trans_genes ON transitions(Genes)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trans_decoy ON transitions(decoy)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trans_qval ON transitions(QValue)")
    return total


def enrich(conn: sqlite3.Connection) -> int:
    """Обогащает метаданные белков из переходов и закрывает 48 пропусков Genes через Protein_Group."""
    cur = conn.execute(
        """
        UPDATE proteins
        SET
            Genes = (
                SELECT t.Genes
                FROM transitions t
                WHERE t.ProteinGroup = proteins.Protein_Group AND t.Genes IS NOT NULL
                LIMIT 1
            ),
            Protein_Names = (
                SELECT t.ProteinName
                FROM transitions t
                WHERE t.ProteinGroup = proteins.Protein_Group AND t.ProteinName IS NOT NULL
                LIMIT 1
            )
        WHERE Genes IS NULL
        """
    )
    enriched_count = cur.rowcount

    conn.execute(
        """
        UPDATE proteins
        SET
            Genes = COALESCE(NULLIF(TRIM(Genes), ''), Protein_Group),
            Protein_Names = COALESCE(NULLIF(TRIM(Protein_Names), ''), Protein_Group)
        WHERE Genes IS NULL OR TRIM(Genes) = ''
        """
    )
    conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_prot_genes ON proteins(Genes)")
    return enriched_count


def build(
    path: Path | str | None = None,
    chunksize: int = 100_000,
    limit: int | None = None,
) -> tuple[int, int, int]:
    """Создает, наполняет и обогащает базу данных SQLite."""
    target = locate(TRANSITIONS_NAME).parent / DB_NAME if path is None else Path(path)
    if target.is_file():
        target.unlink()
    with connect(target) as conn:
        n_prot = populate_matrix(conn)
        n_trans = populate_transitions(conn, chunksize=chunksize, limit=limit)
        n_enriched = enrich(conn)
        with contextlib.suppress(OSError, sqlite3.Error, ValueError, KeyError):
            populate_clinical_cohort(conn)
        return (n_prot, n_trans, n_enriched)


def query(sql: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Выполняет аналитический SQL-запрос и возвращает DataFrame."""
    return pd.read_sql_query(sql, conn)
