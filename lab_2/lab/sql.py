"""Аналитические SQL-запросы к базе данных (выборки, группировки, окна, CASE)."""

from __future__ import annotations

import sqlite3

import pandas as pd


def select_markers(conn: sqlite3.Connection) -> pd.DataFrame:
    """Выбирает ключевые биомаркеры из таблицы белков (WHERE + IN)."""
    return pd.read_sql_query(
        """
        SELECT Protein_Group, Genes, Protein_Names
        FROM proteins
        WHERE Genes IN ('LUM', 'FN1', 'AMBP', 'B2M')
        ORDER BY Genes
        """,
        conn,
    )


def group_trans_by_gene(conn: sqlite3.Connection, limit: int = 10) -> pd.DataFrame:
    """Агрегирует характеристики переходов по генам (GROUP BY + HAVING)."""
    return pd.read_sql_query(
        f"""
        SELECT
            Genes,
            COUNT(*) AS trans_count,
            COUNT(DISTINCT PeptideSequence) AS peptide_count,
            ROUND(AVG(PrecursorMz), 3) AS avg_precursor_mz,
            ROUND(AVG(LibraryIntensity), 4) AS avg_intensity
        FROM transitions
        WHERE Genes IS NOT NULL AND decoy = 0
        GROUP BY Genes
        HAVING COUNT(*) > 10
        ORDER BY trans_count DESC
        LIMIT {limit}
        """,
        conn,
    )


def decoy_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """Сравнивает достоверные мишени и шумовой хвост (Target vs Decoy / FDR > 0.01).

    Учитывает физическую особенность выгрузки DIA-NN (Ограничение #1):
    если строк с decoy = 1 физически 0, но присутствует колонка QValue,
    шумовой хвост формируется по критерию QValue > 0.01 (259 559 переходов).
    """
    info_df = pd.read_sql_query("PRAGMA table_info(transitions)", conn)
    col_names = set(info_df["name"].astype(str).tolist())

    decoy_count_df = pd.read_sql_query(
        "SELECT COUNT(*) AS cnt FROM transitions WHERE decoy = 1", conn
    )
    has_explicit_decoy = int(decoy_count_df["cnt"].iloc[0]) > 0

    if not has_explicit_decoy and "QValue" in col_names:
        return pd.read_sql_query(
            """
            SELECT
                CASE WHEN decoy = 1 OR QValue > 0.01 THEN 1 ELSE 0 END AS decoy,
                CASE
                    WHEN decoy = 1 OR QValue > 0.01 THEN 'Noise_Tail (QValue > 0.01)'
                    ELSE 'Confident_Target (QValue <= 0.01)'
                END AS signal_class,
                COUNT(*) AS total_count,
                ROUND(AVG(LibraryIntensity), 4) AS mean_intensity,
                ROUND(AVG(Tr_recalibrated), 2) AS mean_rt
            FROM transitions
            GROUP BY CASE WHEN decoy = 1 OR QValue > 0.01 THEN 1 ELSE 0 END
            ORDER BY decoy
            """,
            conn,
        )

    return pd.read_sql_query(
        """
        SELECT
            decoy,
            COUNT(*) AS total_count,
            ROUND(AVG(LibraryIntensity), 4) AS mean_intensity,
            ROUND(AVG(Tr_recalibrated), 2) AS mean_rt
        FROM transitions
        GROUP BY decoy
        ORDER BY decoy
        """,
        conn,
    )


def top_transitions(conn: sqlite3.Connection, n: int = 3) -> pd.DataFrame:
    """Выбирает топ-N переходов для каждого маркера через оконную функцию ROW_NUMBER()."""
    return pd.read_sql_query(
        f"""
        WITH ranked AS (
            SELECT
                Genes,
                PeptideSequence,
                ProductMz,
                LibraryIntensity,
                ROW_NUMBER() OVER (
                    PARTITION BY Genes ORDER BY LibraryIntensity DESC
                ) AS rn
            FROM transitions
            WHERE Genes IN ('LUM', 'FN1', 'AMBP', 'B2M') AND decoy = 0
        )
        SELECT Genes, PeptideSequence, ProductMz, LibraryIntensity, rn
        FROM ranked
        WHERE rn <= {n}
        ORDER BY Genes, rn
        """,
        conn,
    )


def join_coverage(conn: sqlite3.Connection) -> pd.DataFrame:
    """Вычисляет пептидное покрытие белков через реляционное объединение JOIN."""
    return pd.read_sql_query(
        """
        SELECT
            p.Genes,
            p.Protein_Group,
            COUNT(t.PeptideSequence) AS total_transitions,
            COUNT(DISTINCT t.PeptideSequence) AS unique_peptides
        FROM proteins p
        JOIN transitions t ON p.Protein_Group = t.ProteinGroup
        WHERE p.Genes IN ('LUM', 'FN1', 'AMBP', 'B2M')
        GROUP BY p.Genes, p.Protein_Group
        ORDER BY total_transitions DESC
        """,
        conn,
    )


def stratify_fdr_tiers(conn: sqlite3.Connection) -> pd.DataFrame:
    """Стратифицирует спектральные переходы по интервальным группам достоверности через CASE WHEN."""
    info_df = pd.read_sql_query("PRAGMA table_info(transitions)", conn)
    col_names = set(info_df["name"].astype(str).tolist())

    if "QValue" in col_names:
        return pd.read_sql_query(
            """
            SELECT
                CASE
                    WHEN QValue <= 0.001 THEN '1_Ultra_High_Confidence (Q <= 0.1%)'
                    WHEN QValue <= 0.010 THEN '2_Standard_Discovery (0.1% < Q <= 1%)'
                    WHEN QValue <= 0.050 THEN '3_Borderline_Relaxed (1% < Q <= 5%)'
                    ELSE '4_Empirical_Noise_Tail (Q > 5%)'
                END AS fdr_tier,
                COUNT(*) AS transition_count,
                COUNT(DISTINCT Genes) AS unique_genes,
                ROUND(AVG(LibraryIntensity), 2) AS mean_ms2_intensity,
                ROUND(AVG(Tr_recalibrated), 2) AS mean_retention_time
            FROM transitions
            WHERE Genes IS NOT NULL
            GROUP BY fdr_tier
            ORDER BY fdr_tier
            """,
            conn,
        )

    return pd.read_sql_query(
        """
        SELECT
            CASE
                WHEN LibraryIntensity >= 10000 THEN '1_High_Intensity'
                WHEN LibraryIntensity >= 1000 THEN '2_Medium_Intensity'
                ELSE '3_Low_Intensity_LOD'
            END AS fdr_tier,
            COUNT(*) AS transition_count,
            COUNT(DISTINCT Genes) AS unique_genes,
            ROUND(AVG(LibraryIntensity), 2) AS mean_ms2_intensity,
            ROUND(AVG(Tr_recalibrated), 2) AS mean_retention_time
        FROM transitions
        GROUP BY fdr_tier
        ORDER BY fdr_tier
        """,
        conn,
    )
