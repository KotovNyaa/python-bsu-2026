"""Априорные биофизические знания UniProt и 6-мерный вектор приборной достоверности MS/MS."""

from __future__ import annotations

import io
import json
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_CORE_GENES = ["LUM", "AMBP", "B2M", "FN1", "TIMP1"]

# Расширенная база знаний UniProt KB по 5 ядерным маркерам и ключевым белкам 6 путей статьи Zhao et al.
BIOPHYSICAL_PRIORS: dict[str, dict[str, Any]] = {
    "LUM": {
        "uniprot_id": "P51884",
        "protein_name": "Lumican",
        "mass_kda": 38.4,
        "length_aa": 338,
        "subcellular_location": "Extracellular matrix, Secreted",
        "is_secreted": 1,
        "is_ecm": 1,
        "signal_peptide": 1,
        "pathway_module": "ECM_Remodeling",
        "biological_role": "Collagen fibril organization, cervical ripening regulator",
    },
    "AMBP": {
        "uniprot_id": "P02760",
        "protein_name": "Protein AMBP (Alpha-1-microglobulin/bikunin precursor)",
        "mass_kda": 39.0,
        "length_aa": 352,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 1,
        "signal_peptide": 1,
        "pathway_module": "ECM_Remodeling",
        "biological_role": "Radical scavenger (Cys34), serine protease inhibitor, ECM repair",
    },
    "B2M": {
        "uniprot_id": "P61769",
        "protein_name": "Beta-2-microglobulin",
        "mass_kda": 13.7,
        "length_aa": 119,
        "subcellular_location": "Secreted, Cell membrane, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 0,
        "signal_peptide": 1,
        "pathway_module": "Immune_Inflammation",
        "biological_role": "MHC class I light chain, infection and immune activation marker",
    },
    "FN1": {
        "uniprot_id": "P02751",
        "protein_name": "Fibronectin 1",
        "mass_kda": 262.6,
        "length_aa": 2386,
        "subcellular_location": "Extracellular matrix, Secreted",
        "is_secreted": 1,
        "is_ecm": 1,
        "signal_peptide": 1,
        "pathway_module": "ECM_Remodeling",
        "biological_role": "Cell adhesion, chorion-decidual extracellular matrix interface glue",
    },
    "TIMP1": {
        "uniprot_id": "P01033",
        "protein_name": "Metalloproteinase inhibitor 1",
        "mass_kda": 23.2,
        "length_aa": 207,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 1,
        "signal_peptide": 1,
        "pathway_module": "ECM_Remodeling",
        "biological_role": "Endogenous MMP inhibitor, cervical collagenolysis control",
    },
    "C3": {
        "uniprot_id": "P01024",
        "protein_name": "Complement C3",
        "mass_kda": 187.1,
        "length_aa": 1663,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 0,
        "signal_peptide": 1,
        "pathway_module": "Complement_Cascade",
        "biological_role": "Central hub of classical and alternative complement activation (sPTB2)",
    },
    "C4A": {
        "uniprot_id": "P0C0L4",
        "protein_name": "Complement C4-A",
        "mass_kda": 192.8,
        "length_aa": 1744,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 0,
        "signal_peptide": 1,
        "pathway_module": "Complement_Cascade",
        "biological_role": "Classical complement pathway activation and opsonization",
    },
    "CFH": {
        "uniprot_id": "P08603",
        "protein_name": "Complement factor H",
        "mass_kda": 139.1,
        "length_aa": 1231,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 0,
        "signal_peptide": 1,
        "pathway_module": "Complement_Cascade",
        "biological_role": "Alternative complement regulator and host cell protection",
    },
    "PLG": {
        "uniprot_id": "P00747",
        "protein_name": "Plasminogen",
        "mass_kda": 90.6,
        "length_aa": 810,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 1,
        "signal_peptide": 1,
        "pathway_module": "Fibrinolysis_Coagulation",
        "biological_role": "Zymogen of plasmin, fibrinolysis and MMP zymogen activation",
    },
    "HRG": {
        "uniprot_id": "P04196",
        "protein_name": "Histidine-rich glycoprotein",
        "mass_kda": 59.6,
        "length_aa": 525,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 0,
        "signal_peptide": 1,
        "pathway_module": "Fibrinolysis_Coagulation",
        "biological_role": "Coagulation/fibrinolysis regulator and divalent metal binder (Zn)",
    },
    "SERPINA1": {
        "uniprot_id": "P01009",
        "protein_name": "Alpha-1-antitrypsin",
        "mass_kda": 46.7,
        "length_aa": 418,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 1,
        "signal_peptide": 1,
        "pathway_module": "Fibrinolysis_Coagulation",
        "biological_role": "Neutrophil elastase and serine protease inhibitor",
    },
    "COL1A2": {
        "uniprot_id": "P08123",
        "protein_name": "Collagen alpha-2(I) chain",
        "mass_kda": 129.3,
        "length_aa": 1366,
        "subcellular_location": "Extracellular matrix, Secreted",
        "is_secreted": 1,
        "is_ecm": 1,
        "signal_peptide": 1,
        "pathway_module": "ECM_Remodeling",
        "biological_role": "Cervical stromal tensile strength and fibrillar scaffolding",
    },
    "THBS1": {
        "uniprot_id": "P07996",
        "protein_name": "Thrombospondin-1",
        "mass_kda": 129.4,
        "length_aa": 1170,
        "subcellular_location": "Extracellular matrix, Secreted",
        "is_secreted": 1,
        "is_ecm": 1,
        "signal_peptide": 1,
        "pathway_module": "ECM_Remodeling",
        "biological_role": "Cell-to-matrix interaction and latent TGF-beta activation",
    },
    "PRG2": {
        "uniprot_id": "P13727",
        "protein_name": "Bone marrow proteoglycan (Major basic protein)",
        "mass_kda": 25.3,
        "length_aa": 222,
        "subcellular_location": "Secreted, Cytoplasmic granule",
        "is_secreted": 1,
        "is_ecm": 0,
        "signal_peptide": 1,
        "pathway_module": "Immune_Inflammation",
        "biological_role": "Eosinophil cytotoxic granule protein, top-ranked sPTB marker",
    },
    "ZG16B": {
        "uniprot_id": "Q96DA0",
        "protein_name": "Zymogen granule protein 16 homolog B",
        "mass_kda": 22.7,
        "length_aa": 208,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 0,
        "signal_peptide": 1,
        "pathway_module": "Immune_Inflammation",
        "biological_role": "Mucosal lectin-like defense and secretory immune response",
    },
    "ALB": {
        "uniprot_id": "P02768",
        "protein_name": "Serum albumin",
        "mass_kda": 69.4,
        "length_aa": 609,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 0,
        "signal_peptide": 1,
        "pathway_module": "Zn_Cu_Targets",
        "biological_role": "Primary zinc/copper carrier and oncotic pressure maintenance",
    },
    "GC": {
        "uniprot_id": "P02774",
        "protein_name": "Vitamin D-binding protein",
        "mass_kda": 52.9,
        "length_aa": 474,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 0,
        "signal_peptide": 1,
        "pathway_module": "Zn_Cu_Targets",
        "biological_role": "Vitamin D transport, extracellular actin scavenging, macrophage activation",
    },
    "TF": {
        "uniprot_id": "P02787",
        "protein_name": "Serotransferrin",
        "mass_kda": 77.1,
        "length_aa": 698,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 0,
        "signal_peptide": 1,
        "pathway_module": "Zn_Cu_Targets",
        "biological_role": "Iron/metal ion transport and bacteriostatic oxidative defense",
    },
    "SOD1": {
        "uniprot_id": "P00441",
        "protein_name": "Superoxide dismutase [Cu-Zn]",
        "mass_kda": 15.9,
        "length_aa": 154,
        "subcellular_location": "Cytoplasm, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 0,
        "signal_peptide": 0,
        "pathway_module": "Zn_Cu_Targets",
        "biological_role": "Cu/Zn-dependent enzymatic superoxide radical detoxification",
    },
    "A1BG": {
        "uniprot_id": "P04217",
        "protein_name": "Alpha-1B-glycoprotein",
        "mass_kda": 54.3,
        "length_aa": 495,
        "subcellular_location": "Secreted, Extracellular space",
        "is_secreted": 1,
        "is_ecm": 0,
        "signal_peptide": 1,
        "pathway_module": "Immune_Inflammation",
        "biological_role": "Acute-phase plasma glycoprotein and CRISP3 binding partner",
    },
}

# Структурные правила HGNC для автоматического маппинга всех 2 699 белков на 6 макро-путей Zhao et al.
PATHWAY_GENE_PATTERNS: list[tuple[str, re.Pattern[str], int, int]] = [
    (
        "Complement_Cascade",
        re.compile(
            r"^(C[1-9][QA-Z0-9]*|CF[HBDI]|CFB|CFD|CFP|MASP[12]|CR[12]|CD55|CD59|CLU|VTN)$"
        ),
        1,
        0,
    ),
    (
        "Fibrinolysis_Coagulation",
        re.compile(
            r"^(SERPIN[A-Z0-9]+|PLG|PLAT|PLAU|PLAUR|F[25789]|F1[0123][AB]?|FG[ABG]|KNG1|KLKB1|HRG|PRO[CS]1|THBD|VWF)$"
        ),
        1,
        0,
    ),
    (
        "ECM_Remodeling",
        re.compile(
            r"^(COL[0-9]+A[0-9]+|MMP[0-9]+|TIMP[1-4]|FN1|LUM|DCN|BGN|FMOD|FBLN[1-7]|FBN[12]|LAM[ABC][1-5]|THBS[1-4]|SPARC|TNC|TNXB|VTN|ITG[AB][0-9A-Z]+|ADAM[TS0-9]+|LOX[L1-4]?|COMP|ECM1|HSPG2|VCAN|AGRN|CST[A-Z0-9]*|KRT[0-9]+|ACT[ABG12]*|MYH[0-9]+|FLN[ABC]|TLN[12]|VCL|ZYX)$"
        ),
        1,
        1,
    ),
    (
        "Zn_Cu_Targets",
        re.compile(
            r"^(SOD[123]|ALB|TF|CP|GC|CA[1-9][0-9]*|MT[1234][A-Z0-9]*|SLC30A[0-9]+|SLC39A[0-9]+|ATOX1|CCS|COMMD1|APO[A-Z][0-9]*|ORM[12]|HP|HPX|AHSG|RBP4|TTR)$"
        ),
        1,
        0,
    ),
    (
        "Immune_Inflammation",
        re.compile(
            r"^(IGH[A-Z0-9]+|IGK[A-Z0-9]+|IGL[A-Z0-9]+|JCHAIN|B2M|HLA-[A-Z0-9]+|S100A[0-9]+|S100P|PRG[234]|DEFA[0-9]+|DEFB[0-9]+|CAMP|LCN2|LTF|ELANE|MPO|AZU1|CTSG|LYZ|SLPI|PI3|WFDC[0-9]+|ZG16B|A1BG|SAA[124]|CRP|IL[0-9]+.*|CXCL[0-9]+|CCL[0-9]+|CD[0-9]+.*|ANXA[0-9]+|LGALS[0-9]+|MUC[0-9A-Z]+)$"
        ),
        1,
        0,
    ),
    (
        "Energy_Metabolism_ATP",
        re.compile(
            r"^(ATP5[A-Z0-9]+|ATP6[A-Z0-9]+|NDUF[A-Z0-9]+|COX[0-9A-Z]+|UQCR[A-Z0-9]+|SDH[ABCD]|CYC[1S]|MDH[12]|IDH[123][ABG]?|ACO[12]|CS|FH|SUCL[A-Z0-9]+|GAPDH|ENO[123]|PKM|PFK[MLP]|ALDO[ABC]|PGK1|PGAM[12]|TPI1|LDHA|LDHB|HK[123]|G6PD|PGD|TKT|TALDO1|PRDX[1-6]|TXN[RD123]*|CAT|GPX[1-8]|GST[A-Z0-9]+|RPL[0-9A-Z]+|RPS[0-9A-Z]+|EEF[12][A-Z0-9]*|EIF[0-9A-Z]+)$"
        ),
        0,
        0,
    ),
]


def classify_gene_pathway_and_location(gene: str) -> tuple[str, int, int]:
    """Определяет макро-путь статьи Zhao et al. и флаги (is_secreted, is_ecm) по символу гена."""
    if gene in BIOPHYSICAL_PRIORS:
        info = BIOPHYSICAL_PRIORS[gene]
        return (
            str(info.get("pathway_module", "Other_Housekeeping")),
            int(info.get("is_secreted", 0)),
            int(info.get("is_ecm", 0)),
        )
    g_upper = str(gene).strip().upper().split(";")[0]
    for pathway_name, pattern, is_sec, is_ecm_flag in PATHWAY_GENE_PATTERNS:
        if pattern.match(g_upper):
            return pathway_name, is_sec, is_ecm_flag
    return "Other_Proteome", 0, 0


def get_biophysical_priors(genes: list[str] | None = None) -> pd.DataFrame:
    """Возвращает таблицу биофизических априорных признаков UniProt."""
    target_genes = list(BIOPHYSICAL_PRIORS.keys()) if genes is None else genes
    base_cols = [
        "uniprot_id",
        "protein_name",
        "mass_kda",
        "length_aa",
        "subcellular_location",
        "is_secreted",
        "is_ecm",
        "signal_peptide",
        "biological_role",
    ]
    records = []
    for g in target_genes:
        if g in BIOPHYSICAL_PRIORS:
            rec = {"gene": g, **{k: BIOPHYSICAL_PRIORS[g][k] for k in base_cols}}
        else:
            pathway, is_sec, is_ecm_flag = classify_gene_pathway_and_location(g)
            rec = {
                "gene": g,
                "uniprot_id": None,
                "protein_name": None,
                "mass_kda": None,
                "length_aa": None,
                "subcellular_location": "Extracellular/Secreted" if is_sec else None,
                "is_secreted": is_sec,
                "is_ecm": is_ecm_flag,
                "signal_peptide": is_sec,
                "biological_role": pathway
                if pathway != "Other_Proteome"
                else "Unannotated",
            }
        records.append(rec)
    return pd.DataFrame(records).set_index("gene")


def fetch_uniprot_online(accession: str, timeout: int = 4) -> dict[str, Any] | None:
    """Запрашивает метаданные одного белка напрямую через REST API UniProt."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    req = urllib.request.Request(url, headers={"User-Agent": "BSU-Proteomics-Lab/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                seq_info = data.get("sequence", {})
                return {
                    "uniprot_id": accession,
                    "mass_kda": round(seq_info.get("molWeight", 0) / 1000.0, 1),
                    "length_aa": seq_info.get("length", 0),
                    "scientific_name": data.get("organism", {}).get("scientificName"),
                }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    return None


def fetch_uniprot_bulk_tsv(
    accessions: list[str],
    chunk_size: int = 350,
    timeout: int = 12,
    cache_path: Path | str | None = None,
) -> pd.DataFrame:
    """Пакетно скачивает биофизические аннотации UniProt для сотен/тысяч белков (по 350 за 1 HTTP-запрос).

    Если указан cache_path и файл уже существует, мгновенно загружает данные из локального кэша.
    """
    if cache_path is not None and Path(cache_path).is_file():
        return pd.read_parquet(cache_path)

    clean_accs = sorted(
        {
            str(a).strip().split(";")[0].split("-")[0]
            for a in accessions
            if a and pd.notna(a) and len(str(a).strip()) >= 6
        }
    )
    if not clean_accs:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    fields = "accession,gene_primary,protein_name,mass,length,cc_subcellular_location,ft_signal"

    for start in range(0, len(clean_accs), chunk_size):
        batch = clean_accs[start : start + chunk_size]
        query_str = " OR ".join(f"accession:{acc}" for acc in batch)
        params = urllib.parse.urlencode(
            {
                "query": f"({query_str})",
                "fields": fields,
                "format": "tsv",
                "size": str(len(batch) + 20),
            }
        )
        url = f"https://rest.uniprot.org/uniprotkb/search?{params}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "BSU-Proteomics-Bulk/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    tsv_text = resp.read().decode("utf-8")
                    if tsv_text.strip():
                        frames.append(pd.read_csv(io.StringIO(tsv_text), sep="\t"))
        except (urllib.error.URLError, TimeoutError, OSError):
            break

    if not frames:
        return pd.DataFrame()

    raw_df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["Entry"], keep="first"
    )
    subcell = raw_df.get(
        "Subcellular location [CC]", pd.Series("", index=raw_df.index)
    ).fillna("")
    sig_pep = raw_df.get("Signal peptide", pd.Series("", index=raw_df.index)).fillna("")
    mass_da = pd.to_numeric(raw_df.get("Mass", 0), errors="coerce").fillna(30000.0)
    length_aa = (
        pd.to_numeric(raw_df.get("Length", 0), errors="coerce").fillna(250).astype(int)
    )

    parsed = pd.DataFrame(
        {
            "uniprot_id": raw_df["Entry"].astype(str),
            "gene": raw_df.get("Gene Names (primary)", raw_df["Entry"])
            .fillna(raw_df["Entry"])
            .astype(str),
            "protein_name": raw_df.get("Protein names", "").fillna("").astype(str),
            "mass_kda": np.round(mass_da / 1000.0, 2),
            "length_aa": length_aa,
            "subcellular_location": subcell.astype(str),
            "is_secreted": subcell.str.contains(
                "Secreted|Extracellular", case=False, regex=True
            ).astype(int),
            "is_ecm": subcell.str.contains(
                "Extracellular matrix|basement membrane|collagen",
                case=False,
                regex=True,
            ).astype(int),
            "signal_peptide": (sig_pep.astype(str).str.len() > 0).astype(int),
        }
    )
    if cache_path is not None and not parsed.empty:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        parsed.to_parquet(cache_path, index=False)

    return parsed


def compute_ms_evidence_vectors(
    conn: sqlite3.Connection,
    genes: list[str] | None = None,
    min_transitions: int = 1,
) -> pd.DataFrame:
    """Вычисляет 6-мерный вектор приборной достоверности MS/MS v_g ∈ [0,1]^6 и сводный вес E_g.

    Учитывает Ограничение #1: при отсутствии в таблице transitions строк с decoy == 1
    распределение приборного шума оценивается по FDR-хвосту QValue > 0.01 (259 559 строк).
    """
    info_df = pd.read_sql_query("PRAGMA table_info(transitions)", conn)
    if info_df.empty:
        return pd.DataFrame()
    cols = set(info_df["name"].astype(str).tolist())

    has_qval = "QValue" in cols
    has_pgqval = "PGQValue" in cols
    has_prot = "Proteotypic" in cols
    has_frag = "FragmentType" in cols
    has_file = "FileName" in cols
    has_pg = "ProteinGroup" in cols

    # 1. Оценка параметров глобального шумового распределения (decoy == 1 ИЛИ QValue > 0.01)
    mu_noise_l, sd_noise_l = 0.35, 0.15
    mu_noise_s, sd_noise_s = 1.3, 0.5
    total_runs = 213

    if has_file:
        run_df = pd.read_sql_query(
            "SELECT COUNT(DISTINCT FileName) AS n_runs FROM transitions", conn
        )
        total_runs = max(int(run_df["n_runs"].iloc[0]), 1)

    if has_qval:
        noise_df = pd.read_sql_query(
            """
            SELECT
                AVG(LibraryIntensity) AS mean_int,
                AVG(QValue) AS mean_q
            FROM transitions
            WHERE decoy = 1 OR QValue > 0.01
            """,
            conn,
        )
        if not noise_df.empty and pd.notna(noise_df["mean_int"].iloc[0]):
            m_int = max(float(noise_df["mean_int"].iloc[0]), 1e-4)
            mu_noise_l = float(np.log2(m_int + 1.0))
            sd_noise_l = max(0.25 * mu_noise_l, 0.08)
            m_q = max(float(noise_df["mean_q"].iloc[0]), 1e-6)
            mu_noise_s = float(-np.log10(m_q))

    # 2. Формирование SQL-условия фильтрации генов
    where_clauses = ["Genes IS NOT NULL", "decoy = 0"]
    params: list[Any] = []
    if genes is not None:
        if len(genes) == 0:
            return pd.DataFrame()
        placeholders = ",".join("?" for _ in genes)
        where_clauses.append(f"Genes IN ({placeholders})")
        params.extend(genes)

    if has_qval and min_transitions >= 2:
        where_clauses.append("QValue <= 0.01")
    where_sql = " AND ".join(where_clauses)

    # 3. Если доступны все колонки DIA-NN, выполняем двухступенчатую агрегацию с межпрогонной дисперсией RT
    if has_qval and has_prot and has_frag and has_file and has_pg:
        pgq_expr = "PGQValue" if has_pgqval else "QValue"
        sql = f"""
            WITH pep_stats AS (
                SELECT
                    Genes AS gene,
                    MAX(ProteinGroup) AS protein_group,
                    PeptideSequence,
                    MAX(Proteotypic) AS is_prot,
                    MIN(QValue) AS pep_min_q,
                    AVG(QValue) AS pep_avg_q,
                    MIN({pgq_expr}) AS pep_min_pgq,
                    AVG({pgq_expr}) AS pep_avg_pgq,
                    COUNT(*) AS pep_trans_cnt,
                    COUNT(DISTINCT FileName) AS pep_run_cnt,
                    SUM(1.0 + CASE WHEN FragmentType = 'y' THEN 0.2 ELSE 0.0 END) AS weighted_tr_sum,
                    AVG(LibraryIntensity) AS pep_mean_int,
                    AVG(Tr_recalibrated) AS pep_mean_rt,
                    MAX(
                        0.0,
                        AVG(Tr_recalibrated * Tr_recalibrated)
                        - AVG(Tr_recalibrated) * AVG(Tr_recalibrated)
                    ) AS pep_rt_var
                FROM transitions
                WHERE {where_sql}
                GROUP BY Genes, PeptideSequence
            )
            SELECT
                gene,
                MAX(protein_group) AS protein_group,
                SUM(pep_trans_cnt) AS trans_count,
                COUNT(DISTINCT PeptideSequence) AS unique_peptides,
                SUM(CASE WHEN is_prot = 1 AND pep_min_q <= 0.01 THEN 1 ELSE 0 END) AS prot_confident_peptides,
                MIN(213, SUM(pep_run_cnt)) AS sample_detection_count,
                ROUND(SUM(pep_mean_int * pep_trans_cnt) / NULLIF(SUM(pep_trans_cnt), 0), 4) AS mean_ms2_intensity,
                ROUND(SUM(pep_mean_rt * pep_trans_cnt) / NULLIF(SUM(pep_trans_cnt), 0), 2) AS mean_rt,
                ROUND(MIN(pep_min_q), 6) AS min_qvalue,
                ROUND(AVG(pep_avg_q), 6) AS mean_qvalue,
                ROUND(MIN(pep_min_pgq), 6) AS min_pgqvalue,
                ROUND(AVG(pep_avg_pgq), 6) AS mean_pgqvalue,
                SUM(weighted_tr_sum) / NULLIF(SUM(pep_run_cnt), 0) AS mean_weighted_tr_per_run,
                AVG(CASE WHEN pep_run_cnt >= 2 THEN pep_rt_var ELSE NULL END) AS mean_pep_rt_var
            FROM pep_stats
            GROUP BY gene
            HAVING SUM(pep_trans_cnt) >= {int(min_transitions)}
            ORDER BY trans_count DESC
        """
        df = pd.read_sql_query(sql, conn, params=params)
    else:
        # Упрощенный fallback для синтетических таблиц в юнит-тестах
        pg_select = (
            "MAX(ProteinGroup) AS protein_group,"
            if has_pg
            else "'PG_Mock' AS protein_group,"
        )
        file_select = (
            "COUNT(DISTINCT FileName) AS sample_detection_count,"
            if has_file
            else "1 AS sample_detection_count,"
        )
        q_select = (
            "ROUND(MIN(QValue), 6) AS min_qvalue, ROUND(AVG(QValue), 6) AS mean_qvalue,"
            if has_qval
            else "0.001 AS min_qvalue, 0.002 AS mean_qvalue,"
        )
        sql = f"""
            SELECT
                Genes AS gene,
                {pg_select}
                COUNT(*) AS trans_count,
                COUNT(DISTINCT PeptideSequence) AS unique_peptides,
                COUNT(DISTINCT PeptideSequence) AS prot_confident_peptides,
                {file_select}
                ROUND(AVG(LibraryIntensity), 4) AS mean_ms2_intensity,
                ROUND(AVG(Tr_recalibrated), 2) AS mean_rt,
                {q_select}
                0.001 AS min_pgqvalue,
                0.002 AS mean_pgqvalue,
                CAST(COUNT(*) AS REAL) / NULLIF(COUNT(DISTINCT PeptideSequence), 0) AS mean_weighted_tr_per_run,
                0.05 AS mean_pep_rt_var
            FROM transitions
            WHERE {where_sql}
            GROUP BY Genes
            HAVING COUNT(*) >= {int(min_transitions)}
            ORDER BY trans_count DESC
        """
        df = pd.read_sql_query(sql, conn, params=params)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "protein_group",
                "trans_count",
                "unique_peptides",
                "sample_detection_count",
                "mean_ms2_intensity",
                "mean_rt",
                "min_qvalue",
                "mean_qvalue",
                "v1_proteotypic",
                "v2_transition_density",
                "v3_qvalue_significance",
                "v4_target_decoy_snr",
                "v5_run_reproducibility",
                "v6_rt_stability",
                "ms_evidence_weight_E",
            ]
        ).set_index("gene")

    # 4. Векторизованный расчет 6 компонент вектора приборной достоверности v_g ∈ [0, 1]^6
    n_pep_total = df["unique_peptides"].to_numpy(dtype=np.float64)
    n_pep_prot = (
        df["prot_confident_peptides"]
        .fillna(df["unique_peptides"])
        .to_numpy(dtype=np.float64)
    )
    f_prot = n_pep_prot / (n_pep_total + 1e-6)
    v1 = np.clip(
        f_prot * (1.0 - np.exp(-np.log1p(n_pep_prot) / np.log1p(3.0))),
        0.0,
        1.0,
    )

    n_tr_bar = df["mean_weighted_tr_per_run"].fillna(4.0).to_numpy(dtype=np.float64)
    v2 = np.clip(n_tr_bar / (n_tr_bar + 6.0), 0.0, 1.0)

    # Используем Series для fillna (исправление ошибки Pandas 2.0 CoW с ndarray)
    eps_q = 1e-6
    s_min_q = df["min_qvalue"].fillna(0.01)
    s_avg_q = df["mean_qvalue"].fillna(0.01)
    q_min_pep = np.clip(s_min_q.to_numpy(dtype=np.float64), eps_q, 1.0)
    q_avg_pep = np.clip(s_avg_q.to_numpy(dtype=np.float64), eps_q, 1.0)
    q_min_pg = np.clip(
        df["min_pgqvalue"].fillna(s_min_q).to_numpy(dtype=np.float64), eps_q, 1.0
    )
    q_avg_pg = np.clip(
        df["mean_pgqvalue"].fillna(s_avg_q).to_numpy(dtype=np.float64), eps_q, 1.0
    )

    s_q = -0.25 * (
        np.log10(q_min_pg)
        + np.log10(q_avg_pg)
        + np.log10(q_min_pep)
        + np.log10(q_avg_pep)
    )
    v3 = np.clip((s_q - 2.0) / (6.0 - 2.0), 0.0, 1.0)

    l_target = np.log2(
        np.clip(
            df["mean_ms2_intensity"].fillna(0.0).to_numpy(dtype=np.float64),
            0.0,
            None,
        )
        + 1.0
    )
    s_target = -np.log10(q_avg_pep)
    snr_td = (s_target - mu_noise_s) / sd_noise_s + 0.5 * np.maximum(
        0.0, (l_target - mu_noise_l) / sd_noise_l
    )
    v4 = np.clip(2.0 * stats.norm.cdf(snr_td / np.sqrt(2.0)) - 1.0, 0.0, 1.0)

    k_det = df["sample_detection_count"].fillna(1).to_numpy(dtype=np.float64)
    v5 = np.clip(k_det / float(total_runs), 0.0, 1.0)

    rt_var = df["mean_pep_rt_var"].fillna(0.08).to_numpy(dtype=np.float64)
    sigma0_sq = max(float(np.nanmedian(rt_var)) * 2.5, 0.25)
    v6 = np.clip(np.exp(-rt_var / sigma0_sq), 0.0, 1.0)

    # 5. Сводный геометрический вес достоверности E_g = prod max(v_{g,m}, 0.05)^{w_m}
    weights_w = np.array([0.20, 0.15, 0.20, 0.15, 0.15, 0.15], dtype=np.float64)
    v_mat = np.column_stack([v1, v2, v3, v4, v5, v6])
    v_clipped = np.maximum(v_mat, 0.05)
    e_g = np.exp(np.sum(np.log(v_clipped) * weights_w[None, :], axis=1))

    df["v1_proteotypic"] = np.round(v1, 4)
    df["v2_transition_density"] = np.round(v2, 4)
    df["v3_qvalue_significance"] = np.round(v3, 4)
    df["v4_target_decoy_snr"] = np.round(v4, 4)
    df["v5_run_reproducibility"] = np.round(v5, 4)
    df["v6_rt_stability"] = np.round(v6, 4)
    df["ms_evidence_weight_E"] = np.round(e_g, 4)

    keep_cols = [
        "gene",
        "protein_group",
        "trans_count",
        "unique_peptides",
        "sample_detection_count",
        "mean_ms2_intensity",
        "mean_rt",
        "min_qvalue",
        "mean_qvalue",
        "v1_proteotypic",
        "v2_transition_density",
        "v3_qvalue_significance",
        "v4_target_decoy_snr",
        "v5_run_reproducibility",
        "v6_rt_stability",
        "ms_evidence_weight_E",
    ]
    return df[keep_cols].set_index("gene")


def get_ms_instrument_priors(
    conn: sqlite3.Connection, genes: list[str]
) -> pd.DataFrame:
    """Вычисляет приборные показатели надежности детекции (MS Evidence) для списка генов."""
    ev_df = compute_ms_evidence_vectors(conn, genes=genes, min_transitions=1)
    if ev_df.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "n_transitions",
                "n_unique_peptides",
                "mean_ms2_intensity",
                "mean_rt",
                "v1_proteotypic",
                "v2_transition_density",
                "v3_qvalue_significance",
                "v4_target_decoy_snr",
                "v5_run_reproducibility",
                "v6_rt_stability",
                "ms_evidence_weight_E",
            ]
        ).set_index("gene")

    out = ev_df.rename(
        columns={
            "trans_count": "n_transitions",
            "unique_peptides": "n_unique_peptides",
        }
    )
    cols = [
        "n_transitions",
        "n_unique_peptides",
        "mean_ms2_intensity",
        "mean_rt",
        "v1_proteotypic",
        "v2_transition_density",
        "v3_qvalue_significance",
        "v4_target_decoy_snr",
        "v5_run_reproducibility",
        "v6_rt_stability",
        "ms_evidence_weight_E",
    ]
    return out[[c for c in cols if c in out.columns]]


def build_priors_table(
    conn: sqlite3.Connection | None = None,
    genes: list[str] | None = None,
) -> pd.DataFrame:
    """Объединяет биофизические аннотации UniProt и 6-мерные приборные приоры MS/MS."""
    target_genes = list(BIOPHYSICAL_PRIORS.keys()) if genes is None else genes
    bio_df = get_biophysical_priors(target_genes)

    ms_cols = [
        "n_transitions",
        "n_unique_peptides",
        "mean_ms2_intensity",
        "mean_rt",
        "v1_proteotypic",
        "v2_transition_density",
        "v3_qvalue_significance",
        "v4_target_decoy_snr",
        "v5_run_reproducibility",
        "v6_rt_stability",
        "ms_evidence_weight_E",
    ]

    if conn is not None:
        ms_df = get_ms_instrument_priors(conn, target_genes)
        full_df = bio_df.join(ms_df, how="left")
    else:
        full_df = bio_df.copy()
        for col in ms_cols:
            full_df[col] = np.nan

    full_df["n_transitions"] = full_df["n_transitions"].fillna(0).astype(int)
    full_df["n_unique_peptides"] = full_df["n_unique_peptides"].fillna(0).astype(int)
    full_df["mean_ms2_intensity"] = full_df["mean_ms2_intensity"].fillna(0.0)
    full_df["mean_rt"] = full_df["mean_rt"].fillna(0.0)

    for v_col in [
        "v1_proteotypic",
        "v2_transition_density",
        "v3_qvalue_significance",
        "v4_target_decoy_snr",
        "v5_run_reproducibility",
        "v6_rt_stability",
        "ms_evidence_weight_E",
    ]:
        if v_col in full_df.columns:
            full_df[v_col] = full_df[v_col].fillna(0.0)

    has_ms = (full_df["n_unique_peptides"] > 0).astype(np.float64)
    ms_score = np.where(
        full_df["ms_evidence_weight_E"] > 0,
        0.5 * has_ms + 0.5 * full_df["ms_evidence_weight_E"],
        has_ms,
    )

    weights = 0.35 * full_df["is_secreted"] + 0.35 * full_df["is_ecm"] + 0.30 * ms_score
    full_df["prior_confidence_weight"] = np.round(weights, 3)

    return full_df


def build_full_proteome_priors(
    conn: sqlite3.Connection,
    fetch_online_batch: bool = False,
    cache_path: Path | str | None = None,
) -> pd.DataFrame:
    """Строит полную таблицу приоров для ВСЕХ 2 699 белков Чунцина (и/или 11 032 генов).

    Объединяет:
      1. 6-мерный вектор достоверности MS/MS (v1..v6, E_g) из transitions;
      2. Курированные данные UniProt + пакетную докачку fetch_uniprot_bulk_tsv (при включенном флаге);
      3. Автономную классификацию по 6 макро-путям статьи Zhao et al. и эмпирическую массу пептидов.
    """
    prot_df = pd.read_sql_query(
        "SELECT Protein_Group, Genes, Protein_Names FROM proteins", conn
    )
    if prot_df.empty:
        return build_priors_table(conn, list(BIOPHYSICAL_PRIORS.keys()))

    raw_genes = prot_df["Genes"].replace("", np.nan)
    prot_df["gene"] = raw_genes.fillna(prot_df["Protein_Group"]).astype(str)
    prot_df = prot_df.drop_duplicates(subset=["gene"], keep="first").set_index("gene")

    ev_df = compute_ms_evidence_vectors(conn, genes=None, min_transitions=1)
    merged = prot_df.join(
        ev_df.drop(columns=["protein_group"], errors="ignore"), how="left"
    )

    # Опциональная пакетная докачка из UniProt REST API (по 350 белков за запрос)
    online_map: dict[str, dict[str, Any]] = {}
    if fetch_online_batch:
        bulk_df = fetch_uniprot_bulk_tsv(
            prot_df["Protein_Group"].astype(str).tolist(),
            cache_path=cache_path,
        )
        if not bulk_df.empty:
            for _, row in bulk_df.iterrows():
                online_map[str(row["uniprot_id"])] = row.to_dict()
                online_map[str(row["gene"])] = row.to_dict()

    pathways: list[str] = []
    is_sec_list: list[int] = []
    is_ecm_list: list[int] = []
    mass_list: list[float] = []
    len_list: list[int] = []

    n_peps = merged["unique_peptides"].fillna(1).to_numpy(dtype=np.float64)

    for idx, (gene_name, row) in enumerate(merged.iterrows()):
        g_str = str(gene_name)
        pg_str = str(row.get("Protein_Group", g_str)).split(";")[0]

        if g_str in BIOPHYSICAL_PRIORS:
            info = BIOPHYSICAL_PRIORS[g_str]
            pathways.append(str(info["pathway_module"]))
            is_sec_list.append(int(info["is_secreted"]))
            is_ecm_list.append(int(info["is_ecm"]))
            mass_list.append(float(info["mass_kda"]))
            len_list.append(int(info["length_aa"]))
        elif g_str in online_map or pg_str in online_map:
            oinfo = online_map.get(g_str, online_map.get(pg_str, {}))
            pw, sec_f, ecm_f = classify_gene_pathway_and_location(g_str)
            pathways.append(pw)
            is_sec_list.append(int(oinfo.get("is_secreted", sec_f)))
            is_ecm_list.append(int(oinfo.get("is_ecm", ecm_f)))
            mass_list.append(float(oinfo.get("mass_kda", 35.0)))
            len_list.append(int(oinfo.get("length_aa", 300)))
        else:
            pw, sec_f, ecm_f = classify_gene_pathway_and_location(g_str)
            pathways.append(pw)
            is_sec_list.append(sec_f)
            is_ecm_list.append(ecm_f)
            # Эмпирическая оценка покрытой длины и массы по числу триптических пептидов
            est_len = int(max(110, min(2500, n_peps[idx] * 24 + 80)))
            len_list.append(est_len)
            mass_list.append(round(est_len * 0.11, 1))

    merged["pathway_module"] = pathways
    merged["is_secreted"] = is_sec_list
    merged["is_ecm"] = is_ecm_list
    merged["mass_kda"] = mass_list
    merged["length_aa"] = len_list

    for col in [
        "trans_count",
        "unique_peptides",
        "sample_detection_count",
        "mean_ms2_intensity",
        "mean_rt",
        "v1_proteotypic",
        "v2_transition_density",
        "v3_qvalue_significance",
        "v4_target_decoy_snr",
        "v5_run_reproducibility",
        "v6_rt_stability",
        "ms_evidence_weight_E",
    ]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    e_vec = merged["ms_evidence_weight_E"].to_numpy(dtype=np.float64)
    sec_vec = merged["is_secreted"].to_numpy(dtype=np.float64)
    ecm_vec = merged["is_ecm"].to_numpy(dtype=np.float64)
    merged["prior_confidence_weight"] = np.round(
        0.30 * sec_vec + 0.25 * ecm_vec + 0.45 * np.maximum(e_vec, 0.10), 4
    )

    return merged
