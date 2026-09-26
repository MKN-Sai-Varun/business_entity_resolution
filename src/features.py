import duckdb
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz
from src.db import get_connection
from src.config import CHECKPOINT_DIR, FUZZY_CHUNK_SIZE

def build_exact_features(con, s1_table, other_table, cand_table, out_table):
    """Pure SQL — no Python loop, no memory risk."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        SELECT
            c.s1_id, c.other_id,
            (a.name_norm = b.name_norm) AS name_exact,
            (a.address_norm = b.address_norm) AS address_exact,
            (a.postal_code = b.postal_code AND a.postal_code IS NOT NULL) AS postal_match,
            (a.house_number = b.house_number AND a.house_number IS NOT NULL) AS house_match,
            (a.country = b.country) AS country_match,
            (a.address_norm IS NULL OR a.address_norm = '') AS s1_addr_missing,
            (b.address_norm IS NULL OR b.address_norm = '') AS other_addr_missing,
            a.name_norm AS s1_name, b.name_norm AS other_name,
            a.address_norm AS s1_addr, b.address_norm AS other_addr
        FROM {cand_table} c
        JOIN {s1_table} a ON c.s1_id = a.entity_id
        JOIN {other_table} b ON c.other_id = b.entity_id
    """)

def _fuzzy_chunk_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized-ish batch of rapidfuzz calls — still Python, but bounded by chunk size."""
    df["name_lev"] = [fuzz.ratio(a, b) for a, b in zip(df.s1_name, df.other_name)]
    df["name_token_sort"] = [fuzz.token_sort_ratio(a, b) for a, b in zip(df.s1_name, df.other_name)]
    df["addr_lev"] = [fuzz.ratio(a or "", b or "") for a, b in zip(df.s1_addr, df.other_addr)]
    df["addr_token_sort"] = [fuzz.token_sort_ratio(a or "", b or "") for a, b in zip(df.s1_addr, df.other_addr)]
    return df.drop(columns=["s1_name", "other_name", "s1_addr", "other_addr"])

def build_fuzzy_features_chunked(con, exact_table, tag):
    """Reads exact-feature table in chunks, runs rapidfuzz, writes one parquet
    checkpoint per chunk. Safe to interrupt and resume — already-written chunks
    are skipped."""
    n_rows = con.execute(f"SELECT count(*) FROM {exact_table}").fetchone()[0]
    n_chunks = (n_rows // FUZZY_CHUNK_SIZE) + 1
    print(f"{exact_table}: {n_rows:,} rows -> {n_chunks} chunks")

    for i in range(n_chunks):
        out_path = CHECKPOINT_DIR / f"{tag}_fuzzy_chunk_{i}.parquet"
        if out_path.exists():
            continue  # RESUME: skip completed chunks
        offset = i * FUZZY_CHUNK_SIZE
        df = con.execute(
            f"SELECT * FROM {exact_table} LIMIT {FUZZY_CHUNK_SIZE} OFFSET {offset}"
        ).fetchdf()
        if df.empty:
            continue
        df = _fuzzy_chunk_features(df)
        df.to_parquet(out_path, index=False)
        del df  # free memory before next chunk
        print(f"  chunk {i+1}/{n_chunks} done -> {out_path.name}")

def assemble_full_features(con, tag, final_table):
    """Loads all checkpointed chunks straight from disk via DuckDB's parquet
    reader (no pandas concat of everything in memory)."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {final_table} AS
        SELECT * FROM read_parquet('{CHECKPOINT_DIR}/{tag}_fuzzy_chunk_*.parquet')
    """)

def run_feature_pipeline():
    con = get_connection()
    build_exact_features(con, "s1_norm", "s2_norm", "cand_s2_topk", "exact_s2")
    build_exact_features(con, "s1_norm", "s3_norm", "cand_s3_topk", "exact_s3")
    build_fuzzy_features_chunked(con, "exact_s2", "s2")
    build_fuzzy_features_chunked(con, "exact_s3", "s3")
    assemble_full_features(con, "s2", "features_s2")
    assemble_full_features(con, "s3", "features_s3")
    con.close()

if __name__ == "__main__":
    run_feature_pipeline()