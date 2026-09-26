import duckdb
from src.db import get_connection
from src.config import RARE_GRAM_MAX_DOC_FREQ, TOP_K_PER_SOURCE

def build_candidates(con, s1_table, other_table, out_table):
    """Union of blocking channels against ONE target source table (S2 or S3)."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_raw AS
        WITH channel_name AS (
            SELECT a.entity_id AS s1_id, b.entity_id AS other_id
            FROM {s1_table} a JOIN {other_table} b ON a.name_norm = b.name_norm AND a.name_norm != ''
        ),
        channel_addr AS (
            SELECT a.entity_id AS s1_id, b.entity_id AS other_id
            FROM {s1_table} a JOIN {other_table} b ON a.address_norm = b.address_norm AND a.address_norm != ''
        ),
        channel_postal AS (
            SELECT a.entity_id AS s1_id, b.entity_id AS other_id
            FROM {s1_table} a JOIN {other_table} b ON a.postal_code = b.postal_code AND a.postal_code IS NOT NULL
        ),
        channel_house_name AS (
            SELECT a.entity_id AS s1_id, b.entity_id AS other_id
            FROM {s1_table} a JOIN {other_table} b
              ON a.house_number = b.house_number AND a.house_number IS NOT NULL
             AND split_part(a.name_norm, ' ', 1) = split_part(b.name_norm, ' ', 1)
        )
        SELECT DISTINCT s1_id, other_id FROM (
            SELECT * FROM channel_name
            UNION SELECT * FROM channel_addr
            UNION SELECT * FROM channel_postal
            UNION SELECT * FROM channel_house_name
        )
    """)
    raw_count = con.execute(f"SELECT count(*) FROM {out_table}_raw").fetchone()[0]
    print(f"{out_table}: raw candidates (4 cheap channels, vs {other_table}) = {raw_count:,}")

def add_rare_gram_channel(con, s1_table, other_table, out_table):
    """5th channel: shared rare 4-grams, done via a Python UDF registered in DuckDB
    so the gram explosion happens once and the join stays a normal hash join."""
    con.execute(f"""
        CREATE OR REPLACE TABLE grams_s1 AS
        SELECT entity_id, unnest(
            [substr(name_norm, i, 4) FOR i IN generate_series(1, greatest(length(name_norm)-3,1))]
        ) AS gram
        FROM {s1_table} WHERE length(name_norm) >= 4
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE grams_other AS
        SELECT entity_id, unnest(
            [substr(name_norm, i, 4) FOR i IN generate_series(1, greatest(length(name_norm)-3,1))]
        ) AS gram
        FROM {other_table} WHERE length(name_norm) >= 4
    """)
    total_s1 = con.execute(f"SELECT count(DISTINCT entity_id) FROM {s1_table}").fetchone()[0]
    con.execute(f"""
        CREATE OR REPLACE TABLE rare_grams AS
        SELECT gram FROM (
            SELECT gram, count(DISTINCT entity_id) AS df FROM grams_s1 GROUP BY gram
        ) WHERE df::DOUBLE / {total_s1} < {RARE_GRAM_MAX_DOC_FREQ}
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_gram AS
        SELECT DISTINCT g1.entity_id AS s1_id, g2.entity_id AS other_id
        FROM grams_s1 g1
        JOIN rare_grams r ON g1.gram = r.gram
        JOIN grams_other g2 ON g1.gram = g2.gram
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table}_raw AS
        SELECT s1_id, other_id FROM {out_table}_raw
        UNION SELECT s1_id, other_id FROM {out_table}_gram
    """)

def coarse_rank_and_cut(con, s1_table, other_table, raw_table, final_table, k):
    """Cheap SQL score (token overlap) -> keep top-K per S1 entity. This is what
    keeps the fuzzy-feature stage from having to touch 100M+ rows."""
    con.execute(f"""
        CREATE OR REPLACE TABLE {final_table} AS
        WITH scored AS (
            SELECT
                r.s1_id, r.other_id,
                len(list_intersect(
                    string_split(a.name_norm, ' '),
                    string_split(b.name_norm, ' ')
                )) AS token_overlap
            FROM {raw_table} r
            JOIN {s1_table} a ON r.s1_id = a.entity_id
            JOIN {other_table} b ON r.other_id = b.entity_id
        ),
        ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY s1_id ORDER BY token_overlap DESC
            ) AS rnk
            FROM scored
        )
        SELECT s1_id, other_id FROM ranked WHERE rnk <= {k}
    """)
    n = con.execute(f"SELECT count(*) FROM {final_table}").fetchone()[0]
    print(f"{final_table}: after top-{k} cut = {n:,} rows")

def run_blocking_train():
    con = get_connection()
    build_candidates(con, "s1_norm", "s2_norm", "cand_s2")
    build_candidates(con, "s1_norm", "s3_norm", "cand_s3")
    add_rare_gram_channel(con, "s1_norm", "s2_norm", "cand_s2")
    add_rare_gram_channel(con, "s1_norm", "s3_norm", "cand_s3")
    coarse_rank_and_cut(con, "s1_norm", "s2_norm", "cand_s2_raw", "cand_s2_topk", TOP_K_PER_SOURCE)
    coarse_rank_and_cut(con, "s1_norm", "s3_norm", "cand_s3_raw", "cand_s3_topk", TOP_K_PER_SOURCE)
    con.close()

if __name__ == "__main__":
    run_blocking_train()