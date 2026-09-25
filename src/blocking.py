import duckdb

def register_table(con, name, df):
    con.register(name, df)

def build_ngram_table(con, src_table, out_table, n=4):
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        SELECT entity_id,
               unnest(
                 list_transform(
                   generate_series(1, greatest(length(name_clean)-{n}+1, 1)),
                   i -> substr(name_clean, i, {n})
                 )
               ) AS ngram
        FROM {src_table}
        WHERE length(name_clean) >= {n}
    """)

def generate_candidates(con, s1_table, other_table, rare_max_df=200, out_table="candidates"):
    con.execute(f"""
        CREATE OR REPLACE TABLE cand_exact AS
        SELECT a.entity_id AS source1_entity_id, b.entity_id AS candidate_entity_id
        FROM {s1_table} a JOIN {other_table} b
          ON a.name_no_suffix = b.name_no_suffix AND a.name_no_suffix != ''
        UNION
        SELECT a.entity_id, b.entity_id FROM {s1_table} a JOIN {other_table} b
          ON a.addr_compact = b.addr_compact AND a.addr_compact != ''
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE cand_postal AS
        SELECT a.entity_id AS source1_entity_id, b.entity_id AS candidate_entity_id
        FROM {s1_table} a JOIN {other_table} b
          ON a.addr_postal = b.addr_postal AND a.addr_postal != ''
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE cand_house AS
        SELECT a.entity_id AS source1_entity_id, b.entity_id AS candidate_entity_id
        FROM {s1_table} a JOIN {other_table} b
          ON a.addr_house_no = b.addr_house_no AND a.addr_house_no != ''
         AND a.name_first_token = b.name_first_token AND a.name_first_token != ''
    """)
    build_ngram_table(con, s1_table, "s1_ngrams")
    build_ngram_table(con, other_table, "other_ngrams")
    con.execute("""
        CREATE OR REPLACE TABLE ngram_freq AS
        SELECT ngram, count(*) AS df FROM other_ngrams GROUP BY ngram
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE cand_ngram AS
        SELECT DISTINCT s.entity_id AS source1_entity_id, o.entity_id AS candidate_entity_id
        FROM s1_ngrams s
        JOIN ngram_freq f ON s.ngram = f.ngram AND f.df <= {rare_max_df}
        JOIN other_ngrams o ON s.ngram = o.ngram
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE {out_table} AS
        SELECT DISTINCT * FROM cand_exact
        UNION SELECT DISTINCT * FROM cand_postal
        UNION SELECT DISTINCT * FROM cand_house
        UNION SELECT DISTINCT * FROM cand_ngram
    """)
    return con.table(out_table)

def candidate_recall_audit(con, candidates_table, gt_pairs_table, source_prefix):
    return con.execute(f"""
        WITH true_pairs AS (
            SELECT * FROM {gt_pairs_table} WHERE matched_entity_id LIKE '{source_prefix}%'
        ),
        hits AS (
            SELECT t.* FROM true_pairs t
            JOIN {candidates_table} c
              ON t.source1_entity_id = c.source1_entity_id
             AND t.matched_entity_id = c.candidate_entity_id
        )
        SELECT (SELECT count(*) FROM hits) AS recovered,
               (SELECT count(*) FROM true_pairs) AS total_true,
               CAST((SELECT count(*) FROM hits) AS DOUBLE) / NULLIF((SELECT count(*) FROM true_pairs), 0) AS recall
    """).df()
