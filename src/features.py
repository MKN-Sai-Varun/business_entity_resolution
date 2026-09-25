import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

def sql_features(con, candidates_table, s1_table, other_table):
    return con.execute(f"""
        SELECT
            c.source1_entity_id, c.candidate_entity_id,
            a.name_clean AS s1_name, b.name_clean AS cand_name,
            a.addr_clean AS s1_addr, b.addr_clean AS cand_addr,
            CAST(a.name_no_suffix = b.name_no_suffix AND a.name_no_suffix != '' AS INT) AS name_exact,
            CAST(a.name_compact = b.name_compact AND a.name_compact != '' AS INT) AS name_compact_exact,
            CAST(a.name_sorted_tokens = b.name_sorted_tokens AND a.name_sorted_tokens != '' AS INT) AS name_sorted_exact,
            CAST(a.addr_compact = b.addr_compact AND a.addr_compact != '' AS INT) AS addr_exact,
            CAST(a.addr_postal = b.addr_postal AND a.addr_postal != '' AS INT) AS addr_postal_match,
            CAST(a.addr_house_no = b.addr_house_no AND a.addr_house_no != '' AS INT) AS addr_house_match,
            CAST(lower(a.country) = lower(b.country) AS INT) AS country_match,
            CAST(a.addr_clean = '' AS INT) AS addr_missing_s1,
            CAST(b.addr_clean = '' AS INT) AS addr_missing_cand,
            length(a.name_clean) AS s1_name_len, length(b.name_clean) AS cand_name_len
        FROM {candidates_table} c
        JOIN {s1_table} a ON c.source1_entity_id = a.entity_id
        JOIN {other_table} b ON c.candidate_entity_id = b.entity_id
    """).df()

def add_fuzzy_features(df: pd.DataFrame) -> pd.DataFrame:
    s1_names = df["s1_name"].tolist(); cand_names = df["cand_name"].tolist()
    s1_addrs = df["s1_addr"].tolist(); cand_addrs = df["cand_addr"].tolist()
    n = len(df)
    name_ratio = np.empty(n, dtype=np.float32)
    name_tsort = np.empty(n, dtype=np.float32)
    name_jw = np.empty(n, dtype=np.float32)
    addr_ratio = np.empty(n, dtype=np.float32)
    addr_jw = np.empty(n, dtype=np.float32)
    for i in range(n):
        a, b = s1_names[i], cand_names[i]
        name_ratio[i] = fuzz.ratio(a, b) / 100.0
        name_tsort[i] = fuzz.token_sort_ratio(a, b) / 100.0
        name_jw[i] = JaroWinkler.similarity(a, b)
        ax, bx = s1_addrs[i], cand_addrs[i]
        addr_ratio[i] = fuzz.ratio(ax, bx) / 100.0
        addr_jw[i] = JaroWinkler.similarity(ax, bx)
    df["name_ratio"] = name_ratio
    df["name_token_sort"] = name_tsort
    df["name_jw"] = name_jw
    df["addr_ratio"] = addr_ratio
    df["addr_jw"] = addr_jw
    df["name_len_ratio"] = ((np.minimum(df["s1_name_len"], df["cand_name_len"]) + 1) /
                             (np.maximum(df["s1_name_len"], df["cand_name_len"]) + 1)).astype(np.float32)
    df["both_strong"] = ((df["name_ratio"] > 0.8) & (df["addr_ratio"] > 0.8)).astype(np.int8)
    df["name_strong_addr_weak"] = ((df["name_ratio"] > 0.85) & (df["addr_ratio"] < 0.4)).astype(np.int8)
    df["name_weak_addr_strong"] = ((df["name_ratio"] < 0.4) & (df["addr_ratio"] > 0.85)).astype(np.int8)
    return df.drop(columns=["s1_name", "cand_name", "s1_addr", "cand_addr"])

def build_features(con, candidates_table, s1_table, other_table, is_s2: bool) -> pd.DataFrame:
    df = sql_features(con, candidates_table, s1_table, other_table)
    df = add_fuzzy_features(df)
    df["is_s2"] = int(is_s2)
    return df
