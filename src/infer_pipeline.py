import duckdb, json, joblib
import pandas as pd, numpy as np

import config
from data import load_source
from normalization import normalize_df
from blocking import register_table, generate_candidates
from features import build_features
from decision import scores_to_pred_dict

def write_tsv(d, all_ids, path, col_name):
    rows = [(eid, ",".join(sorted(d.get(eid, set())))) for eid in all_ids]
    pd.DataFrame(rows, columns=["source1_entity_id", col_name]).to_csv(path, sep="\t", index=False)

def main():
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads TO {config.DUCKDB_THREADS}")

    s1 = normalize_df(load_source(config.DATA_TEST / "test_source1.tsv"))
    s2 = normalize_df(load_source(config.DATA_TEST / "test_source2.tsv"))
    s3 = normalize_df(load_source(config.DATA_TEST / "test_source3.tsv"))
    all_s1_ids = list(s1["entity_id"])

    register_table(con, "s1", s1); register_table(con, "s2", s2); register_table(con, "s3", s3)

    generate_candidates(con, "s1", "s2", rare_max_df=config.RARE_NGRAM_MAX_DF, out_table="candidates_s2")
    generate_candidates(con, "s1", "s3", rare_max_df=config.RARE_NGRAM_MAX_DF, out_table="candidates_s3")

    cand_s2 = con.execute("SELECT * FROM candidates_s2").df()
    cand_s3 = con.execute("SELECT * FROM candidates_s3").df()
    all_candidates = {}
    for eid in all_s1_ids:
        c2 = set(cand_s2.loc[cand_s2.source1_entity_id == eid, "candidate_entity_id"])
        c3 = set(cand_s3.loc[cand_s3.source1_entity_id == eid, "candidate_entity_id"])
        all_candidates[eid] = c2 | c3
    write_tsv(all_candidates, all_s1_ids, config.OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids")

    models = joblib.load(config.EXPERIMENTS_DIR / "models.pkl")
    feature_cols = joblib.load(config.EXPERIMENTS_DIR / "feature_cols.pkl")
    with open(config.EXPERIMENTS_DIR / "decision_config.json") as f:
        cfg = json.load(f)

    feat_s2 = build_features(con, "candidates_s2", "s1", "s2", is_s2=True)
    feat_s3 = build_features(con, "candidates_s3", "s1", "s3", is_s2=False)
    pairs = pd.concat([feat_s2, feat_s3], ignore_index=True)
    X = pairs[feature_cols]

    scores = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)
    scored = pairs[["source1_entity_id", "candidate_entity_id"]].copy()
    scored["score"] = scores

    pred = scores_to_pred_dict(scored, cfg["threshold"], cfg["margin"], cfg["singleton_max_score"])
    write_tsv(pred, all_s1_ids, config.OUTPUT_DIR / "matching_results.tsv", "matched_entity_ids")
    print("Wrote candidate_pairs.tsv and matching_results.tsv to", config.OUTPUT_DIR)

if __name__ == "__main__":
    main()
