import pickle
import lightgbm as lgb
from src.db import get_connection
from src.normalization import build_normalized_test_tables
from src.blocking import build_candidates, add_rare_gram_channel, coarse_rank_and_cut
from src.features import build_exact_features, build_fuzzy_features_chunked, assemble_full_features
from src.decision import apply_decision
from src.config import TOP_K_PER_SOURCE, OUTPUT_DIR

FEATURE_COLS = [
    "name_exact", "address_exact", "postal_match", "house_match", "country_match",
    "s1_addr_missing", "other_addr_missing",
    "name_lev", "name_token_sort", "addr_lev", "addr_token_sort",
    "cosine_sim",
]

def run_blocking_test(con):
    build_candidates(con, "s1_norm_test", "s2_norm_test", "cand_s2_test")
    build_candidates(con, "s1_norm_test", "s3_norm_test", "cand_s3_test")
    add_rare_gram_channel(con, "s1_norm_test", "s2_norm_test", "cand_s2_test")
    add_rare_gram_channel(con, "s1_norm_test", "s3_norm_test", "cand_s3_test")
    coarse_rank_and_cut(con, "s1_norm_test", "s2_norm_test", "cand_s2_test_raw", "cand_s2_test_topk", TOP_K_PER_SOURCE)
    coarse_rank_and_cut(con, "s1_norm_test", "s3_norm_test", "cand_s3_test_raw", "cand_s3_test_topk", TOP_K_PER_SOURCE)

def run_features_test(con):
    build_exact_features(con, "s1_norm_test", "s2_norm_test", "cand_s2_test_topk", "exact_s2_test")
    build_exact_features(con, "s1_norm_test", "s3_norm_test", "cand_s3_test_topk", "exact_s3_test")
    build_fuzzy_features_chunked(con, "exact_s2_test", "s2_test")
    build_fuzzy_features_chunked(con, "exact_s3_test", "s3_test")
    assemble_full_features(con, "s2_test", "features_s2_test")
    assemble_full_features(con, "s3_test", "features_s3_test")

def merge_ann_test(con):
    for tag in ("s2", "s3"):
        con.execute(f"""
            CREATE OR REPLACE TABLE features_{tag} AS
            SELECT f.*, coalesce(a.cosine_sim, 0.0) AS cosine_sim
            FROM features_{tag} f
            LEFT JOIN read_parquet('{OUTPUT_DIR}/ann_candidates_{tag}_test.parquet') a
              ON f.s1_id = a.s1_id AND f.other_id = a.other_id
        """)

def score_and_predict(con, best_config):
    full = con.execute("SELECT * FROM features_s2_test UNION ALL SELECT * FROM features_s3_test").fetchdf()
    full[FEATURE_COLS] = full[FEATURE_COLS].fillna(0)

    fold_preds, i = [], 0
    while (OUTPUT_DIR / f"lgbm_fold{i}.txt").exists():
        model = lgb.Booster(model_file=str(OUTPUT_DIR / f"lgbm_fold{i}.txt"))
        fold_preds.append(model.predict(full[FEATURE_COLS]))
        i += 1
    full["score"] = sum(fold_preds) / len(fold_preds)

    return apply_decision(full[["s1_id", "other_id", "score"]],
                           best_config["threshold"], best_config["margin"], best_config["singleton_cap"])

def write_outputs(con, preds):
    s1_ids = [r[0] for r in con.execute("SELECT entity_id FROM s1_norm_test").fetchall()]

    with open(OUTPUT_DIR / "matching_results.tsv", "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            f.write(f"{s1_id}\t{','.join(sorted(preds.get(s1_id, set())))}\n")

    cand = con.execute("""
        SELECT s1_id, other_id FROM cand_s2_test_topk UNION SELECT s1_id, other_id FROM cand_s3_test_topk
    """).fetchdf()
    grouped = cand.groupby("s1_id")["other_id"].apply(lambda x: ",".join(sorted(x)))
    with open(OUTPUT_DIR / "candidate_pairs.tsv", "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids:
            f.write(f"{s1_id}\t{grouped.get(s1_id, '')}\n")

def main():
    con = get_connection()
    print("Stage 1/4: normalize test data")
    build_normalized_test_tables()
    print("Stage 2/4: block test data")
    run_blocking_test(con)
    print("Stage 3/4: features on test data")
    run_features_test(con)
    merge_ann_test(con)
    print("Stage 4/4: score + decide + write outputs")
    with open(OUTPUT_DIR / "best_decision_config.pkl", "rb") as f:
        best_config = pickle.load(f)
    preds = score_and_predict(con, best_config)
    write_outputs(con, preds)
    con.close()
    print("DONE -> output/matching_results.tsv, output/candidate_pairs.tsv")

if __name__ == "__main__":
    main()