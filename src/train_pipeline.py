import duckdb, json, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import GroupKFold
import lightgbm as lgb

import config
from data import load_source, load_ground_truth
from normalization import normalize_df
from blocking import register_table, generate_candidates, candidate_recall_audit
from features import build_features
from decision import search_thresholds, scores_to_pred_dict
from evaluation import full_report

def explode_ground_truth(gt_df):
    rows = []
    for r in gt_df.itertuples():
        for m in r.match_set:
            rows.append((r.source1_entity_id, m))
    return pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_id"])

def main():
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit='{config.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads TO {config.DUCKDB_THREADS}")

    print("Loading + normalizing...")
    s1 = normalize_df(load_source(config.DATA_TRAIN / "train_source1.tsv"))
    s2 = normalize_df(load_source(config.DATA_TRAIN / "train_source2.tsv"))
    s3 = normalize_df(load_source(config.DATA_TRAIN / "train_source3.tsv"))
    gt = load_ground_truth(config.DATA_TRAIN / "train_ground_truth.tsv")
    gt_pairs = explode_ground_truth(gt)
    print(f"S1={len(s1)} S2={len(s2)} S3={len(s3)} GT={len(gt)}")

    register_table(con, "s1", s1); register_table(con, "s2", s2)
    register_table(con, "s3", s3); register_table(con, "gt_pairs", gt_pairs)

    print("Blocking S1->S2...")
    generate_candidates(con, "s1", "s2", rare_max_df=config.RARE_NGRAM_MAX_DF, out_table="candidates_s2")
    print("Blocking S1->S3...")
    generate_candidates(con, "s1", "s3", rare_max_df=config.RARE_NGRAM_MAX_DF, out_table="candidates_s3")

    r2 = candidate_recall_audit(con, "candidates_s2", "gt_pairs", "S2-")
    r3 = candidate_recall_audit(con, "candidates_s3", "gt_pairs", "S3-")
    print("S1->S2 recall:\n", r2)
    print("S1->S3 recall:\n", r3)
    print("Candidate counts:",
          con.execute("SELECT count(*) FROM candidates_s2").fetchone(),
          con.execute("SELECT count(*) FROM candidates_s3").fetchone())

    print("Building features...")
    feat_s2 = build_features(con, "candidates_s2", "s1", "s2", is_s2=True)
    feat_s3 = build_features(con, "candidates_s3", "s1", "s3", is_s2=False)
    pairs = pd.concat([feat_s2, feat_s3], ignore_index=True)

    gt_lookup = dict(zip(gt["source1_entity_id"], gt["match_set"]))
    pairs["label"] = [int(cid in gt_lookup.get(sid, set()))
                       for sid, cid in zip(pairs["source1_entity_id"], pairs["candidate_entity_id"])]
    print("Total pairs:", len(pairs), "Positives:", pairs["label"].sum())

    feature_cols = [c for c in pairs.columns if c not in ("source1_entity_id", "candidate_entity_id", "label")]
    X = pairs[feature_cols]; y = pairs["label"].to_numpy(); groups = pairs["source1_entity_id"].to_numpy()

    print("Training CV models...")
    gkf = GroupKFold(n_splits=config.N_CV_FOLDS)
    oof = np.zeros(len(y)); models = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        m = lgb.LGBMClassifier(**config.LGBM_PARAMS)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        models.append(m)
        print(f"fold {fold} best_iter={m.best_iteration_}")

    scored = pairs[["source1_entity_id", "candidate_entity_id"]].copy()
    scored["score"] = oof

    all_s1_ids = list(s1["entity_id"])
    true_dict = {eid: gt_lookup.get(eid, set()) for eid in all_s1_ids}

    print("Searching thresholds...")
    best_cfg = search_thresholds(scored, true_dict, all_s1_ids)
    print("Best decision config:", best_cfg)

    pred = scores_to_pred_dict(scored, best_cfg["threshold"], best_cfg["margin"], best_cfg["singleton_max_score"])
    for eid in all_s1_ids: pred.setdefault(eid, set())
    report, _ = full_report(pred, true_dict)
    print("Validation report:", json.dumps(report, indent=2))

    joblib.dump(models, config.EXPERIMENTS_DIR / "models.pkl")
    joblib.dump(feature_cols, config.EXPERIMENTS_DIR / "feature_cols.pkl")
    with open(config.EXPERIMENTS_DIR / "decision_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)
    print("Saved artifacts to", config.EXPERIMENTS_DIR)

if __name__ == "__main__":
    main()