import pickle
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from src.db import get_connection
from src.normalization import build_normalized_tables
from src.blocking import run_blocking_train
from src.features import run_feature_pipeline
from src.evaluation import macro_f0_5
from src.decision import grid_search_decision
from src.config import N_FOLDS, SEED, OUTPUT_DIR
 
FEATURE_COLS = [
    "name_exact", "address_exact", "postal_match", "house_match", "country_match",
    "s1_addr_missing", "other_addr_missing",
    "name_lev", "name_token_sort", "addr_lev", "addr_token_sort",
    "cosine_sim",
]

def load_ground_truth_dict(con) -> dict:
    rows = con.execute("""
        SELECT source1_entity_id, matched_entity_ids
        FROM read_csv('dataset/train/train_ground_truth.tsv', delim='\t', header=true)
    """).fetchall()
    return {s1_id: (set(m.split(",")) if m else set()) for s1_id, m in rows}

def build_ground_truth_table(con):
    con.execute("""
        CREATE OR REPLACE TABLE gt_pairs AS
        SELECT source1_entity_id AS s1_id, unnest(string_split(matched_entity_ids, ',')) AS other_id
        FROM read_csv('dataset/train/train_ground_truth.tsv', delim='\t', header=true)
        WHERE matched_entity_ids != ''
    """)



def merge_ann_candidates(con):
    for tag in ("s2", "s3"):
        con.execute(f"""
            CREATE OR REPLACE TABLE features_{tag} AS
            SELECT f.*, coalesce(a.cosine_sim, 0.0) AS cosine_sim
            FROM features_{tag} f
            LEFT JOIN read_parquet('{OUTPUT_DIR}/ann_candidates_{tag}_train.parquet') a
              ON f.s1_id = a.s1_id AND f.other_id = a.other_id
        """)

def build_labels(con):
    for tag in ("s2", "s3"):
        con.execute(f"""
            CREATE OR REPLACE TABLE labels_{tag} AS
            SELECT f.s1_id, f.other_id,
                   CASE WHEN g.other_id IS NOT NULL THEN 1 ELSE 0 END AS label
            FROM features_{tag} f
            LEFT JOIN gt_pairs g ON f.s1_id = g.s1_id AND f.other_id = g.other_id
        """)

def train_model(con):
    build_ground_truth_table(con)
    build_labels(con)
    gt = load_ground_truth_dict(con)

    full = con.execute("""
        SELECT f.*, l.label FROM (
            SELECT * FROM features_s2 UNION ALL SELECT * FROM features_s3
        ) f
        JOIN (
            SELECT * FROM labels_s2 UNION ALL SELECT * FROM labels_s3
        ) l ON f.s1_id = l.s1_id AND f.other_id = l.other_id
    """).fetchdf()
    full[FEATURE_COLS] = full[FEATURE_COLS].fillna(0)

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_scores = pd.Series(index=full.index, dtype=float)
    models = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(full, groups=full["s1_id"])):
        train_set = lgb.Dataset(full.loc[train_idx, FEATURE_COLS], label=full.loc[train_idx, "label"])
        val_set = lgb.Dataset(full.loc[val_idx, FEATURE_COLS], label=full.loc[val_idx, "label"])
        model = lgb.train(
            {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
             "num_leaves": 31, "seed": SEED, "verbosity": -1},
            train_set, num_boost_round=500, valid_sets=[val_set],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
        )
        oof_scores.loc[val_idx] = model.predict(full.loc[val_idx, FEATURE_COLS])
        models.append(model)
        print(f"Fold {fold} done, best iter {model.best_iteration}")

    full["score"] = oof_scores
    best = grid_search_decision(full[["s1_id", "other_id", "score"]], gt)
    print("Best decision config (measured on out-of-fold predictions):", best)

    with open(OUTPUT_DIR / "best_decision_config.pkl", "wb") as f:
        pickle.dump(best, f)
    for i, model in enumerate(models):
        model.save_model(str(OUTPUT_DIR / f"lgbm_fold{i}.txt"))
    return best

def main():
    print("Stage 1/4: normalization")
    build_normalized_tables()
    print("Stage 2/4: blocking")
    run_blocking_train()
    print("Stage 3/4: features (checkpointed, resumable)")
    run_feature_pipeline()
    con = get_connection()
    print("Stage 3.5/4: merging embedding channel (run after the SageMaker job + download)")
    merge_ann_candidates(con)
    print("Stage 4/4: training + decision tuning")
    best_config = train_model(con)
    con.close()
    print("DONE. Best config:", best_config)

if __name__ == "__main__":
    main()