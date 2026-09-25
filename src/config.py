from pathlib import Path

DATASET_SLUG = "datasets/mknsaivarun/amazon-ml-challenge-ber-data"

ROOT = Path("/kaggle/working/business_entity_resolution")
DATA_TRAIN = Path(f"/kaggle/input/{DATASET_SLUG}/dataset/train")
DATA_TEST = Path(f"/kaggle/input/{DATASET_SLUG}/dataset/test")
EXPERIMENTS_DIR = Path("/kaggle/working/experiments")
OUTPUT_DIR = Path("/kaggle/working/output")

SEED = 42
N_CV_FOLDS = 5
RARE_NGRAM_MAX_DF = 200
DUCKDB_MEMORY_LIMIT = "24GB"
DUCKDB_THREADS = 4

LGBM_PARAMS = dict(
    objective="binary", n_estimators=800, learning_rate=0.03, num_leaves=31,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, n_jobs=-1,
)

EXPERIMENTS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
