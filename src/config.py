import os
from pathlib import Path

# Toggle between local dev and SageMaker Processing paths via env var
RUNTIME = os.environ.get("BER_RUNTIME", "local")  # "local" or "sagemaker"

if RUNTIME == "sagemaker":
    DATA_DIR = Path("/opt/ml/processing/input")
    OUTPUT_DIR = Path("/opt/ml/processing/output")
else:
    DATA_DIR = Path("dataset")
    OUTPUT_DIR = Path("output")

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

DB_PATH = OUTPUT_DIR / "pipeline.duckdb"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Blocking / coarse-cut params — TUNE THESE against measured recall, don't assume
RARE_GRAM_MAX_DOC_FREQ = 0.001   # gram must appear in <0.1% of entities to be used
TOP_K_PER_SOURCE = 25            # coarse cut candidates kept per S1 entity, per source (S2/S3 separately)

# Fuzzy feature batching
FUZZY_CHUNK_SIZE = 300_000       # rows per rapidfuzz batch

SEED = 42
N_FOLDS = 5