# Business Entity Resolution — Amazon ML Challenge

## Setup
pip install -r requirements.txt
Place train files under dataset/train/, test files under dataset/test/
(or point config.py DATA_TRAIN/DATA_TEST at the Kaggle dataset mount).

## Run (from the repo root, not from src/)
python3 -m src.train_pipeline
python3 -m src.infer_pipeline
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
