# Business Entity Resolution — Amazon ML Challenge

## Setup
pip install -r requirements.txt
Place train files under dataset/train/, test files under dataset/test/
(or point config.py DATA_TRAIN/DATA_TEST at the Kaggle dataset mount).

## Run
cd src
python3 train_pipeline.py
python3 infer_pipeline.py
cd ..
python3 utils/own_validate.py output/matching_results.tsv output/candidate_pairs.tsv dataset/test
