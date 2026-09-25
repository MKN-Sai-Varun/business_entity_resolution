import pandas as pd
from pathlib import Path

REQUIRED_COLS = ["entity_id", "business_name", "business_address", "country"]

def load_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    for c in REQUIRED_COLS:
        df[c] = df[c].fillna("").astype(str)
    return df

def load_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")
    df["match_set"] = df["matched_entity_ids"].apply(lambda s: set(x for x in s.split(",") if x))
    return df
