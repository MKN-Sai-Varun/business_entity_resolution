import numpy as np
import pandas as pd
from src.evaluation import macro_f0_5

def apply_decision(scored_df: pd.DataFrame, threshold: float, margin: float, singleton_cap: float):
    """scored_df: columns [s1_id, other_id, score] — one row per candidate, model score attached.
    Returns {s1_id: set(other_id)} predictions."""
    preds = {}
    for s1_id, group in scored_df.groupby("s1_id"):
        top = group["score"].max()
        if top < singleton_cap:
            preds[s1_id] = set()
            continue
        keep = group[group["score"] >= max(threshold, top - margin)]
        preds[s1_id] = set(keep["other_id"])
    return preds

def grid_search_decision(scored_df: pd.DataFrame, ground_truth: dict):
    """Grid search threshold/margin/singleton_cap directly against macro F0.5
    on out-of-fold predictions — never against a proxy metric."""
    best = {"macro_f0_5": -1}
    for singleton_cap in np.arange(0.3, 0.8, 0.05):
        for threshold in np.arange(0.4, 0.9, 0.05):
            for margin in np.arange(0.0, 0.3, 0.05):
                preds = apply_decision(scored_df, threshold, margin, singleton_cap)
                result = macro_f0_5(ground_truth, preds)
                if result["macro_f0_5"] > best["macro_f0_5"]:
                    best = {**result, "threshold": threshold, "margin": margin, "singleton_cap": singleton_cap}
    return best