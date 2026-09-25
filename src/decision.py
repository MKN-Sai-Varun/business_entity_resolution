import numpy as np
from itertools import product
from evaluation import macro_f_beta

def scores_to_pred_dict(scored_df, threshold, margin, singleton_max):
    pred = {}
    for s1_id, group in scored_df.groupby("source1_entity_id"):
        top = group["score"].max()
        if top < singleton_max:
            pred[s1_id] = set(); continue
        keep = group[(group["score"] >= threshold) & (group["score"] >= top - margin)]
        pred[s1_id] = set(keep["candidate_entity_id"])
    return pred

def search_thresholds(scored_df, true_dict, all_s1_ids):
    best = None
    thresholds = np.arange(0.30, 0.85, 0.05)
    margins = [0.05, 0.10, 0.15, 0.20, 0.30]
    singleton_caps = np.arange(0.20, 0.60, 0.05)
    for t, m, s in product(thresholds, margins, singleton_caps):
        if s > t: continue
        pred = scores_to_pred_dict(scored_df, t, m, s)
        for eid in all_s1_ids: pred.setdefault(eid, set())
        macro_f05, _ = macro_f_beta(pred, true_dict, beta=0.5)
        if best is None or macro_f05 > best[0]:
            best = (macro_f05, t, m, s)
    return {"macro_f0.5": best[0], "threshold": best[1], "margin": best[2], "singleton_max_score": best[3]}
