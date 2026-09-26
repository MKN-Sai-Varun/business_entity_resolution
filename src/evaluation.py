import pandas as pd

def per_entity_f0_5(y_true: set, y_pred: set) -> float:
    if not y_true and not y_pred:
        return 1.0  # correct singleton
    if not y_pred:
        return 0.0  # missed everything (recall=0)
    tp = len(y_true & y_pred)
    precision = tp / len(y_pred)
    recall = tp / len(y_true) if y_true else 0.0
    if precision == 0 and recall == 0:
        return 0.0
    beta2 = 0.25
    denom = (beta2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom

def macro_f0_5(ground_truth: dict, predictions: dict) -> dict:
    """ground_truth / predictions: {s1_entity_id: set(matched_ids)}"""
    scores = []
    singleton_scores = []
    tp_total = fp_total = fn_total = 0
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = per_entity_f0_5(true_set, pred_set)
        scores.append(score)
        if not true_set:
            singleton_scores.append(score)
        tp_total += len(true_set & pred_set)
        fp_total += len(pred_set - true_set)
        fn_total += len(true_set - pred_set)

    pooled_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    pooled_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0

    return {
        "macro_f0_5": sum(scores) / len(scores),
        "singleton_f0_5": sum(singleton_scores) / len(singleton_scores) if singleton_scores else None,
        "pooled_precision": pooled_precision,
        "pooled_recall": pooled_recall,
        "n_entities": len(scores),
        "n_singletons": len(singleton_scores),
    }