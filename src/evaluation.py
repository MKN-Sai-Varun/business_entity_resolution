import numpy as np

def entity_f_beta(pred_set: set, true_set: set, beta=0.5) -> float:
    if not true_set and not pred_set:
        return 1.0
    if not pred_set:
        return 0.0
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if not true_set else 0.0)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)

def macro_f_beta(pred_dict: dict, true_dict: dict, beta=0.5):
    scores = [entity_f_beta(pred_dict.get(k, set()), v, beta) for k, v in true_dict.items()]
    return float(np.mean(scores)), scores

def full_report(pred_dict, true_dict):
    macro_f05, _ = macro_f_beta(pred_dict, true_dict, beta=0.5)
    macro_f1, _ = macro_f_beta(pred_dict, true_dict, beta=1.0)
    singleton_ids = [k for k, v in true_dict.items() if len(v) == 0]
    singleton_f05 = float(np.mean([entity_f_beta(pred_dict.get(k, set()), true_dict[k], 0.5)
                                    for k in singleton_ids])) if singleton_ids else float("nan")
    tp = fp = fn = 0
    for s1_id, true_set in true_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        tp += len(pred_set & true_set); fp += len(pred_set - true_set); fn += len(true_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "macro_f0.5": macro_f05, "macro_f1": macro_f1,
        "pooled_precision": precision, "pooled_recall": recall,
        "singleton_f0.5": singleton_f05,
        "n_entities": len(true_dict), "n_singletons": len(singleton_ids),
    }, None
