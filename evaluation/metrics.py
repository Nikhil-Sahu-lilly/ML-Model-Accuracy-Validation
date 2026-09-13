"""Pure metric functions over (true_label, pred_label) arrays. No I/O -- easy to unit test."""
from collections import Counter

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from config import CLASS_ORDER, COST_MATRIX


def confusion(true_labels, pred_labels) -> np.ndarray:
    """3x3 confusion matrix, rows/cols ordered by CLASS_ORDER (severity order)."""
    return confusion_matrix(true_labels, pred_labels, labels=CLASS_ORDER)


def confusion_normalized(true_labels, pred_labels) -> np.ndarray:
    """Row-normalized confusion matrix (each row sums to 1, i.e. % of that true class)."""
    cm = confusion(true_labels, pred_labels).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return cm / row_sums


def per_class_report(true_labels, pred_labels) -> dict:
    """Precision/recall/F1 per class, plus macro-averaged F1."""
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=CLASS_ORDER, zero_division=0
    )
    report = {
        cls: {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": int(support[i]),
        }
        for i, cls in enumerate(CLASS_ORDER)
    }
    report["macro_f1"] = float(np.mean(f1))
    return report


def cost_weighted_matrix(true_labels, pred_labels) -> dict:
    """Apply COST_MATRIX to each (true, pred) pair. Returns total, mean-per-record, and per-cell cost."""
    cell_cost = {t: {p: 0 for p in CLASS_ORDER} for t in CLASS_ORDER}
    total_cost = 0
    n = len(true_labels)
    for t, p in zip(true_labels, pred_labels):
        cost = COST_MATRIX[t][p]
        cell_cost[t][p] += cost
        total_cost += cost
    return {
        "total_cost": total_cost,
        "mean_cost_per_record": total_cost / n if n else 0.0,
        "cell_cost": cell_cost,
    }


def escalation_rates(true_labels, pred_labels) -> dict:
    """Safety KPIs: how often a truly severe PR is routed to a less severe path."""
    true_labels = list(true_labels)
    pred_labels = list(pred_labels)

    def rate(true_set, pred_set):
        eligible = [i for i, t in enumerate(true_labels) if t in true_set]
        if not eligible:
            return None
        misses = sum(1 for i in eligible if pred_labels[i] in pred_set)
        return misses / len(eligible)

    return {
        "ct1_under_escalated": rate({"CT1"}, {"CT2", "CT3"}),
        "ct1_or_ct2_bulk_approved": rate({"CT1", "CT2"}, {"CT3"}),
    }


def roc_auc_ovr(true_labels, proba) -> dict:
    """One-vs-rest ROC-AUC per class, plus (fpr, tpr) curve points and a macro average.

    `proba` maps each class in CLASS_ORDER to an array-like of predicted probabilities
    for that class (e.g. the proba_CT1/CT2/CT3 columns from run_predictions.py).
    """
    true_labels = np.asarray(true_labels)
    per_class = {}
    for cls in CLASS_ORDER:
        y_true_binary = (true_labels == cls).astype(int)
        y_score = np.asarray(proba[cls])
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        per_class[cls] = {"auc": float(roc_auc_score(y_true_binary, y_score)), "fpr": fpr, "tpr": tpr}
    macro_auc = float(np.mean([per_class[cls]["auc"] for cls in CLASS_ORDER]))
    return {"per_class": per_class, "macro_auc": macro_auc}


def weighted_kappa(true_labels, pred_labels) -> float:
    """Linear-weighted Cohen's Kappa -- ordinal agreement, secondary to the cost-weighted matrix."""
    return cohen_kappa_score(true_labels, pred_labels, labels=CLASS_ORDER, weights="linear")


def method_breakdown(true_labels, pred_labels, methods) -> dict:
    """Correctness rate grouped by the decision `method` predict.py used (ml vs. safety-net rule)."""
    counts = Counter(methods)
    correct = Counter()
    for t, p, m in zip(true_labels, pred_labels, methods):
        if t == p:
            correct[m] += 1
    return {
        method: {
            "count": count,
            "correct": correct.get(method, 0),
            "accuracy": correct.get(method, 0) / count,
        }
        for method, count in counts.items()
    }


def evaluate_all(true_labels, pred_labels, methods, proba) -> dict:
    """Run every metric above and return one combined results dict."""
    return {
        "confusion": confusion(true_labels, pred_labels),
        "confusion_normalized": confusion_normalized(true_labels, pred_labels),
        "per_class": per_class_report(true_labels, pred_labels),
        "cost": cost_weighted_matrix(true_labels, pred_labels),
        "escalation": escalation_rates(true_labels, pred_labels),
        "weighted_kappa": weighted_kappa(true_labels, pred_labels),
        "method_breakdown": method_breakdown(true_labels, pred_labels, methods),
        "roc_auc": roc_auc_ovr(true_labels, proba),
    }
