"""Render the evaluation results dict into a markdown report and supporting plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from config import (
    CALIBRATION_CURVE_PNG_NAME,
    CLASS_ORDER,
    CONFUSION_MATRIX_PNG_NAME,
    ROC_CURVE_PNG_NAME,
)


def save_confusion_heatmap(cm_normalized, path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm_normalized, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(len(CLASS_ORDER)):
        for j in range(len(CLASS_ORDER)):
            ax.text(j, i, f"{cm_normalized[i, j]:.0%}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_roc_curve(roc_auc_results: dict, path) -> None:
    """One ROC curve per class (one-vs-rest) plus the chance diagonal."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for cls in CLASS_ORDER:
        entry = roc_auc_results["per_class"][cls]
        ax.plot(entry["fpr"], entry["tpr"], label=f"{cls} (AUC = {entry['auc']:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_calibration_curve(true_labels, proba, path, n_bins: int = 10) -> None:
    """Reliability diagram per class: predicted confidence vs. observed frequency."""
    true_labels = np.asarray(true_labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    for cls in CLASS_ORDER:
        y_true_binary = (true_labels == cls).astype(int)
        y_score = np.asarray(proba[cls])
        frac_positive, mean_predicted = calibration_curve(
            y_true_binary, y_score, n_bins=n_bins, strategy="uniform"
        )
        ax.plot(mean_predicted, frac_positive, marker="o", label=cls)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration curve (reliability diagram)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _markdown_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def render_markdown_report(results: dict, n_records: int) -> str:
    cm = results["confusion"]
    cm_norm = results["confusion_normalized"]
    per_class = results["per_class"]
    cost = results["cost"]
    escalation = results["escalation"]
    method_breakdown = results["method_breakdown"]

    sections = [f"# PR Risk Classifier -- Evaluation Report\n\n{n_records} labeled records evaluated.\n"]

    sections.append("## Confusion matrix (rows = true, cols = predicted)\n")
    header = [""] + [f"pred {c}" for c in CLASS_ORDER]
    rows = [
        [f"true {c}"] + [f"{cm[i][j]} ({cm_norm[i][j]:.0%})" for j in range(len(CLASS_ORDER))]
        for i, c in enumerate(CLASS_ORDER)
    ]
    sections.append(_markdown_table(header, rows))
    sections.append(f"\n![Confusion matrix]({CONFUSION_MATRIX_PNG_NAME})\n")

    sections.append("\n## Per-class precision / recall / F1\n")
    header = ["Class", "Precision", "Recall", "F1", "Support"]
    rows = [
        [c, f"{per_class[c]['precision']:.2f}", f"{per_class[c]['recall']:.2f}",
         f"{per_class[c]['f1']:.2f}", per_class[c]["support"]]
        for c in CLASS_ORDER
    ]
    sections.append(_markdown_table(header, rows))
    sections.append(f"\n**Macro F1: {per_class['macro_f1']:.3f}**")
    sections.append(f"\n**Recall(CT1) -- north-star safety metric: {per_class['CT1']['recall']:.3f}**\n")

    sections.append("## ROC-AUC (one-vs-rest)\n")
    roc_auc = results["roc_auc"]
    header = ["Class", "AUC"]
    rows = [[c, f"{roc_auc['per_class'][c]['auc']:.3f}"] for c in CLASS_ORDER]
    sections.append(_markdown_table(header, rows))
    sections.append(f"\n**Macro AUC: {roc_auc['macro_auc']:.3f}**")
    sections.append(f"\n![ROC curves]({ROC_CURVE_PNG_NAME})\n")

    sections.append("## Calibration\n")
    sections.append(
        "How trustworthy is the model's confidence score? For a well-calibrated model, "
        "when it says \"70% confident\", it should be right about 70% of the time "
        "(points near the dashed diagonal = well calibrated).\n"
    )
    sections.append(f"![Calibration curve]({CALIBRATION_CURVE_PNG_NAME})\n")

    sections.append("## Severity-weighted cost\n")
    sections.append(f"- Total cost: {cost['total_cost']}")
    sections.append(f"- Mean cost per record: {cost['mean_cost_per_record']:.3f}\n")

    sections.append("## Escalation safety KPIs\n")
    ct1_rate = escalation["ct1_under_escalated"]
    bulk_rate = escalation["ct1_or_ct2_bulk_approved"]
    sections.append(
        f"- True CT1 predicted as CT2/CT3 (under-escalated): "
        f"{'n/a' if ct1_rate is None else f'{ct1_rate:.1%}'}"
    )
    sections.append(
        f"- True CT1/CT2 predicted as CT3 (bulk-approved): "
        f"{'n/a' if bulk_rate is None else f'{bulk_rate:.1%}'}\n"
    )

    sections.append(f"## Linear-weighted Cohen's Kappa: {results['weighted_kappa']:.3f}\n")

    sections.append("## Breakdown by decision method\n")
    header = ["Method", "Count", "Correct", "Accuracy"]
    rows = [
        [m, d["count"], d["correct"], f"{d['accuracy']:.2f}"]
        for m, d in sorted(method_breakdown.items())
    ]
    sections.append(_markdown_table(header, rows))

    return "\n".join(sections) + "\n"
