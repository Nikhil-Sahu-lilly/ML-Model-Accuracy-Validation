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
    sections.append(
        "Each row is a true class, each column is where the model routed it. "
        "The diagonal is correct calls; everything off-diagonal is a misroute.\n\n"
        "*Why it matters here:*\n"
        "- Shows the *direction* of errors, not just the count -- CT1 leaking into CT2/CT3 "
        "(a real risk under-caught) is far worse than CT3 leaking into CT1 (just extra review).\n"
        "- Pinpoints exactly which class pairs get confused, so we know where to add training "
        "data or tighten rules instead of guessing.\n"
        "- Gives auditors/compliance an exact, countable trail instead of a single opaque score.\n"
    )
    header = [""] + [f"pred {c}" for c in CLASS_ORDER]
    rows = [
        [f"true {c}"] + [f"{cm[i][j]} ({cm_norm[i][j]:.0%})" for j in range(len(CLASS_ORDER))]
        for i, c in enumerate(CLASS_ORDER)
    ]
    sections.append(_markdown_table(header, rows))
    sections.append(f"\n![Confusion matrix]({CONFUSION_MATRIX_PNG_NAME})\n")

    sections.append("\n## Per-class precision / recall / F1\n")
    sections.append(
        "Precision = of the PRs we called class X, how many really were X. "
        "Recall = of the PRs that really were X, how many we caught. "
        "F1 balances the two; Macro F1 averages the three classes equally so the small "
        "CT3 class can't be hidden by CT1/CT2 volume.\n\n"
        "*Why it matters here:*\n"
        "- **Recall(CT1)** is our north-star safety metric -- it says how many genuinely "
        "high-risk PRs the model actually catches, which is the failure mode we can least afford.\n"
        "- Precision on CT1 tells us reviewer burden: low precision means reviewers keep getting "
        "paged on PRs that turn out to be low-risk.\n"
        "- Macro F1 (not accuracy) keeps CT3 performance visible instead of averaged away by the "
        "larger CT1/CT2 classes.\n"
    )
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
    sections.append(
        "For each class, treats it as \"this class vs. everything else\" and measures how well the "
        "model's confidence score ranks true members above non-members, at every possible threshold.\n\n"
        "*Why it matters here:*\n"
        "- Threshold-independent, so it separates \"is the underlying signal good\" from \"did we pick "
        "the right cutoff\" -- a class can have high AUC but weak precision/recall if the threshold "
        "needs tuning, which is a cheaper fix than retraining.\n"
        "- A high AUC on CT3 despite 0 recall/precision (see table above) is a good demo point: the "
        "model *can* separate CT3, the current decision rule is just conservatively routing it to CT1.\n"
        "- Macro AUC gives one comparable number across model versions without depending on class balance.\n"
    )
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
        "(points near the dashed diagonal = well calibrated).\n\n"
        "*Why it matters here:*\n"
        "- Confidence scores drive the decision method itself (high-conf vs. flag-for-review vs. "
        "low-conf-default) -- if scores are overconfident we auto-approve risk we shouldn't, if "
        "underconfident we over-flag safe PRs and waste reviewer time.\n"
        "- Well-calibrated scores let reviewers treat \"90% CT1\" as a real risk level, supporting a "
        "graduated response instead of a blunt yes/no call.\n"
        "- Miscalibration is invisible in accuracy/F1/AUC alone -- this is the only view that would catch it.\n"
    )
    sections.append(f"![Calibration curve]({CALIBRATION_CURVE_PNG_NAME})\n")

    sections.append("## Severity-weighted cost\n")
    sections.append(
        "Applies the business cost matrix (under-escalating a real risk costs far more than "
        "over-escalating a safe PR; correct calls are free) to every prediction and sums it into "
        "one number.\n\n"
        "*Why it matters here:*\n"
        "- Translates model errors into actual process economics -- a missed CT1 costs 10x an "
        "over-cautious CT2-called-CT3, matching how the business actually experiences the mistake.\n"
        "- Gives one comparable number across model versions that reflects real consequence, not "
        "just raw accuracy.\n"
        "- Easy to justify a model change to stakeholders in terms of expected risk reduction rather "
        "than abstract ML scores.\n"
    )
    sections.append(f"- Total cost: {cost['total_cost']}")
    sections.append(f"- Mean cost per record: {cost['mean_cost_per_record']:.3f}\n")

    sections.append("## Escalation safety KPIs\n")
    sections.append(
        "Two hard safety percentages: how often a truly high-risk PR gets routed to a lower-risk "
        "path, and how often a real CT1/CT2 gets swept into bulk-approve (CT3).\n\n"
        "*Why it matters here:*\n"
        "- Directly answers the question stakeholders actually ask: \"how often does a genuinely "
        "dangerous PR slip through?\"\n"
        "- The bulk-approve rate is a hard gate for us -- even one CT1/CT2 auto-approved as CT3 is a "
        "compliance concern, so holding this at 0% is a key demo talking point.\n"
        "- Plain percentages, easy to explain to non-ML audiences like auditors or leadership without "
        "needing to unpack precision/recall.\n"
    )
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

    sections.append(
        "## Linear-weighted Cohen's Kappa\n\n"
        "Agreement between predicted and true labels, corrected for chance, with partial credit "
        "for \"close\" misses (CT1 vs. CT2) versus \"far\" misses (CT1 vs. CT3), since our classes "
        "are ordered by severity rather than unrelated categories.\n\n"
        "*Why it matters here:*\n"
        "- Corrects for chance agreement and class imbalance, giving a fairer single skill score "
        "than raw accuracy.\n"
        "- The ordinal weighting matches how we actually judge mistakes: calling a CT1 a CT2 is a "
        "smaller error than calling it a CT3, and kappa reflects that instead of treating all misses equally.\n"
        "- One number to track model quality trend across versions over time.\n\n"
        f"**Score: {results['weighted_kappa']:.3f}**\n"
    )

    sections.append("## Breakdown by decision method\n")
    sections.append(
        "Splits accuracy by which rule in the decision pipeline produced the call -- a confident "
        "ML prediction, a flagged-for-review case, a low-confidence default, or the CT3-demoted-to-CT1 "
        "safety net that deliberately overrides the model.\n\n"
        "*Why it matters here:*\n"
        "- Separates \"the model got it wrong\" from \"a guardrail deliberately overrode the model\" -- "
        "e.g. `ml-ct3-demoted-to-ct1` showing 0% accuracy is the safety net working as designed, not a "
        "model failure, and that distinction matters when presenting this number in a demo.\n"
        "- Shows the safety net is actually catching cases, which is the point of having it.\n"
        "- Flags where to invest next: a large, inaccurate `ml-low-conf-default-ct1` bucket points at "
        "threshold tuning, not a full retrain.\n"
    )
    header = ["Method", "Count", "Correct", "Accuracy"]
    rows = [
        [m, d["count"], d["correct"], f"{d['accuracy']:.2f}"]
        for m, d in sorted(method_breakdown.items())
    ]
    sections.append(_markdown_table(header, rows))

    return "\n".join(sections) + "\n"
