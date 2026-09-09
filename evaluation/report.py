"""Render the evaluation results dict into a markdown report and a confusion-matrix heatmap."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CLASS_ORDER


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
