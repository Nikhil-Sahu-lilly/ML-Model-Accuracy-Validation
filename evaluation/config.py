"""Paths and constants shared across the evaluation pipeline."""
from pathlib import Path

EVAL_DIR = Path(__file__).parent
ROOT_DIR = EVAL_DIR.parent

MODEL_DIR = ROOT_DIR / "model"
DATASET_PATH = ROOT_DIR / "dataset" / "pr_dataset.jsonl"
OUTPUT_DIR = EVAL_DIR / "outputs"

PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"
CONFUSION_MATRIX_PNG = OUTPUT_DIR / "confusion_matrix.png"
REPORT_MD = OUTPUT_DIR / "evaluation_report.md"

# Severity order, most severe first. Drives confusion-matrix axis order and the cost matrix below.
CLASS_ORDER = ["CT1", "CT2", "CT3"]

# COST_MATRIX[true][pred]: business cost of a (true, predicted) pair.
# Under-escalation (true is more severe than predicted) is expensive -- a real security/permissions
# change slipping into standard review or bulk-approve. Over-escalation (predicted more severe than
# true) only costs reviewer time, so it is penalized lightly. Diagonal is free.
COST_MATRIX = {
    "CT1": {"CT1": 0, "CT2": 5, "CT3": 10},
    "CT2": {"CT1": 1, "CT2": 0, "CT3": 5},
    "CT3": {"CT1": 1, "CT2": 1, "CT3": 0},
}
