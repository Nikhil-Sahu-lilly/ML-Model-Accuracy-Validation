"""Paths and constants shared across the evaluation pipeline."""
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).parent
ROOT_DIR = EVAL_DIR.parent

MODEL_DIR = ROOT_DIR / "model"
DATASET_PATH = ROOT_DIR / "dataset" / "pr_dataset.jsonl"
OUTPUTS_ROOT = EVAL_DIR / "outputs"
RUNS_DIR = OUTPUTS_ROOT / "runs"
LATEST_DIR = OUTPUTS_ROOT / "latest"


def new_run_dir() -> tuple[str, Path]:
    """Create a fresh timestamped run folder under RUNS_DIR and return (run_id, path)."""
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


# Filenames written inside each run folder (run_dir / <name>).
PREDICTIONS_CSV_NAME = "predictions.csv"
CONFUSION_MATRIX_PNG_NAME = "confusion_matrix.png"
ROC_CURVE_PNG_NAME = "roc_curve.png"
CALIBRATION_CURVE_PNG_NAME = "calibration_curve.png"
REPORT_MD_NAME = "evaluation_report.md"
RUN_METADATA_JSON_NAME = "run_metadata.json"

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
