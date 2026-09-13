"""CLI entrypoint: load labeled data -> run model predictions -> compute metrics -> write report.

Every run is written to its own timestamped folder under evaluation/outputs/runs/, plus a
copy in evaluation/outputs/latest/, so past demo runs stay archived instead of overwritten.

Usage:
    python evaluate.py
"""
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone

import pandas as pd
import sklearn

from config import (
    CALIBRATION_CURVE_PNG_NAME,
    CLASS_ORDER,
    CONFUSION_MATRIX_PNG_NAME,
    DATASET_PATH,
    LATEST_DIR,
    PREDICTIONS_CSV_NAME,
    REPORT_MD_NAME,
    ROC_CURVE_PNG_NAME,
    RUN_METADATA_JSON_NAME,
    new_run_dir,
)
from data_loader import load_labeled_dataset
from metrics import evaluate_all
from report import render_markdown_report, save_calibration_curve, save_confusion_heatmap, save_roc_curve
from run_predictions import run_predictions


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def _write_run_metadata(run_dir, run_id: str, n_records: int) -> None:
    metadata = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit_hash(),
        "dataset_path": str(DATASET_PATH),
        "n_records": n_records,
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    (run_dir / RUN_METADATA_JSON_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _refresh_latest(run_dir) -> None:
    """Overwrite files in place rather than rmtree+copytree -- OneDrive can hold a lock
    on the folder itself, which makes deleting it fail on Windows."""
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for item in run_dir.iterdir():
        shutil.copy2(item, LATEST_DIR / item.name)


def main() -> None:
    run_id, run_dir = new_run_dir()

    records = load_labeled_dataset(DATASET_PATH)
    print(f"Loaded {len(records)} labeled records from {DATASET_PATH}")

    predictions = run_predictions(records)
    predictions.to_csv(run_dir / PREDICTIONS_CSV_NAME, index=False)

    proba = {cls: predictions[f"proba_{cls}"] for cls in CLASS_ORDER}
    results = evaluate_all(
        predictions["true_label"], predictions["pred_label"], predictions["method"], proba
    )

    save_confusion_heatmap(results["confusion_normalized"], run_dir / CONFUSION_MATRIX_PNG_NAME)
    save_roc_curve(results["roc_auc"], run_dir / ROC_CURVE_PNG_NAME)
    save_calibration_curve(predictions["true_label"], proba, run_dir / CALIBRATION_CURVE_PNG_NAME)

    report_text = render_markdown_report(results, len(records))
    (run_dir / REPORT_MD_NAME).write_text(report_text, encoding="utf-8")

    _write_run_metadata(run_dir, run_id, len(records))
    _refresh_latest(run_dir)

    print("\n" + report_text)
    print(f"Run archived to: {run_dir}")
    print(f"Latest copy at:  {LATEST_DIR}")


if __name__ == "__main__":
    main()
