"""CLI entrypoint: load labeled data -> run model predictions -> compute metrics -> write report.

Usage:
    python evaluate.py
"""
from config import CONFUSION_MATRIX_PNG, DATASET_PATH, OUTPUT_DIR, PREDICTIONS_CSV, REPORT_MD
from data_loader import load_labeled_dataset
from metrics import evaluate_all
from report import render_markdown_report, save_confusion_heatmap
from run_predictions import run_predictions


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_labeled_dataset(DATASET_PATH)
    print(f"Loaded {len(records)} labeled records from {DATASET_PATH}")

    predictions = run_predictions(records)
    predictions.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Wrote predictions to {PREDICTIONS_CSV}")

    results = evaluate_all(
        predictions["true_label"], predictions["pred_label"], predictions["method"]
    )

    save_confusion_heatmap(results["confusion_normalized"], CONFUSION_MATRIX_PNG)
    print(f"Wrote confusion matrix heatmap to {CONFUSION_MATRIX_PNG}")

    report_text = render_markdown_report(results, len(records))
    REPORT_MD.write_text(report_text, encoding="utf-8")
    print(f"Wrote report to {REPORT_MD}")

    print("\n" + report_text)


if __name__ == "__main__":
    main()
