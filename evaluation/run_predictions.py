"""Run every labeled PR record through the shipped model/predict.py pipeline."""
import sys

import pandas as pd

from config import CLASS_ORDER, MODEL_DIR

sys.path.insert(0, str(MODEL_DIR))
from predict import classify_pr, load_artifact  # noqa: E402  (path must be set first)


def run_predictions(records: list[dict]) -> pd.DataFrame:
    """Classify each record with the trained artifact and pair the result with ground truth."""
    artifact = load_artifact()
    rows = []
    for record in records:
        decision = classify_pr(record, artifact)
        row = {
            "url": record.get("url"),
            "true_label": record["label"],
            "pred_label": decision["label"],
            "confidence": decision["confidence"],
            "method": decision["method"],
        }
        for cls in CLASS_ORDER:
            row[f"proba_{cls}"] = decision["proba"].get(cls)
        rows.append(row)
    return pd.DataFrame(rows)
