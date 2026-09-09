"""Standalone inference script for the PR risk classifier.

Ship this whole `model/` directory as-is. Needs `pr_features.py` (same feature logic used at
training time) and `triage_classifier.joblib` next to it, plus the packages in
`requirements.txt`.

Usage:
    python predict.py path/to/pr_record.json
    cat pr_record.json | python predict.py -

A PR record is a JSON object with at least these fields:
    {"url": "...", "matched_files": ["template.yml"], "CT3tag": "", "diff": "diff --git ..."}
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from pr_features import NUMERIC_FEATURE_COLUMNS, extract_features

DEFAULT_MODEL_PATH = Path(__file__).parent / "triage_classifier.joblib"


def load_artifact(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    return joblib.load(model_path)


def classify_pr(record: dict, artifact: dict) -> dict:
    """Feature-extract, run the trained pipeline, and route one raw PR record to a final decision."""
    pipeline = artifact["pipeline"]
    feats = extract_features(record)
    row = pd.DataFrame([{col: feats[col] for col in NUMERIC_FEATURE_COLUMNS + ["added_text"]}])

    proba = pipeline.predict_proba(row)[0]
    classes = artifact["classes"]
    top_idx = int(proba.argmax())
    top_label = classes[top_idx]
    top_conf = float(proba[top_idx])

    if top_label == "CT3" and top_conf < artifact["ct3_min_confidence"]:
        label, method = "CT1", "ml-ct3-demoted-to-ct1"
    elif top_conf >= artifact["ml_high_conf_threshold"]:
        label, method = top_label, "ml-high-conf"
    elif top_conf >= artifact["ml_low_conf_threshold"]:
        label, method = top_label, "ml-flag-review"
    else:
        label, method = "CT1", "ml-low-conf-default-ct1"

    return {
        "url": record.get("url"),
        "label": label,
        "confidence": top_conf,
        "method": method,
        "proba": dict(zip(classes, proba.tolist())),
    }


def main():
    if len(sys.argv) != 2:
        print("usage: python predict.py <pr_record.json | ->", file=sys.stderr)
        raise SystemExit(1)

    raw = sys.stdin.read() if sys.argv[1] == "-" else Path(sys.argv[1]).read_text(encoding="utf-8")
    record = json.loads(raw)

    artifact = load_artifact()
    decision = classify_pr(record, artifact)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
