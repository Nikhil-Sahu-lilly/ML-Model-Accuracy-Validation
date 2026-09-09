"""Load and validate the labeled PR dataset used for evaluation."""
import json
from pathlib import Path

from config import CLASS_ORDER

REQUIRED_FIELDS = ("url", "matched_files", "diff", "CT3tag", "label")


def load_labeled_dataset(path: Path) -> list[dict]:
    """Read pr_dataset.jsonl, one JSON object per line, and validate each record."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            _validate_record(record, line_no)
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _validate_record(record: dict, line_no: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"{line_no}: record missing required field(s) {missing}")
    if record["label"] not in CLASS_ORDER:
        raise ValueError(f"{line_no}: label {record['label']!r} not in {CLASS_ORDER}")
