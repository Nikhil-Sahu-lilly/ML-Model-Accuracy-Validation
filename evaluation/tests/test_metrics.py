"""Hand-computable sanity checks for metrics.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics import (  # noqa: E402
    confusion,
    cost_weighted_matrix,
    escalation_rates,
    per_class_report,
    weighted_kappa,
)

# 6 records: CT1 predicted correctly twice, once as CT2 (under-escalated), once as CT3
# (badly under-escalated); CT2 correct once; CT3 correct once.
TRUE = ["CT1", "CT1", "CT1", "CT1", "CT2", "CT3"]
PRED = ["CT1", "CT1", "CT2", "CT3", "CT2", "CT3"]


def test_confusion_matrix_shape_and_diagonal():
    cm = confusion(TRUE, PRED)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(TRUE)
    # CT1 row: 2 correct, 1 to CT2, 1 to CT3
    assert list(cm[0]) == [2, 1, 1]


def test_recall_ct1_is_half():
    report = per_class_report(TRUE, PRED)
    assert report["CT1"]["recall"] == 0.5
    assert report["CT1"]["support"] == 4


def test_cost_weighted_matrix_matches_hand_calc():
    # CT1->CT1 (0) + CT1->CT1 (0) + CT1->CT2 (5) + CT1->CT3 (10) + CT2->CT2 (0) + CT3->CT3 (0) = 15
    result = cost_weighted_matrix(TRUE, PRED)
    assert result["total_cost"] == 15
    assert result["mean_cost_per_record"] == 15 / 6


def test_escalation_rates():
    rates = escalation_rates(TRUE, PRED)
    # 2 of 4 true CT1 predicted as CT2/CT3
    assert rates["ct1_under_escalated"] == 0.5
    # true CT1+CT2 = 5 records, 1 predicted as CT3
    assert rates["ct1_or_ct2_bulk_approved"] == 1 / 5


def test_weighted_kappa_is_between_bounds():
    kappa = weighted_kappa(TRUE, PRED)
    assert -1.0 <= kappa <= 1.0
