# PR Risk Classifier -- Evaluation Report

273 labeled records evaluated.

## Confusion matrix (rows = true, cols = predicted)

Each row is a true class, each column is where the model routed it. The diagonal is correct calls; everything off-diagonal is a misroute.

*Why it matters here:*
- Shows the *direction* of errors, not just the count -- CT1 leaking into CT2/CT3 (a real risk under-caught) is far worse than CT3 leaking into CT1 (just extra review).
- Pinpoints exactly which class pairs get confused, so we know where to add training data or tighten rules instead of guessing.
- Gives auditors/compliance an exact, countable trail instead of a single opaque score.

|  | pred CT1 | pred CT2 | pred CT3 |
| --- | --- | --- | --- |
| true CT1 | 122 (98%) | 2 (2%) | 0 (0%) |
| true CT2 | 17 (13%) | 114 (87%) | 0 (0%) |
| true CT3 | 18 (100%) | 0 (0%) | 0 (0%) |

![Confusion matrix](confusion_matrix.png)


## Per-class precision / recall / F1

Precision = of the PRs we called class X, how many really were X. Recall = of the PRs that really were X, how many we caught. F1 balances the two; Macro F1 averages the three classes equally so the small CT3 class can't be hidden by CT1/CT2 volume.

*Why it matters here:*
- **Recall(CT1)** is our north-star safety metric -- it says how many genuinely high-risk PRs the model actually catches, which is the failure mode we can least afford.
- Precision on CT1 tells us reviewer burden: low precision means reviewers keep getting paged on PRs that turn out to be low-risk.
- Macro F1 (not accuracy) keeps CT3 performance visible instead of averaged away by the larger CT1/CT2 classes.

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| CT1 | 0.78 | 0.98 | 0.87 | 124 |
| CT2 | 0.98 | 0.87 | 0.92 | 131 |
| CT3 | 0.00 | 0.00 | 0.00 | 18 |

**Macro F1: 0.597**

**Recall(CT1) -- north-star safety metric: 0.984**

## ROC-AUC (one-vs-rest)

For each class, treats it as "this class vs. everything else" and measures how well the model's confidence score ranks true members above non-members, at every possible threshold.

*Why it matters here:*
- Threshold-independent, so it separates "is the underlying signal good" from "did we pick the right cutoff" -- a class can have high AUC but weak precision/recall if the threshold needs tuning, which is a cheaper fix than retraining.
- A high AUC on CT3 despite 0 recall/precision (see table above) is a good demo point: the model *can* separate CT3, the current decision rule is just conservatively routing it to CT1.
- Macro AUC gives one comparable number across model versions without depending on class balance.

| Class | AUC |
| --- | --- |
| CT1 | 0.991 |
| CT2 | 0.991 |
| CT3 | 1.000 |

**Macro AUC: 0.994**

![ROC curves](roc_curve.png)

## Calibration

How trustworthy is the model's confidence score? For a well-calibrated model, when it says "70% confident", it should be right about 70% of the time (points near the dashed diagonal = well calibrated).

*Why it matters here:*
- Confidence scores drive the decision method itself (high-conf vs. flag-for-review vs. low-conf-default) -- if scores are overconfident we auto-approve risk we shouldn't, if underconfident we over-flag safe PRs and waste reviewer time.
- Well-calibrated scores let reviewers treat "90% CT1" as a real risk level, supporting a graduated response instead of a blunt yes/no call.
- Miscalibration is invisible in accuracy/F1/AUC alone -- this is the only view that would catch it.

![Calibration curve](calibration_curve.png)

## Severity-weighted cost

Applies the business cost matrix (under-escalating a real risk costs far more than over-escalating a safe PR; correct calls are free) to every prediction and sums it into one number.

*Why it matters here:*
- Translates model errors into actual process economics -- a missed CT1 costs 10x an over-cautious CT2-called-CT3, matching how the business actually experiences the mistake.
- Gives one comparable number across model versions that reflects real consequence, not just raw accuracy.
- Easy to justify a model change to stakeholders in terms of expected risk reduction rather than abstract ML scores.

- Total cost: 45
- Mean cost per record: 0.165

## Escalation safety KPIs

Two hard safety percentages: how often a truly high-risk PR gets routed to a lower-risk path, and how often a real CT1/CT2 gets swept into bulk-approve (CT3).

*Why it matters here:*
- Directly answers the question stakeholders actually ask: "how often does a genuinely dangerous PR slip through?"
- The bulk-approve rate is a hard gate for us -- even one CT1/CT2 auto-approved as CT3 is a compliance concern, so holding this at 0% is a key demo talking point.
- Plain percentages, easy to explain to non-ML audiences like auditors or leadership without needing to unpack precision/recall.

- True CT1 predicted as CT2/CT3 (under-escalated): 1.6%
- True CT1/CT2 predicted as CT3 (bulk-approved): 0.0%

## Linear-weighted Cohen's Kappa

Agreement between predicted and true labels, corrected for chance, with partial credit for "close" misses (CT1 vs. CT2) versus "far" misses (CT1 vs. CT3), since our classes are ordered by severity rather than unrelated categories.

*Why it matters here:*
- Corrects for chance agreement and class imbalance, giving a fairer single skill score than raw accuracy.
- The ordinal weighting matches how we actually judge mistakes: calling a CT1 a CT2 is a smaller error than calling it a CT3, and kappa reflects that instead of treating all misses equally.
- One number to track model quality trend across versions over time.

**Score: 0.648**

## Breakdown by decision method

Splits accuracy by which rule in the decision pipeline produced the call -- a confident ML prediction, a flagged-for-review case, a low-confidence default, or the CT3-demoted-to-CT1 safety net that deliberately overrides the model.

*Why it matters here:*
- Separates "the model got it wrong" from "a guardrail deliberately overrode the model" -- e.g. `ml-ct3-demoted-to-ct1` showing 0% accuracy is the safety net working as designed, not a model failure, and that distinction matters when presenting this number in a demo.
- Shows the safety net is actually catching cases, which is the point of having it.
- Flags where to invest next: a large, inaccurate `ml-low-conf-default-ct1` bucket points at threshold tuning, not a full retrain.

| Method | Count | Correct | Accuracy |
| --- | --- | --- | --- |
| ml-ct3-demoted-to-ct1 | 18 | 0 | 0.00 |
| ml-flag-review | 107 | 104 | 0.97 |
| ml-high-conf | 120 | 120 | 1.00 |
| ml-low-conf-default-ct1 | 28 | 12 | 0.43 |
