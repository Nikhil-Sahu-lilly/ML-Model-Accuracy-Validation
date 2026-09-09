"""Shared feature-extraction logic for the PR risk classifier.

Both test.ipynb (training) and predict.py (inference) import from here, so the exact same
diff -> features logic is used at train time and at prediction time.
"""
import re

TFIDF_TEXT_MAX_CHARS = 3000

# CT1 hard risk signals, matched against ADDED diff lines only
CT1_PATTERNS = {
    "iam_permission_added": re.compile(
        r"(?i)(AWS::IAM::(Role|Policy|ManagedPolicy)|AssumeRolePolicyDocument|"
        r"iam:(CreateRole|AttachRolePolicy|PutRolePolicy|CreatePolicy|AttachUserPolicy)|"
        r"PolicyDocument\s*:|ManagedPolicyName)"
    ),
    "wildcard_grant_added": re.compile(
        r"(?i)((Action|Resource)\s*:\s*(\[)?\s*['\"]?\*)"
    ),
    "s3_access_added": re.compile(
        r"(?i)s3:(GetObject|PutObject|DeleteObject|GetBucketAcl|PutBucketPolicy|ListBucket|GetBucketPolicy)"
    ),
    "cross_account_or_red": re.compile(
        r"(?i)(cross[-_ ]account|\bred[-_ ]data\b|sensitive[-_ ]dataset|restricted[-_ ]dataset)"
    ),
    "data_movement_added": re.compile(
        r"(?i)(AWS::Glue::Job|AWS::AppFlow::|AppFlow|AWS::DataPipeline::|AWS::DMS::)"
    ),
    "network_exposure_added": re.compile(
        r"(?i)(SecurityGroupIngress|SecurityGroupEgress|AWS::EC2::SecurityGroup|"
        r"AWS::ApiGateway::|AWS::EC2::VPC|NetworkAcl|0\.0\.0\.0/0)"
    ),
    "credential_added": re.compile(
        r"(?i)(password\s*=|secret\s*=|api[_-]?key\s*=|token\s*=|BEGIN (RSA|PRIVATE) KEY)"
    ),
    "passrole_added": re.compile(r"iam:PassRole"),
}

SENSITIVE_PATH_PATTERN = re.compile(r"(?i)(^|/)(iam|security|policies|trust)(/|$)")

# CT2 safe-change signals, matched against ADDED diff lines only, and only when no CT1 signal fired
CT2_PATTERNS = {
    "runtime_change_only": re.compile(r"(?i)(Runtime\s*:|MemorySize\s*:|Timeout\s*:|GlueVersion\s*:)"),
    "metadata_only": re.compile(r"(?i)(Tags\s*:|Owner\s*:|CostCenter\s*:|Environment\s*:)"),
    "logging_only": re.compile(r"(?i)(LoggingConfiguration\s*:|CloudWatch|LogGroup)"),
    "cicd_minor": re.compile(r"(?i)(timeout-minutes\s*:|runs-on\s*:|actions/checkout)"),
}

CT1_FEATURES = list(CT1_PATTERNS.keys()) + ["has_sensitive_path", "wildcard_removed"]
CT2_FEATURES = list(CT2_PATTERNS.keys()) + ["only_docs"]
STRUCTURAL_FEATURES = ["file_count", "added_line_count", "has_ct3_tag"]
NUMERIC_FEATURE_COLUMNS = CT1_FEATURES + CT2_FEATURES + STRUCTURAL_FEATURES
assert len(NUMERIC_FEATURE_COLUMNS) == 18, len(NUMERIC_FEATURE_COLUMNS)


def split_diff_lines(diff_text: str) -> tuple[str, str]:
    """Return (added_text, removed_text) extracted from a unified diff."""
    added, removed = [], []
    if not diff_text:
        return "", ""
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added.append(line[1:])
        elif line.startswith("-"):
            removed.append(line[1:])
    return "\n".join(added), "\n".join(removed)


def extract_features(record: dict) -> dict:
    """Compute the 18 numeric risk signals plus the TF-IDF source text for one PR record."""
    added_text, removed_text = split_diff_lines(record.get("diff", ""))
    changed_files = record.get("matched_files") or []

    features = {}

    for name, pattern in CT1_PATTERNS.items():
        features[name] = int(bool(pattern.search(added_text)))
    features["has_sensitive_path"] = int(any(SENSITIVE_PATH_PATTERN.search(f) for f in changed_files))
    features["wildcard_removed"] = int(bool(CT1_PATTERNS["wildcard_grant_added"].search(removed_text)))

    any_ct1_hit = any(features[name] for name in CT1_FEATURES)

    # CT2 signals only count when nothing CT1-risky fired
    for name, pattern in CT2_PATTERNS.items():
        features[name] = int(bool(pattern.search(added_text)) and not any_ct1_hit)

    non_empty_files = [f for f in changed_files if f]
    features["only_docs"] = int(
        len(non_empty_files) > 0
        and all(f.lower().endswith((".md", ".txt", ".rst")) for f in non_empty_files)
        and not any_ct1_hit
    )

    features["file_count"] = len(changed_files)
    features["added_line_count"] = added_text.count("\n") + (1 if added_text else 0)
    features["has_ct3_tag"] = int(bool((record.get("CT3tag") or "").strip()))

    features["added_text"] = added_text[:TFIDF_TEXT_MAX_CHARS]

    return features
