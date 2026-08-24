# ==========================================================
# POLÍTICA DE DATA QUALITY GATES
# ==========================================================

BLOCKING_RULES = [
    "dataset_not_empty",
    "expected_columns",
    "duplicate_rows",
    "valid_dates",
    "valid_times",
    "numeric_columns",
    "temporal_continuity",
]

WARNING_RULES = [
    "empty_rows",
    "encoded_missing_values",
]

def evaluate_quality_gates(results):
    # Evalúa los resultados y determina el estado global del dataset.
    failed_rules = []
    warning_rules = []

    for rule_name in BLOCKING_RULES:
        rule_result = results.get(rule_name, {})

        if not rule_result.get("passed", False):
            failed_rules.append(rule_name)

    for rule_name in WARNING_RULES:
        rule_result = results.get(rule_name, {})

        if (
            not rule_result.get("passed", True)
            or rule_result.get("warning", False)
        ):
            warning_rules.append(rule_name)

    if failed_rules:
        status = "FAIL"
    elif warning_rules:
        status = "WARNING"
    else:
        status = "PASS"

    return {
        "status": status,
        "failed_rules": failed_rules,
        "warning_rules": warning_rules,
    }