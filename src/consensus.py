"""Cohen's Kappa computation for inter-agent agreement."""

import numpy as np
from sklearn.metrics import cohen_kappa_score


def compute_kappa(diag_codes: list[str], audit_codes: list[str]) -> float:
    all_codes = list(set(diag_codes) | set(audit_codes))
    if not all_codes:
        return 0.0

    y_diag = [1 if c in diag_codes else 0 for c in all_codes]
    y_audit = [1 if c in audit_codes else 0 for c in all_codes]

    # Perfect agreement edge case — sklearn returns nan when there's no variance
    if y_diag == y_audit:
        return 1.0

    score = cohen_kappa_score(y_diag, y_audit)
    return float(score) if not np.isnan(score) else 1.0
