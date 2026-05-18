"""Cohen's Kappa computation for inter-agent agreement."""

import numpy as np
from sklearn.metrics import cohen_kappa_score


def compute_kappa(
    diag_codes: list[str],
    audit_codes: list[str],
    all_codes: list[str],
) -> float:
    """Binary label arrays over all trajectory ICD codes — 1 if agent cited, 0 if not.

    all_codes must be the full set of ICD codes in the patient trajectory, not just
    the union of what the two agents cited. This ensures both label vectors have genuine
    zeros (codes neither agent selected), which keeps Cohen's kappa well-defined.
    """
    if not all_codes:
        return 0.0

    y_diag = [1 if c in diag_codes else 0 for c in all_codes]
    y_audit = [1 if c in audit_codes else 0 for c in all_codes]

    if y_diag == y_audit:
        return 1.0

    score = cohen_kappa_score(y_diag, y_audit)
    return float(score) if not np.isnan(score) else 0.0
