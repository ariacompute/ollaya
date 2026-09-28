"""Parity helpers for afm-de-latest (ORT vs PyTorch AFM-D / Laya DecisionModel)."""
from __future__ import annotations


def max_abs_diff(a, b) -> float:
    import numpy as np
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
