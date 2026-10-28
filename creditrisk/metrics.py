from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(np.abs(tpr - fpr)))


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index: <0.1 stable, 0.1-0.25 shifting, >0.25 significant drift."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def decile_table(y, pd_hat, bins: int = 10) -> pd.DataFrame:
    """Calibration: average predicted PD vs observed default rate by risk decile."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(pd_hat)})
    d["decile"] = pd.qcut(d["p"].rank(method="first"), bins, labels=range(1, bins + 1))
    return d.groupby("decile", observed=True).agg(n=("y", "size"), predicted_pd=("p", "mean"), observed_dr=("y", "mean"))


def cutoff_table(y, score, rejects=(0, 10, 20, 30, 40, 50)) -> pd.DataFrame:
    """Business view: reject the riskiest X% of applicants -> approval rate and portfolio bad rate."""
    d = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(score)})
    rows = []
    for r in rejects:
        cut = d["s"].quantile(r / 100)
        kept = d[d["s"] >= cut] if r else d
        rows.append({"reject_%": r, "min_score": round(cut), "approval_rate": len(kept) / len(d),
                     "bad_rate_of_approved": kept["y"].mean(),
                     "bad_loans_avoided_%": 1 - kept["y"].sum() / d["y"].sum()})
    return pd.DataFrame(rows)
