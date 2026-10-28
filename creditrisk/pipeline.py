from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from .data import TARGET
from .metrics import cutoff_table, decile_table, gini, ks_stat, psi
from .scorecard import Scorecard
from .woe import WoEBinner

MIN_IV = 0.02


def split(df: pd.DataFrame, seed: int = 0):
    """Out-of-time split when `issue_month` exists (train on the past, test on the future), else random."""
    if "issue_month" in df:
        cut = df["issue_month"].quantile(0.7)
        return df[df["issue_month"] <= cut], df[df["issue_month"] > cut], "out-of-time"
    tr, te = train_test_split(df, test_size=0.3, stratify=df[TARGET], random_state=seed)
    return tr, te, "random"


def run(df: pd.DataFrame, out: str = "outputs", seed: int = 0, verbose: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tr, te, how = split(df, seed)
    feats = [c for c in df.columns if c not in (TARGET, "issue_month")]
    Xtr, ytr, Xte, yte = tr[feats], tr[TARGET], te[feats], te[TARGET]

    binner = WoEBinner().fit(Xtr, ytr)
    ivt = binner.iv_table()
    keep = ivt.index[ivt["IV"] >= MIN_IV].tolist()
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(binner.transform(Xtr)[keep], ytr)
    sc = Scorecard(binner, lr, keep)
    s_tr, s_te = sc.score(Xtr), sc.score(Xte)
    pd_sc = lr.predict_proba(binner.transform(Xte)[keep])[:, 1]

    gb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, random_state=seed)
    enc = lambda X: pd.get_dummies(X, dtype=float)
    cols = enc(Xtr).columns
    gb.fit(enc(Xtr), ytr)
    pd_gb = gb.predict_proba(enc(Xte).reindex(columns=cols, fill_value=0.0))[:, 1]

    comp = pd.DataFrame({m: {"AUC": roc_auc_score(yte, p), "Gini": gini(yte, p), "KS": ks_stat(yte, p),
                             "Brier": brier_score_loss(yte, p)}
                         for m, p in (("Scorecard (WoE+LR)", pd_sc), ("Gradient boosting", pd_gb))}).T
    sample = te.sample(min(3000, len(te)), random_state=seed)
    Xs = enc(sample[feats]).reindex(columns=cols, fill_value=0.0)
    pi = permutation_importance(gb, Xs, sample[TARGET], scoring="roc_auc", n_repeats=3, random_state=seed)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values(ascending=False).head(10)

    cut, cal = cutoff_table(yte, s_te), decile_table(yte, pd_sc)
    stab = psi(s_tr, s_te)
    tbl = sc.table()
    tbl.to_csv(out / "scorecard_points.csv", index=False)
    ivt.to_csv(out / "information_value.csv")
    comp.round(4).to_csv(out / "model_comparison.csv")
    cut.round(4).to_csv(out / "cutoff_strategy.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for m, p in (("Scorecard", pd_sc), ("Gradient boosting", pd_gb)):
        f, t, _ = roc_curve(yte, p)
        ax[0, 0].plot(f, t, label=f"{m} (AUC {roc_auc_score(yte, p):.3f})")
    ax[0, 0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0, 0].set(title=f"ROC ({how} test set)", xlabel="False positive rate", ylabel="True positive rate")
    ax[0, 0].legend()
    for v, name in ((0, "good"), (1, "bad")):
        ax[0, 1].hist(s_te[yte == v], bins=40, alpha=0.6, density=True, label=name)
    ax[0, 1].set(title="Score distribution by outcome", xlabel="Score")
    ax[0, 1].legend()
    ivt["IV"].sort_values().plot.barh(ax=ax[1, 0], title="Information value by feature")
    ax[1, 1].plot(cal["predicted_pd"], cal["observed_dr"], "o-", label="scorecard")
    mx = max(cal["predicted_pd"].max(), cal["observed_dr"].max())
    ax[1, 1].plot([0, mx], [0, mx], "k--", lw=0.8)
    ax[1, 1].set(title="Calibration by decile", xlabel="Predicted PD", ylabel="Observed default rate")
    fig.tight_layout()
    fig.savefig(out / "credit_risk_report.png", dpi=140)

    if verbose:
        print(f"\nTrain {len(tr):,} / test {len(te):,} ({how}) | default rate train {ytr.mean():.1%}, test {yte.mean():.1%}")
        print("\n== Information value ==\n", ivt.round(3).to_string())
        print(f"\nFeatures kept (IV >= {MIN_IV}): {keep}")
        print("\n== Model comparison on held-out data ==\n", comp.round(4).to_string())
        print("\n== Gradient boosting: permutation importance (AUC drop) ==\n", imp.round(4).to_string())
        print(f"\nScore PSI train->test: {stab:.3f}  ({'stable' if stab < 0.1 else 'shifting' if stab < 0.25 else 'DRIFT'})")
        print("\n== Cut-off strategy ==\n", cut.round(3).to_string(index=False))
        print("\n== Scorecard points (first 15 rows) ==\n", tbl.head(15).round(3).to_string(index=False))
        print(f"\nSaved report + CSVs to {out.resolve()}")
    return {"scorecard": sc, "comparison": comp, "iv": ivt, "psi": stab, "cutoff": cut, "calibration": cal}
