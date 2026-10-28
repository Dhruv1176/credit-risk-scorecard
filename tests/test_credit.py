import numpy as np
import pandas as pd

from creditrisk.data import TARGET, generate_loans
from creditrisk.metrics import cutoff_table, ks_stat, psi
from creditrisk.pipeline import run
from creditrisk.woe import WoEBinner

df = generate_loans(8000, seed=3)


def test_default_rate_calibrated():
    assert abs(df[TARGET].mean() - 0.12) < 0.02


def test_woe_signs_and_iv():
    b = WoEBinner().fit(df.drop(columns=[TARGET, "issue_month"]), df[TARGET])
    t = b.tables_["dti"]
    assert t["woe"].iloc[0] > 0 > t["woe"].iloc[-1]       # lowest-DTI bin is safest, highest-DTI bin riskiest
    assert t["bad_rate"].is_monotonic_increasing
    assert b.iv_["revolving_util"] > 0.02


def test_psi_zero_for_same_distribution_and_positive_for_shift():
    x = np.random.default_rng(0).normal(600, 50, 20000)
    assert psi(x, x) < 1e-6
    assert psi(x, x + 40) > 0.25


def test_pipeline_quality_and_score_math(tmp_path):
    r = run(df, str(tmp_path), verbose=False)
    assert r["comparison"].loc["Scorecard (WoE+LR)", "AUC"] > 0.65
    sc = r["scorecard"]
    X = df.drop(columns=[TARGET, "issue_month"]).head(200)
    pdh = sc.model.predict_proba(sc.binner.transform(X)[sc.features])[:, 1]
    expected = sc.offset - sc.factor * np.log(pdh / (1 - pdh))
    assert np.allclose(sc.score(X), expected)                      # score == offset - factor * logit(PD)
    cut = r["cutoff"]
    assert cut["bad_rate_of_approved"].is_monotonic_decreasing       # rejecting the riskiest lowers bad rate
    assert (tmp_path / "credit_risk_report.png").exists()


def test_ks_perfect_separation():
    assert ks_stat([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
