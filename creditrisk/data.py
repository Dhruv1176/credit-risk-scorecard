"""Synthetic retail-loan generator (seeded) and a CSV loader for real data (e.g. LendingClub, German Credit)."""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "default"


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_loans(n: int = 20000, seed: int = 7, default_rate: float = 0.12) -> pd.DataFrame:
    """Loans issued over 36 months. Default risk depends on a known, realistic set of drivers, plus noise
    and mild vintage drift (later months slightly riskier) so population-stability monitoring has something to find."""
    rng = np.random.default_rng(seed)
    month = rng.integers(0, 36, n)
    age = rng.integers(21, 70, n)
    income = np.round(rng.lognormal(11.0, 0.55, n), -2)
    emp = np.clip(rng.gamma(2.0, 3.0, n), 0, 40).round(1)
    hist = np.clip(age - 18 - rng.gamma(2, 2, n), 0.5, 45).round(1)
    home = rng.choice(["RENT", "MORTGAGE", "OWN"], n, p=[0.45, 0.40, 0.15])
    purpose = rng.choice(["debt_consolidation", "credit_card", "home_improvement", "car", "medical",
                          "small_business", "other"], n, p=[0.38, 0.20, 0.10, 0.10, 0.06, 0.06, 0.10])
    term = rng.choice([36, 60], n, p=[0.7, 0.3])
    amount = np.round(np.clip(income * rng.uniform(0.05, 0.45, n), 1000, 40000), -2)
    dti = np.clip(rng.normal(18, 8, n) + amount / income * 20, 0, 60).round(1)
    util = np.clip(rng.beta(2, 3, n) * 100 + (dti - 18) * 0.4, 0, 100).round(1)
    delinq = rng.poisson(np.clip(0.1 + util / 100 * 0.6, 0.02, 2))
    inq = rng.poisson(0.8 + dti / 60)

    z = (0.045 * (dti - 18) + 0.018 * (util - 40) + 0.35 * delinq + 0.12 * inq - 0.012 * (age - 40)
         - 0.6 * (np.log(income) - 11) + 0.5 * (term == 60) + 0.25 * (home == "RENT") - 0.2 * (home == "OWN")
         + 0.5 * (purpose == "small_business") + 0.2 * (purpose == "medical")
         - 0.03 * emp - 0.02 * hist + 0.25 * month / 36)
    rate = np.clip(8 + 2.0 * (z - z.mean()) + rng.normal(0, 1.5, n), 5, 30).round(2)  # lender's noisy risk-based price
    z = z + rng.normal(0, 0.5, n)
    lo, hi = -10.0, 10.0
    for _ in range(50):  # calibrate intercept to hit the target default rate
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(z + mid).mean() < default_rate else (lo, mid)
    y = (rng.random(n) < _sigmoid(z + mid)).astype(int)

    df = pd.DataFrame({"issue_month": month, "age": age, "annual_income": income, "employment_years": emp,
                       "credit_history_years": hist, "home_ownership": home, "purpose": purpose, "term_months": term,
                       "loan_amount": amount, "interest_rate": rate, "dti": dti, "revolving_util": util,
                       "delinq_2y": delinq, "inquiries_6m": inq, TARGET: y})
    for col, frac in (("employment_years", 0.05), ("credit_history_years", 0.03), ("revolving_util", 0.02)):
        df.loc[rng.random(n) < frac, col] = np.nan  # realistic missingness
    return df


def load_csv(path: str, target: str = TARGET) -> pd.DataFrame:
    """Load your own data: a CSV with a 0/1 default column. Optional `issue_month` enables an out-of-time split."""
    df = pd.read_csv(path)
    if target != TARGET:
        df = df.rename(columns={target: TARGET})
    if not set(df[TARGET].dropna().unique()) <= {0, 1}:
        raise ValueError("target must be binary 0/1 (1 = default)")
    return df.dropna(subset=[TARGET])
