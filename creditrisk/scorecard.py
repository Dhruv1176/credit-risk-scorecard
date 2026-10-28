"""Scale a WoE logistic regression into points (base score at base odds, PDO points to double the odds)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Scorecard:
    def __init__(self, binner, model, features, base_score=600, base_odds=50, pdo=20):
        self.binner, self.model, self.features = binner, model, list(features)
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.coef = dict(zip(self.features, model.coef_[0]))
        self.intercept = float(model.intercept_[0])

    def score(self, X: pd.DataFrame) -> pd.Series:
        woe = self.binner.transform(X)[self.features]
        logit = self.intercept + woe.values @ np.array([self.coef[f] for f in self.features])
        return pd.Series(self.offset - self.factor * logit, index=X.index, name="score")

    def table(self) -> pd.DataFrame:
        """The points table underwriters actually use: one row per feature bin."""
        k, rows = len(self.features), []
        for f in self.features:
            for b, r in self.binner.tables_[f].iterrows():
                pts = -(self.coef[f] * r["woe"] + self.intercept / k) * self.factor + self.offset / k
                rows.append({"feature": f, "bin": b, "share": r["n"], "bad_rate": r["bad_rate"],
                             "woe": r["woe"], "points": round(pts)})
        t = pd.DataFrame(rows)
        t["share"] = t["share"] / t.groupby("feature")["share"].transform("sum")
        return t
