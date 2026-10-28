"""Weight of Evidence / Information Value binning, the backbone of classical credit scorecards."""
from __future__ import annotations

import numpy as np
import pandas as pd

MISSING = "MISSING"


def iv_strength(iv: float) -> str:
    return ("useless" if iv < 0.02 else "weak" if iv < 0.1 else "medium" if iv < 0.3
            else "strong" if iv < 0.5 else "suspicious (check leakage)")


class WoEBinner:
    """WoE = ln(%good / %bad) per bin (target 1 = bad), with +0.5 smoothing. Numeric features use quantile
    bins; missing values get their own bin; rare categories are pooled into OTHER."""

    def __init__(self, n_bins: int = 6, rare_frac: float = 0.02):
        self.n_bins, self.rare_frac = n_bins, rare_frac

    def _label(self, col: str, s: pd.Series) -> pd.Series:
        spec = self.specs_[col]
        if spec["type"] == "num":
            lab = pd.cut(s, spec["edges"], right=True).astype(str)
            lab[s.isna().values] = MISSING
            return lab
        lab = s.astype(object).where(s.isin(spec["keep"]), "OTHER")
        lab[s.isna().values] = MISSING
        return lab.astype(str)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.specs_, self.tables_, self.iv_ = {}, {}, {}
        y = pd.Series(np.asarray(y), index=X.index)
        good_t, bad_t = (y == 0).sum(), (y == 1).sum()
        for col in X.columns:
            s = X[col]
            if pd.api.types.is_numeric_dtype(s):
                q = np.unique(np.quantile(s.dropna(), np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                self.specs_[col] = {"type": "num", "edges": np.r_[-np.inf, q, np.inf]}
            else:
                freq = s.value_counts(normalize=True)
                self.specs_[col] = {"type": "cat", "keep": set(freq[freq >= self.rare_frac].index)}
            lab = self._label(col, s)
            if self.specs_[col]["type"] == "num":
                order = [str(c) for c in pd.cut(pd.Series([0.0]), self.specs_[col]["edges"], right=True).cat.categories]
                order = order + [MISSING]
            else:
                order = None
            g = pd.DataFrame({"bin": lab.values, "bad": y.values}).groupby("bin", sort=False)["bad"].agg(["count", "sum"])
            g.columns = ["n", "bad"]
            g = g.loc[[o for o in order if o in g.index]] if order else g.loc[(g["bad"] / g["n"]).sort_values().index]
            g["good"] = g["n"] - g["bad"]
            g["pct_good"] = (g["good"] + 0.5) / good_t
            g["pct_bad"] = (g["bad"] + 0.5) / bad_t
            g["woe"] = np.log(g["pct_good"] / g["pct_bad"])
            g["bad_rate"] = g["bad"] / g["n"]
            g["iv"] = (g["pct_good"] - g["pct_bad"]) * g["woe"]
            self.tables_[col] = g[["n", "bad_rate", "woe", "iv"]]
            self.iv_[col] = float(g["iv"].sum())
        return self

    def labels(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: self._label(c, X[c]) for c in self.specs_}, index=X.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in self.specs_:
            m = self.tables_[c]["woe"].to_dict()
            out[c] = self._label(c, X[c]).map(m).fillna(0.0).astype(float)
        return pd.DataFrame(out, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        t = pd.Series(self.iv_, name="IV").sort_values(ascending=False).to_frame()
        t["strength"] = t["IV"].map(iv_strength)
        return t
