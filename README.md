# Credit Risk Scorecard: Probability of Default with WoE + Logistic Regression

![CI](https://github.com/YOUR_USERNAME/credit-risk-scorecard/actions/workflows/ci.yml/badge.svg)

An end-to-end retail credit model built the way banks actually build it: **WoE/IV binning → logistic regression →
points-based scorecard → validation (AUC, Gini, KS, calibration, PSI) → cut-off strategy**, benchmarked against
gradient boosting on an **out-of-time** test set.

![Credit risk report](docs/credit_risk_report.png)

## Why a scorecard and not just XGBoost?
Regulated lenders need models that are explainable, monotonic and stable. A scorecard gives underwriters a
points table they can read, and regulators a model they can validate. The benchmark shows what (little) you give up.

## Pipeline
1. **Data:** seeded synthetic loan book (20k loans, ~12% default, realistic missing values, vintage drift) or your own CSV (`--csv loans.csv --target loan_status`), e.g. LendingClub or German Credit.
2. **Binning:** quantile bins for numerics, missing values as their own bin, rare categories pooled; **Weight of Evidence** per bin and **Information Value** per feature with standard strength labels and a leakage warning.
3. **Model:** logistic regression on WoE features (IV ≥ 0.02).
4. **Scaling:** 600 points at 50:1 good:bad odds, 20 points to double the odds (PDO), producing the points table `outputs/scorecard_points.csv`.
5. **Benchmark:** gradient boosting + permutation importance.
6. **Validation:** AUC, Gini, KS, Brier, decile calibration, **PSI** (train vs test scores).
7. **Business view:** reject the riskiest X% → approval rate, bad rate of approved, % of bad loans avoided.

## Quickstart
```bash
pip install -r requirements.txt
python -m creditrisk                                   # synthetic data
python -m creditrisk --csv my_loans.csv --target default   # your data (0/1 target)
python -m pytest -q
```
Outputs land in `outputs/`: `credit_risk_report.png`, `scorecard_points.csv`, `information_value.csv`,
`model_comparison.csv`, `cutoff_strategy.csv`.

## Results (replace after running on real data)
| Model | AUC | Gini | KS |
|---|---|---|---|
| Scorecard (WoE + LR) | _0.xx_ | _0.xx_ | _0.xx_ |
| Gradient boosting | _0.xx_ | _0.xx_ | _0.xx_ |

Cut-off example: "Rejecting the lowest-scoring 20% avoids ~_xx%_ of defaults while approving 80% of applicants."

## Design choices worth discussing in interviews
- **Out-of-time split** (train on early vintages, test on later) is stricter and more realistic than a random split.
- **Smoothing (+0.5)** in WoE avoids infinite values on empty bins.
- **A suspiciously high IV (> 0.5)** usually means target leakage (for example, a field populated only after default).
- The synthetic `interest_rate` is a noisy proxy for lender risk-pricing; in real life, check whether using it creates circularity.

## Limitations
Synthetic data demonstrates the method, not real-world performance. No reject inference, no macro variables, no
IFRS 9 / CECL provisioning (PD × LGD × EAD) and no fairness analysis yet.

## Roadmap
Monotonic binning · reject inference · LGD/EAD models and expected-loss calculation · SHAP explanations · fairness / adverse-impact testing · FastAPI scoring endpoint

MIT License. Educational use only.

## Author

**Dhruv Singh**

GitHub: [@Dhruv1176](https://github.com/Dhruv1176)
