# Customer Churn Prediction

End-to-end educational project: predict whether a telco subscription customer will churn, then connect **loss functions**, **metrics**, **thresholds**, and **business value**.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.download
python -m src.models.train          # logistic regression
python -m src.models.compare        # optional: also train HGB
pytest
python -m src.predict --input examples/customer.json
```

Explore learning notebooks under [`notebooks/`](notebooks/) (phases 01–08).

## Business problem

Retain customers before they leave. A false **negative** (missed churner) is usually far more costly than a false **positive** (unnecessary retention offer). The model estimates churn risk; a **threshold** turns risk into an action.

## Dataset

[IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) public sample (~7k customers). Download:

```bash
python -m src.data.download
```

Label: `Churn` (Yes/No). This is a cross-sectional snapshot used as a binary classification proxy for “will churn”; a production 30-day label would be built from event timestamps.

## Features

Numeric: `tenure`, `MonthlyCharges`, `TotalCharges` (blank tenure-0 charges imputed).  
Categorical: demographics, services, `Contract`, `PaymentMethod`, `InternetService`, etc.  
Dropped: `customerID`. Target: `Churn` ∈ {0, 1}.

## Modeling approach

1. Stratified train/test split (seed 42, 20% test).  
2. Baselines: majority class + month-to-month / high-charge rule.  
3. Sklearn `Pipeline`: impute → scale / one-hot → classifier.  
4. Models: **Logistic Regression** (final) and **HistGradientBoosting** (comparison).  
5. Thresholds chosen from holdout probability sweeps + business cost model.

## Loss function — why log loss?

Binary cross-entropy / log loss matches probabilistic logistic regression: it penalizes confident wrong probabilities and supports ranking and expected-value decisions. Accuracy is not the training objective.

## Evaluation metrics — why these?

| Metric | Role |
|---|---|
| Accuracy | Overall correctness — **misleading** under ~27% churn |
| Precision / Recall / F1 | Quality of the positive (churn) decision at a threshold |
| ROC-AUC | Ranking across thresholds |
| PR-AUC | Positive-class focus under imbalance |
| Log loss / Brier | Probability quality / calibration |

## Threshold selection

0.5 is not inherently optimal. We compare:

- default **0.5**
- **F1-optimal**
- **business-value-optimal** using costs in [`configs/default.yaml`](configs/default.yaml)

**Final operating point:** threshold **0.15** (see [`reports/final_selection.json`](reports/final_selection.json)).

## Business assumptions (defaults)

- False-positive cost: $50 (wasted offer)  
- False-negative cost: $500 (lost customer)  
- Retention value: $200 (successful save, net)

Change these in YAML and re-run business analysis to update the preferred threshold.

## Model comparison (holdout @ 0.5)

| Model | F1 | PR-AUC | Log loss | Brier |
|---|---|---|---|---|
| Logistic regression | 0.613 | 0.632 | 0.497 | 0.169 |
| HistGradientBoosting | 0.632 | 0.651 | 0.483 | 0.161 |

HGB wins several ML metrics; under our cost model, **LR at 0.15** still achieves higher estimated business value than HGB at its BV-optimal threshold — illustrating that the best metric ≠ best business outcome.

## Error analysis

- **FP**: often month-to-month / risky-looking customers who stay.  
- **FN**: some longer-tenure / longer-contract customers who still leave.  
- Missing real-world signals: tickets, usage drops, payment failures.

## Final model

- **Model:** `logistic_regression` (`reports/models/logistic_regression.joblib`)  
- **Threshold:** `0.15`  
- **Inference:**

```bash
python -m src.predict --input examples/customer.json
```

Output: churn probability, threshold, boolean prediction, recommended action (`offer retention` / `no action`).

## Seven questions (project checklist)

1. **Why this loss?** Log loss fits probabilistic binary classification and supports calibrated decisions.  
2. **Why these metrics?** Imbalance + asymmetric costs require recall/PR-AUC/log loss, not accuracy alone.  
3. **Why this threshold?** Maximizes expected business value under explicit FP/FN/retention assumptions.  
4. **False positive?** Offer retention to someone who would have stayed — wasted cost.  
5. **False negative?** Miss a churner — no intervention, lost value.  
6. **Where does it fail?** FP-heavy in month-to-month; FN among some longer-tenure contracts; no behavioral features.  
7. **Best ML metric = best business?** **No** — HGB looks better on F1/PR-AUC/log loss, but LR + BV threshold wins on our cost model.

## Limitations

- Snapshot label ≠ true 30-day forward churn.  
- Costs are illustrative.  
- No extensive hyperparameter tuning or production monitoring.  
- Single random split (no nested CV).

## Future improvements

- Time-aware labels and leakage-safe feature windows  
- Cost/benefit sensitivity analysis  
- Calibration (Platt/isotonic) after model selection  
- Champion/challenger monitoring in production

## Project layout

See [PRD.md](PRD.md). Core code lives in `src/`; reports and figures in `reports/`; tests in `tests/`.
