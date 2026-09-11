# Customer Churn Prediction — PRD

## Goal

Build an end-to-end beginner data science project that predicts whether a customer will churn within 30 days.

The primary learning goals are:

* Binary classification
* Loss functions
* Evaluation metrics
* Classification thresholds
* Error analysis
* Translating ML performance into business value

Use Python, pandas, NumPy, scikit-learn, matplotlib/seaborn, Jupyter, pytest, and YAML configuration.

Do not introduce deep learning, Spark, Ray, Docker, or cloud infrastructure.

---

## Project Structure

Create:

```text
customer-churn/
├── data/raw/
├── data/processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── business/
├── tests/
├── reports/
├── configs/
├── README.md
├── PRD.md
└── requirements.txt
```

---

# Phase 1 — Data + EDA

Use a public customer churn dataset such as IBM Telco Customer Churn.

Implement:

* Data loading
* Data validation
* Missing-value analysis
* Duplicate detection
* Data-type inspection
* Target distribution
* Basic feature distributions
* Churn relationships with important features
* Leakage investigation

Document important observations and their potential modeling implications.

### Commit

```text
feat: initialize project and complete exploratory analysis
```

---

# Phase 2 — Baselines

Create a reproducible stratified train/test split.

Implement:

* Majority-class baseline
* Simple baseline prediction

Evaluate:

* Accuracy
* Precision
* Recall
* F1

Use the baselines to establish what "better than trivial" means.

### Commit

```text
feat: establish classification baselines
```

---

# Phase 3 — Logistic Regression

Build an sklearn pipeline containing:

* Missing-value handling
* Numerical preprocessing
* Categorical encoding
* Logistic regression

Train using binary cross-entropy/log loss.

Evaluate:

* Log loss
* Accuracy
* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC
* Confusion matrix

Document:

* What the model outputs
* Why binary cross-entropy is appropriate
* Difference between training loss and evaluation metrics

### Commit

```text
feat: train logistic regression churn model
```

---

# Phase 4 — Metric Analysis

Compare all evaluation metrics.

Explain what each metric rewards and when it can be misleading.

Specifically investigate situations where:

* Accuracy improves but recall decreases
* Precision and recall trade off
* ROC-AUC and PR-AUC tell different stories
* Two models have similar accuracy but different log loss

### Commit

```text
feat: add comprehensive classification evaluation
```

---

# Phase 5 — Threshold Analysis

Use predicted probabilities from the model.

Evaluate thresholds from 0.05–0.95.

For each threshold calculate:

* TP
* FP
* TN
* FN
* Precision
* Recall
* F1
* Accuracy
* Positive prediction rate

Create plots showing how these metrics change with the threshold.

Explain why 0.5 is not inherently the optimal threshold.

### Commit

```text
feat: add classification threshold analysis
```

---

# Phase 6 — Business Cost

Create a configurable business cost model.

Include:

* False-positive cost
* False-negative cost
* Successful retention value

Calculate expected business value at each classification threshold.

Compare:

* Default 0.5 threshold
* F1-optimal threshold
* Business-value-optimal threshold

Select the threshold based on explicit business assumptions rather than simply choosing the highest ML metric.

### Commit

```text
feat: optimize classification threshold for business value
```

---

# Phase 7 — Nonlinear Model

Train one nonlinear model, preferably Random Forest or Gradient Boosting.

Compare it against logistic regression using:

* Log loss
* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC
* Calibration
* Business value

Consider:

* Predictive performance
* Interpretability
* Complexity
* Training/inference cost

Do not perform extensive hyperparameter tuning.

### Commit

```text
feat: compare linear and nonlinear models
```

---

# Phase 8 — Calibration + Error Analysis

Evaluate probability calibration using:

* Calibration curve
* Brier score
* Log loss

Analyze:

* True positives
* True negatives
* False positives
* False negatives

Identify where the model performs poorly and potential data-quality or feature issues.

### Commit

```text
feat: add probability calibration and error analysis
```

---

# Phase 9 — Final Model + Inference

Select the final model and threshold based on:

1. Predictive performance
2. Probability quality
3. Business value
4. Interpretability
5. Complexity

Create a simple prediction interface:

```text
python -m src.predict --input customer.json
```

Output:

* Churn probability
* Selected threshold
* Churn prediction
* Recommended action

Add tests for preprocessing, metrics, threshold calculations, business calculations, and inference.

### Commit

```text
feat: finalize model and prediction interface
```

---

# Final README

Document:

* Business problem
* Dataset
* Features
* Modeling approach
* Loss function and why it was chosen
* Evaluation metrics and why they matter
* Threshold selection
* Business assumptions
* Model comparison
* Error analysis
* Final model
* Limitations
* Future improvements

The project is complete only when you can clearly answer:

1. Why this loss function?
2. Why these evaluation metrics?
3. Why this threshold?
4. What does a false positive mean?
5. What does a false negative mean?
6. Where does the model fail?
7. Does the best ML metric correspond to the best business outcome?
