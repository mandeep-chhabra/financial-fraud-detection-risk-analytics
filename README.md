# Financial Fraud Detection & Transaction Risk Analytics

An end-to-end fraud risk analytics and decision system built on the **IEEE-CIS Fraud Detection benchmark** (590,540 real-world transactions across 182 days): leak-safe feature engineering, multi-family model comparison across five imbalance strategies, threshold optimization under manual-review capacity constraints, SHAP explanations, SQL warehouse analytics, and a six-page Streamlit dashboard.

---

## Results at a glance

Evaluated **once** on the held-out, strictly future time period (day 156 to day 183) at an operating threshold chosen on the validation period (day 128 to day 156):

| Metric | Validation (day 128–156) | Test (held out, day 156–183) | Baseline (threshold = 0.5) |
|---|---:|---:|---:|
| **Operating Threshold** | **0.2300** | **0.2300** | 0.5000 |
| **Precision** | 66.54% | 59.39% | 79.96% |
| **Recall** | 46.35% | **46.12%** | 32.29% |
| **F1 Score** | 0.5464 | **0.5192** | 0.4600 |
| **PR-AUC** | 0.5816 | **0.5296** | 0.5296 |
| **ROC-AUC** | 0.9240 | **0.9095** | 0.9095 |
| **False-positive rate (FPR)** | 0.84% | **1.14%** | 0.29% |
| **Alert Volume** | 1,823 | 2,083 | 1,083 |
| **Fraud Value Captured** | 41.00% | **41.55% ($170,180)** | 32.29% ($132,239) |
| **Net Financial Benefit** | $159,036 | **$161,848** | $127,907 |

> [!NOTE]
> Optimizing the decision threshold from the naive 0.5 default to the policy-constrained **0.2300** produces a **+42.8% relative lift in fraud recall** (from 32.29% to 46.12%) and captures **+$37,941 in additional fraudulent value**, maintaining an FPR within the ~1% operational review capacity.

### Feature-set ablation (validation period)

| Feature set | Features | Validation PR-AUC | Recall @ 1% FPR | Top Gain Drivers |
|---|---:|---:|---:|---|
| **Core transaction** | 405 | 0.4869 | 40.43% | `V258`, `C1`, `V243`, `card1_freq`, `C13`, `txn_hour` |
| **Core + behavioural history** | 415 | 0.4884 | 39.78% | `V258`, `C1`, `C14`, `card1_prior_amt_std`, `card2` |
| **Full (with device & identity)** | 458 | **0.5013** | **48.18%** | `V258`, `C13`, `C1`, `id_02`, `DeviceInfo_freq`, `id_31` |

Adding identity and device features yields an honest **+0.0129 PR-AUC gain** and boosts Recall @ 1% FPR by **+8.4 percentage points**, even though identity data is available for only 24.4% of transactions.

---

## Architecture & System Overview

```
IEEE-CIS Raw CSVs (train_transaction + train_identity)
        │
        ▼
src/data/load.py ── schema validation, quality audit, memory-efficient float32 downcasting
        │
        ▼
src/features/engineering.py ── 458 features: amount shape, calendar, card/address,
        │                      train-only frequency encoding, strictly past-only velocity
        │
        ├──────────────► SQLite Warehouse (590k rows + scores) ──► 17 analytical SQL queries
        ▼
src/models/train.py ── strict chronological split (70% train / 15% val / 15% test)
        │              Logistic Regression / Random Forest / XGBoost / LightGBM
        │              × 5 imbalance strategies × feature ablation
        ▼
src/evaluation/ ── PR-AUC, recall@1%FPR, calibration, threshold curves, error analysis
        │
        ├──► models/fraud_model.joblib + risk_thresholds.json + feature_encoder.json
        ├──► reports/ (metrics, ablation, figures, evaluation report)
        └──► src/explainability/ ── TreeSHAP global + waterfall local ──► dashboard
```

### Six Dashboard Pages

1. **Overview** — Portfolio KPIs, risk band distributions, test confusion matrix and value captured.
2. **Exploratory Analytics** — Class imbalance, amount distributions, fraud rate by product code, card types, missingness, and temporal trends.
3. **Model Performance** — Multi-model leaderboard, PR and ROC curves, calibration analysis, interactive threshold simulator.
4. **Transaction Risk Assessment** — Live single-transaction risk scoring with partial-data default handling and SHAP local attribution; batch CSV scoring.
5. **Explainability** — Global TreeSHAP feature importance ranking, beeswarm summaries, dependence plots, and representative case studies.
6. **SQL Insights** — 17 named parameterised SQL queries executed directly against SQLite for real-time reporting.

---

## Dataset

[IEEE-CIS Fraud Detection Benchmark](https://www.kaggle.com/c/ieee-fraud-detection) (Vesta Corporation):
- **590,540 labelled transactions** spanning 182 days (TransactionDT 86,400 to 15,811,131 seconds)
- **20,663 fraudulent transactions (3.499%)**, imbalance ratio ≈ 1:28
- **394 transaction columns + 41 identity columns** (device type, browser, OS, IP/region proxies, Vesta C/D/V engineered feature blocks)
- Identity match in **24.42%** of transactions; identity absence is modelled explicitly as an informative feature (`has_identity`, `identity_missing_count`)

---

## Quick start

```bash
# 1. Environment setup
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run end-to-end pipeline (dataset extraction -> EDA -> training -> DB -> SHAP)
python scripts/run_all.py

# 3. Launch dashboard
streamlit run app.py
```

### Individual Pipeline Stages

```bash
python scripts/build_dataset.py       # Builds ieee_features.parquet & encoders
python scripts/run_eda.py             # Generates EDA plots and summary JSON
python scripts/run_training.py        # Model comparison, thresholding, ablation
python scripts/build_database.py      # Populates SQLite warehouse (fraud.db)
python scripts/run_explainability.py  # Computes TreeSHAP explanations
python -m src.evaluation.report       # Compiles reports/EVALUATION_REPORT.md
pytest                                # 54 tests across all modules
```

---

## Key Design Principles

1. **Strict Chronological Splitting, Never Random:** Split by `TransactionDT` (train first 70% ≤ day 128.4; validation next 15% to day 155.7; test held-out final 15% to day 183.0). Prevents lookahead leakage and mirrors real-world production deployment.
2. **Leakage-Safe Behavioural History:** Account history features (`card1_prior_count`, `card1_prior_amt_mean`, `amount_vs_card1_mean`, `uid_prior_count`) use cumulative stats strictly excluding the current row. Verified via automated perturbation tests.
3. **Train-Only Categorical & Frequency Encoding:** High-cardinality encodings (`card1_freq`, `addr1_freq`, `P_emaildomain_freq`, `DeviceInfo_freq`) are fitted solely on training data. Unseen test categories safely map to default bins.
4. **Policy-Constrained Thresholding:** Threshold is optimized on validation data to maximize F1 under an operational 1% False Positive Rate review budget ($4.00 per manual investigation cost).
5. **Derived Three-Tier Risk Bands:**
   - **High Risk (score ≥ 0.2300):** Block or prioritize for manual investigation.
   - **Medium Risk (score ≥ 0.0950):** Route to step-up verification (OTP/2FA, sized to 5% customer capacity).
   - **Low Risk (score < 0.0950):** Auto-approve payment.
6. **Robust Missingness Handling:** Missing values are native signals in tree models (LightGBM/XGBoost). Missingness counters (`core_missing_count`, `identity_missing_count`) provide structural cues to tree splitters.

---

## Project Structure

```
fraud-detection/
├── app.py                      # Streamlit entry point (6 pages)
├── dashboard/                  # Page modules & cached data access
│   ├── overview.py             # Portfolio KPIs & summary
│   ├── eda_page.py             # Exploratory analytics visualizations
│   ├── model_performance.py    # Benchmark leaderboard, curves & threshold explorer
│   ├── risk_assessment.py     # Real-time transaction scoring & batch CSV upload
│   ├── explainability_page.py  # Global and local SHAP analysis
│   └── sql_insights.py         # Live parameterised SQL queries
├── src/
│   ├── config.py               # Paths, schema definitions, policy constants
│   ├── data/load.py            # Data loading, schema validation, quality audit
│   ├── preprocessing/          # Forward temporal splits, input schema validation
│   ├── features/engineering.py # 458 engineered features, past-only velocity
│   ├── models/                 # Registry (LGBM, XGBoost, RF, LR), train & predict
│   ├── evaluation/             # Metrics, threshold sweep, error analysis, report
│   ├── explainability/         # TreeSHAP global importance and local waterfalls
│   └── database/               # SQLite schema + 17 analytical queries
├── scripts/                    # End-to-end pipeline scripts
├── tests/                      # 54 automated pytest tests
├── models/                     # Serialised model, thresholds, encoder mapping
├── reports/                    # Metrics JSONs, ablation report, figures, markdown

```

---

## Documentation

- [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md) — Comprehensive technical documentation (business problem, architecture, feature dictionary, modelling decisions, error analysis).
- [`reports/EVALUATION_REPORT.md`](reports/EVALUATION_REPORT.md) — Experiment report compiled from pipeline artifacts.

## Tech Stack

Python 3.14 · pandas · scikit-learn · LightGBM · XGBoost · imbalanced-learn · TreeSHAP · SQLite · Streamlit · Plotly · pytest
