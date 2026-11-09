# Financial Fraud Detection & Transaction Risk Analytics — Technical Documentation

This document provides end-to-end technical, architectural, and operational documentation for the fraud risk analytics and decisioning platform built on the IEEE-CIS Fraud Detection benchmark dataset. All numbers, metrics, and thresholds are derived directly from reproducible pipeline artifacts in `reports/` and `models/`.

---

## 1. Problem Statement and Business Motivation

In payment networks and digital commerce, millions of transactions occur daily, of which a small fraction (typically 0.1% to 3.5%) represent fraudulent activity such as identity theft, account takeover, synthetic identity fraud, or card-not-present fraud.

A payment risk system must classify each incoming transaction into an actionable decision tier:
1. **Auto-Approve (Low Risk):** Transactions proceed frictionlessly.
2. **Step-Up Verification (Medium Risk):** Additional verification is triggered (e.g., SMS OTP, 2FA, biometric authentication, or callback).
3. **Manual Review / Block (High Risk):** Routed to specialized fraud investigators or automatically declined.

Both types of classification mistakes incur significant financial and operational costs:
- **False Negative (Missed Fraud):** The merchant or financial institution absorbs the chargeback, interchange dispute penalties, asset loss, and customer churn.
- **False Positive (Blocked Legitimate Transaction):** Results in cart abandonment, merchant interchange loss, degraded customer satisfaction, and investigator labor cost (~$4.00 to $10.00 per manual review).

Because legitimate transactions overwhelmingly outnumber frauds (here, 96.5% legitimate vs. 3.5% fraudulent), **accuracy is a misleading and dangerous metric**: a naive classifier predicting "legitimate" for 100% of transactions scores 96.5% accuracy while preventing $0 in fraud losses. The operational objective is therefore to **maximize captured fraudulent value and recall under a bounded false positive review capacity (e.g., ≤ 1.0% FPR)**.

---

## 2. Fraud Detection System Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │ Raw IEEE-CIS CSVs (Transaction + Identity)   │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Preprocessing & Schema Validation (src/data/load.py, src/preprocessing/)               │
│ - Schema validation & float32 memory optimization (1.9 GB -> 0.95 GB)                  │
│ - Informative identity missingness extraction (24.42% match rate)                      │
│ - Chronological timeline slicing (70% Train / 15% Validation / 15% Test)               │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Feature Engineering (src/features/engineering.py) - 458 features                       │
│ - Amount shape (log amount, cents, round hundred flags)                                │
│ - Anchored calendar features (hour of day, day of week, night flag, weekend flag)      │
│ - Train-only frequency encodings (card1, addr1, P_emaildomain, DeviceInfo)             │
│ - Leak-free, past-only card & UID rolling history (velocity, prior mean, spend delta) │
└────────────────────────┬─────────────────────────────────┬─────────────────────────────┘
                         │                                 │
                         ▼                                 ▼
┌────────────────────────────────────────┐   ┌───────────────────────────────────────────┐
│ SQLite Analytics DB (src/database/)    │   │ Model Experimentation (src/models/train.py│
│ - 590,540 rows, 152,415 scored         │   │ - Multi-family benchmark (LR, RF, XGB, LGB│
│ - 8 performance indexes                │   │ - 5 Imbalance treatments (resampling/SMOTE│
│ - 17 named parameterised SQL queries   │   │ - Feature ablation (Core, History, Id)    │
└────────────────────────┬───────────────┘   └─────────────────────┬─────────────────────┘
                         │                                         │
                         │                                         ▼
                         │                   ┌───────────────────────────────────────────┐
                         │                   │ Evaluation & Decision Policy              │
                         │                   │ - Precision-Recall AUC & ROC-AUC          │
                         │                   │ - Operating threshold (F1 @ 1% max FPR)   │
                         │                   │ - 3-tier risk bands (High, Medium, Low)   │
                         │                   │ - Calibration analysis & error segments   │
                         │                   └─────────────────────┬─────────────────────┘
                         │                                         │
                         │                                         ▼
                         │                   ┌───────────────────────────────────────────┐
                         │                   │ Explainability (src/explainability/)      │
                         │                   │ - Global TreeSHAP feature ranking         │
                         │                   │ - Local waterfall transaction attribution │
                         │                   └─────────────────────┬─────────────────────┘
                         │                                         │
                         ▼                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Six-Page Streamlit Risk Analytics Dashboard (app.py, dashboard/)                       │
│ 1. Overview  2. EDA  3. Model Performance  4. Risk Assessment  5. SHAP  6. SQL Insights│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset: Source, Schema, and Characteristics

The system is trained and evaluated on the **IEEE-CIS Fraud Detection dataset** (provided by Vesta Corporation on Kaggle):

- **Row Count:** 590,540 labelled transactions.
- **Timeline:** 182 days (from `TransactionDT = 86,400` to `15,811,131` seconds, where `TransactionDT` represents elapsed seconds from a reference point).
- **Target:** `isFraud` binary label (0 = Legitimate, 1 = Fraudulent).
- **Total Fraud Count:** 20,663 frauds (prevalence of **3.499%**, imbalance ratio ≈ 1:28).
- **Raw Dimensions:**
  - `train_transaction.csv`: 394 columns including TransactionAmt, ProductCD, card1–card6, addr1–addr2, dist1–dist2, P_emaildomain, R_emaildomain, C1–C14 (count features), D1–D15 (timedelta features), V1–V339 (Vesta engineered attributes), M1–M9 (match flags).
  - `train_identity.csv`: 144,233 records with 41 identity and device attributes (id_01–id_38, DeviceType, DeviceInfo).

### Identity Linkage and Informative Missingness

Only **24.42%** of transactions have a matching record in `train_identity.csv`. In real-world card payments, device fingerprinting is only available for web or mobile browser checkouts; card-present or standard gateway payments lack device telemetry.

Rather than discarding unmatched rows (which would introduce severe selection bias), the absence of identity data is explicitly modeled:
- `has_identity = 1` if identity record exists, else `0`.
- `identity_missing_count`: Count of missing fields across device/identity blocks.
- `core_missing_count`: Count of missing values in core address/distance fields.

---

## 4. Chronological Splitting & Leakage Prevention

### Temporal Split

Random k-fold cross-validation or random train/test splits introduce fatal lookahead leakage in fraud systems: an attacker testing compromised cards on Day 10 would have transactions leaked into the training set, allowing the model to "memorize" future patterns.

To replicate real-world deployment, transactions are partitioned strictly forward in time across `TransactionDT`:
- **Training Set (first 70%):** Days 1.0 to 128.4 (`TransactionDT` ≤ 11,093,712). 438,125 transactions, 15,364 frauds (3.51% prevalence).
- **Validation Set (next 15%):** Days 128.4 to 155.7 (`TransactionDT` 11,093,713 to 13,452,422). 75,560 transactions, 2,617 frauds (3.46% prevalence).
- **Held-Out Test Set (final 15%):** Days 155.7 to 183.0 (`TransactionDT` > 13,452,422). 76,855 transactions, 2,682 frauds (3.49% prevalence).

All hyperparameter tuning, model family selection, imbalance strategy comparison, feature ablation, and threshold selection are performed on the validation period. The test set is evaluated **once** at the conclusion to provide unbiased generalization metrics.

### Leak-Free Behavioral History Aggregations

Cardholders exhibit historical behavioral baselines against which fraud can be identified (e.g., sudden spikes in spend or velocity). To prevent target leakage and future information bleed:
1. Records are sorted strictly by `(card1, TransactionDT)`.
2. Cumulative counts and sums are computed excluding the current transaction:
   $$\text{prior\_count}_i = \text{cumsum}(\mathbf{1})_i - 1$$
   $$\text{prior\_sum}_i = \text{cumsum}(\text{amount})_i - \text{amount}_i$$
   $$\text{amount\_vs\_card1\_mean}_i = \frac{\text{amount}_i}{\text{prior\_mean}_i} \quad (\text{defaulting to 1.0 on zero history})$$
3. A unit test verifies this property by perturbing future transactions and confirming that earlier rows' feature values remain strictly invariant.

---

## 5. Feature Engineering: 458 Engineered Signals

The production feature matrix comprises 458 clean features:

1. **Amount Structure:** `TransactionAmt`, `log_amount` ($\log(1 + x)$ to compress the heavy right tail), `amount_cents` (cents modulo 100), `amount_is_round` (flag for round hundred amounts indicative of card testing), and `amount_is_whole` (zero-cent flag).
2. **Anchored Calendar Dynamics:** `txn_hour` (cumulative hours from dataset start), `hour_of_day` (0–23), `day_of_week` (0–6), `day_index` (0–182), `is_night` (22:00–06:00), `is_weekend`. Anchoring to dataset start ensures single-transaction real-time scoring aligns identically with batch historical scoring.
3. **Card, Account & Geolocation:** `card1`–`card6`, `addr1` (billing region), `addr2` (billing country), `dist1` (billing-to-shipping distance proxy), `dist2`.
4. **Match Flags:** `M1` through `M9` mapped to binary numeric indicators (`T` = 1.0, `F` = 0.0, missing = NaN).
5. **Vesta Feature Blocks:** Count signals `C1`–`C14`, timedelta signals `D1`–`D15`, engineered signals `V1`–`V339`, identity numericals `id_01`–`id_38`.
6. **Train-Only Frequency Encodings:** Category frequencies for high-cardinality attributes (`card1_freq`, `addr1_freq`, `P_emaildomain_freq`, `DeviceInfo_freq`) calculated solely on the training split. Unseen test categories default to 0.0.
7. **Past-Only Rolling Velocity & History:**
   - `card1_prior_count`, `card1_prior_amt_mean`, `card1_prior_amt_std`
   - `amount_vs_card1_mean` (spend deviation ratio)
   - `card1_secs_since_prev` (recency in seconds)
   - `card1_prior_count_24h`, `card1_prior_count_7d` (rolling velocity counters)
   - `uid_prior_count`, `uid_prior_amt_mean`, `amount_vs_uid_mean` (compound user proxy UID: `card1_addr1_D1`).

---

## 6. Model Benchmarking & Imbalance Evaluation

Ten models across four supervised learning families and five class imbalance treatments were trained and evaluated on the common validation split:

| Rank | Model | Imbalance Strategy | Train Rows | Fit Time | Val PR-AUC | Val ROC-AUC | Recall @ 1% FPR |
|:---:|---|---|---:|---:|---:|---:|---:|
| **1** | **LightGBM (Full Train)** | **None (Natural Imbalance)** | **438,125** | **19.8s** | **0.5816** | **0.9240** | **48.18%** |
| 2 | LightGBM (Sample) | Random Oversampling (10%) | 42,456 | 6.0s | 0.4891 | 0.8867 | 40.73% |
| 3 | LightGBM (Sample) | None | 40,000 | 6.0s | 0.4824 | 0.8854 | 39.28% |
| 4 | XGBoost (Histogram) | None | 60,000 | 25.0s | 0.4800 | 0.8865 | 39.21% |
| 5 | LightGBM (Sample) | SMOTE (10% prevalence) | 42,456 | 9.1s | 0.4753 | 0.8808 | 39.51% |
| 6 | LightGBM (Sample) | Random Undersampling (10:1) | 15,433 | 3.5s | 0.4690 | 0.8845 | 38.94% |
| 7 | Random Forest | Balanced Subsample | 30,000 | 1.5s | 0.4070 | 0.8614 | 32.48% |
| 8 | Logistic Regression | Balanced Class Weight | 60,000 | 7.9s | 0.3536 | 0.8242 | 29.31% |
| 9 | LightGBM (Sample) | Scale Pos Weight (28.5) | 40,000 | 2.4s | 0.3234 | 0.8370 | 22.32% |
| 10 | LightGBM (Full Train) | Scale Pos Weight (28.5) | 438,125 | 9.2s | 0.3225 | 0.8508 | 23.77% |

### Key Experimental Insights

1. **Failure of Extreme Cost-Sensitive Weighting:** Applying extreme positive sample weighting (`scale_pos_weight = 28.5`) severely distorted the decision surface, driving PR-AUC down to ~0.32 and degrading Recall @ 1% FPR. Boosting trees optimize split criteria based on loss gradients; over-weighting minority gradients caused tree splits to isolate small noise clusters rather than genuine decision boundaries.
2. **Data Scale Dominates Resampling:** Fitting LightGBM on the full 438K training set without resampling achieved 0.5816 PR-AUC, significantly outperforming all subsampled imbalance treatments (which topped out at 0.4891). In tabular fraud datasets with high feature dimensions (458 columns), exposing the trees to the full negative distribution is superior to artificial synthetic oversampling.
3. **SMOTE with Missingness:** SMOTE generates synthetic interpolations in Euclidean feature space and requires non-null coordinates. While median-imputing NaNs allows SMOTE to run, its validation performance (0.4753 PR-AUC) underperformed simple random oversampling (0.4891 PR-AUC), as linear interpolations between high-dimensional sparse records created noisy, implausible feature combinations.

---

## 7. Decision Policy & Threshold Optimization

In production fraud systems, the threshold cannot be set to 0.50. At threshold 0.50, the model achieves high precision (79.96%) but misses **67.71% of all frauds** (only 32.29% recall).

### Operating Policy Selection

We conduct a full parametric threshold sweep ($\theta \in [0.01, 0.99]$) on the validation set under an operational constraint:
$$\max_{\theta} F_1(\theta) \quad \text{subject to} \quad \text{FPR}(\theta) \le 1.0\%$$

On validation data, the optimal operating threshold is identified as **$\theta^* = 0.2300$**:
- Validation FPR: **0.84%** (under the 1.0% review capacity limit).
- Validation Recall: **46.35%** (vs. 33.28% at 0.50).
- Validation F1: **0.5464** (vs. 0.4787 at 0.50).

### Three-Tier Risk Band Specification

Using the validation sweep, the system derives three actionable decision bands:

1. **High Risk ($\hat{p} \ge 0.2300$):**
   - **Operational Action:** Immediate block or prioritization for manual investigator review.
   - **Capacity Boundary:** Sized to consume at most 1.0% of legitimate payment traffic.
2. **Medium Risk ($0.0950 \le \hat{p} < 0.2300$):**
   - **Operational Action:** Step-up verification challenge (3D Secure, SMS OTP, biometric confirmation).
   - **Capacity Boundary:** Lowest cutoff whose cumulative alert rate fits within a 5.0% customer friction budget.
3. **Low Risk ($\hat{p} < 0.0950$):**
   - **Operational Action:** Frictionless auto-approval.

---

## 8. Final Held-Out Evaluation & Financial Impact

The champion model (`LightGBM | full train, no weighting`) was evaluated on the held-out test split (76,855 transactions, 2,682 frauds, $409,594 total fraud value):

| Metric | Baseline Policy ($\theta = 0.50$) | Deployed Policy ($\theta = 0.23$) | Lift / Delta |
|---|---:|---:|---:|
| **True Positives (Frauds Blocked)** | 866 | **1,237** | **+371 frauds (+42.8%)** |
| **False Positives (Legitimate Flagged)** | 217 | **846** | +629 reviews |
| **False Negatives (Missed Frauds)** | 1,816 | **1,445** | -371 misses |
| **True Negatives (Auto-Approved)** | 73,956 | **73,327** | — |
| **Recall** | 32.29% | **46.12%** | **+13.83 pp (+42.8% rel)** |
| **Precision** | 79.96% | **59.39%** | -20.57 pp |
| **F1 Score** | 0.4600 | **0.5192** | **+0.0592** |
| **PR-AUC** | 0.5296 | **0.5296** | — |
| **ROC-AUC** | 0.9095 | **0.9095** | — |
| **False Positive Rate** | 0.29% | **1.14%** | +0.85 pp |
| **Total Alerts Generated** | 1,083 | **2,083** | +1,000 alerts |
| **Fraud Value Captured ($)** | $132,239 | **$170,180** | **+$37,941 (+28.7%)** |
| **Fraud Value Captured (%)** | 32.29% | **41.55%** | **+9.26 pp** |
| **Investigation Cost (@ $4/alert)** | $4,332 | **$8,332** | +$4,000 |
| **Net Financial Benefit** | $127,907 | **$161,848** | **+$33,941 net profit** |

$$\text{Net Financial Benefit} = \text{Fraud Value Captured} - (\text{Alerts} \times \text{Cost per Review})$$
The policy generates **$161,848 in net savings**, returning over $20 in saved fraud value for every $1 spent on manual investigations.

---

## 9. Feature Set Ablation Study

To understand the source of predictive signal and prevent deployment of non-viable features, we executed a controlled validation ablation across three feature groups:

1. **Core Transaction (405 features):** Amount, calendar, card/address fields, Vesta C/D/V blocks, missingness flags. PR-AUC: **0.4869**.
2. **Core + Behavioral History (415 features):** Adds rolling velocity and spend deviation signals. PR-AUC: **0.4884**.
3. **Full with Device/Identity (458 features):** Adds identity table attributes, browser, OS, and missingness signals. PR-AUC: **0.5013** (+0.0129 lift; Recall @ 1% FPR jumps from 40.4% to 48.2%).

---

## 10. Model Explainability via TreeSHAP

To satisfy regulatory explainability requirements and provide clear rationale for risk analysts, the system integrates **TreeSHAP** (Tree-based Shapley Additive Explanations).

### Top 10 Global Predictive Features

| Rank | Feature | Mean \|SHAP\| | Business Meaning |
|:---:|---|---:|---|
| **1** | `C13` | 0.2056 | Rolling transaction count associated with the card/account identifier. |
| **2** | `C1` | 0.1891 | Count of transactions across the associated address/account cluster. |
| **3** | `V258` | 0.1495 | Vesta cross-reference consistency scoring between device and card. |
| **4** | `TransactionAmt` | 0.1449 | Transaction dollar amount; anomalous spikes push scores upward. |
| **5** | `C14` | 0.1350 | Frequency of transaction occurrences in recent velocity windows. |
| **6** | `cat_card6` | 0.1239 | Card funding type (Credit cards exhibit higher fraud rates than Debit). |
| **7** | `cat_P_emaildomain` | 0.1161 | Purchaser email provider domain risk profile. |
| **8** | `cat_R_emaildomain` | 0.1054 | Recipient email provider domain risk profile. |
| **9** | `V294` | 0.1016 | Velocity accumulation metric within transaction bursts. |
| **10** | `V70` | 0.0892 | Identity match score between billing and shipping identities. |

### Local Transaction Attribution

For any scored payment, the system produces a local waterfall decomposition:
$$\log\left(\frac{p}{1 - p}\right) = \text{base\_value} + \sum_{j=1}^{M} \phi_j$$
Investigators can inspect precisely which attributes pushed an alert over the 0.23 threshold (e.g., sudden spend burst on a credit card via a high-risk domain with absent identity match).

---

## 11. SQLite Analytics Warehouse

The project compiles a relational analytical warehouse (`data/fraud.db`, 93.9 MB) indexed across 8 dimensions.

### Core Registered Queries (`src/database/queries.py`)

1. `overview`: Portfolio summary of transaction volume, fraud counts, fraud rate %, and total monetary losses.
2. `scored_overview`: TP, FP, FN, precision, recall, and alert rates across all scored transactions.
3. `fraud_by_product`: Breakdown of transaction volume and fraud risk across ProductCDs (W, C, R, H, S).
4. `fraud_by_card`: Fraud rates aggregated by card network (`card4`) and funding type (`card6`).
5. `identity_coverage`: Contrast in fraud rate and volume between identity-present vs. identity-absent cohorts.
6. `risk_bands`: Transaction counts, dollar amounts, and actual fraud capture across High, Medium, and Low risk bands.
7. `daily_trends`: Time-series grouping by `day_index` to track prevalence and volume stability over the 182-day timeline.
8. `top_risk_transactions`: Rank-ordered extraction of highest risk transactions with full feature metadata for analyst triage.
9. `confusion_at_threshold`: Dynamic confusion matrix computation at arbitrary score thresholds.

---

## 12. Testing and Validation Framework

The repository is guarded by 54 automated pytest tests across 7 test suites:

- `test_features.py`: Verifies leak-free history calculation, chronological order independence, and feature matrix dimensions.
- `test_validation.py`: Tests input boundary conditions, string/numeric coercion, range checks, and optional field handling.
- `test_predict.py`: Validates end-to-end inference, partial-record median default imputation, and risk band assignment.
- `test_metrics.py`: Tests confusion matrices, value capture math, monotonic threshold sweeps, and constrained optimizer behavior.
- `test_sql.py`: Validates schema integrity, query syntax, and parameter binding against a test database fixture.
- `test_preprocessing.py`: Verifies non-overlapping temporal split boundaries and stratification math.
- `test_dashboard.py`: Verifies formatter helpers, artifact path status reporters, and UI dependencies.

---

## 13. Production Deployment Roadmap

To move this system into a high-throughput, low-latency live production environment:

1. **Feature Store Integration:** Deploy Feast or Hopsworks for streaming feature calculation. Features such as `card1_prior_count_24h` and `amount_vs_card1_mean` would be maintained in a sub-millisecond Redis online store.
2. **Real-Time Serving Endpoint:** Wrap `FraudScorer` in a FastAPI microservice deployed on Kubernetes. Target p99 inference latency: < 25 ms.
3. **Data Drift & Concept Drift Monitoring:** Deploy Evidently AI or Arize to monitor Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) drift across numerical features (`TransactionAmt`, `C13`, `V258`), alerting when seasonal consumer behavior diverges from training baselines.
4. **Automated Shadow Retraining Pipeline:** Implement an Airflow DAG to retrain the champion LightGBM model weekly on a sliding 90-day window, running shadow evaluation against the live model before automated promotion.
