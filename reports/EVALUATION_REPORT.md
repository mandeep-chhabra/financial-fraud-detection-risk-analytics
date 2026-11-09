# Evaluation Report

Generated from the persisted experiment artefacts. Every number below is
reproducible with `python scripts/run_all.py`; nothing is hand-entered.

## 1. Experimental setup

- **Selected model:** LightGBM | full train, no weighting
- **Feature set:** full (transaction + identity + past-only history)
- **Model training rows:** 438,125 (sampled from the full training period)
- **Validation rows:** 75,560
- **Test rows:** 76,855
- **Operating threshold:** 0.2300 (chosen on validation)
- **Policy:** maximise F1 (fraud caught vs alert precision) subject to FPR <= 1.00% manual-review capacity on the validation period

Temporal split: train (first 70%), validation (next 15%), test (final 15%).
The test period is evaluated exactly once, after all decisions are frozen.

## 2. Class imbalance

Fraud prevalence is 3.499% overall (20,663 of 590,540), an imbalance of about 1:28. A constant 'legitimate' predictor would score 96.501% accuracy while catching zero fraud, which is why accuracy is not used for model selection.

| Period | Rows | Frauds | Prevalence |
|---|---:|---:|---:|
| train | 438,125 | 15,364 | 3.5068% |
| validation | 75,560 | 2,617 | 3.4635% |
| test | 76,855 | 2,682 | 3.4897% |

## 3. Model comparison (validation period)

 rank                                       model  train_rows  precision   recall       f1   pr_auc  roc_auc  recall_at_1pct_fpr
    1         LightGBM | full train, no weighting      438125   0.852250 0.332824 0.478703 0.581631 0.923984            0.481849
    2              LightGBM | random oversampling       42456   0.768131 0.287352 0.418242 0.489078 0.886687            0.407337
    3                            LightGBM | SMOTE       42456   0.832461 0.243026 0.376220 0.475290 0.880762            0.395109
    4                    LightGBM | no resampling       40000   0.839332 0.249522 0.384683 0.482437 0.885369            0.392816
    5             XGBoost | full train, histogram       60000   0.815725 0.253726 0.387059 0.480017 0.886507            0.392052
    6             LightGBM | random undersampling       15433   0.667908 0.342759 0.453030 0.469043 0.884480            0.389377
    7          Random Forest | balanced_subsample       30000   0.483068 0.370653 0.419459 0.407014 0.861397            0.324799
    8 Logistic Regression | class_weight=balanced       60000   0.114351 0.704241 0.196755 0.353609 0.824170            0.293084
    9         LightGBM | full train, class weight      438125   0.000000 0.000000 0.000000 0.322458 0.850797            0.237677
   10                     LightGBM | class weight       40000   0.000000 0.000000 0.000000 0.323398 0.836977            0.223156

**Selection rule:** Within 0.005 recall of the best at FPR<=1%, keep models within 0.01 PR-AUC of that group's best, then prefer the largest training set (tie-break PR-AUC).

A 0.01 PR-AUC band is narrower than the sampling error of ~3,100 validation frauds; selecting on the fourth decimal would fit noise.

## 4. Held-out test results

At the validation-tuned threshold of **0.2300**:

| Metric | Validation | Test |
|---|---:|---:|
| Precision | 0.6654 | 0.5939 |
| Recall | 0.4635 | 0.4612 |
| F1 | 0.5464 | 0.5192 |
| PR-AUC | 0.5816 | 0.5296 |
| ROC-AUC | 0.9240 | 0.9095 |
| False positive rate | 0.0084 | 0.0114 |
| Alerts | 1,823 | 2,083 |
| Fraud value captured | 41.00% | 41.55% |

Confusion matrix (test): TP 1,237 · FP 846 · FN 1,445 · TN 73,327.

Recall at a 1% false-positive budget: **0.4485** (FPR 0.994%).

### Prevalence and temporal stability

Split prevalence is stable across the timeline (~3.5% across train, validation, and test). The forward temporal split evaluates model performance on future transactions without temporal leakage.

## 5. Feature-set ablation (validation only)

Trained on a 60,000-row stratified sample.

| Feature set | Features | PR-AUC | Recall @ 1% FPR | Precision @ 0.5 | FPR @ 0.5 |
|---|---:|---:|---:|---:|---:|
| core_transaction | 405 | 0.4869 | 0.4043 | 0.8223 | 0.00206 |
| core_plus_history | 415 | 0.4884 | 0.3978 | 0.8397 | 0.00185 |
| full_with_identity | 458 | 0.5013 | 0.4238 | 0.8461 | 0.00174 |

- **core_transaction:** Amount, calendar, card/address, Vesta C/D/V blocks and identity numbers. This is what a single transaction record supports with no memory of the card's past.
- **core_plus_history:** Adds past-only card and uid velocity/spend-deviation features. The gain is the value of behavioural history.
- **full_with_identity:** Adds the device/identity block (only 24.4% of rows have it) and the identity-presence flags. The gap quantifies how much device data is worth in this dataset.

## 6. Threshold analysis

- High-risk cutoff: **0.2300**
- Medium-risk cutoff: **0.0950**
- Rule: High risk: p >= high_threshold (validation-optimal threshold under a 1% FPR review budget). Medium risk: p >= medium_threshold, the lowest score cutoff whose cumulative alert volume still fits the 5% step-up-verification capacity. Low risk: everything below. Bands are policy, not ground truth.

| Threshold | Precision | Recall | FPR | Alerts | F1 |
|---:|---:|---:|---:|---:|---:|
| 0.010 | 0.0898 | 0.9373 | 0.34091 | 27,320 | 0.1639 |
| 0.110 | 0.4811 | 0.5896 | 0.02281 | 3,207 | 0.5299 |
| 0.210 | 0.6386 | 0.4746 | 0.00964 | 1,945 | 0.5445 |
| 0.310 | 0.7457 | 0.4169 | 0.00510 | 1,463 | 0.5348 |
| 0.410 | 0.8045 | 0.3665 | 0.00319 | 1,192 | 0.5035 |
| 0.510 | 0.8581 | 0.3282 | 0.00195 | 1,001 | 0.4748 |
| 0.610 | 0.8860 | 0.2851 | 0.00132 | 842 | 0.4313 |
| 0.710 | 0.9180 | 0.2438 | 0.00078 | 695 | 0.3853 |
| 0.810 | 0.9400 | 0.2037 | 0.00047 | 567 | 0.3348 |
| 0.910 | 0.9549 | 0.1376 | 0.00023 | 377 | 0.2405 |

## 7. Calibration

- Brier score (raw, test): **0.022408**
- Isotonic recalibration would change Brier by -0.000090; the raw score is kept for ranking and thresholding (documented in `calibration.json`).
- Mean predicted probability 0.031584 vs observed fraud rate 0.034897.

## 8. Error analysis (test)

- False positives: 846 (11.41 per 1,000 legitimate transactions), total value 181,933.
- False negatives: 1,445, 58.45% of fraud value missed.

### By product code

segment  transactions  frauds  fraud_rate  precision   recall      fpr   pr_auc  fraud_value_captured_pct
      C          8378    1117    0.133325   0.692441 0.713518 0.048754 0.733240                  0.757820
      H          2289     130    0.056793   0.466258 0.584615 0.040296 0.555858                  0.650762
      R          3785     179    0.047292   0.686869 0.759777 0.017194 0.810373                  0.826794
      S          2757     123    0.044614   0.666667 0.390244 0.009112 0.538776                  0.305225
      W         59646    1133    0.018995   0.360721 0.158870 0.005452 0.196286                  0.270213

### By amount quartile

    segment  transactions  frauds  fraud_rate  precision   recall      fpr   pr_auc  fraud_value_captured_pct
Q1 smallest         19215     892    0.046422   0.664948 0.578475 0.014190 0.636010                  0.542944
         Q2         19304     467    0.024192   0.710843 0.379015 0.003822 0.492423                  0.375977
         Q3         19549     544    0.027828   0.553623 0.351103 0.008103 0.423210                  0.325523
 Q4 largest         18787     779    0.041465   0.495091 0.453145 0.019991 0.503775                  0.425354

### Threshold sensitivity

 threshold  precision   recall      fpr  alerts  fraud_value_captured_pct
    0.1150   0.418117 0.564504 0.028407    3621                  0.555220
    0.1725   0.514231 0.505220 0.017257    2635                  0.477724
    0.2300   0.593855 0.461223 0.011406    2083                  0.415485
    0.2875   0.650255 0.428412 0.008332    1767                  0.364812
    0.3450   0.711155 0.399329 0.005865    1506                  0.323350
    0.4600   0.776541 0.338180 0.003519    1168                  0.238228

## 9. Limitations

- Synthetic identity matching: 24.4% of transactions link to identity records; lack of match is informative but not conclusive.
- High cardinality categoricals (e.g. card1, DeviceInfo) require careful frequency/history encoding to prevent overfitting.
- Temporal drift: while prevalence is stable (~3.5%), fraud patterns and transaction attributes shift over time.
- Review cost ($4.00) and review capacity (1% FPR) are illustrative operating assumptions.
- The model is designed for pre-authorisation assessment without post-settlement chargeback knowledge.

## 10. Reproducibility

```bash
python scripts/download_data.py
python scripts/run_all.py
pytest
streamlit run app.py
```

Thresholds file: `risk_thresholds.json` · comparison CSV: `model_comparison.csv`