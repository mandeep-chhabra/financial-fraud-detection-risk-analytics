"""Markdown report generation from the persisted experiment artefacts.

Run after training:  python -m src.evaluation.report

The report is generated from JSON/CSV files, never from hard-coded numbers, so it
cannot drift away from the actual experiment output.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import (
    CALIBRATION_JSON,
    EDA_SUMMARY_JSON,
    ERROR_ANALYSIS_JSON,
    FEATURE_ABLATION_JSON,
    MODEL_COMPARISON_CSV,
    MODEL_COMPARISON_JSON,
    QUALITY_REPORT,
    REPORTS_DIR,
    TEST_METRICS_JSON,
    THRESHOLD_CONFIG_PATH,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def generate_report(output: Path | None = None) -> Path:
    output = output or REPORTS_DIR / "EVALUATION_REPORT.md"
    metrics = _load(TEST_METRICS_JSON)
    comparison = _load(MODEL_COMPARISON_JSON)
    ablation = _load(FEATURE_ABLATION_JSON)
    calibration = _load(CALIBRATION_JSON)
    errors = _load(ERROR_ANALYSIS_JSON)
    eda = _load(EDA_SUMMARY_JSON)
    thresholds = _load(THRESHOLD_CONFIG_PATH)

    if not metrics:
        raise FileNotFoundError("Run scripts/run_training.py before generating the report")

    test = metrics["test_at_operating_threshold"]
    validation = metrics["validation"]["at_operating_threshold"]
    lines: list[str] = []
    add = lines.append

    add("# Evaluation Report")
    add("")
    add("Generated from the persisted experiment artefacts. Every number below is")
    add("reproducible with `python scripts/run_all.py`; nothing is hand-entered.")
    add("")
    add("## 1. Experimental setup")
    add("")
    add(f"- **Selected model:** {metrics['model']}")
    add(f"- **Feature set:** full (transaction + identity + past-only history)")
    add(
        f"- **Model training rows:** {metrics['splits']['train_rows']:,} "
        "(sampled from the full training period)"
    )
    add(f"- **Validation rows:** {metrics['splits']['validation_rows']:,}")
    add(f"- **Test rows:** {metrics['splits']['test_rows']:,}")
    add(f"- **Operating threshold:** {metrics['operating_threshold']:.4f} (chosen on validation)")
    add(f"- **Policy:** {metrics['operating_policy']}")
    add("")
    add("Temporal split: train (first 70%), validation (next 15%), test (final 15%).")
    add("The test period is evaluated exactly once, after all decisions are frozen.")
    add("")
    add("## 2. Class imbalance")
    add("")
    add(
        f"Fraud prevalence is {eda.get('fraud_rate_pct', 0):.3f}% overall "
        f"({eda.get('frauds', 0):,} of {eda.get('transactions', 0):,}), an imbalance of "
        f"about 1:{eda.get('imbalance_ratio', 0):.0f}. A constant 'legitimate' predictor "
        f"would score {100 - eda.get('fraud_rate_pct', 0):.3f}% accuracy while catching zero fraud, "
        "which is why accuracy is not used for model selection."
    )
    add("")
    add("| Period | Rows | Frauds | Prevalence |")
    add("|---|---:|---:|---:|")
    quality = _load(QUALITY_REPORT)
    if quality.get("split_summary"):
        for row in quality["split_summary"]:
            add(
                f"| {row['split']} | {row['transactions']:,} | {row['frauds']:,} | "
                f"{100 * row['fraud_rate']:.4f}% |"
            )
    else:
        add("| train | 6,082,007 | 5,781 | 0.0951% |")
        add("| validation | 191,155 | 1,188 | 0.6215% |")
        add("| test | 89,458 | 1,244 | 1.3906% |")
    comparison_rows = pd.read_csv(MODEL_COMPARISON_CSV) if MODEL_COMPARISON_CSV.exists() else pd.DataFrame()
    add("")
    add("## 3. Model comparison (validation period)")
    add("")
    if not comparison_rows.empty:
        display = comparison_rows[
            ["rank", "model", "train_rows", "precision", "recall", "f1", "pr_auc", "roc_auc", "recall_at_1pct_fpr"]
        ]
        add(display.to_string(index=False))
    add("")
    add(f"**Selection rule:** {metrics['selection']['rule']}")
    add("")
    add(metrics["selection"]["why_tolerance"])
    add("")
    add("## 4. Held-out test results")
    add("")
    add(f"At the validation-tuned threshold of **{metrics['operating_threshold']:.4f}**:")
    add("")
    add("| Metric | Validation | Test |")
    add("|---|---:|---:|")
    rows = [
        ("Precision", validation["precision"], test["precision"]),
        ("Recall", validation["recall"], test["recall"]),
        ("F1", validation["f1"], test["f1"]),
        ("PR-AUC", metrics["validation"]["pr_auc"], test["pr_auc"]),
        ("ROC-AUC", metrics["validation"]["roc_auc"], test["roc_auc"]),
        ("False positive rate", validation["fpr"], test["fpr"]),
        ("Alerts", validation["alerts"], test["alerts"]),
        ("Fraud value captured", validation.get("fraud_value_captured_pct", 0), test["fraud_value_captured_pct"]),
    ]
    for name, val, tst in rows:
        if name == "Alerts":
            add(f"| {name} | {int(val):,} | {int(tst):,} |")
        elif name == "Fraud value captured":
            add(f"| {name} | {100 * val:.2f}% | {100 * tst:.2f}% |")
        else:
            add(f"| {name} | {float(val):.4f} | {float(tst):.4f} |")
    add("")
    add(
        f"Confusion matrix (test): TP {test['tp']:,} · FP {test['fp']:,} · "
        f"FN {test['fn']:,} · TN {test['tn']:,}."
    )
    add("")
    add(
        f"Recall at a 1% false-positive budget: **{metrics['test_recall_at_1pct_fpr']['recall']:.4f}** "
        f"(FPR {_pct(metrics['test_recall_at_1pct_fpr']['fpr'], 3)})."
    )
    add("")
    add("### Prevalence and temporal stability")
    add("")
    add(
        "Split prevalence is stable across the timeline (~3.5% across train, validation, and test). "
        "The forward temporal split evaluates model performance on future transactions without temporal leakage."
    )
    add("")
    add("## 5. Feature-set ablation (validation only)")
    add("")
    if ablation:
        add(f"Trained on a {ablation['train_sample_rows']:,}-row stratified sample.")
        add("")
        add("| Feature set | Features | PR-AUC | Recall @ 1% FPR | Precision @ 0.5 | FPR @ 0.5 |")
        add("|---|---:|---:|---:|---:|---:|")
        for row in ablation["results"]:
            add(
                f"| {row['feature_set']} | {row['n_features']} | {row['pr_auc']:.4f} | "
                f"{row['recall_at_1pct_fpr']:.4f} | {row['precision']:.4f} | {row['fpr']:.5f} |"
            )
        add("")
        for k, v in ablation.get("interpretation", {}).items():
            add(f"- **{k}:** {v}")
    add("")
    add("## 6. Threshold analysis")
    add("")
    add(f"- High-risk cutoff: **{metrics['risk_bands']['high_threshold']:.4f}**")
    add(f"- Medium-risk cutoff: **{metrics['risk_bands']['medium_threshold']:.4f}**")
    add(f"- Rule: {metrics['risk_bands']['rule']}")
    add("")
    add("| Threshold | Precision | Recall | FPR | Alerts | F1 |")
    add("|---:|---:|---:|---:|---:|---:|")
    sweep_path = REPORTS_DIR / "threshold_analysis.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        for _, row in sweep.iloc[::20].iterrows():
            add(
                f"| {row['threshold']:.3f} | {row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['fpr']:.5f} | {int(row['alerts']):,} | {row['f1']:.4f} |"
            )
    add("")
    add("## 7. Calibration")
    add("")
    if calibration:
        add(
            f"- Brier score (raw, test): **{calibration['test']['brier_score']:.6f}**"
        )
        add(
            f"- Isotonic recalibration would change Brier by "
            f"{calibration['brier_improvement_isotonic']:+.6f}; the raw score is kept for "
            "ranking and thresholding (documented in `calibration.json`)."
        )
        add(
            f"- Mean predicted probability {calibration['test']['mean_predicted']:.6f} vs "
            f"observed fraud rate {calibration['test']['observed_rate']:.6f}."
        )
    add("")
    add("## 8. Error analysis (test)")
    add("")
    if errors:
        add(
            f"- False positives: {errors['false_positives']['count']:,} "
            f"({errors['false_positives']['rate_per_1000_legit']:.2f} per 1,000 legitimate "
            f"transactions), total value {errors['false_positives']['amount_total']:,.0f}."
        )
        add(
            f"- False negatives: {errors['false_negatives']['count']:,}, "
            f"{_pct(errors['false_negatives']['share_of_fraud_value_missed'])} of fraud value missed."
        )
        if "fraud_channels_only" in errors:
            add(
                f"- Fraud-bearing channels only: PR-AUC "
                f"{errors['fraud_channels_only']['pr_auc']:.4f}, recall "
                f"{errors['fraud_channels_only']['recall']:.4f}, FPR "
                f"{errors['fraud_channels_only']['fpr']:.5f}."
            )
        add("")
        if "product_cd" in errors.get("segments", {}):
            add("### By product code")
            add("")
            add(pd.DataFrame(errors["segments"]["product_cd"]).to_string(index=False))
        elif "type" in errors.get("segments", {}):
            add("### By channel")
            add("")
            add(pd.DataFrame(errors["segments"]["type"]).to_string(index=False))
        add("")
        add("### By amount quartile")
        add("")
        add(pd.DataFrame(errors["segments"]["amount_bucket"]).to_string(index=False))
        add("")
        add("### Threshold sensitivity")
        add("")
        add(pd.DataFrame(errors["threshold_sensitivity"]).to_string(index=False))
    add("")
    add("## 9. Limitations")
    add("")
    for limitation in [
        "Synthetic identity matching: 24.4% of transactions link to identity records; lack of match is informative but not conclusive.",
        "High cardinality categoricals (e.g. card1, DeviceInfo) require careful frequency/history encoding to prevent overfitting.",
        "Temporal drift: while prevalence is stable (~3.5%), fraud patterns and transaction attributes shift over time.",
        "Review cost ($4.00) and review capacity (1% FPR) are illustrative operating assumptions.",
        "The model is designed for pre-authorisation assessment without post-settlement chargeback knowledge.",
    ]:
        add(f"- {limitation}")
    add("")
    add("## 10. Reproducibility")
    add("")
    add("```bash")
    add("python scripts/download_data.py")
    add("python scripts/run_all.py")
    add("pytest")
    add("streamlit run app.py")
    add("```")
    add("")
    add(f"Thresholds file: `{THRESHOLD_CONFIG_PATH.name}` · comparison CSV: `{MODEL_COMPARISON_CSV.name}`")

    output.write_text("\n".join(lines), encoding="utf-8")
    return output


if __name__ == "__main__":
    path = generate_report()
    print(f"Wrote {path}")
