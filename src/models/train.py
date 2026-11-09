"""Training orchestration: model comparison, imbalance experiments, threshold
selection and held-out evaluation.

Split discipline
----------------
* train       days 1-128    -> model fitting only
* validation  days 128-156  -> imbalance strategy, model choice, operating
                               threshold and risk bands
* test        days 156-183  -> evaluated exactly once, at the end

No test row is used to choose a model, a hyper-parameter or a threshold.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from lightgbm import early_stopping, log_evaluation

from src.config import (
    CALIBRATION_JSON,
    CATEGORICAL_CODE_FEATURES,
    DEFAULT_INVESTIGATION_COST,
    DEFAULT_MAX_FPR,
    ERROR_ANALYSIS_JSON,
    FEATURE_ABLATION_JSON,
    FEATURE_COLUMNS,
    FIGURES_DIR,
    HISTORY_FEATURES,
    HIGH_CARD_CATEGORICALS,
    LOW_CARD_CATEGORICALS,
    MODEL_COMPARISON_CSV,
    MODEL_COMPARISON_JSON,
    MODEL_PATH,
    NO_IDENTITY_COLUMNS,
    PREDICTIONS_PARQUET,
    PRODUCTION_FEATURES,
    RANDOM_STATE,
    TEST_METRICS_JSON,
    THRESHOLD_ANALYSIS_CSV,
    THRESHOLD_CONFIG_PATH,
    TRAIN_END_DT,
    VAL_END_DT,
)
from src.evaluation.error_analysis import run_error_analysis
from src.evaluation.metrics import (
    calibrate_and_report,
    choose_risk_bands,
    compute_metrics,
    pr_curve_points,
    recall_at_fpr,
    roc_curve_points,
    select_operating_threshold,
    threshold_sweep,
)
from src.evaluation.plots import (
    plot_calibration,
    plot_confusion,
    plot_model_comparison,
    plot_pr_curve,
    plot_roc_curve,
    plot_threshold_curves,
)
from src.models.registry import (
    build_lightgbm,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
)
from src.preprocessing.splits import split_xy, stratified_subsample
from src.utils import ensure_dir, get_logger, save_json, set_seed

logger = get_logger("train")

META_COLUMNS = [
    "txn_id", "transaction_dt", "txn_hour", "day_index", "product_cd", "card4",
    "card6", "card1", "addr1", "p_emaildomain", "has_identity", "amount",
    "isFraud", "split",
]

LR_NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_CODE_FEATURES]
LR_CATEGORICAL_COLUMNS = [f"cat_{c}" for c in LOW_CARD_CATEGORICALS]
HIGH_CARD_CODE_COLUMNS = [f"cat_{c}" for c in HIGH_CARD_CATEGORICALS]


@dataclass
class Experiment:
    name: str
    imbalance_strategy: str
    train_rows: int
    fit_seconds: float
    validation: dict
    model: object = field(repr=False, default=None)
    params: dict = field(default_factory=dict)


def _score(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _filters(name: str) -> list[tuple]:
    if name == "train":
        return [("transaction_dt", "<=", TRAIN_END_DT)]
    if name == "validation":
        return [("transaction_dt", ">", TRAIN_END_DT), ("transaction_dt", "<=", VAL_END_DT)]
    if name == "test":
        return [("transaction_dt", ">", VAL_END_DT)]
    raise ValueError(f"Unknown split {name!r}")


def _load_xy(path, name: str):
    """Read features and labels with row-group filters, never a whole-frame copy."""
    filters = _filters(name)
    meta = pd.read_parquet(path, columns=META_COLUMNS, filters=filters)
    X = pd.read_parquet(path, columns=FEATURE_COLUMNS, filters=filters)
    y = meta["isFraud"].to_numpy(dtype="int8")
    for column in CATEGORICAL_CODE_FEATURES:
        X[column] = X[column].astype("int32")
    return X, y, meta


def _validation_metrics(name, scores, y_val, amount_val, seconds, train_rows, params) -> Experiment:
    metrics = compute_metrics(y_val, scores, 0.5, amount_val)
    at_budget = recall_at_fpr(y_val, scores, DEFAULT_MAX_FPR)
    metrics.update(
        {
            "model": name,
            "fit_seconds": round(seconds, 1),
            "train_rows": int(train_rows),
            "recall_at_1pct_fpr": at_budget["recall"],
            "threshold_at_1pct_fpr": at_budget["threshold"],
            "fpr_at_1pct_recall_threshold": at_budget["fpr"],
        }
    )
    return Experiment(
        name=name,
        imbalance_strategy=str(params.get("_strategy", "")),
        train_rows=int(train_rows),
        fit_seconds=seconds,
        validation=metrics,
        params={k: v for k, v in params.items() if k != "_strategy"},
    )


def _feature_set_ablation(X_train, y_train, X_val, y_val, amount_val, quick: bool = False) -> dict:
    """Quantify what identity data and behavioural history are worth.

    Validation only; the test period stays untouched. A 150k-row stratified
    sample is used so three extra fits stay cheap, and every set is fitted on the
    same rows so the comparison is like-for-like.
    """
    sample_size = 60_000 if quick else 150_000
    sample = stratified_subsample(
        pd.concat([X_train, pd.Series(y_train, name="isFraud")], axis=1), sample_size
    )
    y_sample = sample.pop("isFraud").to_numpy(dtype="int8")

    feature_sets = {
        "core_transaction": [
            c
            for c in NO_IDENTITY_COLUMNS
            if c not in set(HISTORY_FEATURES)
        ],
        "core_plus_history": NO_IDENTITY_COLUMNS,
        "full_with_identity": FEATURE_COLUMNS,
    }
    results = []
    for name, columns in feature_sets.items():
        logger.info("Ablation: %s (%d features)", name, len(columns))
        cat_cols = [c for c in columns if c in CATEGORICAL_CODE_FEATURES]
        model = build_lightgbm(n_estimators=150 if quick else 400, learning_rate=0.05)
        model.fit(
            sample[columns],
            y_sample,
            eval_set=[(X_val[columns], y_val)],
            eval_metric="average_precision",
            categorical_feature=cat_cols,
            callbacks=[early_stopping(40, verbose=False), log_evaluation(0)],
        )
        scores = model.predict_proba(X_val[columns])[:, 1]
        metrics = compute_metrics(y_val, scores, 0.5, amount_val)
        importance = (
            pd.Series(model.booster_.feature_importance("gain"), index=columns)
            .sort_values(ascending=False)
            .head(8)
        )
        results.append(
            {
                "feature_set": name,
                "n_features": len(columns),
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "fpr": metrics["fpr"],
                "alerts": metrics["alerts"],
                "recall_at_1pct_fpr": recall_at_fpr(y_val, scores, DEFAULT_MAX_FPR)["recall"],
                "top_gain_features": {k: float(v) for k, v in importance.items()},
            }
        )
        del model
        gc.collect()

    payload = {
        "evaluated_on": "validation period (test not used)",
        "train_sample_rows": int(len(sample)),
        "results": results,
        "interpretation": {
            "core_transaction": (
                "Amount, calendar, card/address, Vesta C/D/V blocks and identity "
                "numbers. This is what a single transaction record supports with no "
                "memory of the card's past."
            ),
            "core_plus_history": (
                "Adds past-only card and uid velocity/spend-deviation features. The "
                "gain is the value of behavioural history."
            ),
            "full_with_identity": (
                "Adds the device/identity block (only 24.4% of rows have it) and the "
                "identity-presence flags. The gap quantifies how much device data is "
                "worth in this dataset."
            ),
        },
    }
    save_json(payload, FEATURE_ABLATION_JSON)
    return payload


def _defaults_from_training(X_train: pd.DataFrame, n: int = 60_000) -> tuple[dict, dict]:
    """Population defaults used when scoring a transaction with partial input.

    Numeric defaults are training medians, categorical defaults the most frequent
    level. They are persisted with the model so the dashboard never invents
    values silently.
    """
    sample = X_train.sample(n=min(n, len(X_train)), random_state=RANDOM_STATE)
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_CODE_FEATURES]
    feature_defaults = {c: float(sample[c].median()) for c in numeric}
    for column in CATEGORICAL_CODE_FEATURES:
        mode = sample[column].mode(dropna=True)
        feature_defaults[column] = int(mode.iloc[0]) if len(mode) else -1

    # Raw-value defaults so the same feature engineering path can run for a
    # single record (see src/models/predict.py).
    raw_numeric = [
        "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1", "dist2",
        *[f"C{i}" for i in range(1, 15)],
        *[f"D{i}" for i in range(1, 16)],
        *[f"V{i}" for i in range(1, 340)],
        *[f"id_{i:02d}" for i in range(1, 12)], "id_13", "id_14",
        *[f"id_{i:02d}" for i in range(17, 23)], "id_24", "id_25", "id_26", "id_32",
    ]
    raw_defaults = {c: float(sample[c].median()) for c in raw_numeric if c in sample.columns}
    return raw_defaults, feature_defaults


def run_training(
    features_path,
    quick: bool = False,
    cost_per_investigation: float = DEFAULT_INVESTIGATION_COST,
) -> dict:
    """Execute the experiment matrix, select a model and evaluate on test."""
    set_seed(RANDOM_STATE)
    started = time.time()
    ensure_dir(FIGURES_DIR)

    logger.info("Loading splits from %s", features_path)
    X_train, y_train, _ = _load_xy(features_path, "train")
    X_val, y_val, val_meta = _load_xy(features_path, "validation")
    X_test, y_test, test_meta = _load_xy(features_path, "test")

    if int(val_meta["transaction_dt"].min()) <= TRAIN_END_DT:
        raise AssertionError("Train and validation periods overlap")
    if int(test_meta["transaction_dt"].min()) <= VAL_END_DT:
        raise AssertionError("Validation and test periods overlap")
    if X_train.shape[0] != len(y_train) or X_test.shape[0] != len(y_test):
        raise AssertionError("Feature and label rows are misaligned")

    logger.info(
        "Rows -> train %s | validation %s | test %s",
        f"{len(X_train):,}", f"{len(X_val):,}", f"{len(X_test):,}",
    )
    logger.info(
        "Fraud prevalence -> train %.4f%% | validation %.4f%% | test %.4f%%",
        100 * y_train.mean(), 100 * y_val.mean(), 100 * y_test.mean(),
    )

    amount_val = val_meta["amount"].to_numpy(dtype="float64")
    amount_test = test_meta["amount"].to_numpy(dtype="float64")
    cat_features = [c for c in FEATURE_COLUMNS if c in CATEGORICAL_CODE_FEATURES]

    experiments: list[Experiment] = []
    lgb_fit = {
        "eval_set": [(X_val, y_val)],
        "eval_metric": "average_precision",
        "categorical_feature": cat_features,
        "callbacks": [early_stopping(60, verbose=False), log_evaluation(0)],
    }

    def run(name, factory, X_fit, y_fit, notes, fit_kwargs=None, **factory_kwargs):
        logger.info("Training %s (%s rows)", name, f"{len(X_fit):,}")
        model = factory(**factory_kwargs)
        t0 = time.time()
        model.fit(X_fit, y_fit, **(fit_kwargs or {}))
        seconds = time.time() - t0
        scores = _score(model, X_val)
        exp = _validation_metrics(
            name, scores, y_val, amount_val, seconds, len(X_fit),
            {**factory_kwargs, "_strategy": notes},
        )
        exp.model = model
        experiments.append(exp)
        logger.info(
            "  %.1fs | val PR-AUC %.4f | recall@1%%FPR %.4f | precision %.4f | recall %.4f",
            seconds, exp.validation["pr_auc"], exp.validation["recall_at_1pct_fpr"],
            exp.validation["precision"], exp.validation["recall"],
        )
        return exp

    # ------------------------------------------------------------- baselines
    lr_rows = 60_000 if quick else 200_000
    lr_frame = stratified_subsample(
        pd.concat([X_train[LR_NUMERIC_COLUMNS + LR_CATEGORICAL_COLUMNS], pd.Series(y_train, name="isFraud")], axis=1),
        lr_rows,
    )
    lr_y = lr_frame.pop("isFraud").to_numpy(dtype="int8")
    run(
        "Logistic Regression | class_weight=balanced",
        lambda **kw: build_logistic_regression(
            numeric_columns=LR_NUMERIC_COLUMNS, categorical_columns=LR_CATEGORICAL_COLUMNS, **kw
        ),
        lr_frame, lr_y,
        f"class_weight=balanced, stratified {lr_rows:,}-row subsample, one-hot low-cardinality block",
    )
    del lr_frame, lr_y
    gc.collect()

    rf_rows = 30_000 if quick else 120_000
    rf_frame = stratified_subsample(
        pd.concat([X_train, pd.Series(y_train, name="isFraud")], axis=1), rf_rows
    )
    rf_y = rf_frame.pop("isFraud").to_numpy(dtype="int8")
    run(
        "Random Forest | balanced_subsample",
        build_random_forest,
        rf_frame, rf_y,
        f"stratified {rf_rows:,}-row subsample (runtime constraint), median imputation",
        n_estimators=80 if quick else 300,
    )
    del rf_frame, rf_y
    gc.collect()

    # ------------------------------------ LightGBM: imbalance strategy comparison
    # One common 120k-row stratified sample so the *only* thing that changes
    # between runs is the imbalance treatment.
    imb_rows = 40_000 if quick else 120_000
    sub = stratified_subsample(
        pd.concat([X_train, pd.Series(y_train, name="isFraud")], axis=1), imb_rows
    )
    y_sub = sub.pop("isFraud").to_numpy(dtype="int8")
    X_sub = sub
    n_fraud = int(y_sub.sum())
    logger.info(
        "Imbalance experiment sample: %s rows (%s frauds, %.2f%%)",
        f"{len(X_sub):,}", f"{n_fraud:,}", 100 * y_sub.mean(),
    )
    rounds = 200 if quick else 600

    run(
        "LightGBM | no resampling",
        build_lightgbm, X_sub, y_sub,
        "no imbalance treatment",
        fit_kwargs=lgb_fit, n_estimators=rounds,
    )

    pos_weight = float((y_sub == 0).sum() / max(n_fraud, 1))
    run(
        "LightGBM | class weight",
        build_lightgbm, X_sub, y_sub,
        f"scale_pos_weight={pos_weight:.1f}",
        fit_kwargs=lgb_fit, n_estimators=rounds, scale_pos_weight=pos_weight,
    )

    rng = np.random.default_rng(RANDOM_STATE)
    legit_idx = np.flatnonzero(y_sub == 0)
    keep_legit = rng.choice(legit_idx, size=min(n_fraud * 10, len(legit_idx)), replace=False)
    under_idx = np.concatenate([np.flatnonzero(y_sub == 1), keep_legit])
    run(
        "LightGBM | random undersampling",
        build_lightgbm, X_sub.iloc[under_idx], y_sub[under_idx],
        "10:1 legitimate:fraud undersampling of the training sample",
        fit_kwargs=lgb_fit, n_estimators=rounds,
    )

    ros = RandomOverSampler(sampling_strategy=0.1, random_state=RANDOM_STATE)
    X_ros, y_ros = ros.fit_resample(X_sub, y_sub)
    run(
        "LightGBM | random oversampling",
        build_lightgbm, X_ros, y_ros,
        "random oversampling to 10% fraud prevalence",
        fit_kwargs=lgb_fit, n_estimators=rounds,
    )
    del X_ros, y_ros
    gc.collect()

    smote = SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE, k_neighbors=5)
    X_sub_filled = X_sub.fillna(-999.0)
    X_sm, y_sm = smote.fit_resample(X_sub_filled, y_sub)
    if not isinstance(X_sm, pd.DataFrame):
        X_sm = pd.DataFrame(X_sm, columns=X_sub.columns)
    run(
        "LightGBM | SMOTE",
        build_lightgbm, X_sm, y_sm,
        "SMOTE to 10% fraud prevalence (synthetic frauds only, training data only)",
        fit_kwargs=lgb_fit, n_estimators=rounds,
    )
    del X_sm, y_sm, X_sub, y_sub
    gc.collect()

    # --------------------------------------------- full-period boosting models
    full_rounds = 200 if quick else 1200
    run(
        "LightGBM | full train, no weighting",
        build_lightgbm, X_train, y_train,
        f"full training period ({len(X_train):,} rows), no imbalance treatment",
        fit_kwargs=lgb_fit, n_estimators=full_rounds,
    )

    full_weight = float((y_train == 0).sum() / max(int(y_train.sum()), 1))
    run(
        "LightGBM | full train, class weight",
        build_lightgbm, X_train, y_train,
        f"full training period ({len(X_train):,} rows), scale_pos_weight={full_weight:.1f}",
        fit_kwargs=lgb_fit, n_estimators=full_rounds, scale_pos_weight=full_weight,
    )

    xgb_rows = 60_000 if quick else len(X_train)
    xgb_frame = X_train if xgb_rows >= len(X_train) else X_train.sample(n=xgb_rows, random_state=RANDOM_STATE)
    xgb_y = y_train if xgb_rows >= len(X_train) else np.asarray(y_train)[xgb_frame.index.to_numpy()]
    run(
        "XGBoost | full train, histogram",
        build_xgboost, xgb_frame, xgb_y,
        f"hist tree method, {xgb_rows:,} rows, no imbalance treatment",
        fit_kwargs={"eval_set": [(X_val, y_val)], "verbose": False},
        n_estimators=150 if quick else 900,
    )
    del xgb_frame
    gc.collect()

    # ------------------------------------------------- feature-set ablation
    ablation = _feature_set_ablation(X_train, y_train, X_val, y_val, amount_val, quick=quick)
    logger.info(
        "Ablation PR-AUC -> %s",
        {r["feature_set"]: round(r["pr_auc"], 4) for r in ablation["results"]},
    )

    # -------------------------------------------------------------- leaderboard
    leaderboard = (
        pd.DataFrame([e.validation for e in experiments])
        .sort_values(["recall_at_1pct_fpr", "pr_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1))
    leaderboard.to_csv(MODEL_COMPARISON_CSV, index=False)

    # Selection rule, business-oriented and tolerant to noise:
    #   1. keep models within 0.005 recall of the best at FPR <= 1%;
    #   2. keep those within 0.01 PR-AUC of that group's best;
    #   3. prefer the model trained on the most data (stability), then PR-AUC.
    best_recall = float(leaderboard["recall_at_1pct_fpr"].max())
    candidates = leaderboard[leaderboard["recall_at_1pct_fpr"] >= best_recall - 0.005]
    best_pr_auc = float(candidates["pr_auc"].max())
    near_best = candidates[candidates["pr_auc"] >= best_pr_auc - 0.01]
    selected_row = near_best.sort_values(["train_rows", "pr_auc"], ascending=False).iloc[0]
    best_name = selected_row["model"]
    best = next(e for e in experiments if e.name == best_name)
    selection_notes = {
        "rule": (
            "Within 0.005 recall of the best at FPR<=1%, keep models within 0.01 "
            "PR-AUC of that group's best, then prefer the largest training set "
            "(tie-break PR-AUC)."
        ),
        "best_recall_at_1pct_fpr": best_recall,
        "candidates": near_best[["model", "train_rows", "pr_auc", "recall_at_1pct_fpr"]].to_dict(
            orient="records"
        ),
        "why_not_plain_accuracy": (
            "Predicting 'legitimate' for every row already scores 96.5% accuracy, so "
            "accuracy cannot discriminate between these models."
        ),
        "why_tolerance": (
            "A 0.01 PR-AUC band is narrower than the sampling error of ~3,100 "
            "validation frauds; selecting on the fourth decimal would fit noise."
        ),
    }
    logger.info("Selected model: %s (train rows %s)", best_name, f"{best.train_rows:,}")

    # ------------------------------------------- operating point (validation only)
    val_scores = _score(best.model, X_val)
    thresholds = np.round(np.arange(0.01, 0.995, 0.005), 4)
    val_sweep = threshold_sweep(
        y_val, val_scores, thresholds=thresholds, amounts=amount_val,
        cost_per_investigation=cost_per_investigation,
    )
    operating = select_operating_threshold(
        val_sweep, cost_per_investigation=cost_per_investigation,
        max_fpr=DEFAULT_MAX_FPR, objective="f1",
    )
    bands = choose_risk_bands(
        val_sweep, operating["threshold"], medium_fpr=0.05, medium_alert_rate=0.05
    )
    val_sweep.to_csv(THRESHOLD_ANALYSIS_CSV, index=False)

    # ------------------------------------------------------------------ test set
    test_scores = _score(best.model, X_test)
    test_operating = compute_metrics(
        y_test, test_scores, operating["threshold"], amount_test, cost_per_investigation
    )
    test_default = compute_metrics(y_test, test_scores, 0.5, amount_test, cost_per_investigation)
    budget_test = recall_at_fpr(y_test, test_scores, DEFAULT_MAX_FPR)
    test_sweep = threshold_sweep(
        y_test, test_scores, thresholds=thresholds, amounts=amount_test,
        cost_per_investigation=cost_per_investigation,
    )
    matched = test_sweep.iloc[int(np.argmin(np.abs(test_sweep["fpr"] - budget_test["fpr"])))]

    test_metrics = {
        "model": best.name,
        "operating_threshold": operating["threshold"],
        "operating_policy": operating["objective"],
        "operating_constraint_relaxed": operating["constraint_relaxed"],
        "selection": selection_notes,
        "risk_bands": bands,
        "splits": {
            "train_rows": int(best.train_rows),
            "train_rows_available": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
        },
        "validation": {
            "at_operating_threshold": operating["validation"],
            "pr_auc": best.validation["pr_auc"],
            "roc_auc": best.validation["roc_auc"],
            "recall_at_1pct_fpr": best.validation["recall_at_1pct_fpr"],
        },
        "test_at_operating_threshold": test_operating,
        "test_at_0.5": test_default,
        "test_recall_at_1pct_fpr": budget_test,
        "test_metrics_at_matched_fpr": matched.to_dict(),
        "interpretation": (
            "Test metrics are reported once, at the threshold chosen on validation. "
            "They are the only performance numbers used in the resume bullets and docs."
        ),
    }
    save_json(test_metrics, TEST_METRICS_JSON)

    raw_defaults, feature_defaults = _defaults_from_training(X_train)
    model_bundle = {
        "model": best.model,
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": cat_features,
        "feature_set": "full (transaction + identity + past-only history)",
        "thresholds": bands,
        "operating_threshold": operating["threshold"],
        "cost_per_investigation": cost_per_investigation,
        "model_name": best.name,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_rows": int(best.train_rows),
        "raw_defaults": raw_defaults,
        "feature_defaults": feature_defaults,
    }
    ensure_dir(MODEL_PATH.parent)
    joblib.dump(model_bundle, MODEL_PATH)
    save_json(
        {
            **bands,
            "operating_threshold": operating["threshold"],
            "cost_per_investigation": cost_per_investigation,
            "max_fpr": DEFAULT_MAX_FPR,
            "selected_model": best.name,
            "feature_set": model_bundle["feature_set"],
            "selection_rule": selection_notes["rule"],
            "validation_metrics": operating["validation"],
            "test_metrics": test_operating,
        },
        THRESHOLD_CONFIG_PATH,
    )

    # -------------------------------------------------------------- predictions
    def _pred_frame(meta, scores, split):
        return pd.DataFrame(
            {
                "txn_id": meta["txn_id"].to_numpy(),
                "transaction_dt": meta["transaction_dt"].to_numpy(),
                "txn_hour": meta["txn_hour"].to_numpy(),
                "product_cd": meta["product_cd"].astype(str).to_numpy(),
                "amount": meta["amount"].to_numpy(dtype="float64"),
                "isFraud": meta["isFraud"].to_numpy(dtype="int8"),
                "fraud_probability": scores,
                "split": split,
            }
        )

    predictions = pd.concat(
        [_pred_frame(test_meta, test_scores, "test"), _pred_frame(val_meta, val_scores, "validation")],
        ignore_index=True,
    )
    predictions["risk_band"] = np.select(
        [
            predictions["fraud_probability"] >= bands["high_threshold"],
            predictions["fraud_probability"] >= bands["medium_threshold"],
        ],
        ["High Risk", "Medium Risk"],
        default="Low Risk",
    )
    predictions["alerted"] = (predictions["fraud_probability"] >= operating["threshold"]).astype("int8")
    ensure_dir(PREDICTIONS_PARQUET.parent)
    predictions.to_parquet(PREDICTIONS_PARQUET, index=False)

    # ----------------------------------------------------- calibration + errors
    calibration = calibrate_and_report(y_val, val_scores, y_test, test_scores)
    save_json(calibration, CALIBRATION_JSON)

    error_analysis = run_error_analysis(
        y_true=y_test,
        scores=test_scores,
        amounts=amount_test,
        types=test_meta["product_cd"].astype(str).to_numpy(),
        days=test_meta["day_index"].to_numpy(),
        threshold=operating["threshold"],
        medium_threshold=bands["medium_threshold"],
        identity=test_meta["has_identity"].to_numpy(),
        segment_column_name="product_cd",
    )
    save_json(error_analysis, ERROR_ANALYSIS_JSON)

    # -------------------------------------------------------------------- plots
    plot_confusion(test_operating, FIGURES_DIR / "confusion_matrix_test.png")
    plot_pr_curve(pr_curve_points(y_test, test_scores), FIGURES_DIR / "pr_curve_test.png")
    plot_roc_curve(roc_curve_points(y_test, test_scores), FIGURES_DIR / "roc_curve_test.png")
    plot_threshold_curves(test_sweep, operating["threshold"], FIGURES_DIR / "threshold_curves_test.png")
    plot_calibration(calibration["test"], FIGURES_DIR / "calibration_test.png")
    plot_model_comparison(leaderboard, FIGURES_DIR / "model_comparison.png")

    save_json(
        {
            "experiments": [
                {
                    "model": e.name,
                    "imbalance_strategy": e.imbalance_strategy,
                    "train_rows": e.train_rows,
                    "fit_seconds": round(e.fit_seconds, 1),
                    "params": e.params,
                    "validation": {k: v for k, v in e.validation.items() if k != "model"},
                }
                for e in experiments
            ],
            "selection_rule": selection_notes["rule"],
            "selection_notes": selection_notes,
            "selected_model": best_name,
            "runtime_seconds": round(time.time() - started, 1),
            "cost_per_investigation": cost_per_investigation,
            "quick_mode": quick,
        },
        MODEL_COMPARISON_JSON,
    )

    logger.info(
        "Test @%.3f | precision %.4f | recall %.4f | PR-AUC %.4f | value captured %.2f%%",
        operating["threshold"], test_operating["precision"], test_operating["recall"],
        test_operating["pr_auc"], 100 * test_operating["fraud_value_captured_pct"],
    )
    logger.info("Artifacts written in %.1fs", time.time() - started)

    return {"leaderboard": leaderboard, "test_metrics": test_metrics, "predictions": predictions}


if __name__ == "__main__":
    from src.config import PROCESSED_PARQUET

    run_training(PROCESSED_PARQUET)
