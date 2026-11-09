"""SHAP explainability.

Produces
--------
* global importance (mean absolute SHAP value)
* beeswarm summary plot
* dependence plots for the strongest features
* local waterfall explanations for representative transactions
* a compact artefact the dashboard can use to explain any sampled transaction

SHAP values are additive contributions in log-odds space. They explain the
model's output, not the causal effect of a feature on fraud. That caveat matters
doubly here because the Vesta V-block features are correlated by construction.
"""

from __future__ import annotations

import time

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    FIGURES_DIR,
    MODEL_PATH,
    PREDICTIONS_PARQUET,
    PROCESSED_PARQUET,
    RANDOM_STATE,
    SHAP_DIR,
)
from src.evaluation.plots import plot_feature_importance
from src.utils import ensure_dir, get_logger, save_json

logger = get_logger("shap")


def _sample_test_transactions(n: int, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Sample scored test transactions, keeping a share of frauds.

    All frauds are kept when the sample is large enough so the local explanations
    and the error narrative are not dominated by legitimate rows.
    """
    predictions = pd.read_parquet(
        PREDICTIONS_PARQUET,
        columns=["txn_id", "split", "fraud_probability", "isFraud", "product_cd", "amount"],
    )
    predictions = predictions[predictions["split"] == "test"]
    rng = np.random.default_rng(seed)

    fraud_ids = predictions.loc[predictions["isFraud"] == 1, "txn_id"].to_numpy()
    legit_ids = predictions.loc[predictions["isFraud"] == 0, "txn_id"].to_numpy()
    fraud_target = min(len(fraud_ids), n // 3)
    if len(fraud_ids) > fraud_target:
        fraud_ids = rng.choice(fraud_ids, size=fraud_target, replace=False)
    n_legit = max(n - len(fraud_ids), 1)
    if len(legit_ids) > n_legit:
        legit_ids = rng.choice(legit_ids, size=n_legit, replace=False)
    ids = np.concatenate([fraud_ids, legit_ids])
    return predictions[predictions["txn_id"].isin(ids)].reset_index(drop=True)


def _load_features(txn_ids: np.ndarray, feature_columns: list[str]) -> pd.DataFrame:
    import pyarrow.dataset as ds

    dataset = ds.dataset(PROCESSED_PARQUET, format="parquet")
    table = dataset.to_table(
        columns=["txn_id", *feature_columns],
        filter=ds.field("txn_id").isin(txn_ids.tolist()),
    )
    frame = table.to_pandas()
    return frame.set_index("txn_id").loc[txn_ids].reset_index()


def _explainer_for(model, background: pd.DataFrame):
    """Return (explainer, preprocessing callable or None)."""
    from sklearn.pipeline import Pipeline

    if isinstance(model, Pipeline):
        preprocess = Pipeline(model.steps[:-1])
        clf = model.named_steps["clf"]
        transformed = preprocess.transform(background)
        if hasattr(clf, "coef_"):
            return shap.LinearExplainer(clf, transformed), preprocess
        return shap.TreeExplainer(clf), preprocess

    kind = type(model).__name__
    if kind == "LGBMClassifier":
        return shap.TreeExplainer(model), None
    return shap.TreeExplainer(model), None


def _binary_shap_values(raw, explainer) -> tuple[np.ndarray, float]:
    if isinstance(raw, list):
        values = np.asarray(raw[1])
        expected = float(np.ravel(explainer.expected_value)[1])
    else:
        values = np.asarray(raw)
        expected = np.ravel(explainer.expected_value)
        expected = float(expected[1] if expected.size > 1 else expected[0])
        if values.ndim == 3:
            values = values[:, :, 1]
    return values, expected


def run_shap_analysis(
    model_path: str = MODEL_PATH,
    n_samples: int = 2000,
    seed: int = RANDOM_STATE,
) -> dict:
    ensure_dir(SHAP_DIR)
    ensure_dir(FIGURES_DIR)
    started = time.time()

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    logger.info("Sampling %s test transactions", f"{n_samples:,}")
    sample = _sample_test_transactions(n_samples, seed)
    features = _load_features(sample["txn_id"].to_numpy(), feature_columns)
    X = features[feature_columns]
    for column in feature_columns:
        if column.startswith("cat_"):
            X[column] = X[column].astype("int32")

    logger.info("Computing SHAP values")
    explainer, preprocess = _explainer_for(model, X.sample(n=min(200, len(X)), random_state=seed))
    X_explain = preprocess.transform(X) if preprocess is not None else X
    shap_values, base_value = _binary_shap_values(explainer.shap_values(X_explain), explainer)

    shap_values = np.asarray(shap_values, dtype="float32")
    if shap_values.shape[0] != len(X):
        raise RuntimeError(f"Unexpected SHAP shape {shap_values.shape} for {len(X)} rows")

    # ------------------------------------------------------------ global views
    importance = (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                "mean_shap": shap_values.mean(axis=0),
                "mean_value": np.nanmean(X.to_numpy(dtype="float64"), axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance.to_csv(SHAP_DIR / "global_importance.csv", index=False)
    plot_feature_importance(importance, FIGURES_DIR / "shap_importance.png", top_n=15)

    X_plot = pd.DataFrame(X_explain, columns=feature_columns)
    plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_plot, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    top_features = importance["feature"].head(4).tolist()
    for feature in top_features:
        plt.figure(figsize=(7, 4))
        shap.dependence_plot(feature, shap_values, X_plot, show=False, alpha=0.5)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_dependence_{feature}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------- local views
    sample = sample.set_index("txn_id").loc[features["txn_id"]].reset_index()
    probabilities = model.predict_proba(X)[:, 1]

    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(len(X), base_value),
        data=X.to_numpy(dtype="float64"),
        feature_names=feature_columns,
    )
    picks = {}
    fraud_idx = np.flatnonzero(sample["isFraud"].to_numpy() == 1)
    legit_idx = np.flatnonzero(sample["isFraud"].to_numpy() == 0)
    if len(fraud_idx):
        picks["top_fraud"] = int(fraud_idx[np.argmax(probabilities[fraud_idx])])
        if len(fraud_idx) > 1:
            picks["lowest_scored_fraud"] = int(fraud_idx[np.argmin(probabilities[fraud_idx])])
    if len(legit_idx):
        picks["top_legitimate_alert"] = int(legit_idx[np.argmax(probabilities[legit_idx])])

    for label, index in picks.items():
        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(explanation[index], max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------ dashboard artefacts
    np.save(SHAP_DIR / "shap_values.npy", shap_values)
    save_json(
        {
            "base_value": base_value,
            "feature_columns": feature_columns,
            "n_samples": int(len(X)),
            "top_features": top_features,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        SHAP_DIR / "meta.json",
    )

    dashboard_frame = features.copy()
    dashboard_frame["isFraud"] = sample["isFraud"].to_numpy()
    dashboard_frame["fraud_probability"] = probabilities
    dashboard_frame["product_cd"] = sample["product_cd"].to_numpy()
    dashboard_frame["amount_txn"] = sample["amount"].to_numpy()
    dashboard_frame["explanation_index"] = np.arange(len(X))
    dashboard_frame.to_parquet(SHAP_DIR / "sample.parquet", index=False)

    local = []
    for label, index in picks.items():
        contributions = (
            pd.DataFrame(
                {
                    "feature": feature_columns,
                    "value": X.iloc[index].to_numpy(),
                    "shap_value": shap_values[index],
                }
            )
            .assign(abs_value=lambda d: d["shap_value"].abs())
            .sort_values("abs_value", ascending=False)
            .head(10)
        )
        local.append(
            {
                "case": label,
                "txn_id": int(features["txn_id"].iloc[index]),
                "probability": float(probabilities[index]),
                "actual_fraud": int(sample["isFraud"].iloc[index]),
                "product_cd": str(sample["product_cd"].iloc[index]),
                "amount": float(sample["amount"].iloc[index]),
                "top_contributions": contributions.drop(columns="abs_value").to_dict(orient="records"),
            }
        )
    save_json(
        {
            "global_top_10": importance.head(10).to_dict(orient="records"),
            "local_explanations": local,
            "disclaimer": (
                "SHAP values describe how features move the model output relative to "
                "the average prediction. They are not causal effects. Several Vesta "
                "V-columns are correlated, so their attribution is shared arbitrarily."
            ),
        },
        SHAP_DIR / "explanation_summary.json",
    )

    logger.info("SHAP analysis finished in %.1fs", time.time() - started)
    return {"importance": importance, "sample": dashboard_frame, "shap_values": shap_values}


if __name__ == "__main__":
    run_shap_analysis()
