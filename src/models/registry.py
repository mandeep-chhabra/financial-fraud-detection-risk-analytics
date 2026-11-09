"""Model definitions.

Four supervised families are compared:

* Logistic Regression - linear, fully interpretable baseline.
* Random Forest       - bagged non-linear trees.
* LightGBM            - gradient-boosted trees with native categorical handling.
* XGBoost             - second boosted-tree implementation, as a robustness check
                        that the result is not an artefact of one library.

Isolation Forest is deliberately *not* used: anomaly detection answers "is this
unusual", not "is this fraud", and the task here is supervised classification.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


def build_logistic_regression(
    numeric_columns: list[str],
    categorical_columns: list[str],
    seed: int = 42,
    C: float = 0.3,
    class_weight: str | None = "balanced",
) -> Pipeline:
    """Median-imputed, standardised numeric block + one-hot low-cardinality block.

    Scaling matters because TransactionAmt, V-columns and day counters live on
    very different ranges and the L2 penalty is not scale invariant. The high
    cardinality categoricals are *not* one-hot encoded (card1 alone has 13,000
    levels); they enter through frequency and history features instead.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", fill_value=0, keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=50,
                    sparse_output=False,
                    dtype="float32",
                ),
            )
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_columns),
            ("cat", categorical_pipe, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight=class_weight,
                    max_iter=400,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def build_random_forest(
    seed: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = 24,
    min_samples_leaf: int = 10,
    class_weight: str | None = "balanced_subsample",
) -> Pipeline:
    """Bagged trees; trained on a stratified subsample for runtime reasons.

    scikit-learn forests cannot consume NaN, so a median imputer is part of the
    estimator. The boosting models handle missing values natively and do not get
    one - that difference is documented rather than hidden.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", fill_value=0, keep_empty_features=True)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    max_features="sqrt",
                    class_weight=class_weight,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_lightgbm(
    seed: int = 42,
    n_estimators: int = 1200,
    learning_rate: float = 0.04,
    num_leaves: int = 127,
    min_child_samples: int = 60,
    scale_pos_weight: float = 1.0,
    **overrides,
) -> LGBMClassifier:
    """Gradient boosting with histogram binning and early stopping support."""
    params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=-1,
        min_child_samples=min_child_samples,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.6,
        reg_lambda=5.0,
        reg_alpha=0.0,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)


def build_xgboost(
    seed: int = 42,
    n_estimators: int = 900,
    learning_rate: float = 0.04,
    max_depth: int = 8,
    min_child_weight: float = 5.0,
    scale_pos_weight: float = 1.0,
    **overrides,
) -> XGBClassifier:
    params = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=5.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )
    params.update(overrides)
    return XGBClassifier(**params)
