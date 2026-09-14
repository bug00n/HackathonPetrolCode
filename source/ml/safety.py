"""Safety-first sulfur forecasting helpers.

The point forecast, conservative upper estimate and exceedance alarm are kept
separate deliberately: none of them is allowed to masquerade as an action
effect model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from source.data.prepare import PreparedData
from source.ml.artifacts import ModelBundle, save_model
from source.ml.diagnostics import exclude_confirmed_lims_outliers
from source.ml.evaluate import expanding_purged_folds, make_outer_split
from source.ml.features import SupervisedDataset
from source.ml.uncertainty import ApplicabilityResult

SULFUR_LIMIT = 10.0
POSITIVE_WEIGHTS = (1.0, 2.0, 5.0, 10.0, 20.0)
FALSE_ALARM_BUDGET = 0.20
TARGET_FALSE_NEGATIVE_RATE = 0.10


@dataclass(frozen=True)
class AlarmPolicy:
    """Validation-selected operating point for a calibrated risk score."""

    threshold: float
    false_alarm_budget: float
    target_false_negative_rate: float


@dataclass(frozen=True)
class JointApplicabilityModel:
    """Train-only joint feature domain in whitened PCA space."""

    feature_names: tuple[str, ...]
    required_features: tuple[str, ...]
    imputer: SimpleImputer
    scaler: StandardScaler
    pca: PCA
    max_distance_squared: float

    def availability_mask(self, features: pd.DataFrame) -> np.ndarray:
        """Vectorized applicability result for offline evaluation."""
        if tuple(str(name) for name in features.columns) != self.feature_names:
            raise ValueError("applicability feature order mismatch")
        required_available = features.loc[:, list(self.required_features)].notna().all(axis=1)
        transformed = self.pca.transform(self.scaler.transform(self.imputer.transform(features)))
        distances = np.square(transformed).sum(axis=1)
        return cast(
            np.ndarray,
            required_available.to_numpy()
            & np.isfinite(distances)
            & (distances <= self.max_distance_squared),
        )

    def assess(self, features: pd.DataFrame) -> ApplicabilityResult:
        """Reject missing required inputs and flag joint distribution drift."""
        if tuple(str(name) for name in features.columns) != self.feature_names:
            raise ValueError("applicability feature order mismatch")
        if len(features) != 1:
            raise ValueError("applicability check requires exactly one feature row")
        missing = tuple(name for name in self.required_features if pd.isna(features.iloc[0][name]))
        if missing:
            return ApplicabilityResult(False, "FEATURES_UNAVAILABLE", missing)
        if not self.availability_mask(features)[0]:
            return ApplicabilityResult(False, "OUT_OF_DOMAIN", ("joint_pca_distance",))
        return ApplicabilityResult(True, None)


@dataclass(frozen=True)
class PlattCalibrator:
    """Small serializable sigmoid calibration layer."""

    estimator: LogisticRegression

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
        return cast(np.ndarray, self.estimator.predict_proba(logits)[:, 1])


@dataclass(frozen=True)
class CompositeSafetyPredictor:
    """One persisted predictor exposing point, upper, risk and applicability."""

    point_predictor: Any
    upper_predictor: Any
    risk_predictor: Any
    risk_calibrator: PlattCalibrator
    alarm_policy: AlarmPolicy
    applicability: JointApplicabilityModel

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.point_predictor.predict(features), dtype=float)

    def predict_upper(self, features: pd.DataFrame) -> np.ndarray:
        point = self.predict(features)
        upper = np.asarray(self.upper_predictor.predict_upper(features), dtype=float)
        return np.maximum(point, upper)

    def predict_exceedance_probability(self, features: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.risk_predictor.predict_proba(features)[:, 1], dtype=float)
        return self.risk_calibrator.predict(raw)

    def predict_alarm(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_exceedance_probability(features) >= self.alarm_policy.threshold

    def check_applicability(self, features: pd.DataFrame) -> ApplicabilityResult:
        return self.applicability.assess(features)


@dataclass(frozen=True)
class SafetyFitResult:
    predictor: CompositeSafetyPredictor
    report: dict[str, Any]


@dataclass(frozen=True)
class LimsCorrectionResult:
    """Sparse PAK-to-LIMS correction kept separate from the operational target."""

    model_name: str
    point_estimator: Any
    upper_estimator: Any
    report: dict[str, Any]


def published_lims_features(
    data: PreparedData, as_of: pd.Series, *, depth: int = 3
) -> pd.DataFrame:
    """Return only LIMS values whose publication time is not later than each decision."""
    if depth < 1:
        raise ValueError("LIMS history depth must be positive")
    quality = data.quality
    values = pd.to_numeric(quality["value"], errors="coerce")
    source = quality["source"].map(lambda value: str(getattr(value, "value", value)))
    validity = quality["validity"].map(lambda value: str(getattr(value, "value", value)))
    rows = quality[
        quality["signal_id"].astype(str).eq("ht:2:Mg.Sulfur")
        & source.eq("lims")
        & validity.eq("valid")
        & quality["unit"].astype(str).str.casefold().eq("mg/kg")
        & values.notna()
        & np.isfinite(values)
    ].copy()
    rows["value"] = values.loc[rows.index].astype(float)
    rows, _ = exclude_confirmed_lims_outliers(rows)
    rows["available_at"] = pd.to_datetime(rows["available_at"], utc=True)
    rows["measured_at"] = pd.to_datetime(rows["measured_at"], utc=True)
    rows.sort_values(["available_at", "measured_at", "observation_id"], inplace=True)
    history = rows.loc[:, ["available_at"]].copy()
    for position in range(depth):
        history[f"lims_value_{position}"] = rows["value"].shift(position)
        history[f"lims_measured_at_{position}"] = rows["measured_at"].shift(position)
    queries = pd.DataFrame(
        {"position": np.arange(len(as_of)), "as_of": pd.to_datetime(as_of, utc=True)}
    ).sort_values("as_of", kind="stable")
    merged = pd.merge_asof(
        queries,
        history,
        left_on="as_of",
        right_on="available_at",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("position", kind="stable")
    result: dict[str, np.ndarray] = {}
    for position in range(depth):
        result[f"lims_published_lag_{position}"] = merged[f"lims_value_{position}"].to_numpy(
            dtype=float
        )
        result[f"lims_age_minutes_{position}"] = (
            (merged["as_of"] - merged[f"lims_measured_at_{position}"])
            .dt.total_seconds()
            .div(60)
            .to_numpy(dtype=float)
        )
    return pd.DataFrame(result)


def binary_alarm_metrics(
    y_true: Sequence[bool] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    """Return explicit safety and calibration metrics for one alarm threshold."""
    actual = np.asarray(y_true, dtype=bool)
    probability = np.asarray(probabilities, dtype=float)
    if actual.ndim != 1 or probability.shape != actual.shape or not np.isfinite(probability).all():
        raise ValueError("alarm inputs must be equal-length finite vectors")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("alarm threshold must be between zero and one")
    alarm = probability >= threshold
    positives = int(actual.sum())
    negatives = int((~actual).sum())
    false_negatives = int((actual & ~alarm).sum())
    false_positives = int((~actual & alarm).sum())
    true_positives = int((actual & alarm).sum())
    return {
        "n": int(len(actual)),
        "positives": positives,
        "false_negatives": false_negatives,
        "false_negative_rate": false_negatives / positives if positives else None,
        "false_positives": false_positives,
        "false_positive_rate": false_positives / negatives if negatives else None,
        "precision": true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else None,
        "average_precision": float(average_precision_score(actual, probability))
        if positives and negatives
        else None,
        "brier": float(brier_score_loss(actual, probability)),
    }


def select_alarm_threshold(
    y_true: Sequence[bool] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    false_alarm_budget: float = FALSE_ALARM_BUDGET,
) -> tuple[AlarmPolicy, dict[str, float | int | None]]:
    """Minimize misses subject to the configured false-alarm budget."""
    if not 0.0 <= false_alarm_budget < 1.0:
        raise ValueError("false_alarm_budget must be in [0, 1)")
    actual = np.asarray(y_true, dtype=bool)
    probability = np.asarray(probabilities, dtype=float)
    if actual.ndim != 1 or probability.shape != actual.shape or not np.isfinite(probability).all():
        raise ValueError("alarm inputs must be equal-length finite vectors")
    positives = int(actual.sum())
    negatives = int((~actual).sum())
    if not positives or not negatives:
        raise ValueError("threshold selection needs both sulfur classes")
    order = np.argsort(-probability, kind="stable")
    sorted_probability = probability[order]
    sorted_actual = actual[order]
    cumulative_true = np.cumsum(sorted_actual)
    cumulative_false = np.cumsum(~sorted_actual)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_probability) != 0), len(actual) - 1]
    brier = float(brier_score_loss(actual, probability))
    average_precision = float(average_precision_score(actual, probability))
    feasible: list[tuple[float, dict[str, float | int | None]]] = []
    for index in group_ends:
        true_positives = int(cumulative_true[index])
        false_positives = int(cumulative_false[index])
        false_positive_rate = false_positives / negatives
        if false_positive_rate <= false_alarm_budget + 1e-12:
            feasible.append(
                (
                    float(sorted_probability[index]),
                    {
                        "n": int(len(actual)),
                        "positives": positives,
                        "false_negatives": positives - true_positives,
                        "false_negative_rate": (positives - true_positives) / positives,
                        "false_positives": false_positives,
                        "false_positive_rate": false_positive_rate,
                        "precision": true_positives / (true_positives + false_positives),
                        "average_precision": average_precision,
                        "brier": brier,
                    },
                )
            )
    if not feasible:
        raise ValueError("no alarm threshold satisfies the false-alarm budget")

    def ordering(item: tuple[float, Mapping[str, float | int | None]]) -> tuple[float, float]:
        threshold, metrics = item
        fnr = metrics["false_negative_rate"]
        return (float("inf") if fnr is None else float(fnr), -threshold)

    threshold, metrics = min(feasible, key=ordering)
    return AlarmPolicy(threshold, false_alarm_budget, TARGET_FALSE_NEGATIVE_RATE), metrics


def fit_joint_applicability(
    features: pd.DataFrame,
    *,
    required_features: Sequence[str],
    coverage: float = 0.99,
) -> JointApplicabilityModel:
    """Fit a compact joint domain without compounding 54 marginal cutoffs."""
    if not 0.5 < coverage < 1.0:
        raise ValueError("coverage must be between 0.5 and 1")
    names = tuple(str(name) for name in features.columns)
    required = tuple(str(name) for name in required_features)
    if not set(required).issubset(names):
        raise ValueError("required applicability features are absent")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(features)
    scaled = scaler.fit_transform(imputed)
    pca = PCA(n_components=0.99, whiten=True, svd_solver="full").fit(scaled)
    transformed = pca.transform(scaled)
    distances = np.square(transformed).sum(axis=1)
    return JointApplicabilityModel(
        names,
        required,
        imputer,
        scaler,
        pca,
        float(np.quantile(distances, coverage)),
    )


def _risk_estimator(family: str, seed: int) -> BaseEstimator:
    if family == "logistic":
        return Pipeline(
            (
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=200, random_state=seed, solver="liblinear"),
                ),
            )
        )
    if family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    if family == "lightgbm":
        return Pipeline(
            (
                ("to_array", FunctionTransformer(np.asarray, validate=False)),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        learning_rate=0.05,
                        n_estimators=200,
                        num_leaves=15,
                        reg_lambda=1.0,
                        random_state=seed,
                        n_jobs=1,
                        verbosity=-1,
                    ),
                ),
            )
        )
    raise ValueError(f"unknown risk family: {family}")


def _fit_risk(
    family: str,
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    positive_weight: float,
    seed: int,
) -> BaseEstimator:
    target = pd.to_numeric(frame["y"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(target) & frame.loc[:, list(feature_names)].notna().any(axis=1).to_numpy()
    labels = target[valid] > SULFUR_LIMIT
    if len(np.unique(labels)) != 2:
        raise ValueError("risk training needs both sulfur classes")
    weights = np.where(labels, positive_weight, 1.0)
    if family == "lightgbm":
        weights *= np.where((target[valid] >= 8.0) & (target[valid] <= 12.0), 2.0, 1.0)
    estimator = _risk_estimator(family, seed)
    estimator.fit(
        frame.loc[valid, list(feature_names)], labels, **_sample_weight_argument(family, weights)
    )
    return estimator


def _sample_weight_argument(family: str, weights: np.ndarray) -> dict[str, np.ndarray]:
    if family in {"logistic", "lightgbm"}:
        return {"model__sample_weight": weights}
    return {"sample_weight": weights}


def _fit_calibrator(probabilities: np.ndarray, labels: np.ndarray, seed: int) -> PlattCalibrator:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    estimator = LogisticRegression(random_state=seed).fit(logits, labels)
    return PlattCalibrator(estimator)


def fit_safety_model(
    dataset: SupervisedDataset,
    point_upper_model: ModelBundle,
    *,
    source_timezone: str = "Europe/Moscow",
    train_end: str = "2025-01-01",
    validation_end: str = "2026-01-01",
    false_alarm_budget: float = FALSE_ALARM_BUDGET,
    seed: int = 42,
    include_lightgbm: bool = False,
    candidate_families: Sequence[str] | None = None,
) -> SafetyFitResult:
    """Fit risk/OOD heads without reading test labels during selection."""
    frame = dataset.frame.reset_index(drop=True)
    feature_names = tuple(dataset.feature_names)
    split = make_outer_split(
        frame,
        source_timezone=source_timezone,
        train_end=train_end,
        validation_end=validation_end,
    )
    train = frame.iloc[split.train].reset_index(drop=True)
    validation = frame.iloc[split.validation].reset_index(drop=True)
    test = frame.iloc[split.test].reset_index(drop=True)
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("safety fitting needs non-empty train, validation and test")

    folds = expanding_purged_folds(train, n_splits=3)

    def score_candidate(family: str, positive_weight: float) -> dict[str, Any]:
        actual_parts: list[np.ndarray] = []
        probability_parts: list[np.ndarray] = []
        for fit_indices, score_indices in folds:
            estimator = _fit_risk(
                family,
                train.iloc[fit_indices],
                feature_names,
                positive_weight,
                seed,
            )
            score = train.iloc[score_indices]
            actual_parts.append(
                pd.to_numeric(score["y"], errors="coerce").to_numpy(dtype=float) > SULFUR_LIMIT
            )
            probability_parts.append(
                np.asarray(estimator.predict_proba(score.loc[:, list(feature_names)])[:, 1])
            )
        actual = np.concatenate(actual_parts)
        probability = np.concatenate(probability_parts)
        policy, metrics = select_alarm_threshold(
            actual, probability, false_alarm_budget=false_alarm_budget
        )
        return {
            "family": family,
            "positive_weight": positive_weight,
            "threshold": policy.threshold,
            "metrics": metrics,
        }

    families = tuple(candidate_families or ("logistic", "hist_gradient_boosting"))
    if include_lightgbm and "lightgbm" not in families:
        families += ("lightgbm",)
    unknown = set(families) - {
        "logistic",
        "hist_gradient_boosting",
        "lightgbm",
    }
    if unknown:
        raise ValueError(f"unknown risk families: {sorted(unknown)}")
    candidates = Parallel(n_jobs=2, prefer="threads")(
        delayed(score_candidate)(family, positive_weight)
        for family in families
        for positive_weight in POSITIVE_WEIGHTS
    )

    complexity = {
        "logistic": 0,
        "hist_gradient_boosting": 1,
        "lightgbm": 2,
    }
    selected = min(
        candidates,
        key=lambda item: (
            float(item["metrics"]["false_negative_rate"]),
            float(item["metrics"]["brier"]),
            complexity[item["family"]],
            item["positive_weight"],
        ),
    )
    risk = _fit_risk(
        selected["family"],
        train,
        feature_names,
        selected["positive_weight"],
        seed,
    )
    middle = len(validation) // 2
    calibration = validation.iloc[:middle]
    policy_frame = validation.iloc[middle:]
    calibration_labels = (
        pd.to_numeric(calibration["y"], errors="coerce").to_numpy(dtype=float) > SULFUR_LIMIT
    )
    raw_calibration = np.asarray(
        risk.predict_proba(calibration.loc[:, list(feature_names)])[:, 1], dtype=float
    )
    calibrator = _fit_calibrator(raw_calibration, calibration_labels, seed)
    policy_labels = (
        pd.to_numeric(policy_frame["y"], errors="coerce").to_numpy(dtype=float) > SULFUR_LIMIT
    )
    raw_policy = np.asarray(
        risk.predict_proba(policy_frame.loc[:, list(feature_names)])[:, 1], dtype=float
    )
    policy, validation_metrics = select_alarm_threshold(
        policy_labels,
        calibrator.predict(raw_policy),
        false_alarm_budget=false_alarm_budget,
    )
    required = (str(dataset.baseline_feature),)
    applicability = fit_joint_applicability(
        train.loc[:, list(feature_names)], required_features=required
    )
    predictor = CompositeSafetyPredictor(
        point_predictor=point_upper_model.predictor,
        upper_predictor=point_upper_model.predictor,
        risk_predictor=risk,
        risk_calibrator=calibrator,
        alarm_policy=policy,
        applicability=applicability,
    )

    test_features = test.loc[:, list(feature_names)]
    test_labels = pd.to_numeric(test["y"], errors="coerce").to_numpy(dtype=float) > SULFUR_LIMIT
    test_probability = predictor.predict_exceedance_probability(test_features)
    test_metrics = binary_alarm_metrics(test_labels, test_probability, policy.threshold)
    baseline = pd.to_numeric(test[required[0]], errors="coerce").to_numpy(dtype=float)
    transition = (baseline <= SULFUR_LIMIT) & test_labels
    transition_recall = (
        float(np.mean(predictor.predict_alarm(test_features)[transition]))
        if transition.any()
        else None
    )
    in_domain = applicability.availability_mask(test_features)
    promoted = (
        validation_metrics["false_negative_rate"] is not None
        and float(validation_metrics["false_negative_rate"]) <= TARGET_FALSE_NEGATIVE_RATE
        and validation_metrics["false_positive_rate"] is not None
        and float(validation_metrics["false_positive_rate"]) <= false_alarm_budget
    )
    report = {
        "schema_version": "1.1",
        "selected_family": selected["family"],
        "positive_weight": selected["positive_weight"],
        "near_threshold_weight": 2.0 if selected["family"] == "lightgbm" else 1.0,
        "alarm_threshold": policy.threshold,
        "false_alarm_budget": false_alarm_budget,
        "target_false_negative_rate": TARGET_FALSE_NEGATIVE_RATE,
        "validation_policy": validation_metrics,
        "test": test_metrics,
        "test_transition_recall": transition_recall,
        "test_applicability_rate": float(in_domain.mean()),
        "promotion_eligible": promoted,
        "test_used_for_selection": False,
        "candidates": candidates,
    }
    return SafetyFitResult(predictor, report)


def fit_lims_correction(
    dataset: SupervisedDataset,
    pak_model: ModelBundle,
    *,
    data: PreparedData | None = None,
    source_timezone: str = "Europe/Moscow",
    train_end: str = "2025-01-01",
    validation_end: str = "2026-01-01",
    seed: int = 42,
) -> LimsCorrectionResult:
    """Fit a sparse, explicitly LIMS-targeted residual correction model."""
    frame = dataset.frame.reset_index(drop=True).copy()
    frame, excluded_outliers = exclude_confirmed_lims_outliers(frame)
    feature_names = tuple(dataset.feature_names)
    original_features = frame.loc[:, list(feature_names)]
    base = np.asarray(pak_model.predict(original_features), dtype=float)
    frame["pak_point"] = base
    pak_feature_names = ["pak_point"]
    if pak_model.metadata.capabilities.supports_uncertainty:
        frame["pak_upper"] = pak_model.predict_upper(original_features)
        pak_feature_names.append("pak_upper")
    if pak_model.metadata.capabilities.supports_exceedance_probability:
        frame["pak_exceedance_probability"] = pak_model.predict_exceedance_probability(
            original_features
        )
        pak_feature_names.append("pak_exceedance_probability")
    lims_feature_names: tuple[str, ...] = ()
    if data is not None:
        lims_features = published_lims_features(data, frame["as_of"])
        frame = pd.concat([frame.reset_index(drop=True), lims_features], axis="columns")
        lims_feature_names = tuple(lims_features.columns)
    frame["residual_target"] = pd.to_numeric(frame["y"], errors="coerce") - base
    split = make_outer_split(
        frame,
        source_timezone=source_timezone,
        train_end=train_end,
        validation_end=validation_end,
    )
    names = (
        *feature_names,
        *pak_feature_names,
        *lims_feature_names,
    )
    train = frame.iloc[split.train]
    validation = frame.iloc[split.validation]
    test = frame.iloc[split.test]
    candidates: dict[str, Any] = {
        "ridge": Pipeline(
            (
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            )
        ),
        "huber": Pipeline(
            (
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", HuberRegressor(max_iter=2000)),
            )
        ),
    }
    y_train = train["residual_target"].to_numpy(dtype=float)
    y_validation = validation["residual_target"].to_numpy(dtype=float)
    fitted: dict[str, Any] = {}
    validation_mae: dict[str, float] = {}
    for name, estimator in candidates.items():
        estimator.fit(train.loc[:, list(names)], y_train)
        fitted[name] = estimator
        prediction = estimator.predict(validation.loc[:, list(names)])
        validation_mae[name] = float(np.mean(np.abs(y_validation - prediction)))
    selected_name = min(validation_mae, key=lambda name: (validation_mae[name], name))
    point = fitted[selected_name]
    upper = GradientBoostingRegressor(
        loss="quantile", alpha=0.95, n_estimators=100, max_depth=2, random_state=seed
    ).fit(train.loc[:, list(names)].fillna(train.loc[:, list(names)].median()), y_train)
    x_test = test.loc[:, list(names)]
    median = train.loc[:, list(names)].median()
    corrected = test["pak_point"].to_numpy(dtype=float) + point.predict(x_test)
    upper_prediction = test["pak_point"].to_numpy(dtype=float) + upper.predict(
        x_test.fillna(median)
    )
    actual = pd.to_numeric(test["y"], errors="coerce").to_numpy(dtype=float)
    report: dict[str, Any] = {
        "schema_version": "1.1",
        "target_source": "lims",
        "confirmed_outlier_policy": "exclude_by_observation_id",
        "excluded_outlier_count": len(excluded_outliers),
        "excluded_outlier_ids": excluded_outliers,
        "selected_model": selected_name,
        "published_lims_feature_count": len(lims_feature_names),
        "validation_mae": validation_mae,
        "test_rows": int(len(test)),
        "pak_point_mae": float(np.mean(np.abs(actual - test["pak_point"].to_numpy(dtype=float)))),
        "corrected_mae": float(np.mean(np.abs(actual - corrected))),
        "upper_coverage": float(np.mean(actual <= np.maximum(corrected, upper_prediction))),
        "test_used_for_selection": False,
    }
    report["promotion_eligible"] = (
        report["corrected_mae"] <= 0.95 * report["pak_point_mae"]
        and report["upper_coverage"] >= 0.95
    )
    return LimsCorrectionResult(selected_name, point, upper, report)


def save_safety_model(
    directory: Path,
    dataset: SupervisedDataset,
    point_upper_model: ModelBundle,
    *,
    source_timezone: str = "Europe/Moscow",
    train_end: str = "2025-01-01",
    validation_end: str = "2026-01-01",
    false_alarm_budget: float = FALSE_ALARM_BUDGET,
    seed: int = 42,
    allow_unpromoted: bool = False,
    include_lightgbm: bool = False,
) -> tuple[ModelBundle, SafetyFitResult]:
    """Persist schema-1.1 safety heads after the validation promotion gate."""
    if not point_upper_model.metadata.capabilities.supports_uncertainty:
        raise ValueError("safety training requires an uncertainty-capable point artifact")
    fitted = fit_safety_model(
        dataset,
        point_upper_model,
        source_timezone=source_timezone,
        train_end=train_end,
        validation_end=validation_end,
        false_alarm_budget=false_alarm_budget,
        seed=seed,
        include_lightgbm=include_lightgbm,
    )
    if not fitted.report["promotion_eligible"] and not allow_unpromoted:
        raise ValueError("safety model failed the validation promotion gate")
    metadata = point_upper_model.metadata.model_dump(mode="python")
    metadata.update(
        {
            "model_id": directory.name,
            "schema_version": "1.1",
            "model_type": f"{point_upper_model.metadata.model_type}+safety_classifier",
            "capabilities": {
                "supports_forecast": True,
                "supports_actions": False,
                "supports_uncertainty": True,
                "supports_exceedance_probability": True,
            },
            "processing": {
                **point_upper_model.metadata.processing,
                "safety_policy": {
                    "risk_family": fitted.report["selected_family"],
                    "positive_weight": fitted.report["positive_weight"],
                    "calibration": "platt_first_validation_half",
                    "threshold_selection": "minimum_fnr_subject_to_fpr_budget",
                    "threshold_validation_half": "second",
                    "alarm_threshold": fitted.report["alarm_threshold"],
                    "false_alarm_budget": fitted.report["false_alarm_budget"],
                    "target_false_negative_rate": fitted.report["target_false_negative_rate"],
                    "pak_metrics": {
                        "validation": fitted.report["validation_policy"],
                        "test": fitted.report["test"],
                        "transition_recall": fitted.report["test_transition_recall"],
                    },
                    "lims_metrics": "separate_target_not_promoted",
                },
            },
            "applicability": {
                "method": "joint_pca_mahalanobis",
                "required_features": list(fitted.predictor.applicability.required_features),
                "coverage": 0.99,
                "max_distance_squared": fitted.predictor.applicability.max_distance_squared,
                "out_of_domain_behavior": "unavailable",
                "purpose": "60-minute sulfur point, upper and exceedance-risk forecast",
                "target_source": point_upper_model.metadata.target_source,
                "action_comparison": "forbidden",
            },
            "reports": tuple(point_upper_model.metadata.reports) + ("metrics.json",),
        }
    )
    bundle = save_model(directory, fitted.predictor, metadata, fitted.report)
    return bundle, fitted


__all__ = [
    "AlarmPolicy",
    "CompositeSafetyPredictor",
    "FALSE_ALARM_BUDGET",
    "JointApplicabilityModel",
    "LimsCorrectionResult",
    "SafetyFitResult",
    "binary_alarm_metrics",
    "fit_joint_applicability",
    "fit_lims_correction",
    "published_lims_features",
    "fit_safety_model",
    "save_safety_model",
    "select_alarm_threshold",
]
