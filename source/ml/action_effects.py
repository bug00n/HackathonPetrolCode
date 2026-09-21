"""Transparent setpoint consequences for Stage-3 model-only scenarios."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, cast

import numpy as np
import pandas as pd

from source.contracts import CandidateAction, CandidateKind, ControlSpec, ProcessState, Validity
from source.data.prepare import PreparedData
from source.ml.controls import (
    ACTION_EFFECT_HORIZONS_MINUTES,
    ActionEffectEvidence,
    JointControlDomain,
    assess_action_capability,
    validate_action_artifact_metadata,
)

if TYPE_CHECKING:
    from source.ml.safety import JointApplicabilityModel

HISTORICAL_ACTION_CONTROLS = ("ht:P8", "ht:F19")
HISTORICAL_CONTEXT_SIGNALS = ("ht:T11", "ht:F26")
ACTION_CONTROL_UNITS = {"ht:P8": "MPa", "ht:F19": "t/h"}
HISTORICAL_SIGNAL_MEANINGS = {
    "ht:P8": "R-202 differential pressure, MPa",
    "ht:F19": "gasoline flow to K-201, t/h",
    "ht:T11": "R-202 outlet product temperature, degC",
    "ht:F26": "hydrotreated diesel volumetric output, m3/h; context only",
}
ACTION_HORIZONS = ACTION_EFFECT_HORIZONS_MINUTES
# Allowed process-to-quality observation lags from the physical review.  These
# are metadata for the study, not a licence to search arbitrary offsets.
PHYSICAL_LAG_MINUTES = (0, 60, 120, 180)
LEGACY_ACTION_STATE_FEATURES = (
    "baseline_sulfur",
    "sulfur_slope_60m",
    "ht:P8",
    "ht:F19",
    "ht:F26",
)
ACTION_STATE_FEATURES = (
    "baseline_sulfur",
    "sulfur_slope_60m",
    "ht:P8",
    "ht:F19",
    "ht:T11",
    "ht:F26",
)
ACTION_MODEL_FEATURES = (*ACTION_STATE_FEATURES, "delta_ht:P8", "delta_ht:F19")
LEGACY_ACTION_MODEL_FEATURES = (*LEGACY_ACTION_STATE_FEATURES, "delta_ht:P8", "delta_ht:F19")


@dataclass(frozen=True)
class ActionModelBundle:
    """Action model kept separate from an ordinary forecast artifact."""

    predictor: object
    control_ids: tuple[str, ...]
    outcome_names: tuple[str, ...]
    horizons_minutes: tuple[int, ...]
    joint_domain: JointControlDomain
    evidence: ActionEffectEvidence
    supports_actions: bool = False

    def __post_init__(self) -> None:
        if not self.control_ids or not self.outcome_names or not self.horizons_minutes:
            raise ValueError("action bundle needs controls, outcomes and horizons")
        if self.supports_actions and not (
            self.evidence.shadow_replay_passed and self.evidence.pilot_approved
        ):
            raise ValueError("action capability requires shadow replay and technologist pilot")


@dataclass(frozen=True)
class HistoricalActionDataset:
    """Matched observational action episodes; never an authorization to act."""

    frame: pd.DataFrame
    thresholds: dict[str, float]
    observed_delta_bounds: dict[str, tuple[float, float]]
    match_distance_limit: float


@dataclass(frozen=True)
class HistoricalActionEstimate:
    """UI-ready historical effect estimate explicitly separated from advice."""

    control_id: str
    proposed_delta: float
    sulfur_by_horizon: dict[int, float]
    sulfur_upper_by_horizon: dict[int, float]
    effect_by_horizon: dict[int, float]
    applicable: bool
    reason_codes: tuple[str, ...]
    within_observed_domain: bool = False
    model_validated: bool = False
    safety_passes: bool = False

    def as_ui_payload(self) -> dict[str, object]:
        return {
            "title": "Модельный эффект по историческим эпизодам",
            "disclaimer": "Оценка не является советом по изменению уставки.",
            "control_id": self.control_id,
            "control_unit": ACTION_CONTROL_UNITS.get(self.control_id, "unknown"),
            "proposed_delta": self.proposed_delta,
            "sulfur_change": {
                str(horizon): self.effect_by_horizon[horizon] for horizon in ACTION_HORIZONS
            },
            "predicted_sulfur": {
                str(horizon): self.sulfur_by_horizon[horizon] for horizon in ACTION_HORIZONS
            },
            "sulfur_upper": {
                str(horizon): self.sulfur_upper_by_horizon[horizon] for horizon in ACTION_HORIZONS
            },
            "applicable": self.applicable,
            "within_observed_domain": self.within_observed_domain,
            "model_validated": self.model_validated,
            "safety_passes": self.safety_passes,
            "advisory": False,
            "reason_codes": self.reason_codes,
        }


@dataclass(frozen=True)
class HistoricalActionEffectModel:
    """Matched-episode sulfur model for shadow display, not control advice."""

    point_estimators: Mapping[int, Any]
    upper_estimators: Mapping[int, Any]
    upper_shifts: Mapping[int, float]
    applicability: JointApplicabilityModel
    observed_delta_bounds: Mapping[str, tuple[float, float]]
    report: Mapping[str, Any]
    supports_actions: bool = False

    def __post_init__(self) -> None:
        if self.supports_actions:
            raise ValueError("historical action-effect artifacts are research-only")

    def estimate(
        self,
        state: Mapping[str, float],
        control_id: str,
        proposed_delta: float,
        *,
        sulfur_limit: float = 10.0,
    ) -> HistoricalActionEstimate:
        if control_id not in HISTORICAL_ACTION_CONTROLS:
            raise ValueError("historical action model supports only ht:P8 or ht:F19")
        if not np.isfinite(proposed_delta):
            raise ValueError("proposed action delta must be finite")
        configured_state_features = tuple(
            str(name) for name in self.report.get("state_features", LEGACY_ACTION_STATE_FEATURES)
        )
        configured_model_features = tuple(
            str(name) for name in self.report.get("model_features", LEGACY_ACTION_MODEL_FEATURES)
        )
        missing = set(configured_state_features).difference(state)
        if missing:
            raise ValueError(f"historical action state is missing: {sorted(missing)}")
        action = {"delta_ht:P8": 0.0, "delta_ht:F19": 0.0}
        action[f"delta_{control_id}"] = float(proposed_delta)
        row = pd.DataFrame([{**state, **action}], columns=configured_model_features)
        reasons: list[str] = []
        model_validated = bool(self.report.get("evidence_gate_passed", False))
        if not model_validated:
            reasons.append("ACTION_EFFECT_VALIDATION_FAILED")
        lower, upper = self.observed_delta_bounds[control_id]
        delta_in_range = lower <= proposed_delta <= upper
        if not delta_in_range:
            reasons.append("ACTION_DELTA_OUT_OF_OBSERVED_RANGE")
        applicability = self.applicability.assess(row.loc[:, list(configured_state_features)])
        state_in_domain = bool(applicability.available)
        if not state_in_domain:
            reasons.append(applicability.reason_code or "OUT_OF_DOMAIN")
        hold = row.copy()
        hold.loc[:, ["delta_ht:P8", "delta_ht:F19"]] = 0.0
        sulfur: dict[int, float] = {}
        sulfur_upper: dict[int, float] = {}
        effect: dict[int, float] = {}
        for horizon in ACTION_HORIZONS:
            point = float(self.point_estimators[horizon].predict(row)[0])
            hold_point = float(self.point_estimators[horizon].predict(hold)[0])
            upper_point = float(self.upper_estimators[horizon].predict(row)[0]) + float(
                self.upper_shifts[horizon]
            )
            sulfur[horizon] = point
            sulfur_upper[horizon] = max(point, upper_point)
            effect[horizon] = point - hold_point
            if sulfur_upper[horizon] > sulfur_limit:
                reasons.append(f"SULFUR_UPPER_LIMIT_{horizon}M")
        safety_passes = all(value <= sulfur_limit for value in sulfur_upper.values())
        return HistoricalActionEstimate(
            control_id,
            proposed_delta,
            sulfur,
            sulfur_upper,
            effect,
            not reasons,
            tuple(dict.fromkeys(reasons)),
            delta_in_range and state_in_domain,
            model_validated,
            safety_passes,
        )


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_historical_action_model(
    directory: Path, model: HistoricalActionEffectModel, *, training_dataset_id: str
) -> Path:
    """Persist the shadow model for local research UI; it never enables actions."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"action model directory already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        model_path = temporary / "model.joblib"
        _joblib().dump(model, model_path, compress=3)
        metadata = {
            "schema_version": "1.0",
            "artifact_kind": "historical_action_effect_shadow",
            "bundle_kind": "ActionModelBundle",
            "training_dataset_id": training_dataset_id,
            "model_sha256": _artifact_sha256(model_path),
            "supports_actions": False,
            "controls": list(model.report.get("controls", HISTORICAL_ACTION_CONTROLS)),
            "signal_meanings": HISTORICAL_SIGNAL_MEANINGS,
            "outcomes": [f"sulfur_{horizon}m" for horizon in ACTION_HORIZONS],
            "horizons_minutes": list(ACTION_HORIZONS),
            "physical_lag_minutes": list(PHYSICAL_LAG_MINUTES),
            "state_features": list(
                model.report.get("state_features", LEGACY_ACTION_STATE_FEATURES)
            ),
            "model_features": list(
                model.report.get("model_features", LEGACY_ACTION_MODEL_FEATURES)
            ),
            "observed_delta_bounds": {
                control: list(bounds) for control, bounds in model.observed_delta_bounds.items()
            },
            "joint_domain": {
                "method": "joint_pca_mahalanobis",
                "max_distance_squared": getattr(model.applicability, "max_distance_squared", None),
            },
            "validation_evidence": dict(model.report.get("gate", {})),
            "engineering_limits": None,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        (temporary / "metrics.json").write_text(
            json.dumps(dict(model.report), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return directory


def load_historical_action_model(
    directory: Path, *, trusted: bool = False, expected_dataset_id: str | None = None
) -> HistoricalActionEffectModel:
    """Load a checksum-verified local shadow artifact from a trusted directory."""
    if not trusted:
        raise ValueError("action artifacts may be loaded only from an explicitly trusted path")
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model_path = directory / "model.joblib"
    if metadata.get("artifact_kind") != "historical_action_effect_shadow":
        raise ValueError("artifact is not a historical action-effect model")
    if (
        expected_dataset_id is not None
        and metadata.get("training_dataset_id") != expected_dataset_id
    ):
        raise ValueError("action artifact was trained on a different prepared dataset")
    if metadata.get("supports_actions") is not False:
        raise ValueError("action artifact must not enable actions")
    if metadata.get("model_sha256") != _artifact_sha256(model_path):
        raise ValueError("action artifact checksum mismatch")
    model = _joblib().load(model_path)
    if not isinstance(model, HistoricalActionEffectModel) or model.supports_actions:
        raise ValueError("action artifact has incompatible capability")
    return model


def load_verified_action_model(
    directory: Path,
    *,
    trusted: bool = False,
    expected_dataset_id: str | None = None,
    expected_config_sha256: str | None = None,
    expected_tag_dictionary_sha256: str | None = None,
    expected_telemetry_rules_sha256: str | None = None,
) -> VerifiedActionEffectModel:
    """Load a production action artifact only after full metadata gate validation."""
    if not trusted:
        raise ValueError("action artifacts may be loaded only from an explicitly trusted path")
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_id") != directory.name:
        raise ValueError("action metadata model_id does not match the artifact directory")
    controls = validate_action_artifact_metadata(
        metadata,
        expected_dataset_id=expected_dataset_id,
        expected_config_sha256=expected_config_sha256,
        expected_tag_dictionary_sha256=expected_tag_dictionary_sha256,
        expected_telemetry_rules_sha256=expected_telemetry_rules_sha256,
    )
    model_path = directory / "model.joblib"
    if metadata.get("model_sha256") != _artifact_sha256(model_path):
        raise ValueError("action artifact checksum mismatch")
    model = _joblib().load(model_path)
    if not isinstance(model, VerifiedActionEffectModel) or not model.supports_actions:
        raise ValueError("action artifact has incompatible capability")
    if model.model_id != metadata.get("model_id"):
        raise ValueError("action model_id does not match metadata")
    loaded_controls = tuple(sorted(control.signal_id for control in model.controls))
    metadata_controls = tuple(sorted(control.signal_id for control in controls))
    if loaded_controls != metadata_controls:
        raise ValueError("action model controls do not match metadata")
    return model


def _joblib() -> Any:
    import joblib

    return joblib


def _pak_sulfur(data: PreparedData) -> pd.DataFrame:
    quality = data.quality
    value = pd.to_numeric(quality["value"], errors="coerce")
    selected = quality[
        quality["signal_id"].astype(str).eq("ht:2:Mg.Sulfur")
        & quality["source"].map(lambda item: str(getattr(item, "value", item))).eq("pak")
        & quality["validity"]
        .map(lambda item: str(getattr(item, "value", item)))
        .eq(Validity.VALID.value)
        & quality["unit"].astype(str).eq("mg/kg")
        & value.notna()
        & np.isfinite(value)
    ].copy()
    selected["baseline_sulfur"] = value.loc[selected.index].astype(float)
    selected["timestamp"] = pd.to_datetime(selected["measured_at"], utc=True)
    return cast(
        pd.DataFrame,
        selected.loc[:, ["timestamp", "baseline_sulfur"]].sort_values("timestamp"),
    )


def _action_timeline(data: PreparedData) -> pd.DataFrame:
    signals = (*HISTORICAL_ACTION_CONTROLS, *HISTORICAL_CONTEXT_SIGNALS)
    missing = set(signals).difference(data.telemetry.columns)
    if missing:
        raise ValueError(f"telemetry is missing historical action signals: {sorted(missing)}")
    telemetry = data.telemetry.loc[:, ["timestamp", *signals]].copy()
    telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"], utc=True)
    for signal in signals:
        telemetry[signal] = pd.to_numeric(telemetry[signal], errors="coerce")
    timeline = pd.merge_asof(
        telemetry.sort_values("timestamp"),
        _pak_sulfur(data),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(minutes=10),
    )
    timeline["sulfur_slope_60m"] = (
        timeline["baseline_sulfur"] - timeline["baseline_sulfur"].shift(6)
    ) / 60.0
    indexed = timeline.set_index("timestamp")
    for horizon in ACTION_HORIZONS:
        future = indexed["baseline_sulfur"].reindex(indexed.index + pd.Timedelta(minutes=horizon))
        timeline[f"sulfur_{horizon}m"] = future.to_numpy(dtype=float)
    return timeline


def build_historical_action_dataset(
    data: PreparedData,
    *,
    train_end: str = "2025-01-01",
    threshold_quantile: float = 0.95,
) -> HistoricalActionDataset:
    """Extract isolated P8/F19 changes and pair them with similar calm states."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    if not 0.5 < threshold_quantile < 1.0:
        raise ValueError("action threshold quantile must be between 0.5 and 1")
    frame = _action_timeline(data)
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    changes = frame.loc[:, list(HISTORICAL_ACTION_CONTROLS)].diff()
    train = timestamp < pd.Timestamp(train_end, tz="UTC")
    thresholds: dict[str, float] = {}
    for signal in HISTORICAL_ACTION_CONTROLS:
        positive_steps = changes.loc[train, signal].abs()
        positive_steps = positive_steps[positive_steps > 0].dropna()
        if positive_steps.empty:
            raise ValueError(f"historical action signal {signal} has no observed changes")
        thresholds[signal] = float(positive_steps.quantile(threshold_quantile))
    notable = pd.DataFrame(
        {signal: changes[signal].abs().ge(threshold) for signal, threshold in thresholds.items()}
    )
    stable_before = (
        changes.abs().rolling(6, min_periods=6).max().shift(1).lt(pd.Series(thresholds)).all(axis=1)
    )
    quiet_after = (
        notable.iloc[::-1].rolling(18, min_periods=18).sum().iloc[::-1].shift(-1).fillna(1).eq(0)
    ).all(axis=1)
    outcomes = [f"sulfur_{horizon}m" for horizon in ACTION_HORIZONS]
    complete = frame[[*ACTION_STATE_FEATURES, *outcomes]].notna().all(axis=1)
    treated_mask = notable.sum(axis=1).eq(1) & stable_before & quiet_after & complete
    calm_mask = notable.rolling(18, min_periods=18).sum().eq(0).all(axis=1) & complete
    treated = frame.loc[treated_mask].copy()
    calm = frame.loc[calm_mask].copy()
    if treated.empty or calm.empty:
        raise ValueError("no complete isolated action/control episodes were found")

    scaler = StandardScaler().fit(frame.loc[train & complete, list(ACTION_STATE_FEATURES)])
    treated_parts: list[pd.DataFrame] = []
    matched_parts: list[pd.DataFrame] = []
    distance_parts: list[np.ndarray] = []
    for year in sorted(pd.to_datetime(treated["timestamp"], utc=True).dt.year.unique()):
        treated_year = treated.loc[pd.to_datetime(treated["timestamp"], utc=True).dt.year.eq(year)]
        calm_year = calm.loc[pd.to_datetime(calm["timestamp"], utc=True).dt.year.eq(year)]
        if calm_year.empty:
            continue
        treated_state = scaler.transform(treated_year.loc[:, list(ACTION_STATE_FEATURES)])
        calm_state = scaler.transform(calm_year.loc[:, list(ACTION_STATE_FEATURES)])
        distances, indices = (
            NearestNeighbors(n_neighbors=1).fit(calm_state).kneighbors(treated_state)
        )
        treated_parts.append(treated_year)
        matched_parts.append(calm_year.iloc[indices[:, 0]].copy())
        distance_parts.append(distances[:, 0])
    if not treated_parts:
        raise ValueError("no same-year matched action/control episodes were found")
    treated = pd.concat(treated_parts).copy()
    matched = pd.concat(matched_parts).copy()
    matched_distances = np.concatenate(distance_parts)
    treated["pair_id"] = np.arange(len(treated))
    matched["pair_id"] = np.arange(len(treated))
    treated["is_action_episode"] = True
    matched["is_action_episode"] = False
    treated["control_id"] = np.where(
        notable.loc[treated.index, "ht:P8"].to_numpy(), "ht:P8", "ht:F19"
    )
    matched["control_id"] = treated["control_id"].to_numpy()
    for signal in HISTORICAL_ACTION_CONTROLS:
        treated[f"delta_{signal}"] = np.where(
            treated["control_id"].eq(signal),
            changes.loc[treated.index, signal].to_numpy(dtype=float),
            0.0,
        )
        matched[f"delta_{signal}"] = 0.0
    treated["match_distance"] = matched_distances
    matched["match_distance"] = matched_distances
    result = pd.concat([treated, matched], ignore_index=True).sort_values(
        ["timestamp", "pair_id"], kind="stable"
    )
    bounds = {
        signal: (
            float(treated.loc[treated[f"delta_{signal}"].ne(0), f"delta_{signal}"].min()),
            float(treated.loc[treated[f"delta_{signal}"].ne(0), f"delta_{signal}"].max()),
        )
        for signal in HISTORICAL_ACTION_CONTROLS
    }
    return HistoricalActionDataset(
        result.reset_index(drop=True),
        thresholds,
        bounds,
        float(np.quantile(matched_distances, 0.99)),
    )


def fit_historical_action_model(
    dataset: HistoricalActionDataset,
    *,
    train_end: str = "2025-01-01",
    validation_end: str = "2026-01-01",
    seed: int = 42,
) -> HistoricalActionEffectModel:
    """Fit shadow-only matched action models and evaluate 2026 without selection."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from source.ml.safety import fit_joint_applicability

    frame = dataset.frame
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    train = timestamp < pd.Timestamp(train_end, tz="UTC")
    validation = (timestamp >= pd.Timestamp(train_end, tz="UTC")) & (
        timestamp < pd.Timestamp(validation_end, tz="UTC")
    )
    audit = timestamp >= pd.Timestamp(validation_end, tz="UTC")
    if min(int(train.sum()), int(validation.sum()), int(audit.sum())) == 0:
        raise ValueError("historical action model needs train, validation and audit episodes")
    features = list(ACTION_MODEL_FEATURES)
    point_estimators: dict[int, Any] = {}
    upper_estimators: dict[int, Any] = {}
    upper_shifts: dict[int, float] = {}
    validation_metrics: dict[str, Any] = {}
    audit_metrics: dict[str, Any] = {}
    for horizon in ACTION_HORIZONS:
        target = f"sulfur_{horizon}m"
        point = Pipeline(
            (
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            )
        ).fit(frame.loc[train, features], frame.loc[train, target])
        upper = Pipeline(
            (
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        loss="quantile", alpha=0.95, max_depth=2, random_state=seed
                    ),
                ),
            )
        ).fit(frame.loc[train, features], frame.loc[train, target])
        validation_upper = upper.predict(frame.loc[validation, features])
        shift = max(
            0.0,
            float(np.quantile(frame.loc[validation, target] - validation_upper, 0.95)),
        )
        point_estimators[horizon] = point
        upper_estimators[horizon] = upper
        upper_shifts[horizon] = shift
        for label, selected, destination in (
            ("validation", validation, validation_metrics),
            ("audit", audit, audit_metrics),
        ):
            predicted = point.predict(frame.loc[selected, features])
            conservative = np.maximum(
                predicted, upper.predict(frame.loc[selected, features]) + shift
            )
            actual = frame.loc[selected, target].to_numpy(dtype=float)
            destination[str(horizon)] = {
                "period": label,
                "rows": int(selected.sum()),
                "mae": float(mean_absolute_error(actual, predicted)),
                "hold_mae": float(
                    mean_absolute_error(actual, frame.loc[selected, "baseline_sulfur"])
                ),
                "upper_coverage": float(np.mean(actual <= conservative)),
            }
    applicability = fit_joint_applicability(
        frame.loc[train, list(ACTION_STATE_FEATURES)],
        required_features=ACTION_STATE_FEATURES,
    )
    per_control_pairs = {
        signal: int(frame.loc[frame["is_action_episode"], "control_id"].eq(signal).sum())
        for signal in HISTORICAL_ACTION_CONTROLS
    }
    statistical_mae_gate = all(
        values["mae"] <= 0.90 * values["hold_mae"] for values in validation_metrics.values()
    )
    coverage_gate = all(values["upper_coverage"] >= 0.95 for values in validation_metrics.values())
    audit_coverage_gate = all(values["upper_coverage"] >= 0.95 for values in audit_metrics.values())
    episode_count_gate = all(count >= 100 for count in per_control_pairs.values())
    # These are deliberately false until engineering limits and temporal sign stability
    # are confirmed outside this observational benchmark.
    sign_stability_gate = False
    engineering_bounds_gate = False
    evidence_gate_passed = (
        episode_count_gate
        and statistical_mae_gate
        and coverage_gate
        and audit_coverage_gate
        and sign_stability_gate
        and engineering_bounds_gate
    )
    gate_reasons: list[str] = []
    if not episode_count_gate:
        gate_reasons.append("INSUFFICIENT_PER_CONTROL_EPISODES")
    if not statistical_mae_gate:
        gate_reasons.append("ACTION_MODEL_GAIN_BELOW_10_PERCENT")
    if not coverage_gate:
        gate_reasons.append("ACTION_INTERVAL_COVERAGE_INSUFFICIENT")
    if not audit_coverage_gate:
        gate_reasons.append("ACTION_AUDIT_INTERVAL_COVERAGE_INSUFFICIENT")
    gate_reasons.extend(
        (
            "ACTION_EFFECT_SIGN_UNSTABLE",
            "CONTROL_LIMITS_UNCONFIRMED",
            "SHADOW_REPLAY_MISSING",
            "TECHNOLOGIST_PILOT_MISSING",
        )
    )
    report = {
        "basis": "matched_historical_episodes_not_causal_guarantee",
        "controls": list(HISTORICAL_ACTION_CONTROLS),
        "context_only": list(HISTORICAL_CONTEXT_SIGNALS),
        "physical_lag_minutes": list(PHYSICAL_LAG_MINUTES),
        "thresholds": dataset.thresholds,
        "state_features": list(ACTION_STATE_FEATURES),
        "model_features": list(ACTION_MODEL_FEATURES),
        "episode_rows": int(len(frame)),
        "independent_pairs": int(frame["pair_id"].nunique()),
        "per_control_pairs": per_control_pairs,
        "match_distance_limit": dataset.match_distance_limit,
        "gate": {
            "minimum_100_each_control": episode_count_gate,
            "mae_10_percent_better_than_hold": statistical_mae_gate,
            "upper_coverage_at_least_95_percent": coverage_gate,
            "audit_upper_coverage_at_least_95_percent": audit_coverage_gate,
            "effect_sign_stable_in_three_folds": sign_stability_gate,
            "engineering_bounds_confirmed": engineering_bounds_gate,
            "shadow_replay_passed": False,
            "technologist_pilot_approved": False,
        },
        "gate_failure_reasons": tuple(dict.fromkeys(gate_reasons)),
        "validation_2025": validation_metrics,
        "audit_2026": audit_metrics,
        "evidence_gate_passed": evidence_gate_passed,
        "supports_actions": False,
        "ui_label": "Модельный эффект по историческим эпизодам",
        "disclaimer": "Не является советом по изменению уставки.",
    }
    return HistoricalActionEffectModel(
        point_estimators,
        upper_estimators,
        upper_shifts,
        applicability,
        dataset.observed_delta_bounds,
        report,
    )


def evaluate_temporal_residualization(
    dataset: HistoricalActionDataset,
    *,
    train_end: str = "2025-01-01",
    validation_end: str = "2026-01-01",
) -> dict[str, Any]:
    """Evaluate an action/outcome residual model with forward-only cross-fitting.

    This is evidence about conditional historical associations.  It is deliberately
    separate from ``HistoricalActionEffectModel`` because residualization alone does
    not establish a causal action effect.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    frame = dataset.frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamp = pd.to_datetime(frame["timestamp"], utc=True)
    train = frame.loc[timestamp < pd.Timestamp(train_end, tz="UTC")].reset_index(drop=True)
    validation = frame.loc[
        (timestamp >= pd.Timestamp(train_end, tz="UTC"))
        & (timestamp < pd.Timestamp(validation_end, tz="UTC"))
    ].reset_index(drop=True)
    if len(train) < 40 or validation.empty:
        raise ValueError("residualization needs non-empty temporal train and validation periods")
    states = list(ACTION_STATE_FEATURES)
    actions = ["delta_ht:P8", "delta_ht:F19"]

    def pipeline() -> Pipeline:
        return Pipeline(
            (
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            )
        )

    splitter = TimeSeriesSplit(n_splits=3)
    oof_action = np.full((len(train), len(actions)), np.nan)
    oof_outcomes = {horizon: np.full(len(train), np.nan) for horizon in ACTION_HORIZONS}
    fold_residuals: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
        horizon: [] for horizon in ACTION_HORIZONS
    }
    for fold_train, fold_valid in splitter.split(train):
        action_model = pipeline().fit(
            train.iloc[fold_train][states], train.iloc[fold_train][actions]
        )
        oof_action[fold_valid] = action_model.predict(train.iloc[fold_valid][states])
        for horizon in ACTION_HORIZONS:
            target = f"sulfur_{horizon}m"
            outcome_model = pipeline().fit(
                train.iloc[fold_train][states], train.iloc[fold_train][target]
            )
            outcome_prediction = outcome_model.predict(train.iloc[fold_valid][states])
            oof_outcomes[horizon][fold_valid] = outcome_prediction
            fold_residuals[horizon].append(
                (
                    train.iloc[fold_valid][actions].to_numpy(dtype=float) - oof_action[fold_valid],
                    train.iloc[fold_valid][target].to_numpy(dtype=float) - outcome_prediction,
                )
            )
    valid_oof = np.isfinite(oof_action).all(axis=1)
    action_residual = train.loc[valid_oof, actions].to_numpy(dtype=float) - oof_action[valid_oof]
    report: dict[str, Any] = {
        "basis": "temporal_cross_fitted_residualization_not_causal_guarantee",
        "train_period": f"before {train_end}",
        "validation_period": f"{train_end}/{validation_end}",
        "cross_fit_folds": 3,
        "oof_rows": int(valid_oof.sum()),
        "horizons": {},
        "supports_actions": False,
        "promotion_eligible": False,
    }
    final_action_model = pipeline().fit(train[states], train[actions])
    validation_action_residual = validation[actions].to_numpy(
        dtype=float
    ) - final_action_model.predict(validation[states])
    for horizon in ACTION_HORIZONS:
        target = f"sulfur_{horizon}m"
        target_oof = train.loc[valid_oof, target].to_numpy(dtype=float)
        outcome_oof = oof_outcomes[horizon][valid_oof]
        outcome_residual = target_oof - outcome_oof
        effect_model = pipeline().fit(action_residual, outcome_residual)
        final_outcome_model = pipeline().fit(train[states], train[target])
        outcome_prediction = final_outcome_model.predict(validation[states])
        effect_prediction = effect_model.predict(validation_action_residual)
        validation_prediction = outcome_prediction + effect_prediction
        actual = validation[target].to_numpy(dtype=float)
        hold = validation["baseline_sulfur"].to_numpy(dtype=float)
        control_effects: dict[str, float] = {}
        signs: dict[str, int] = {}
        for position, control in enumerate(HISTORICAL_ACTION_CONTROLS):
            scale = float(np.nanquantile(np.abs(action_residual[:, position]), 0.75))
            probe = np.zeros((1, len(actions)))
            probe[0, position] = scale
            effect = float(
                effect_model.predict(probe)[0] - effect_model.predict(np.zeros_like(probe))[0]
            )
            control_effects[control] = effect
            signs[control] = int(np.sign(effect))
        fold_signs: list[dict[str, int]] = []
        for fold_action_residual, fold_outcome_residual in fold_residuals[horizon]:
            fold_model = pipeline().fit(fold_action_residual, fold_outcome_residual)
            fold_result: dict[str, int] = {}
            for position, control in enumerate(HISTORICAL_ACTION_CONTROLS):
                scale = float(np.nanquantile(np.abs(fold_action_residual[:, position]), 0.75))
                probe = np.zeros((1, len(actions)))
                probe[0, position] = scale
                fold_result[control] = int(
                    np.sign(
                        fold_model.predict(probe)[0] - fold_model.predict(np.zeros_like(probe))[0]
                    )
                )
            fold_signs.append(fold_result)
        sign_stable = {
            control: len({fold[control] for fold in fold_signs}) == 1
            and fold_signs[0][control] != 0
            for control in HISTORICAL_ACTION_CONTROLS
        }
        report["horizons"][str(horizon)] = {
            "mae": float(mean_absolute_error(actual, validation_prediction)),
            "hold_mae": float(mean_absolute_error(actual, hold)),
            "effect_for_train_q75_residual_action": control_effects,
            "effect_sign": signs,
            "effect_sign_by_fold": fold_signs,
            "effect_sign_stable": sign_stable,
        }
    return report


@dataclass(frozen=True)
class LinearActionEffectModel:
    """Small scenario model; coefficients are explicit assumptions, not causality proof."""

    current_setpoints: dict[str, float]
    baseline_sulfur: float
    baseline_sulfur_upper: float | None
    baseline_risk_index: float
    baseline_throughput: float
    baseline_cost_proxy: float
    sulfur_coefficients: dict[str, float]
    risk_coefficients: dict[str, float]
    throughput_coefficients: dict[str, float]
    cost_coefficients: dict[str, float]
    evidence_ref: str


@dataclass(frozen=True)
class VerifiedActionEffectModel:
    """Production action-effect model; construction is gated by external evidence."""

    model_id: str
    controls: tuple[ControlSpec, ...]
    joint_domain: JointControlDomain
    evidence: ActionEffectEvidence
    sulfur_coefficients: Mapping[str, float]
    risk_coefficients: Mapping[str, float]
    throughput_coefficients: Mapping[str, float]
    cost_coefficients: Mapping[str, float]
    evidence_ref: str
    baseline_risk_index: float = 0.5
    baseline_throughput: float = 1.0
    baseline_cost_proxy: float = 1.0
    horizons_minutes: tuple[int, ...] = ACTION_HORIZONS
    supports_actions: bool = True

    def __post_init__(self) -> None:
        if not self.supports_actions:
            raise ValueError("verified action model must declare supports_actions=true")
        control_ids = tuple(control.signal_id for control in self.controls)
        if set(control_ids) != set(HISTORICAL_ACTION_CONTROLS) or len(control_ids) != len(
            set(control_ids)
        ):
            raise ValueError("verified action model may enable only ht:P8 and ht:F19")
        if set(self.joint_domain.signal_ids) != set(control_ids):
            raise ValueError("verified action domain must match enabled controls")
        if self.horizons_minutes != ACTION_HORIZONS:
            raise ValueError("verified action model horizons must be 60/120/180 minutes")
        report = assess_action_capability(self.controls, self.evidence)
        if not report.supports_actions:
            raise ValueError(f"action capability gates failed: {report.reason_codes}")
        if not self.evidence_ref:
            raise ValueError("verified action model needs evidence_ref")

    def evaluate(
        self,
        state: ProcessState,
        candidate: CandidateAction,
        *,
        baseline_sulfur: float,
        baseline_sulfur_upper: float | None,
        sulfur_upper_limit: float = 10.0,
    ) -> "ActionOutcome":
        current_setpoints = _current_setpoints(state, self.controls)
        linear = LinearActionEffectModel(
            current_setpoints=current_setpoints,
            baseline_sulfur=baseline_sulfur,
            baseline_sulfur_upper=baseline_sulfur_upper,
            baseline_risk_index=self.baseline_risk_index,
            baseline_throughput=self.baseline_throughput,
            baseline_cost_proxy=self.baseline_cost_proxy,
            sulfur_coefficients=dict(self.sulfur_coefficients),
            risk_coefficients=dict(self.risk_coefficients),
            throughput_coefficients=dict(self.throughput_coefficients),
            cost_coefficients=dict(self.cost_coefficients),
            evidence_ref=self.evidence_ref,
        )
        return evaluate_linear_action(
            candidate,
            linear,
            self.controls,
            self.joint_domain,
            sulfur_upper_limit=sulfur_upper_limit,
        )


@dataclass(frozen=True)
class CombinedModelCapabilities:
    supports_forecast: bool
    supports_actions: bool
    supports_uncertainty: bool = False
    supports_exceedance_probability: bool = False
    supports_multi_horizon: bool = False


@dataclass(frozen=True)
class ActionEnabledMetadata:
    """Forecast metadata view with action capability supplied by a separate artifact."""

    forecast_metadata: object
    action_model_id: str
    model_id: str
    capabilities: CombinedModelCapabilities

    def __getattr__(self, name: str) -> object:
        return getattr(self.forecast_metadata, name)


@dataclass(frozen=True)
class ActionEnabledForecastModel:
    """Runtime wrapper combining a forecast artifact and a verified action artifact."""

    forecast_model: Any
    action_model: VerifiedActionEffectModel
    metadata: ActionEnabledMetadata

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(self.forecast_model.feature_names)

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return cast(np.ndarray, self.forecast_model.predict(features))

    def predict_upper(self, features: pd.DataFrame) -> np.ndarray:
        return cast(np.ndarray, self.forecast_model.predict_upper(features))

    def predict_exceedance_probability(self, features: pd.DataFrame) -> np.ndarray:
        return cast(np.ndarray, self.forecast_model.predict_exceedance_probability(features))

    def predict_alarm(self, features: pd.DataFrame) -> np.ndarray:
        return cast(np.ndarray, self.forecast_model.predict_alarm(features))

    def check_applicability(self, features: pd.DataFrame) -> object:
        return self.forecast_model.check_applicability(features)


def _capability_bool(capabilities: object, name: str) -> bool:
    if isinstance(capabilities, Mapping):
        return capabilities.get(name) is True
    return getattr(capabilities, name, False) is True


def combine_forecast_action_model(
    forecast_model: object,
    action_model: VerifiedActionEffectModel,
) -> ActionEnabledForecastModel:
    """Attach a separately verified action artifact to a trusted forecast bundle."""
    forecast_metadata = getattr(forecast_model, "metadata", None)
    if forecast_metadata is None:
        raise ValueError("forecast model needs metadata before action attachment")
    capabilities = getattr(forecast_metadata, "capabilities", {})
    combined = CombinedModelCapabilities(
        supports_forecast=_capability_bool(capabilities, "supports_forecast"),
        supports_actions=True,
        supports_uncertainty=_capability_bool(capabilities, "supports_uncertainty"),
        supports_exceedance_probability=_capability_bool(
            capabilities, "supports_exceedance_probability"
        ),
        supports_multi_horizon=_capability_bool(capabilities, "supports_multi_horizon"),
    )
    if not combined.supports_forecast:
        raise ValueError("action attachment requires a forecast-capable model")
    metadata = ActionEnabledMetadata(
        forecast_metadata=forecast_metadata,
        action_model_id=action_model.model_id,
        model_id=f"{forecast_metadata.model_id}+{action_model.model_id}",
        capabilities=combined,
    )
    return ActionEnabledForecastModel(forecast_model, action_model, metadata)


def _current_setpoints(
    state: ProcessState,
    controls: tuple[ControlSpec, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for control in controls:
        snapshot = state.signals.get(control.signal_id)
        if snapshot is None or snapshot.selected is None or snapshot.selected.value is None:
            raise ValueError(f"current setpoint is unavailable for {control.signal_id}")
        values[control.signal_id] = float(snapshot.selected.value)
    return values


@dataclass(frozen=True)
class ActionOutcome:
    candidate_id: str
    sulfur: float
    sulfur_upper: float | None
    risk_index: float
    throughput: float
    cost_proxy: float
    change_size: float
    feasible: bool
    reason_codes: tuple[str, ...]

    @property
    def rank_key(self) -> tuple[float, float, float, float, str] | None:
        if not self.feasible:
            return None
        return (
            self.risk_index,
            -self.throughput,
            self.cost_proxy,
            self.change_size,
            self.candidate_id,
        )


def _metric_delta(
    changes: dict[str, float], coefficients: dict[str, float], *, metric: str
) -> float:
    unknown = set(coefficients).difference(changes)
    if unknown:
        raise ValueError(f"{metric} coefficients reference unknown controls: {sorted(unknown)}")
    return sum(coefficients.get(signal_id, 0.0) * delta for signal_id, delta in changes.items())


def evaluate_linear_action(
    candidate: CandidateAction,
    model: LinearActionEffectModel,
    controls: tuple[ControlSpec, ...],
    domain: JointControlDomain,
    *,
    sulfur_upper_limit: float = 10.0,
) -> ActionOutcome:
    """Evaluate and hard-filter one hold/setpoint candidate."""
    if candidate.kind is CandidateKind.HOLD:
        setpoints = dict(model.current_setpoints)
    elif candidate.kind is CandidateKind.SETPOINTS:
        setpoints = dict(candidate.setpoints)
    else:
        raise ValueError("linear setpoint model cannot evaluate blending")
    if set(setpoints) != set(model.current_setpoints):
        raise ValueError("candidate setpoints must match the action model controls")

    specs = {control.signal_id: control for control in controls}
    if set(specs) != set(setpoints):
        raise ValueError("control specifications must match model setpoints")
    reasons: list[str] = []
    for signal_id, value in setpoints.items():
        spec = specs[signal_id]
        if not spec.enabled or spec.lower is None or spec.upper is None or spec.max_step is None:
            reasons.append("CONTROL_LIMITS_UNCONFIRMED")
            continue
        if not spec.lower <= value <= spec.upper:
            reasons.append("CONTROL_BOUND_VIOLATION")
        if abs(value - model.current_setpoints[signal_id]) > spec.max_step + 1e-12:
            reasons.append("CONTROL_STEP_VIOLATION")
    if not domain.contains(setpoints):
        reasons.append("OUT_OF_DOMAIN")

    changes = {
        signal_id: value - model.current_setpoints[signal_id]
        for signal_id, value in setpoints.items()
    }
    sulfur_delta = _metric_delta(changes, model.sulfur_coefficients, metric="sulfur")
    sulfur = model.baseline_sulfur + sulfur_delta
    sulfur_upper = (
        None if model.baseline_sulfur_upper is None else model.baseline_sulfur_upper + sulfur_delta
    )
    risk = max(
        0.0,
        min(
            1.0,
            model.baseline_risk_index
            + _metric_delta(changes, model.risk_coefficients, metric="risk"),
        ),
    )
    throughput = model.baseline_throughput + _metric_delta(
        changes, model.throughput_coefficients, metric="throughput"
    )
    cost = model.baseline_cost_proxy + _metric_delta(
        changes, model.cost_coefficients, metric="cost"
    )
    change_size = 0.0
    for signal_id, delta in changes.items():
        step = specs[signal_id].step
        if step is None:
            reasons.append("CONTROL_LIMITS_UNCONFIRMED")
        else:
            change_size += abs(delta) / step
    if not all(isfinite(value) for value in (sulfur, risk, throughput, cost, change_size)) or (
        sulfur_upper is not None and not isfinite(sulfur_upper)
    ):
        reasons.append("NON_FINITE_ACTION_FORECAST")
    if sulfur_upper is None:
        reasons.append("UNCERTAINTY_UNAVAILABLE")
    elif sulfur_upper > sulfur_upper_limit:
        reasons.append("QUALITY_LIMIT_VIOLATION")
    return ActionOutcome(
        candidate_id=candidate.id,
        sulfur=sulfur,
        sulfur_upper=sulfur_upper,
        risk_index=risk,
        throughput=throughput,
        cost_proxy=cost,
        change_size=change_size,
        feasible=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def rank_linear_actions(
    candidates: tuple[CandidateAction, ...],
    model: LinearActionEffectModel,
    controls: tuple[ControlSpec, ...],
    domain: JointControlDomain,
    *,
    sulfur_upper_limit: float = 10.0,
) -> tuple[ActionOutcome, ...]:
    """Filter infeasible actions before applying the documented rank key."""
    outcomes = tuple(
        evaluate_linear_action(
            candidate,
            model,
            controls,
            domain,
            sulfur_upper_limit=sulfur_upper_limit,
        )
        for candidate in candidates
    )
    feasible = (outcome for outcome in outcomes if outcome.feasible)
    return tuple(sorted(feasible, key=lambda outcome: outcome.rank_key or ()))


__all__ = [
    "ACTION_HORIZONS",
    "PHYSICAL_LAG_MINUTES",
    "ACTION_MODEL_FEATURES",
    "ActionModelBundle",
    "ActionOutcome",
    "ActionEnabledForecastModel",
    "ActionEnabledMetadata",
    "CombinedModelCapabilities",
    "HISTORICAL_ACTION_CONTROLS",
    "ACTION_CONTROL_UNITS",
    "HISTORICAL_CONTEXT_SIGNALS",
    "HISTORICAL_SIGNAL_MEANINGS",
    "HistoricalActionDataset",
    "HistoricalActionEffectModel",
    "HistoricalActionEstimate",
    "LinearActionEffectModel",
    "VerifiedActionEffectModel",
    "build_historical_action_dataset",
    "combine_forecast_action_model",
    "evaluate_linear_action",
    "evaluate_temporal_residualization",
    "fit_historical_action_model",
    "rank_linear_actions",
    "save_historical_action_model",
    "load_historical_action_model",
    "load_verified_action_model",
]
