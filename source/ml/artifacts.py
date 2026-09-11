"""Trusted local model artifacts for stage-2 sulfur forecasts."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Self, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.base import BaseEstimator, RegressorMixin

MODEL_METADATA_VERSION: Literal["1.0"] = "1.0"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ModelCapabilities(BaseModel):
    """Capabilities that callers must check before using a forecast."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    supports_forecast: bool
    supports_actions: bool
    supports_uncertainty: bool = False


class ModelMetadata(BaseModel):
    """Strict JSON metadata stored next to a trusted sklearn predictor."""

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    model_id: str
    schema_version: Literal["1.0"] = MODEL_METADATA_VERSION
    model_sha256: str = Field(pattern=_SHA256_PATTERN)
    model_type: str
    training_dataset_id: str
    git_commit: str
    python_version: str
    sklearn_version: str
    target_signal: str
    target_source: str
    target_unit: str
    horizon_minutes: int = Field(gt=0)
    feature_names: tuple[str, ...]
    baseline_feature: str
    tag_dictionary_sha256: str = Field(pattern=_SHA256_PATTERN)
    feature_schema_hash: str = Field(pattern=_SHA256_PATTERN)
    processing: dict[str, Any]
    time_boundaries: dict[str, str | None]
    seed: int
    capabilities: ModelCapabilities
    applicability: dict[str, Any]
    reports: tuple[str, ...]

    @model_validator(mode="after")
    def validate_features_and_capabilities(self) -> Self:
        """Keep the serving schema explicit and stage-2 actions disabled."""
        if not self.feature_names:
            raise ValueError("feature_names cannot be empty")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("feature_names must be unique")
        if self.baseline_feature not in self.feature_names:
            raise ValueError("baseline_feature must be present in feature_names")
        definition = self.processing.get("feature_definition", {})
        if not isinstance(definition, Mapping):
            raise ValueError("processing.feature_definition must be an object")
        definition_horizon = definition.get("horizon_minutes")
        if definition_horizon is not None and definition_horizon != self.horizon_minutes:
            raise ValueError("feature recipe horizon does not match model horizon")
        if self.feature_schema_hash != feature_schema_hash(self.feature_names, definition):
            raise ValueError("feature_schema_hash does not match the stored feature recipe")
        for name in ("source_timezone", "train_end_local", "validation_end_local"):
            value = self.time_boundaries.get(name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"time_boundaries.{name} must be a non-empty string")
        if not self.capabilities.supports_forecast:
            raise ValueError("a stage-2 artifact must support forecasting")
        if self.capabilities.supports_actions:
            raise ValueError("a stage-2 forecast cannot claim action support")
        return self


class LastValueRegressor(BaseEstimator, RegressorMixin):
    """Forecast persistence by returning one named current-value feature."""

    def __init__(self, feature_name: str = "baseline") -> None:
        self.feature_name = feature_name

    def fit(self, x: pd.DataFrame, y: object = None) -> Self:
        """Validate the baseline column and remember the fitted schema."""
        _ = y
        if self.feature_name not in x.columns:
            raise ValueError(f"missing baseline feature: {self.feature_name}")
        self.feature_names_in_ = np.asarray(x.columns, dtype=object)
        self.n_features_in_ = len(x.columns)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        """Return the current value as the 60-minute persistence forecast."""
        if self.feature_name not in x.columns:
            raise ValueError(f"missing baseline feature: {self.feature_name}")
        return x[self.feature_name].to_numpy(dtype=float, copy=True)


@dataclass(frozen=True)
class ModelBundle:
    """Runtime predictor plus validated compatibility metadata."""

    predictor: Any
    metadata: ModelMetadata

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Expose the exact serving order without leaking estimator details."""
        return self.metadata.feature_names

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Reject train/serve schema drift before calling the predictor."""
        actual = tuple(str(name) for name in features.columns)
        if actual != self.metadata.feature_names:
            raise ValueError(
                f"feature order mismatch: expected {self.metadata.feature_names}, received {actual}"
            )
        prediction = np.asarray(self.predictor.predict(features), dtype=float)
        if prediction.shape != (len(features),):
            raise ValueError("predictor must return one value per feature row")
        if not np.isfinite(prediction).all():
            raise ValueError("predictor returned a non-finite value")
        return prediction

    def predict_upper(self, features: pd.DataFrame) -> np.ndarray:
        """Return a validated empirical upper estimate when the artifact supports it."""
        if not self.metadata.capabilities.supports_uncertainty:
            raise ValueError("model artifact does not declare uncertainty support")
        actual = tuple(str(name) for name in features.columns)
        if actual != self.metadata.feature_names:
            raise ValueError(
                f"feature order mismatch: expected {self.metadata.feature_names}, received {actual}"
            )
        predict_upper = getattr(self.predictor, "predict_upper", None)
        if not callable(predict_upper):
            raise ValueError("uncertainty-capable predictor must expose predict_upper(features)")
        prediction = np.asarray(predict_upper(features), dtype=float)
        if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
            raise ValueError("upper predictor must return one finite value per feature row")
        point = self.predict(features)
        if np.any(prediction < point):
            raise ValueError("upper prediction cannot be below point prediction")
        return prediction

    def check_applicability(self, features: pd.DataFrame) -> object:
        """Check persisted train-only feature bounds before a Stage-5 forecast."""
        from source.ml.uncertainty import ApplicabilityResult, check_applicability

        raw_bounds = self.metadata.applicability.get("feature_bounds")
        if not isinstance(raw_bounds, Mapping):
            return ApplicabilityResult(False, "APPLICABILITY_UNAVAILABLE")
        if len(features) != 1:
            raise ValueError("applicability check requires exactly one feature row")
        bounds: dict[str, tuple[float, float]] = {}
        for name in self.feature_names:
            value = raw_bounds.get(name)
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes))
                or len(value) != 2
            ):
                return ApplicabilityResult(False, "APPLICABILITY_UNAVAILABLE", (name,))
            bounds[name] = (float(value[0]), float(value[1]))
        row = features.iloc[0]
        values = {
            name: None if pd.isna(row[name]) else float(row[name]) for name in self.feature_names
        }
        return check_applicability(values, bounds)


def sha256_file(path: Path) -> str:
    """Calculate a streaming digest for an artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_schema_hash(
    feature_names: Sequence[str], feature_definition: Mapping[str, Any] | None = None
) -> str:
    """Hash ordered names and the feature recipe used by train and serve."""
    payload = {
        "feature_names": list(feature_names),
        "feature_definition": dict(feature_definition or {}),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_model(
    directory: Path,
    predictor: Any,
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> ModelBundle:
    """Atomically persist one predictor, its strict metadata and metrics."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"model directory already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)

    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    try:
        model_path = temporary / "model.joblib"
        joblib.dump(predictor, model_path, compress=3)
        payload = dict(metadata)
        payload["model_sha256"] = sha256_file(model_path)
        validated = ModelMetadata.model_validate(payload)
        if validated.model_id != directory.name:
            raise ValueError("metadata model_id must match the artifact directory name")
        _write_json(temporary / "metadata.json", validated.model_dump(mode="json"))
        _write_json(temporary / "metrics.json", dict(metrics))
        temporary.replace(directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ModelBundle(predictor=predictor, metadata=validated)


def load_model(
    directory: Path,
    *,
    trusted: bool = False,
    expected_schema_version: str = MODEL_METADATA_VERSION,
    expected_horizon_minutes: int | None = None,
    expected_feature_names: Sequence[str] | None = None,
    expected_feature_schema_hash: str | None = None,
    expected_tag_dictionary_sha256: str | None = None,
    expected_target_signal: str | None = None,
    expected_target_source: str | None = None,
    expected_target_unit: str | None = None,
) -> ModelBundle:
    """Load a verified local joblib artifact only after explicit trust."""
    if not trusted:
        raise ValueError("joblib artifacts may be loaded only from an explicitly trusted path")
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    model_path = directory / "model.joblib"
    metadata = ModelMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
    if metadata.model_id != directory.name:
        raise ValueError("metadata model_id does not match the artifact directory")
    if metadata.schema_version != expected_schema_version:
        raise ValueError("model schema version is incompatible")
    if _python_major_minor(metadata.python_version) != platform.python_version_tuple()[:2]:
        raise ValueError("model Python version is incompatible")
    if metadata.sklearn_version != sklearn.__version__:
        raise ValueError("model scikit-learn version is incompatible")
    if sha256_file(model_path) != metadata.model_sha256:
        raise ValueError("model checksum mismatch")

    _check_expected("horizon_minutes", metadata.horizon_minutes, expected_horizon_minutes)
    _check_expected(
        "feature_names",
        metadata.feature_names,
        tuple(expected_feature_names) if expected_feature_names is not None else None,
    )
    _check_expected(
        "feature_schema_hash", metadata.feature_schema_hash, expected_feature_schema_hash
    )
    _check_expected(
        "tag_dictionary_sha256",
        metadata.tag_dictionary_sha256,
        expected_tag_dictionary_sha256,
    )
    _check_expected("target_signal", metadata.target_signal, expected_target_signal)
    _check_expected("target_source", metadata.target_source, expected_target_source)
    _check_expected("target_unit", metadata.target_unit, expected_target_unit)
    predictor = joblib.load(model_path)
    return ModelBundle(predictor=predictor, metadata=metadata)


def _python_major_minor(version: str) -> tuple[str, str]:
    parts = version.split(".")
    if len(parts) < 2:
        raise ValueError("invalid Python version in model metadata")
    return parts[0], parts[1]


def _check_expected(name: str, actual: object, expected: object | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"model {name} is incompatible")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


__all__ = [
    "MODEL_METADATA_VERSION",
    "LastValueRegressor",
    "ModelBundle",
    "ModelCapabilities",
    "ModelMetadata",
    "feature_schema_hash",
    "load_model",
    "save_model",
    "sha256_file",
]
