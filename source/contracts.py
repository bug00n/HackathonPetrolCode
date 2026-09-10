"""Strict version-1 contracts shared by backend and ML code.

These models implement the DTOs from DESIGN.md sections 5 and 6.  They reject
unknown fields, naive timestamps and non-finite numbers so invalid data cannot
quietly cross a layer boundary.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Literal["1.0"] = "1.0"
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]


class ContractModel(BaseModel):
    """Common validation and serialization policy for public contracts."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, protected_namespaces=(), validate_default=True
    )


class Severity(StrEnum):
    WARNING = "warning"
    BLOCKING = "blocking"


class Stage(StrEnum):
    AVT = "avt"
    HYDROTREATMENT = "ht"
    BLEND = "blend"


class SourceKind(StrEnum):
    TELEMETRY = "telemetry"
    LIMS = "lims"
    PAK = "pak"
    VAK = "vak"
    SCENARIO = "scenario"


class Validity(StrEnum):
    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"
    CONFLICT = "conflict"


class Unit(StrEnum):
    """Known canonical units; DTO fields remain strings as required by DESIGN."""

    CELSIUS = "degC"
    DENSITY = "kg/m3"
    PERCENT_VOLUME = "vol%"
    PERCENT_MASS = "mass%"
    MG_KG = "mg/kg"
    PPM = "ppm"
    CETANE = "cetane_number"
    MM2_S = "mm2/s"
    KPA = "kPa"
    MPA = "MPa"
    THOUSAND_M3_H = "1000*m3/h"
    M3_H = "m3/h"
    T_H = "t/h"
    PERCENT = "%"
    DIMENSIONLESS = "1"
    PROXY = "proxy_unit"
    RISK_INDEX = "index_0_1"
    UNKNOWN = "unknown"


class OperationMode(StrEnum):
    HISTORY = "history"
    MODEL_DEMO = "model_demo"
    HYBRID = "hybrid"


class CandidateKind(StrEnum):
    HOLD = "hold"
    SETPOINTS = "setpoints"
    BLEND = "blend"


class AssessmentAgent(StrEnum):
    QUALITY = "quality"
    RELIABILITY = "reliability"
    OPTIMIZER = "optimizer"


class AssessmentStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class EstimateBasis(StrEnum):
    MEASURED = "measured"
    FORECAST = "forecast"
    FORMULA = "formula"
    PROXY = "proxy"


class IntervalKind(StrEnum):
    NONE = "none"
    EMPIRICAL = "empirical"
    SCENARIO_BOUND = "scenario_bound"


class ConstraintStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class ConstraintBasis(StrEnum):
    TERMS_OF_REFERENCE = "tz"
    CONFIRMED = "confirmed"
    MODEL_ASSUMPTION = "model_assumption"


class RecommendationStatus(StrEnum):
    RECOMMEND = "recommend"
    HOLD = "hold"
    ABSTAIN = "abstain"


class MappingStatus(StrEnum):
    CONFIRMED = "confirmed"
    AMBIGUOUS = "ambiguous"
    EXCLUDED = "excluded"


class Conversion(StrEnum):
    NONE = "none"
    PPM_MASS_TO_MG_KG = "ppm_mass_to_mg_kg"
    MASS_PERCENT_TO_MG_KG = "mass_percent_to_mg_kg"


class Issue(ContractModel):
    code: str
    severity: Severity
    signal_id: str | None
    detail: str
    source_ref: str | None


class Observation(ContractModel):
    id: str
    signal_id: str
    stage: Stage
    source: SourceKind
    measured_at: AwareDatetime
    available_at: AwareDatetime
    value: FiniteFloat | None
    unit: str
    validity: Validity
    source_ref: str

    @model_validator(mode="after")
    def validate_times_and_value(self) -> "Observation":
        """Ensure publication time, value presence and validity agree."""
        if self.available_at < self.measured_at:
            raise ValueError("available_at cannot precede measured_at")
        if self.validity is Validity.VALID and self.value is None:
            raise ValueError("valid observation must have a value")
        if self.validity is Validity.MISSING and self.value is not None:
            raise ValueError("missing observation must not have a value")
        return self


class SignalSnapshot(ContractModel):
    selected: Observation | None
    alternatives: tuple[Observation, ...] = ()
    age_seconds: NonNegativeFloat | None
    fresh: bool
    issues: tuple[Issue, ...] = ()

    @model_validator(mode="after")
    def validate_freshness(self) -> "SignalSnapshot":
        """Require the selected observation and age whenever data is fresh."""
        if self.fresh and (self.selected is None or self.age_seconds is None):
            raise ValueError("fresh snapshot needs a selected observation and age")
        return self


class ProcessState(ContractModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    state_id: str
    as_of: AwareDatetime
    dataset_id: str
    mode: OperationMode
    signals: dict[str, SignalSnapshot]
    issues: tuple[Issue, ...] = ()


class CandidateAction(ContractModel):
    id: str
    kind: CandidateKind
    setpoints: dict[str, FiniteFloat] = Field(default_factory=dict)
    blend_mass_fractions: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    horizon_minutes: PositiveInt
    is_model_scenario: bool = False

    @model_validator(mode="after")
    def validate_payload(self) -> "CandidateAction":
        """Check that the candidate payload matches its declared action kind."""
        if self.kind is CandidateKind.HOLD and (self.setpoints or self.blend_mass_fractions):
            raise ValueError("hold candidate cannot contain changes")
        if self.kind is CandidateKind.SETPOINTS and (
            not self.setpoints or self.blend_mass_fractions
        ):
            raise ValueError("setpoints candidate needs only setpoints")
        if self.kind is CandidateKind.BLEND:
            if self.setpoints or not self.blend_mass_fractions:
                raise ValueError("blend candidate needs only a complete recipe")
            if abs(sum(self.blend_mass_fractions.values()) - 1.0) > 1e-9:
                raise ValueError("blend mass fractions must sum to one")
        return self


class MetricEstimate(ContractModel):
    value: FiniteFloat | None
    lower: FiniteFloat | None
    upper: FiniteFloat | None
    unit: str
    basis: EstimateBasis
    interval_kind: IntervalKind
    interval_level: Annotated[FiniteFloat, Field(gt=0, lt=1)] | None
    reference: str
    assumptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_interval(self) -> "MetricEstimate":
        """Validate interval ordering and the metadata describing its confidence."""
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("lower cannot exceed upper")
        if self.interval_kind is IntervalKind.EMPIRICAL and self.interval_level is None:
            raise ValueError("empirical interval needs interval_level")
        if self.interval_kind is not IntervalKind.EMPIRICAL and self.interval_level is not None:
            raise ValueError("interval_level is only valid for empirical intervals")
        return self


class AgentAssessment(ContractModel):
    agent: AssessmentAgent
    state_id: str
    candidate_id: str
    evaluated_for: AwareDatetime
    status: AssessmentStatus
    metrics: dict[str, MetricEstimate]
    issues: tuple[Issue, ...] = ()


class ConstraintResult(ContractModel):
    constraint_id: str
    candidate_id: str
    status: ConstraintStatus
    actual: FiniteFloat | None
    lower: FiniteFloat | None
    upper: FiniteFloat | None
    unit: str | None
    basis: ConstraintBasis
    evidence_ref: str
    reason_code: str


RankKey = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat, str]


class CandidateEvaluation(ContractModel):
    candidate: CandidateAction
    assessments: tuple[AgentAssessment, ...] = ()
    checks: tuple[ConstraintResult, ...] = ()
    feasible: bool
    rank_key: RankKey | None

    @model_validator(mode="after")
    def validate_consistency(self) -> "CandidateEvaluation":
        """Keep candidate identifiers and feasibility status consistent."""
        candidate_id = self.candidate.id
        if any(item.candidate_id != candidate_id for item in self.assessments):
            raise ValueError("assessment candidate_id mismatch")
        if any(item.candidate_id != candidate_id for item in self.checks):
            raise ValueError("constraint candidate_id mismatch")
        if self.feasible and any(item.status is not ConstraintStatus.PASS for item in self.checks):
            raise ValueError("feasible candidate cannot have failed or unknown checks")
        if not self.feasible and self.rank_key is not None:
            raise ValueError("infeasible candidate cannot have rank_key")
        return self


class Recommendation(ContractModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    run_id: str
    state_id: str
    as_of: AwareDatetime
    scenario_id: str
    mode: OperationMode
    status: RecommendationStatus
    baseline: CandidateEvaluation | None
    selected: CandidateEvaluation | None
    alternatives: Annotated[tuple[CandidateEvaluation, ...], Field(max_length=3)] = ()
    reason_codes: tuple[str, ...]
    explanation: str
    assumptions: tuple[str, ...] = ()
    model_id: str | None

    @model_validator(mode="after")
    def validate_result(self) -> "Recommendation":
        """Ensure the selected result matches the recommendation status."""
        if self.status is RecommendationStatus.ABSTAIN:
            if self.selected is not None or not self.reason_codes:
                raise ValueError("abstain needs selected=None and at least one reason")
            return self
        if self.selected is None or not self.selected.feasible:
            raise ValueError("hold/recommend needs a feasible selected candidate")
        if self.status is RecommendationStatus.HOLD:
            if self.selected.candidate.kind is not CandidateKind.HOLD:
                raise ValueError("hold must select a hold candidate")
        elif self.selected.candidate.kind is CandidateKind.HOLD:
            raise ValueError("recommend must select a non-hold candidate")
        return self


class RunFailure(ContractModel):
    run_id: str
    occurred_at: AwareDatetime
    error_code: str
    stage: str
    message: str


class ControlSpec(ContractModel):
    signal_id: str
    unit: str
    lower: FiniteFloat | None
    upper: FiniteFloat | None
    max_step: NonNegativeFloat | None
    step: Annotated[FiniteFloat, Field(gt=0)] | None
    evidence_ref: str
    basis: ConstraintBasis
    enabled: bool = False

    @model_validator(mode="after")
    def validate_enabled(self) -> "ControlSpec":
        """Require complete limits and evidence for an enabled control."""
        fields = (self.lower, self.upper, self.max_step, self.step)
        if self.enabled and (any(item is None for item in fields) or not self.evidence_ref):
            raise ValueError("enabled control needs limits, step and evidence")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("control lower cannot exceed upper")
        return self


class ConstraintSpec(ContractModel):
    id: str
    metric: str
    stage: Stage
    lower: FiniteFloat | None
    upper: FiniteFloat | None
    unit: str
    use_upper_estimate: bool
    required: bool
    basis: ConstraintBasis
    evidence_ref: str

    @model_validator(mode="after")
    def validate_bounds(self) -> "ConstraintSpec":
        """Require evidence and at least one valid lower or upper bound."""
        if self.lower is None and self.upper is None:
            raise ValueError("constraint needs at least one bound")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("constraint lower cannot exceed upper")
        if not self.evidence_ref:
            raise ValueError("constraint needs evidence_ref")
        return self


class BlendComponent(ContractModel):
    id: str
    sulfur: MetricEstimate
    available_mass_t: NonNegativeFloat
    cost_proxy_per_t: NonNegativeFloat
    risk_index: Annotated[FiniteFloat, Field(ge=0, le=1)]
    source_state_id: str | None


class ScenarioConfig(ContractModel):
    id: str
    mode: OperationMode
    required_signals: tuple[str, ...] = ()
    controls: tuple[ControlSpec, ...] = ()
    constraints: tuple[ConstraintSpec, ...] = ()
    blend_components: tuple[BlendComponent, ...] = ()
    total_mass_t: NonNegativeFloat | None
    current_blend_mass_fractions: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    active_criteria: tuple[str, ...] = ()
    materiality_thresholds: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    require_upper_bound: bool = True
    action_cooldown_minutes: Annotated[int, Field(ge=0)] = 60
    assumptions: tuple[str, ...] = ()


class DecisionContext(ContractModel):
    last_recommended_at: AwareDatetime | None = None


class RuntimeConfig(ContractModel):
    source_timezone: str = "Europe/Moscow"
    lims_delay_hours: NonNegativeFloat = 6.0
    freshness_minutes: dict[SourceKind, PositiveInt] = Field(
        default_factory=lambda: {
            SourceKind.TELEMETRY: 20,
            SourceKind.PAK: 30,
            SourceKind.LIMS: 48 * 60,
        }
    )
    horizon_minutes: PositiveInt = 60
    seed: int = 42
    max_candidates: PositiveInt = 125
    data_dir: Path = Path("data/processed")
    materials_dir: Path = Path("materials")
    tag_dictionary_path: Path = Path("config/tags.csv")
    models_dir: Path = Path("artifacts/models")
    reports_dir: Path = Path("reports")
    runs_dir: Path = Path("runs")


class TagMeta(ContractModel):
    signal_id: str
    raw_name: str
    stage: Stage
    meaning: str
    raw_unit: str | None
    canonical_unit: str
    conversion: Conversion
    mapping_status: MappingStatus
    controllable: bool
    evidence_ref: str

    @model_validator(mode="after")
    def validate_mapping(self) -> "TagMeta":
        """Prevent unsupported mappings from being treated as controllable."""
        if self.mapping_status is MappingStatus.CONFIRMED and not self.evidence_ref:
            raise ValueError("confirmed mapping needs evidence_ref")
        if self.controllable and self.mapping_status is not MappingStatus.CONFIRMED:
            raise ValueError("controllable signal must have confirmed mapping")
        return self


class SourceArtifact(ContractModel):
    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    size_bytes: Annotated[int, Field(ge=0)]


class DatasetManifest(ContractModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    dataset_id: Annotated[str, Field(min_length=12, max_length=12)]
    preparation_version: str
    created_at: AwareDatetime
    source_timezone: str
    assumptions: tuple[str, ...]
    sources: tuple[SourceArtifact, ...]
    config_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tag_dictionary_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    row_counts: dict[str, Annotated[int, Field(ge=0)]]
    time_ranges: dict[str, tuple[AwareDatetime, AwareDatetime] | None]


__all__ = [
    "SCHEMA_VERSION",
    "AgentAssessment",
    "AssessmentAgent",
    "AssessmentStatus",
    "BlendComponent",
    "CandidateAction",
    "CandidateEvaluation",
    "CandidateKind",
    "ConstraintBasis",
    "ConstraintResult",
    "ConstraintSpec",
    "ConstraintStatus",
    "ControlSpec",
    "Conversion",
    "DatasetManifest",
    "DecisionContext",
    "EstimateBasis",
    "IntervalKind",
    "Issue",
    "MappingStatus",
    "MetricEstimate",
    "Observation",
    "OperationMode",
    "ProcessState",
    "Recommendation",
    "RecommendationStatus",
    "RunFailure",
    "RuntimeConfig",
    "ScenarioConfig",
    "Severity",
    "SignalSnapshot",
    "SourceArtifact",
    "SourceKind",
    "Stage",
    "TagMeta",
    "Unit",
    "Validity",
]
