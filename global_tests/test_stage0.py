<<<<<<< HEAD
"""Acceptance checks for the complete stage-0 contract boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from source.config import load_runtime_config, load_scenario, load_tag_dictionary
from source.contracts import (
    CandidateAction,
    CandidateKind,
    DatasetManifest,
    MappingStatus,
    Observation,
    ProcessState,
    Recommendation,
    Stage,
    TagMeta,
    Unit,
    Validity,
)
from source.data.ingest import read_lims, read_pak, read_telemetry_csv
from source.data.prepare import PreparedData, known_feature_order
from source.data.state import build_state

FIXTURES = Path(__file__).parent / "fixtures"


def test_all_versioned_configs_load() -> None:
    """Verify runtime settings and every checked-in scenario load successfully."""
    runtime = load_runtime_config("config/runtime.toml")
    assert runtime.source_timezone == "Europe/Moscow"
    assert runtime.lims_delay_hours == 4
    assert runtime.horizon_minutes == 60
    assert runtime.max_candidates == 125
    assert {load_scenario(path).id for path in Path("config/scenarios").glob("*.json")} == {
        "history",
        "blend_normal",
        "blend_risk",
        "blend_t95_risk",
        "blend_cetane_risk",
        "blend_missing",
        "hybrid_blend",
    }


def test_tag_dictionary_uses_confirmed_expert_clarifications() -> None:
    """Verify expert-confirmed mappings, conversion and controllable signals."""
    tags = load_tag_dictionary("config/tags.csv")
    assert len(tags) == 170
    assert {tag.signal_id for tag in tags.values() if tag.controllable} == {
        "ht:F19",
        "ht:P8",
        "ht:T11",
    }
    pak_sulfur = tags["24-2000:Mg.Sulfur"]
    assert pak_sulfur.mapping_status is MappingStatus.CONFIRMED
    assert pak_sulfur.signal_id == "ht:2:Mg.Sulfur"
    assert pak_sulfur.canonical_unit == Unit.MG_KG.value
    assert pak_sulfur.conversion.value == "ppm_mass_to_mg_kg"
    assert tags["ЛИМС:Гидроочистка.2:Mg.Sulfur"].mapping_status is MappingStatus.CONFIRMED


def test_ml_feature_order_excludes_unconfirmed_mappings() -> None:
    """Unknown tags stay inspectable but cannot silently become model inputs."""
    tags = load_tag_dictionary("config/tags.csv")

    feature_order = known_feature_order(tags)

    assert "ht:2:Mg.Sulfur" in feature_order
    assert "ht:sulfur" not in feature_order


def test_serialized_contract_examples_validate() -> None:
    """Verify the representative serialized state and recommendation contracts."""
    ProcessState.model_validate_json(
        (FIXTURES / "contracts/process_state.json").read_text(encoding="utf-8")
    )
    Recommendation.model_validate_json(
        (FIXTURES / "contracts/recommendation.json").read_text(encoding="utf-8")
    )
    legacy = json.loads((FIXTURES / "contracts/recommendation.json").read_text(encoding="utf-8"))
    legacy["schema_version"] = "1.0"
    legacy["selected"]["candidate"].pop("additive_mass_fraction")
    Recommendation.model_validate(legacy)


@pytest.mark.parametrize(
    "name",
    [
        "blend_normal",
        "blend_risk",
        "blend_t95_risk",
        "blend_cetane_risk",
        "blend_missing",
    ],
)
def test_model_demo_state_fixtures_validate(name: str) -> None:
    """Verify each model-demo fixture contains a valid process state and status."""
    fixture = json.loads((FIXTURES / f"model_demo/{name}.json").read_text(encoding="utf-8"))
    ProcessState.model_validate(fixture["state"])
    assert fixture["expected"]["status"] in {"hold", "recommend", "abstain"}


def test_model_demo_numbers_match_design() -> None:
    """Verify fixture sulfur values match the numerical examples in DESIGN.md."""
    normal = load_scenario("config/scenarios/blend_normal.json")
    risk = load_scenario("config/scenarios/blend_risk.json")

    def sulfur(scenario, field: str) -> float:
        """Calculate weighted sulfur for the scenario's current blend."""
        return sum(
            scenario.current_blend_mass_fractions[item.id] * getattr(item.sulfur, field)
            for item in scenario.blend_components
        )

    assert sulfur(normal, "value") == pytest.approx(8.316)
    assert sulfur(normal, "upper") == pytest.approx(9.306)
    assert sulfur(risk, "value") == pytest.approx(13.068)
    assert sulfur(risk, "upper") == pytest.approx(14.058)
    missing = load_scenario("config/scenarios/blend_missing.json")
    assert missing.blend_components[0].sulfur.value is None
    assert missing.blend_components[0].sulfur.upper is None


def test_contracts_reject_extra_fields_naive_time_and_nan() -> None:
    """Verify contracts reject unknown fields, naive timestamps and NaN values."""
    payload = json.loads((FIXTURES / "contracts/process_state.json").read_text(encoding="utf-8"))
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ProcessState.model_validate(payload)

    observation = {
        "id": "invalid",
        "signal_id": "ht:sulfur",
        "stage": "ht",
        "source": "pak",
        "measured_at": "2026-01-01T00:00:00",
        "available_at": "2026-01-01T00:00:00Z",
        "value": 1.0,
        "unit": "mg/kg",
        "validity": "valid",
        "source_ref": "fixture",
    }
    with pytest.raises(ValidationError):
        Observation.model_validate(observation)
    observation["measured_at"] = "2026-01-01T00:00:00Z"
    observation["value"] = float("nan")
    with pytest.raises(ValidationError):
        Observation.model_validate(observation)


def test_candidate_contract_enforces_kind_and_recipe() -> None:
    """Verify candidate payloads cannot contradict their declared action kind."""
    with pytest.raises(ValidationError):
        CandidateAction(
            id="bad-hold",
            kind=CandidateKind.HOLD,
            setpoints={"ht:T6": 300.0},
            horizon_minutes=60,
        )
    with pytest.raises(ValidationError):
        CandidateAction(
            id="bad-blend",
            kind=CandidateKind.BLEND,
            blend_mass_fractions={"A": 0.8, "B": 0.3},
            horizon_minutes=60,
        )


def test_telemetry_is_utc_namespaced_and_reports_conflicts(tmp_path: Path) -> None:
    """Verify telemetry normalization, namespacing and duplicate conflict reporting."""
    path = tmp_path / "telemetry.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "date": ["2026-01-15 12:00:00"] * 2 + ["broken"],
            "T1": [130.0, 131.0, 132.0],
        }
    ).to_csv(path, index=False)
    tags = {
        "avt:T1": TagMeta(
            signal_id="avt:T1",
            raw_name="avt:T1",
            stage=Stage.AVT,
            meaning="temperature fixture",
            raw_unit=None,
            canonical_unit="unknown",
            conversion="none",
            mapping_status="ambiguous",
            controllable=False,
            evidence_ref="fixture",
        )
    }
    result = read_telemetry_csv(path, "avt", tags)
    assert list(result.frame.columns) == ["avt:T1", "timestamp"] or list(result.frame.columns) == [
        "timestamp",
        "avt:T1",
    ]
    assert len(result.frame) == 1
    assert result.frame.loc[0, "timestamp"].tzinfo is not None
    assert pd.isna(result.frame.loc[0, "avt:T1"])
    assert {issue.code for issue in result.issues} >= {
        "INVALID_TIMESTAMP",
        "SOURCE_CONFLICT",
        "TAG_UNCONFIRMED",
    }


def test_pak_and_lims_keep_time_semantics_and_invalid_values(tmp_path: Path) -> None:
    """Verify PAK and LIMS preserve timestamps, invalid values and delay semantics."""
    pak_path = tmp_path / "pak.xlsx"
    pd.DataFrame(
        [
            ["24-2000:D15", None],
            ["кг/м3", None],
            [datetime(2026, 1, 15, 12), 830.0],
            [datetime(2026, 1, 15, 12, 10), "Pt Created"],
        ]
    ).to_excel(pak_path, header=False, index=False)
    lims_path = tmp_path / "lims.xlsx"
    section = "Установка 'Гидроочистка'. Точка отбора '2'. Продукт 'ДТ'"
    pd.DataFrame(
        [
            [section, None],
            ["Mg.Sulfur", None],
            ["мг/кг", None],
            ["Количество значений:", 1],
            [datetime(2026, 1, 15, 12), 8.4],
        ]
    ).to_excel(lims_path, header=False, index=False)
    tags = {
        "24-2000:D15": TagMeta(
            signal_id="ht:density_15c",
            raw_name="24-2000:D15",
            stage="ht",
            meaning="density",
            raw_unit="кг/м3",
            canonical_unit="kg/m3",
            conversion="none",
            mapping_status="confirmed",
            controllable=False,
            evidence_ref="fixture",
        ),
        "ЛИМС:Гидроочистка.2:Mg.Sulfur": TagMeta(
            signal_id="ht:2:Mg.Sulfur",
            raw_name="ЛИМС:Гидроочистка.2:Mg.Sulfur",
            stage="ht",
            meaning="sulfur",
            raw_unit="мг/кг",
            canonical_unit="mg/kg",
            conversion="none",
            mapping_status="confirmed",
            controllable=False,
            evidence_ref="fixture",
        ),
    }
    pak = read_pak(pak_path, tags)
    lims = read_lims(lims_path, tags, lims_delay_hours=6)
    assert pak.observations[0].measured_at == datetime(2026, 1, 15, 9, tzinfo=UTC)
    assert pak.observations[0].available_at == pak.observations[0].measured_at
    assert pak.observations[1].validity is Validity.INVALID
    assert pak.observations[1].value is None
    assert "INVALID_VALUE" in {issue.code for issue in pak.issues}
    assert lims.observations[0].available_at - lims.observations[0].measured_at == pd.Timedelta(
        hours=6
    )


def test_expert_confirmed_pak_conversion_and_lims_unit_override(tmp_path: Path) -> None:
    """Apply only the PAK conversion and bad-header override confirmed by experts."""
    pak_path = tmp_path / "pak_sulfur.xlsx"
    pd.DataFrame(
        [
            ["24-2000:Mg.Sulfur", None],
            ["ppm", None],
            [datetime(2026, 1, 15, 12), 7.5],
        ]
    ).to_excel(pak_path, header=False, index=False)
    lims_path = tmp_path / "lims_bad_unit.xlsx"
    section = "Установка 'АВТ'. Точка отбора '1'. Продукт 'ДТ'"
    pd.DataFrame(
        [
            [section, None],
            ["50%.T", None],
            ["кг/м3", None],
            ["Количество значений:", 1],
            [datetime(2026, 1, 15, 12), 250.0],
        ]
    ).to_excel(lims_path, header=False, index=False)
    tags = {
        "24-2000:Mg.Sulfur": TagMeta(
            signal_id="ht:2:Mg.Sulfur",
            raw_name="24-2000:Mg.Sulfur",
            stage="ht",
            meaning="sulfur",
            raw_unit="ppm",
            canonical_unit="mg/kg",
            conversion="ppm_mass_to_mg_kg",
            mapping_status="confirmed",
            controllable=False,
            evidence_ref="DESIGN §16",
        ),
        "ЛИМС:АВТ.1:50%.T": TagMeta(
            signal_id="avt:1:50%.T",
            raw_name="ЛИМС:АВТ.1:50%.T",
            stage="avt",
            meaning="50 percent boiling temperature",
            raw_unit="кг/м3",
            canonical_unit="degC",
            conversion="none",
            mapping_status="confirmed",
            controllable=False,
            evidence_ref="DESIGN §16",
        ),
    }

    pak = read_pak(pak_path, tags)
    lims = read_lims(lims_path, tags, lims_delay_hours=4)

    assert pak.observations[0].signal_id == "ht:2:Mg.Sulfur"
    assert pak.observations[0].unit == "mg/kg"
    assert pak.observations[0].value == pytest.approx(7.5)
    assert pak.observations[0].validity is Validity.VALID
    assert lims.observations[0].unit == "degC"
    assert lims.observations[0].validity is Validity.VALID
    assert {issue.code for issue in lims.issues} == {"UNIT_HEADER_OVERRIDDEN"}


def test_build_state_cannot_see_delayed_lims() -> None:
    """Verify state selection respects measured and publication availability times."""
    quality = pd.read_csv(FIXTURES / "data/quality.csv")
    manifest = DatasetManifest.model_validate_json(
        (FIXTURES / "data/manifest.json").read_text(encoding="utf-8")
    )
    data = PreparedData(
        telemetry=pd.read_csv(FIXTURES / "data/telemetry.csv"),
        quality=quality,
        issues=pd.read_csv(FIXTURES / "data/issues.csv"),
        manifest=manifest,
        feature_order=tuple(
            json.loads((FIXTURES / "data/feature_order.json").read_text(encoding="utf-8"))
        ),
    )
    scenario = load_scenario("config/scenarios/history.json")
    config = load_runtime_config("config/runtime.toml")

    early = build_state(data, datetime(2026, 1, 15, 8, 10, tzinfo=UTC), scenario, config)
    assert early.signals["ht:2:Mg.Sulfur"].selected.source.value == "pak"
    later = build_state(data, datetime(2026, 1, 15, 14, 10, tzinfo=UTC), scenario, config)
    assert later.signals["ht:2:Mg.Sulfur"].selected.source.value == "lims"
=======
"""Acceptance checks for the complete stage-0 contract boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from source.config import load_runtime_config, load_scenario, load_tag_dictionary
from source.contracts import (
    CandidateAction,
    CandidateKind,
    DatasetManifest,
    MappingStatus,
    Observation,
    ProcessState,
    Recommendation,
    Stage,
    TagMeta,
    Unit,
    Validity,
)
from source.data.ingest import read_lims, read_pak, read_telemetry_csv
from source.data.prepare import PreparedData, known_feature_order
from source.data.state import build_state

FIXTURES = Path(__file__).parent / "fixtures"


def test_all_versioned_configs_load() -> None:
    """Verify runtime settings and every checked-in scenario load successfully."""
    runtime = load_runtime_config("config/runtime.toml")
    assert runtime.source_timezone == "Europe/Moscow"
    assert runtime.lims_delay_hours == 4
    assert runtime.horizon_minutes == 60
    assert runtime.max_candidates == 125
    assert {load_scenario(path).id for path in Path("config/scenarios").glob("*.json")} == {
        "history",
        "blend_normal",
        "blend_risk",
        "blend_t95_risk",
        "blend_cetane_risk",
        "blend_missing",
        "hybrid_blend",
    }


def test_tag_dictionary_uses_confirmed_expert_clarifications() -> None:
    """Verify expert-confirmed mappings and fail-closed action metadata."""
    tags = load_tag_dictionary("config/tags.csv")
    assert len(tags) == 170
    # Historical action-effect research is observational; no tag is yet
    # authorized as a production control (supports_actions remains false).
    assert {tag.signal_id for tag in tags.values() if tag.controllable} == set()
    pak_sulfur = tags["24-2000:Mg.Sulfur"]
    assert pak_sulfur.mapping_status is MappingStatus.CONFIRMED
    assert pak_sulfur.signal_id == "ht:2:Mg.Sulfur"
    assert pak_sulfur.canonical_unit == Unit.MG_KG.value
    assert pak_sulfur.conversion.value == "ppm_mass_to_mg_kg"
    assert tags["ЛИМС:Гидроочистка.2:Mg.Sulfur"].mapping_status is MappingStatus.CONFIRMED


def test_ml_feature_order_excludes_unconfirmed_mappings() -> None:
    """Unknown tags stay inspectable but cannot silently become model inputs."""
    tags = load_tag_dictionary("config/tags.csv")

    feature_order = known_feature_order(tags)

    assert "ht:2:Mg.Sulfur" in feature_order
    assert "ht:sulfur" not in feature_order


def test_serialized_contract_examples_validate() -> None:
    """Verify the representative serialized state and recommendation contracts."""
    ProcessState.model_validate_json(
        (FIXTURES / "contracts/process_state.json").read_text(encoding="utf-8")
    )
    Recommendation.model_validate_json(
        (FIXTURES / "contracts/recommendation.json").read_text(encoding="utf-8")
    )
    legacy = json.loads((FIXTURES / "contracts/recommendation.json").read_text(encoding="utf-8"))
    legacy["schema_version"] = "1.0"
    legacy["selected"]["candidate"].pop("additive_mass_fraction")
    Recommendation.model_validate(legacy)


@pytest.mark.parametrize(
    "name",
    [
        "blend_normal",
        "blend_risk",
        "blend_t95_risk",
        "blend_cetane_risk",
        "blend_missing",
    ],
)
def test_model_demo_state_fixtures_validate(name: str) -> None:
    """Verify each model-demo fixture contains a valid process state and status."""
    fixture = json.loads((FIXTURES / f"model_demo/{name}.json").read_text(encoding="utf-8"))
    ProcessState.model_validate(fixture["state"])
    assert fixture["expected"]["status"] in {"hold", "recommend", "abstain"}


def test_model_demo_numbers_match_design() -> None:
    """Verify fixture sulfur values match the numerical examples in DESIGN.md."""
    normal = load_scenario("config/scenarios/blend_normal.json")
    risk = load_scenario("config/scenarios/blend_risk.json")

    def sulfur(scenario, field: str) -> float:
        """Calculate weighted sulfur for the scenario's current blend."""
        return sum(
            scenario.current_blend_mass_fractions[item.id] * getattr(item.sulfur, field)
            for item in scenario.blend_components
        )

    assert sulfur(normal, "value") == pytest.approx(8.316)
    assert sulfur(normal, "upper") == pytest.approx(9.306)
    assert sulfur(risk, "value") == pytest.approx(13.068)
    assert sulfur(risk, "upper") == pytest.approx(14.058)
    missing = load_scenario("config/scenarios/blend_missing.json")
    assert missing.blend_components[0].sulfur.value is None
    assert missing.blend_components[0].sulfur.upper is None


def test_contracts_reject_extra_fields_naive_time_and_nan() -> None:
    """Verify contracts reject unknown fields, naive timestamps and NaN values."""
    payload = json.loads((FIXTURES / "contracts/process_state.json").read_text(encoding="utf-8"))
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        ProcessState.model_validate(payload)

    observation = {
        "id": "invalid",
        "signal_id": "ht:sulfur",
        "stage": "ht",
        "source": "pak",
        "measured_at": "2026-01-01T00:00:00",
        "available_at": "2026-01-01T00:00:00Z",
        "value": 1.0,
        "unit": "mg/kg",
        "validity": "valid",
        "source_ref": "fixture",
    }
    with pytest.raises(ValidationError):
        Observation.model_validate(observation)
    observation["measured_at"] = "2026-01-01T00:00:00Z"
    observation["value"] = float("nan")
    with pytest.raises(ValidationError):
        Observation.model_validate(observation)


def test_candidate_contract_enforces_kind_and_recipe() -> None:
    """Verify candidate payloads cannot contradict their declared action kind."""
    with pytest.raises(ValidationError):
        CandidateAction(
            id="bad-hold",
            kind=CandidateKind.HOLD,
            setpoints={"ht:T6": 300.0},
            horizon_minutes=60,
        )
    with pytest.raises(ValidationError):
        CandidateAction(
            id="bad-blend",
            kind=CandidateKind.BLEND,
            blend_mass_fractions={"A": 0.8, "B": 0.3},
            horizon_minutes=60,
        )


def test_telemetry_is_utc_namespaced_and_reports_conflicts(tmp_path: Path) -> None:
    """Verify telemetry normalization, namespacing and duplicate conflict reporting."""
    path = tmp_path / "telemetry.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "date": ["2026-01-15 12:00:00"] * 2 + ["broken"],
            "T1": [130.0, 131.0, 132.0],
        }
    ).to_csv(path, index=False)
    tags = {
        "avt:T1": TagMeta(
            signal_id="avt:T1",
            raw_name="avt:T1",
            stage=Stage.AVT,
            meaning="temperature fixture",
            raw_unit=None,
            canonical_unit="unknown",
            conversion="none",
            mapping_status="ambiguous",
            controllable=False,
            evidence_ref="fixture",
        )
    }
    result = read_telemetry_csv(path, "avt", tags)
    assert list(result.frame.columns) == ["avt:T1", "timestamp"] or list(result.frame.columns) == [
        "timestamp",
        "avt:T1",
    ]
    assert len(result.frame) == 1
    assert result.frame.loc[0, "timestamp"].tzinfo is not None
    assert pd.isna(result.frame.loc[0, "avt:T1"])
    assert {issue.code for issue in result.issues} >= {
        "INVALID_TIMESTAMP",
        "SOURCE_CONFLICT",
        "TAG_UNCONFIRMED",
    }


def test_pak_and_lims_keep_time_semantics_and_invalid_values(tmp_path: Path) -> None:
    """Verify PAK and LIMS preserve timestamps, invalid values and delay semantics."""
    pak_path = tmp_path / "pak.xlsx"
    pd.DataFrame(
        [
            ["24-2000:D15", None],
            ["кг/м3", None],
            [datetime(2026, 1, 15, 12), 830.0],
            [datetime(2026, 1, 15, 12, 10), "Pt Created"],
        ]
    ).to_excel(pak_path, header=False, index=False)
    lims_path = tmp_path / "lims.xlsx"
    section = "Установка 'Гидроочистка'. Точка отбора '2'. Продукт 'ДТ'"
    pd.DataFrame(
        [
            [section, None],
            ["Mg.Sulfur", None],
            ["мг/кг", None],
            ["Количество значений:", 1],
            [datetime(2026, 1, 15, 12), 8.4],
        ]
    ).to_excel(lims_path, header=False, index=False)
    tags = {
        "24-2000:D15": TagMeta(
            signal_id="ht:density_15c",
            raw_name="24-2000:D15",
            stage="ht",
            meaning="density",
            raw_unit="кг/м3",
            canonical_unit="kg/m3",
            conversion="none",
            mapping_status="confirmed",
            controllable=False,
            evidence_ref="fixture",
        ),
        "ЛИМС:Гидроочистка.2:Mg.Sulfur": TagMeta(
            signal_id="ht:2:Mg.Sulfur",
            raw_name="ЛИМС:Гидроочистка.2:Mg.Sulfur",
            stage="ht",
            meaning="sulfur",
            raw_unit="мг/кг",
            canonical_unit="mg/kg",
            conversion="none",
            mapping_status="confirmed",
            controllable=False,
            evidence_ref="fixture",
        ),
    }
    pak = read_pak(pak_path, tags)
    lims = read_lims(lims_path, tags, lims_delay_hours=6)
    assert pak.observations[0].measured_at == datetime(2026, 1, 15, 9, tzinfo=UTC)
    assert pak.observations[0].available_at == pak.observations[0].measured_at
    assert pak.observations[1].validity is Validity.INVALID
    assert pak.observations[1].value is None
    assert "INVALID_VALUE" in {issue.code for issue in pak.issues}
    assert lims.observations[0].available_at - lims.observations[0].measured_at == pd.Timedelta(
        hours=6
    )


def test_expert_confirmed_pak_conversion_and_lims_unit_override(tmp_path: Path) -> None:
    """Apply only the PAK conversion and bad-header override confirmed by experts."""
    pak_path = tmp_path / "pak_sulfur.xlsx"
    pd.DataFrame(
        [
            ["24-2000:Mg.Sulfur", None],
            ["ppm", None],
            [datetime(2026, 1, 15, 12), 7.5],
        ]
    ).to_excel(pak_path, header=False, index=False)
    lims_path = tmp_path / "lims_bad_unit.xlsx"
    section = "Установка 'АВТ'. Точка отбора '1'. Продукт 'ДТ'"
    pd.DataFrame(
        [
            [section, None],
            ["50%.T", None],
            ["кг/м3", None],
            ["Количество значений:", 1],
            [datetime(2026, 1, 15, 12), 250.0],
        ]
    ).to_excel(lims_path, header=False, index=False)
    tags = {
        "24-2000:Mg.Sulfur": TagMeta(
            signal_id="ht:2:Mg.Sulfur",
            raw_name="24-2000:Mg.Sulfur",
            stage="ht",
            meaning="sulfur",
            raw_unit="ppm",
            canonical_unit="mg/kg",
            conversion="ppm_mass_to_mg_kg",
            mapping_status="confirmed",
            controllable=False,
            evidence_ref="DESIGN §16",
        ),
        "ЛИМС:АВТ.1:50%.T": TagMeta(
            signal_id="avt:1:50%.T",
            raw_name="ЛИМС:АВТ.1:50%.T",
            stage="avt",
            meaning="50 percent boiling temperature",
            raw_unit="кг/м3",
            canonical_unit="degC",
            conversion="none",
            mapping_status="confirmed",
            controllable=False,
            evidence_ref="DESIGN §16",
        ),
    }

    pak = read_pak(pak_path, tags)
    lims = read_lims(lims_path, tags, lims_delay_hours=4)

    assert pak.observations[0].signal_id == "ht:2:Mg.Sulfur"
    assert pak.observations[0].unit == "mg/kg"
    assert pak.observations[0].value == pytest.approx(7.5)
    assert pak.observations[0].validity is Validity.VALID
    assert lims.observations[0].unit == "degC"
    assert lims.observations[0].validity is Validity.VALID
    assert {issue.code for issue in lims.issues} == {"UNIT_HEADER_OVERRIDDEN"}


def test_build_state_cannot_see_delayed_lims() -> None:
    """Verify state selection respects measured and publication availability times."""
    quality = pd.read_csv(FIXTURES / "data/quality.csv")
    manifest = DatasetManifest.model_validate_json(
        (FIXTURES / "data/manifest.json").read_text(encoding="utf-8")
    )
    data = PreparedData(
        telemetry=pd.read_csv(FIXTURES / "data/telemetry.csv"),
        quality=quality,
        issues=pd.read_csv(FIXTURES / "data/issues.csv"),
        manifest=manifest,
        feature_order=tuple(
            json.loads((FIXTURES / "data/feature_order.json").read_text(encoding="utf-8"))
        ),
    )
    scenario = load_scenario("config/scenarios/history.json")
    config = load_runtime_config("config/runtime.toml")

    early = build_state(data, datetime(2026, 1, 15, 8, 10, tzinfo=UTC), scenario, config)
    assert early.signals["ht:2:Mg.Sulfur"].selected.source.value == "pak"
    later = build_state(data, datetime(2026, 1, 15, 14, 10, tzinfo=UTC), scenario, config)
    assert later.signals["ht:2:Mg.Sulfur"].selected.source.value == "lims"
>>>>>>> d0bbf23 (Update project sources and materials)
