"""Stage-2 feature tests with small, source-specific synthetic histories."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from source.contracts import DatasetManifest, OperationMode, ProcessState, SourceKind
from source.data.prepare import PreparedData
from source.ml.features import (
    baseline_feature_name,
    build_features,
    build_supervised_dataset,
    prepare_feature_batch,
)

FIXTURES = Path(__file__).parent / "fixtures"
QUALITY_COLUMNS = [
    "observation_id",
    "signal_id",
    "stage",
    "source",
    "measured_at",
    "available_at",
    "value",
    "unit",
    "validity",
    "source_ref",
]


def _observation(
    observation_id: str,
    measured_at: str,
    value: float,
    *,
    source: str = "pak",
    signal_id: str = "ht:sulfur",
    available_at: str | None = None,
    unit: str = "mg/kg",
    validity: str = "valid",
) -> dict[str, object]:
    return {
        "observation_id": observation_id,
        "signal_id": signal_id,
        "stage": "ht",
        "source": source,
        "measured_at": measured_at,
        "available_at": available_at or measured_at,
        "value": value,
        "unit": unit,
        "validity": validity,
        "source_ref": f"fixture:{observation_id}",
    }


def _prepared(
    telemetry: pd.DataFrame,
    observations: list[dict[str, object]],
    feature_order: tuple[str, ...] = (),
) -> PreparedData:
    manifest = DatasetManifest.model_validate_json(
        (FIXTURES / "data/manifest.json").read_text(encoding="utf-8")
    )
    return PreparedData(
        telemetry=telemetry,
        quality=pd.DataFrame(observations, columns=QUALITY_COLUMNS),
        issues=pd.DataFrame(columns=["source_ref", "code", "detail"]),
        manifest=manifest,
        feature_order=feature_order,
    )


def test_backward_lags_and_windows_end_at_as_of() -> None:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="10min")
    telemetry = pd.DataFrame(
        {
            "timestamp": timestamps,
            "ht:F1": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10_000.0],
        }
    )
    data = _prepared(
        telemetry,
        [_observation("target", "2026-01-01T02:00:00Z", 8.0)],
        ("ht:F1",),
    )

    dataset = build_supervised_dataset(
        data, "ht:sulfur", SourceKind.PAK, telemetry_signals=("ht:F1",)
    )
    row = dataset.frame.iloc[0]

    assert row["as_of"] == pd.Timestamp("2026-01-01T01:00:00Z")
    assert row["ht:F1__lag_0m"] == pytest.approx(60.0)
    assert row["ht:F1__lag_10m"] == pytest.approx(50.0)
    assert row["ht:F1__lag_30m"] == pytest.approx(30.0)
    assert row["ht:F1__lag_60m"] == pytest.approx(0.0)
    assert pd.isna(row["ht:F1__lag_180m"])
    assert row["ht:F1__delta_10m"] == pytest.approx(10.0)
    assert row["ht:F1__delta_30m"] == pytest.approx(30.0)
    assert row["ht:F1__delta_60m"] == pytest.approx(60.0)
    assert pd.isna(row["ht:F1__delta_180m"])
    assert row["ht:F1__mean_60m"] == pytest.approx(30.0)
    assert row["ht:F1__std_60m"] == pytest.approx(np.std(np.arange(0.0, 70.0, 10.0)))
    assert row["ht:F1__mean_180m"] == pytest.approx(30.0)


def test_pak_targets_are_exact_valid_rows_and_never_forward_filled() -> None:
    data = _prepared(
        pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")]}),
        [
            _observation("valid-1", "2026-01-01T01:00:00Z", 7.0),
            _observation("invalid", "2026-01-01T01:10:00Z", 999.0, validity="invalid"),
            _observation("wrong-unit", "2026-01-01T01:20:00Z", 9.0, unit="ppm"),
            _observation("valid-2", "2026-01-01T01:30:00Z", 8.0),
            _observation(
                "lims",
                "2026-01-01T01:40:00Z",
                6.0,
                source="lims",
                signal_id="ht:2:Mg.Sulfur",
            ),
        ],
    )

    dataset = build_supervised_dataset(data, "ht:sulfur", "pak", telemetry_signals=())

    assert list(dataset.frame["target_at"]) == [
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T01:30:00Z"),
    ]
    assert list(dataset.frame["as_of"]) == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-01-01T00:30:00Z"),
    ]
    assert list(dataset.frame["y"]) == [7.0, 8.0]
    assert set(dataset.frame["target_source"]) == {"pak"}


def test_pak_autoregressive_lags_and_windows_exclude_future_target() -> None:
    data = _prepared(
        pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")]}),
        [
            _observation("history-0", "2026-01-01T00:00:00Z", 1.0),
            _observation("history-30", "2026-01-01T00:30:00Z", 3.0),
            _observation("history-60", "2026-01-01T01:00:00Z", 5.0),
            _observation("future-target", "2026-01-01T02:00:00Z", 999.0),
        ],
    )

    dataset = build_supervised_dataset(data, "ht:sulfur", "pak", telemetry_signals=())
    row = dataset.frame.loc[dataset.frame["observation_id"] == "future-target"].iloc[0]

    assert row["ht:sulfur__pak_lag_0m"] == pytest.approx(5.0)
    assert row["ht:sulfur__pak_lag_30m"] == pytest.approx(3.0)
    assert row["ht:sulfur__pak_lag_60m"] == pytest.approx(1.0)
    assert row["ht:sulfur__pak_delta_60m"] == pytest.approx(4.0)
    assert row["ht:sulfur__pak_mean_60m"] == pytest.approx(3.0)
    assert row["ht:sulfur__pak_mean_60m"] != pytest.approx(999.0)


def test_lims_autoregression_respects_publication_delay() -> None:
    signal_id = "ht:2:Mg.Sulfur"
    data = _prepared(
        pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")]}),
        [
            _observation(
                "old-visible",
                "2026-01-01T00:00:00Z",
                5.0,
                source="lims",
                signal_id=signal_id,
                available_at="2026-01-01T06:00:00Z",
            ),
            _observation(
                "measured-but-hidden",
                "2026-01-01T06:30:00Z",
                999.0,
                source="lims",
                signal_id=signal_id,
                available_at="2026-01-01T12:30:00Z",
            ),
            _observation(
                "target",
                "2026-01-01T08:00:00Z",
                8.4,
                source="lims",
                signal_id=signal_id,
                available_at="2026-01-01T14:00:00Z",
            ),
        ],
    )

    dataset = build_supervised_dataset(data, signal_id, "lims", telemetry_signals=())
    target = dataset.frame.loc[
        dataset.frame["target_at"] == pd.Timestamp("2026-01-01T08:00:00Z")
    ].iloc[0]
    last_name = baseline_feature_name(signal_id, SourceKind.LIMS)

    assert target["as_of"] == pd.Timestamp("2026-01-01T07:00:00Z")
    assert target["target_available_at"] == pd.Timestamp("2026-01-01T14:00:00Z")
    assert target["baseline"] == pytest.approx(5.0)
    assert target[last_name] == pytest.approx(5.0)
    assert target[f"{signal_id}__lims_age_minutes"] == pytest.approx(7 * 60)
    assert len(dataset.frame) == 3  # one example per real analysis, never forward-filled


def test_training_and_serving_use_identical_features_and_model_order() -> None:
    timestamps = pd.date_range("2026-01-01T00:00:00Z", periods=7, freq="10min")
    data = _prepared(
        pd.DataFrame({"timestamp": timestamps, "ht:F1": np.arange(7, dtype=float)}),
        [
            _observation("history", "2026-01-01T00:30:00Z", 6.0),
            _observation("target", "2026-01-01T02:00:00Z", 8.0),
        ],
        ("ht:F1",),
    )
    dataset = build_supervised_dataset(data, "ht:sulfur", "pak", telemetry_signals=("ht:F1",))
    expected_row = dataset.frame.loc[
        dataset.frame["target_at"] == pd.Timestamp("2026-01-01T02:00:00Z")
    ]
    last_name = baseline_feature_name("ht:sulfur", "pak")
    feature_names = ("ht:F1__mean_60m", last_name, "ht:F1__lag_0m")
    model = SimpleNamespace(
        metadata=SimpleNamespace(
            feature_names=feature_names,
            target_signal="ht:sulfur",
            target_source="pak",
            target_unit="mg/kg",
        )
    )
    as_of = datetime(2026, 1, 1, 1, tzinfo=UTC)
    state = ProcessState(
        state_id="fixture",
        as_of=as_of,
        dataset_id=data.manifest.dataset_id,
        mode=OperationMode.HISTORY,
        signals={},
    )

    served = build_features(data, as_of, state, model)
    expected = expected_row.loc[:, list(feature_names)].reset_index(drop=True)

    assert tuple(served.columns) == feature_names
    pd.testing.assert_frame_equal(served, expected)


def test_batched_history_features_match_single_points_with_delayed_lims() -> None:
    signal = "ht:2:Mg.Sulfur"
    data = _prepared(
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=8, freq="1h", tz="UTC"),
                "ht:F1": np.arange(8, dtype=float),
            }
        ),
        [
            _observation(
                "old",
                "2026-01-01T00:00:00Z",
                5.0,
                source="lims",
                signal_id=signal,
                available_at="2026-01-01T02:00:00Z",
            ),
            _observation(
                "new",
                "2026-01-01T04:00:00Z",
                7.0,
                source="lims",
                signal_id=signal,
                available_at="2026-01-01T06:00:00Z",
            ),
        ],
        ("ht:F1",),
    )
    model = SimpleNamespace(
        metadata=SimpleNamespace(
            feature_names=(
                "ht:F1__lag_0m",
                baseline_feature_name(signal, "lims"),
                f"{signal}__lims_mean_60m",
            ),
            target_signal=signal,
            target_source="lims",
            target_unit="mg/kg",
        )
    )
    queries = pd.date_range("2026-01-01T01:00:00Z", periods=4, freq="2h")
    batch = prepare_feature_batch(data, queries, model)
    for query in queries:
        state = ProcessState(
            state_id="fixture",
            as_of=query.to_pydatetime(),
            dataset_id=data.manifest.dataset_id,
            mode=OperationMode.HISTORY,
            signals={},
        )
        single = build_features(data, query.to_pydatetime(), state, model)
        batched = build_features(data, query.to_pydatetime(), state, model, batch=batch)
        pd.testing.assert_frame_equal(single, batched)
    assert np.isnan(batch.frame.iloc[0][baseline_feature_name(signal, "lims")])
    assert batch.frame.iloc[1][baseline_feature_name(signal, "lims")] == 5.0
    assert batch.frame.iloc[-1][baseline_feature_name(signal, "lims")] == 7.0


def test_build_features_rejects_naive_time_and_state_mismatch() -> None:
    data = _prepared(
        pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")]}),
        [_observation("target", "2026-01-01T01:00:00Z", 8.0)],
    )
    model = SimpleNamespace(
        metadata={
            "feature_names": (baseline_feature_name("ht:sulfur", "pak"),),
            "target_signal_id": "ht:sulfur",
            "target_source": "pak",
            "target_unit": "mg/kg",
        }
    )
    state = ProcessState(
        state_id="fixture",
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        dataset_id=data.manifest.dataset_id,
        mode="history",
        signals={},
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        build_features(data, datetime(2026, 1, 1), state, model)
    with pytest.raises(ValueError, match="must match"):
        build_features(data, datetime(2026, 1, 1, 1, tzinfo=UTC), state, model)


def test_explicit_telemetry_signals_must_be_in_prepared_feature_order() -> None:
    data = _prepared(
        pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")],
                "ht:ambiguous": [1.0],
            }
        ),
        [_observation("target", "2026-01-01T01:00:00Z", 8.0)],
        feature_order=(),
    )

    with pytest.raises(ValueError, match="feature_order"):
        build_supervised_dataset(
            data,
            "ht:sulfur",
            "pak",
            telemetry_signals=("ht:ambiguous",),
        )


def test_default_feature_set_is_autoregressive_only() -> None:
    """Avoid materializing every confirmed KIP tag for dense PAK targets by accident."""
    data = _prepared(
        pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")],
                "ht:F1": [1.0],
            }
        ),
        [_observation("target", "2026-01-01T01:00:00Z", 8.0)],
        feature_order=("ht:F1",),
    )

    dataset = build_supervised_dataset(data, "ht:sulfur", "pak")

    assert all(not name.startswith("ht:F1__") for name in dataset.feature_names)
    assert dataset.feature_names[:2] == (
        "ht:sulfur__pak_last",
        "ht:sulfur__pak_age_minutes",
    )
    assert "ht:sulfur__pak_lag_60m" in dataset.feature_names
    assert "ht:sulfur__pak_mean_180m" in dataset.feature_names


def test_lims_control_targets_can_use_pak_history_without_merging_sources() -> None:
    """Evaluate a PAK feature schema on separate laboratory target timestamps."""
    signal_id = "ht:2:Mg.Sulfur"
    data = _prepared(
        pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01T00:00:00Z")]}),
        [
            _observation(
                "pak-history",
                "2026-01-01T06:30:00Z",
                7.2,
                source="pak",
                signal_id=signal_id,
            ),
            _observation(
                "lims-target",
                "2026-01-01T08:00:00Z",
                8.4,
                source="lims",
                signal_id=signal_id,
                available_at="2026-01-01T12:00:00Z",
            ),
        ],
    )

    control = build_supervised_dataset(
        data,
        signal_id,
        "lims",
        telemetry_signals=(),
        feature_source="pak",
    )

    assert control.target_source is SourceKind.LIMS
    assert control.feature_source is SourceKind.PAK
    assert set(control.frame["target_source"]) == {"lims"}
    assert control.frame.loc[0, "ht:2:Mg.Sulfur__pak_last"] == pytest.approx(7.2)
    assert control.frame.loc[0, "baseline"] == pytest.approx(7.2)
