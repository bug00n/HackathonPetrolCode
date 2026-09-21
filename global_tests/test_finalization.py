"""Release regressions for time safety, bounded snapshots and cancellation."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from source.config import load_runtime_config, load_scenario
from source.data.prepare import write_prepared_dataset
from source.data.state import build_state
from source.main import history_interval_command, run_model_demo
from source.ml.features import _last_available_quality


def test_fast_publication_lookup_matches_reference_for_delayed_samples() -> None:
    rng = np.random.default_rng(24)
    measured = pd.date_range("2025-01-01", periods=1000, freq="10min", tz="UTC")
    observations = pd.DataFrame(
        {
            "measured_at": measured,
            "available_at": measured + pd.to_timedelta(rng.integers(0, 300, 1000), unit="min"),
            "observation_id": [f"row-{i}" for i in range(1000)],
            "value": rng.normal(10, 2, 1000),
        }
    ).sample(frac=1, random_state=24)
    queries = pd.DatetimeIndex(
        [measured[30], measured[0] - pd.Timedelta(days=1), measured[800], measured[1], measured[30]]
    )
    values, ages = _last_available_quality(observations, queries)
    for i, query in enumerate(queries):
        eligible = observations[observations.available_at <= query]
        if eligible.empty:
            assert np.isnan(values[i]) and np.isnan(ages[i])
        else:
            selected = eligible.sort_values("measured_at").iloc[-1]
            assert values[i] == selected.value
            assert ages[i] == (query - selected.measured_at).total_seconds() / 60


def test_state_keeps_source_snapshot_not_whole_history() -> None:
    from global_tests.test_stage7_history_serving import _prepared_data

    data = _prepared_data()
    row = data.quality.iloc[0].copy()
    records = []
    for i in range(1000):
        item = row.copy()
        item["observation_id"] = f"sample-{i}"
        item["measured_at"] = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=i)
        item["available_at"] = item["measured_at"]
        item["value"] = 8.0
        records.append(item)
    data.quality.drop(data.quality.index, inplace=True)
    for key in row.index:
        data.quality[key] = [item[key] for item in records]
    state = build_state(
        data,
        datetime(2026, 1, 15, 9, tzinfo=UTC),
        load_scenario("config/scenarios/history.json"),
        load_runtime_config("config/runtime.toml"),
    )
    snapshot = state.signals[str(row["signal_id"])]
    assert snapshot.selected.id == "sample-999"
    assert len(snapshot.alternatives) < 3


def test_interval_cancellation_keeps_only_completed_points(tmp_path: Path) -> None:
    from global_tests.test_stage7_history_serving import _prepared_data, _write_point_model

    data = _prepared_data()
    dataset = write_prepared_dataset(data, tmp_path / "data")
    model = _write_point_model(tmp_path / "models", data.manifest.tag_dictionary_sha256)
    calls = []
    report = history_interval_command(
        dataset,
        model,
        datetime(2026, 1, 15, 9, tzinfo=UTC),
        datetime(2026, 1, 15, 12, tzinfo=UTC),
        run_dir=tmp_path / "runs",
        progress=lambda done, total: calls.append((done, total)),
        cancelled=lambda: bool(calls and calls[-1][0] == 1),
    )
    assert report["cancelled"] is True
    assert report["requested_points"] == 4 and report["points"] == 1
    assert calls == [(0, 4), (1, 4)]


def test_tradeoff_prefers_lower_model_risk_despite_higher_cost(tmp_path: Path) -> None:
    result = run_model_demo("blend_tradeoff", run_dir=tmp_path)
    assert result.status.value == "recommend"
    assert result.baseline.feasible and result.selected.feasible
    assert result.baseline.rank_key[0] == pytest.approx(0.8)
    assert result.selected.rank_key[0] == pytest.approx(0.73)
    assert result.selected.rank_key[2] > result.baseline.rank_key[2]
    assert "MATERIAL_IMPROVEMENT" in result.reason_codes
