"""Tests for read-only ML data diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from source.contracts import SourceKind
from source.ml.diagnostics import (
    CONFIRMED_LIMS_OUTLIER_IDS,
    engineered_feature_screen,
    exclude_confirmed_lims_outliers,
    pak_lims_alignment,
    regime_diagnostics,
    temporal_drift,
)
from source.ml.features import SupervisedDataset


def _dataset() -> SupervisedDataset:
    times = pd.to_datetime(
        [
            "2024-01-01T00:00:00Z",
            "2024-06-01T00:00:00Z",
            "2025-01-01T00:00:00Z",
            "2025-06-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ]
    )
    frame = pd.DataFrame(
        {
            "as_of": times,
            "target_at": times + pd.Timedelta(minutes=60),
            "target_available_at": times + pd.Timedelta(minutes=60),
            "y": [9.0, 11.0, 8.0, 12.0, 9.5, 10.5],
            "baseline": [8.0, 9.0, 8.0, 9.0, 9.0, 9.0],
            "trend": [1.0, 2.0, 0.0, 3.0, 0.5, 1.5],
        }
    )
    return SupervisedDataset(
        frame=frame,
        feature_names=("baseline", "trend"),
        target_signal_id="ht:2:Mg.Sulfur",
        target_unit="mg/kg",
        target_source=SourceKind.PAK,
        feature_source=SourceKind.PAK,
        baseline_feature="baseline",
        feature_definition={},
        excluded_counts={},
    )


def test_alignment_keeps_raw_rows_and_labels_sensitivity_exclusions() -> None:
    times = pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"])
    quality = pd.DataFrame(
        {
            "signal_id": ["ht:2:Mg.Sulfur"] * 4,
            "observation_id": [
                "pak-1",
                "pak-2",
                "lims-normal",
                next(iter(CONFIRMED_LIMS_OUTLIER_IDS)),
            ],
            "source": ["pak", "pak", "lims", "lims"],
            "measured_at": [*times, *times],
            "value": [8.0, 9.0, 8.2, 120.0],
        }
    )
    data = SimpleNamespace(quality=quality)

    report = pak_lims_alignment(data, offsets_minutes=(0,))  # type: ignore[arg-type]

    assert report["raw"][0]["n"] == 2
    assert report["training_policy"]["excluded_values"] == [120.0]
    assert "prepared data" in report["training_policy"]["interpretation"]


def test_outlier_policy_uses_confirmed_ids_not_a_blanket_value_cap() -> None:
    confirmed = next(iter(CONFIRMED_LIMS_OUTLIER_IDS))
    frame = pd.DataFrame({"observation_id": [confirmed, "future-real-high"], "y": [120.0, 120.0]})

    eligible, excluded = exclude_confirmed_lims_outliers(frame)

    assert excluded == (confirmed,)
    assert eligible["observation_id"].tolist() == ["future-real-high"]


def test_temporal_drift_reports_transition_counts_per_fixed_period() -> None:
    report = temporal_drift(_dataset())

    assert report["train"]["n"] == 2
    assert report["validation"]["transition_count"] == 1
    assert report["test"]["transition_count"] == 1


def test_engineered_screen_uses_train_association_and_reports_later_signs() -> None:
    report = engineered_feature_screen(_dataset(), top_n=2, min_samples=2)

    by_feature = {row["feature"]: row for row in report}
    assert "trend" in by_feature
    assert set(by_feature["trend"]) >= {
        "train_correlation",
        "validation_correlation",
        "test_correlation",
        "stable_sign",
    }


def test_regime_diagnostics_keeps_near_limit_band_separate() -> None:
    report = regime_diagnostics(_dataset())

    assert report["validation"]["8_to_10"]["transition_count"] == 1
    assert report["test"]["8_to_10"]["transition_count"] == 1
