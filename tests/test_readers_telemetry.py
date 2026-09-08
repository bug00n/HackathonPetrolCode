"""Юнит-тесты чтения телеметрии CSV."""

from datetime import datetime

import pandas as pd
import pytest

from source.readers_telemetry import (
    iter_telemetry_chunks,
    read_telemetry_csv,
    telemetry_sample,
)


@pytest.fixture(scope="module")
def avt_csv(tmp_path_factory):
    """Синтетический CSV АВТ: служебные Unnamed-колонки + date + теги."""
    path = tmp_path_factory.mktemp("telemetry") / "avt_tags.csv"
    df = pd.DataFrame(
        {
            "Unnamed: 0.1": [0, 1, 2],
            "Unnamed: 0": [0, 1, 2],
            "date": ["2023-01-01 00:00:00", "2023-01-01 00:10:00", "2023-01-01 00:20:00"],
            "T1": [130.37, 130.31, 130.25],
            "P2": [3.54, 3.53, 3.52],
            "F3": [68.64, 68.76, 68.88],
        }
    )
    df.to_csv(path, index=False)
    return path


class TestReadTelemetry:
    def test_service_columns_dropped(self, avt_csv):
        df = read_telemetry_csv(avt_csv, "АВТ")
        assert not any(c.startswith("Unnamed") for c in df.columns)

    def test_namespaces_applied(self, avt_csv):
        df = read_telemetry_csv(avt_csv, "АВТ")
        assert {"date", "АВТ:T1", "АВТ:P2", "АВТ:F3"} == set(df.columns)

    def test_dates_parsed(self, avt_csv):
        df = read_telemetry_csv(avt_csv, "АВТ")
        assert df["date"].iloc[0] == datetime(2023, 1, 1, 0, 0)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_values_preserved(self, avt_csv):
        df = read_telemetry_csv(avt_csv, "АВТ")
        assert df["АВТ:T1"].iloc[0] == pytest.approx(130.37)
        assert len(df) == 3

    def test_chunked_reading(self, avt_csv):
        chunks = list(iter_telemetry_chunks(avt_csv, "АВТ", chunksize=2))
        assert len(chunks) == 2
        total = sum(len(c) for c in chunks)
        assert total == 3
        assert all(set(c.columns) == {"date", "АВТ:T1", "АВТ:P2", "АВТ:F3"} for c in chunks)

    def test_sample_preview(self, avt_csv):
        df = telemetry_sample(avt_csv, "АВТ", rows=2)
        assert len(df) == 2
