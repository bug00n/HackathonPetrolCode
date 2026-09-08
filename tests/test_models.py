"""Юнит-тесты моделей данных."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from source.models import ReliabilityFlag, Sample, SourceKind, TagMeta, Unit


class TestSample:
    def test_valid_sample(self):
        s = Sample(
            tag="АВТ:T1",
            source=SourceKind.TELEMETRY,
            unit=Unit.CELSIUS,
            timestamp=datetime(2023, 1, 1, 0, 0),
            value=130.37,
            namespace="АВТ",
        )
        assert s.tag == "АВТ:T1"
        assert s.is_usable is True
        assert s.reliability is ReliabilityFlag.RELIABLE

    def test_missing_value_is_not_usable(self):
        s = Sample(
            tag="АВТ:T1",
            source=SourceKind.TELEMETRY,
            unit=Unit.CELSIUS,
            timestamp=datetime(2023, 1, 1),
            value=None,
        )
        assert s.is_usable is False

    def test_missing_timestamp_is_not_usable(self):
        s = Sample(
            tag="АВТ:T1",
            source=SourceKind.TELEMETRY,
            unit=Unit.CELSIUS,
            timestamp=None,
            value=10.0,
        )
        assert s.is_usable is False

    def test_stale_requires_reason(self):
        with pytest.raises(ValidationError, match="unreliability_reason"):
            Sample(
                tag="24-2000:Mg.Sulfur",
                source=SourceKind.PAK,
                unit=Unit.PPM,
                timestamp=datetime(2023, 1, 1),
                value=6.5,
                reliability=ReliabilityFlag.STALE,
            )

    def test_stale_with_reason_is_valid(self):
        s = Sample(
            tag="24-2000:Mg.Sulfur",
            source=SourceKind.PAK,
            unit=Unit.PPM,
            timestamp=datetime(2023, 1, 1),
            value=6.5,
            reliability=ReliabilityFlag.STALE,
            unreliability_reason="возраст анализа 30 суток",
        )
        assert s.is_usable is False
        assert s.unreliability_reason is not None

    def test_reliability_enum_values(self):
        assert {f.value for f in ReliabilityFlag} == {
            "reliable",
            "stale",
            "missing",
            "conflicting",
            "unverified_tag",
            "service_value",
        }

    def test_measured_at_defaults_to_none(self):
        s = Sample(
            tag="ЛИМС:Гидроочистка.1:Mg.Sulfur",
            source=SourceKind.LIMS,
            unit=Unit.MG_KG,
            timestamp=datetime(2023, 5, 1, 14, 0),
            value=5.2,
        )
        assert s.measured_at is None


class TestTagMeta:
    def test_valid_tag_meta(self):
        t = TagMeta(
            tag_id="T1",
            description="Температура верха К1",
            unit=Unit.CELSIUS,
            namespace="АВТ",
        )
        assert t.controllable is False
        assert t.source_kind is None

    def test_controllable_tag(self):
        t = TagMeta(
            tag_id="T6",
            description="Температура",
            unit=Unit.CELSIUS,
            namespace="24-2000",
            controllable=True,
        )
        assert t.controllable is True

    def test_missing_namespace_raises(self):
        with pytest.raises(ValidationError):
            TagMeta(tag_id="T1", description="Без namespace")


class TestUnits:
    def test_units_from_data_present(self):
        # Единицы, реально встречающиеся в файлах выгрузки
        assert Unit("°С").value == "°С"
        assert Unit("кг/м3").value == "кг/м3"
        assert Unit("мг/кг").value == "мг/кг"
        assert Unit("ppm").value == "ppm"
        assert Unit("ед.цет.ч.").value == "ед.цет.ч."
        assert Unit("% масс.").value == "% масс."

    def test_unknown_unit_for_invalid(self):
        with pytest.raises(ValueError):
            Unit("фурлонги")
