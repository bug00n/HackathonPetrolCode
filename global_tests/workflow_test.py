"""Интеграционные (workflow) тесты на реальных данных.

Проверяют сквозную готовность данных этапа 0: все источники читаются,
структуры совпадают с документацией плана, чтение воспроизводимо
(требование ТЗ: повтор одного запуска даёт одинаковые расчёты).

Помечены маркером `real_data`: требуют наличия materials/ и data/.
"""

from collections import Counter

import pytest

from source.models import ReliabilityFlag, SourceKind, Unit
from source.readers_lims import read_lims
from source.readers_pak import read_pak
from source.readers_telemetry import iter_telemetry_chunks, telemetry_sample
from source.storage import DEFAULT_LIMS, DEFAULT_PAK, DataProvider

pytestmark = pytest.mark.real_data


def test_smoke():
    """Заглушка сохранена на случай окружения без данных."""
    assert True


@pytest.mark.skipif(not DEFAULT_PAK.exists(), reason="нет файла выгрузки ПАК")
class TestPakReal:
    def test_counts_match_plan(self):
        samples = read_pak(DEFAULT_PAK)
        cnt = Counter(s.tag for s in samples)
        # Наблюдения плана: ~189 тыс. точек серы, плотность с марта 2025
        assert cnt["24-2000:Mg.Sulfur"] > 180_000
        density = [s for s in samples if s.tag == "24-2000:D15"]
        assert len(density) > 50_000
        assert min(s.timestamp for s in density).year == 2025

    def test_units_and_reliability(self):
        samples = read_pak(DEFAULT_PAK)
        assert all(s.source is SourceKind.PAK for s in samples)
        assert all(s.reliability is ReliabilityFlag.RELIABLE for s in samples)
        sulfur = next(s for s in samples if "Sulfur" in s.tag)
        assert sulfur.unit is Unit.PPM


@pytest.mark.skipif(not DEFAULT_LIMS.exists(), reason="нет файла ЛИМС")
class TestLimsReal:
    def test_counts_match_plan(self):
        samples = read_lims(DEFAULT_LIMS)
        cnt = Counter(s.tag for s in samples)
        # План: 1462 измерения серы ГО, 42 цетанового числа, max серы 2120
        assert cnt["ЛИМС:Гидроочистка.2:Mg.Sulfur"] == 1462
        assert cnt["ЛИМС:Гидроочистка.2:CetaneNumber"] == 42
        sulfur_max = max(s.value for s in samples if s.tag == "ЛИМС:Гидроочистка.2:Mg.Sulfur")
        assert sulfur_max == 2120.0

    def test_sections_and_namespaces(self):
        samples = read_lims(DEFAULT_LIMS)
        namespaces = {s.namespace for s in samples}
        assert namespaces == {
            "АВТ.1",
            "АВТ.2",
            "АВТ.2.1",
            "АВТ.3",
            "Гидроочистка.1",
            "Гидроочистка.2",
        }
        assert all(s.tag.startswith("ЛИМС:") for s in samples)


class TestTelemetryReal:
    def test_avt_preview(self):
        df = telemetry_sample("data/avt_tags.csv", "АВТ", rows=10)
        assert len(df) == 10
        assert "АВТ:T1" in df.columns
        assert not any("Unnamed" in c for c in df.columns)

    def test_242000_row_count(self):
        # План: 189 217 строк телеметрии
        total = sum(
            len(chunk)
            for chunk in iter_telemetry_chunks("data/242000_tags.csv", "24-2000", 100_000)
        )
        assert total == 189_217


@pytest.mark.skipif(not DEFAULT_PAK.exists() or not DEFAULT_LIMS.exists(), reason="нет материалов")
class TestWorkflow:
    def test_data_provider_summary(self):
        provider = DataProvider()
        summary = provider.summary()
        # Телеметрия обоих участков загружена
        assert summary["АВТ"]["rows"] > 180_000
        assert summary["24-2000"]["rows"] == 189_217
        assert summary["ЛИМС"]["samples"] > 30_000
        assert summary["ПАК"]["samples"] > 200_000
        assert summary["теги"]["verified"] == 99

    def test_reproducibility(self):
        """Повторное чтение даёт идентичный результат (ТЗ: воспроизводимость)."""
        first = read_lims(DEFAULT_LIMS)
        second = read_lims(DEFAULT_LIMS)
        assert first == second

    def test_all_samples_usable(self):
        """Все прочитанные измерения проходят проверку пригодности."""
        for sample in read_pak(DEFAULT_PAK)[:1000] + read_lims(DEFAULT_LIMS)[:1000]:
            assert sample.is_usable
            assert sample.unreliability_reason is None
