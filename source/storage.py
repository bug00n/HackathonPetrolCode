"""Единый слой доступа к данным (backend, этап 0).

DataProvider объединяет все источники за одним интерфейсом и кэширует
результаты в памяти. Повторные обращения не перечитывают файлы —
это основа воспроизводимости и экономии времени (~340 МБ CSV).

Пути к данным задаются параметрами конструктора: код не зависит от
расположения материалов, исходники никогда не изменяются.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from source.models import Sample, TagMeta
from source.readers_lims import read_lims
from source.readers_pak import read_pak
from source.readers_telemetry import iter_telemetry_chunks
from source.tags import load_tag_dictionary

# Дефолтное расположение данных относительно корня репозитория
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MATERIALS_DIR = PROJECT_ROOT / "materials"

DEFAULT_TELEMETRY = {
    "АВТ": DATA_DIR / "avt_tags.csv",
    "24-2000": DATA_DIR / "242000_tags.csv",
}
DEFAULT_PAK = MATERIALS_DIR / "Выгрузка ПАК 01.01.2023 - н.в_.xlsx"
DEFAULT_LIMS = MATERIALS_DIR / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx"
DEFAULT_TAGS = MATERIALS_DIR / "Теги_хакатон.xlsx"


class DataProvider:
    """Единая точка доступа к телеметрии, ЛИМС, ПАК и справочнику тегов."""

    def __init__(
        self,
        telemetry_paths: dict[str, Path] | None = None,
        pak_path: Path | None = None,
        lims_path: Path | None = None,
        tags_path: Path | None = None,
    ) -> None:
        self.telemetry_paths = telemetry_paths or DEFAULT_TELEMETRY
        self.pak_path = pak_path or DEFAULT_PAK
        self.lims_path = lims_path or DEFAULT_LIMS
        self.tags_path = tags_path or DEFAULT_TAGS
        self._cache: dict[str, object] = {}

    # --- Телеметрия -------------------------------------------------

    def get_telemetry(self, namespace: str) -> pd.DataFrame:
        """Телеметрия целиком (кэшируется). Для 247 МБ используйте iter_telemetry."""
        key = f"telemetry:{namespace}"
        if key not in self._cache:
            frames = [chunk for chunk in self.iter_telemetry(namespace, chunksize=100_000)]
            self._cache[key] = pd.concat(frames, ignore_index=True)
        assert isinstance(self._cache[key], pd.DataFrame)
        return self._cache[key]

    def iter_telemetry(self, namespace: str, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
        """Итератор по чанкам телеметрии (без кэша, для потоковой обработки)."""
        path = self.telemetry_paths[namespace]
        return iter_telemetry_chunks(path, namespace, chunksize=chunksize)

    def telemetry_sample(self, namespace: str, rows: int = 100) -> pd.DataFrame:
        """Быстрый preview телеметрии (без кэша и полной загрузки)."""
        path = self.telemetry_paths[namespace]
        from source.readers_telemetry import telemetry_sample

        return telemetry_sample(path, namespace, rows=rows)

    # --- Анализы ----------------------------------------------------

    def get_pak(self) -> list[Sample]:
        """Поточные анализаторы (кэшируется)."""
        key = "pak"
        if key not in self._cache:
            self._cache[key] = read_pak(self.pak_path)
        assert isinstance(self._cache[key], list)
        return self._cache[key]

    def get_lims(self) -> list[Sample]:
        """Лабораторные анализы (кэшируются)."""
        key = "lims"
        if key not in self._cache:
            self._cache[key] = read_lims(self.lims_path)
        assert isinstance(self._cache[key], list)
        return self._cache[key]

    # --- Справочник ---------------------------------------------------

    def get_tag_dictionary(self) -> dict[str, TagMeta]:
        """Справочник тегов КИП+ПАК (кэшируется)."""
        key = "tags"
        if key not in self._cache:
            self._cache[key] = load_tag_dictionary(self.tags_path)
        assert isinstance(self._cache[key], dict)
        return self._cache[key]

    # --- Служебное ----------------------------------------------------

    def clear_cache(self) -> None:
        """Сбрасывает кэш (для тестов и повторного чтения изменённых файлов)."""
        self._cache.clear()

    def summary(self) -> dict[str, object]:
        """Краткая сводка по источникам: размеры и диапазоны времени.

        Полезна как smoke-проверка готовности данных перед этапом 1.
        """
        result: dict[str, object] = {}
        for namespace in self.telemetry_paths:
            df = self.get_telemetry(namespace)
            result[namespace] = {
                "rows": len(df),
                "tags": len(df.columns) - 1,
                "from": df["date"].min(),
                "to": df["date"].max(),
            }
        lims = self.get_lims()
        result["ЛИМС"] = {
            "samples": len(lims),
            "parameters": len({s.tag for s in lims}),
        }
        pak = self.get_pak()
        result["ПАК"] = {
            "samples": len(pak),
            "parameters": len({s.tag for s in pak}),
        }
        result["теги"] = {"verified": len(self.get_tag_dictionary())}
        return result


__all__ = ["DataProvider"]
