"""Доступные данные и кэш (DESIGN.md §3: data/state.py).

DataProvider объединяет все источники за одним интерфейсом и кэширует
результаты в памяти. Повторные обращения не перечитывают файлы —
это основа воспроизводимости и экономии времени (~340 МБ CSV).

Пути задаются через DI (``config.DataPaths``): код не зависит от
расположения материалов, исходники никогда не изменяются.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from source.config import DEFAULT_PATHS, DataPaths
from source.contracts import Sample, TagMeta
from source.data.ingest import (
    iter_telemetry_chunks,
    read_lims,
    read_pak,
    telemetry_sample,
)
from source.data.prepare import load_tag_dictionary


class DataProvider:
    """Единая точка доступа к телеметрии, ЛИМС, ПАК и справочнику тегов."""

    def __init__(self, paths: DataPaths | None = None) -> None:
        self.paths = paths or DEFAULT_PATHS
        self._cache: dict[str, object] = {}

    # --- Телеметрия -------------------------------------------------

    def get_telemetry(self, namespace: str) -> pd.DataFrame:
        """Телеметрия целиком (кэшируется). Для 247 МБ используйте iter_telemetry."""
        key = f"telemetry:{namespace}"
        if key not in self._cache:
            frames = list(self.iter_telemetry(namespace, chunksize=100_000))
            self._cache[key] = pd.concat(frames, ignore_index=True)
        assert isinstance(self._cache[key], pd.DataFrame)
        return self._cache[key]

    def iter_telemetry(self, namespace: str, chunksize: int = 50_000) -> Iterator[pd.DataFrame]:
        """Итератор по чанкам телеметрии (без кэша, для потоковой обработки)."""
        return iter_telemetry_chunks(self.paths.telemetry[namespace], namespace, chunksize)

    def telemetry_sample(self, namespace: str, rows: int = 100) -> pd.DataFrame:
        """Быстрый preview телеметрии (без кэша и полной загрузки)."""
        return telemetry_sample(self.paths.telemetry[namespace], namespace, rows=rows)

    # --- Анализы ----------------------------------------------------

    def get_pak(self) -> list[Sample]:
        """Поточные анализаторы (кэшируются)."""
        key = "pak"
        if key not in self._cache:
            self._cache[key] = read_pak(self.paths.pak)
        assert isinstance(self._cache[key], list)
        return self._cache[key]

    def get_lims(self) -> list[Sample]:
        """Лабораторные анализы (кэшируются)."""
        key = "lims"
        if key not in self._cache:
            self._cache[key] = read_lims(self.paths.lims)
        assert isinstance(self._cache[key], list)
        return self._cache[key]

    # --- Справочник ---------------------------------------------------

    def get_tag_dictionary(self) -> dict[str, TagMeta]:
        """Справочник тегов КИП+ПАК (кэшируется)."""
        key = "tags"
        if key not in self._cache:
            self._cache[key] = load_tag_dictionary(self.paths.tags)
        assert isinstance(self._cache[key], dict)
        return self._cache[key]

    # --- Служебное ----------------------------------------------------

    def clear_cache(self) -> None:
        """Сбрасывает кэш (для тестов и повторного чтения изменённых файлов)."""
        self._cache.clear()

    def summary(self) -> dict[str, object]:
        """Краткая сводка по источникам: размеры и диапазоны времени.

        Источники с отсутствующими файлами помечаются как missing
        (CI без LFS-данных).
        """
        result: dict[str, object] = {}
        for namespace, fpath in self.paths.telemetry.items():
            if not Path(fpath).exists():
                result[namespace] = {"missing": True}
                continue
            df = self.get_telemetry(namespace)
            result[namespace] = {
                "rows": len(df),
                "tags": len(df.columns) - 1,
                "from": df["date"].min(),
                "to": df["date"].max(),
            }
        if self.paths.lims.exists():
            lims = self.get_lims()
            result["ЛИМС"] = {
                "samples": len(lims),
                "parameters": len({s.tag for s in lims}),
            }
        else:
            result["ЛИМС"] = {"missing": True}
        if self.paths.pak.exists():
            pak = self.get_pak()
            result["ПАК"] = {
                "samples": len(pak),
                "parameters": len({s.tag for s in pak}),
            }
        else:
            result["ПАК"] = {"missing": True}
        if self.paths.tags.exists():
            result["теги"] = {"verified": len(self.get_tag_dictionary())}
        else:
            result["теги"] = {"missing": True}
        return result


__all__ = ["DataProvider"]
