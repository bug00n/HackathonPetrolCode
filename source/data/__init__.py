"""Слой данных (backend, этап 0): чтение, нормализация, состояние.

Публичное API слоя:
- :class:`DataProvider` — доступ к телеметрии/ЛИМС/ПАК/справочнику;
- читатели из :mod:`source.data.ingest`;
- нормализация и справочник тегов из :mod:`source.data.prepare`.
"""

from source.config import DEFAULT_PATHS, DataPaths
from source.data.ingest import (
    NS_LIMS,
    TELEMETRY_FILES,
    iter_telemetry_chunks,
    normalize_section,
    read_lims,
    read_pak,
    read_telemetry_csv,
    telemetry_sample,
)
from source.data.prepare import (
    NS_AV,
    NS_GODT,
    load_tag_dictionary,
    resolve_unit,
)
from source.data.state import DataProvider

__all__ = [
    "DEFAULT_PATHS",
    "DataPaths",
    "DataProvider",
    "NS_AV",
    "NS_GODT",
    "NS_LIMS",
    "TELEMETRY_FILES",
    "iter_telemetry_chunks",
    "load_tag_dictionary",
    "normalize_section",
    "read_lims",
    "read_pak",
    "read_telemetry_csv",
    "resolve_unit",
    "telemetry_sample",
]
