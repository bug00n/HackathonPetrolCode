"""Конфигурация путей к данным (backend, этап 0).

Зависит только от стандартной библиотеки (DESIGN §69): пути задаются
явно через DI, код не зависит от расположения материалов, исходники
никогда не изменяются.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MATERIALS_DIR = PROJECT_ROOT / "materials"


@dataclass(frozen=True)
class DataPaths:
    """Пути ко всем источникам данных этапа 0.

    ``telemetry`` — отображение пространства имён на путь CSV
    (по ТЗ временной ключ — ``date``, служебные ``Unnamed:*`` отбрасываются).
    """

    telemetry: dict[str, Path]
    pak: Path
    lims: Path
    tags: Path


DEFAULT_PATHS = DataPaths(
    telemetry={
        "АВТ": DATA_DIR / "avt_tags.csv",
        "24-2000": DATA_DIR / "242000_tags.csv",
    },
    pak=MATERIALS_DIR / "Выгрузка ПАК 01.01.2023 - н.в_.xlsx",
    lims=MATERIALS_DIR / "ЛИМСы 01.01.2023 - н.в_ (2).xlsx",
    tags=MATERIALS_DIR / "Теги_хакатон.xlsx",
)


__all__ = ["DATA_DIR", "DEFAULT_PATHS", "MATERIALS_DIR", "PROJECT_ROOT", "DataPaths"]
