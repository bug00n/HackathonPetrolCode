"""Чтение исходных CSV и Excel (DESIGN.md §3: data/ingest.py).

Источники этапа 0:
- ``data/avt_tags.csv`` и ``data/242000_tags.csv`` — синхронные 10-минутные
  ряды телеметрии (~189 тыс. точек с 01.01.2023); колонки ``Unnamed:*`` —
  служебные индексы (ТЗ), временной ключ — ``date``;
- «Выгрузка ПАК…xlsx» — пары колонок «timestamp — значение», строка 0 —
  теги, строка 1 — единицы;
- «ЛИМСы…xlsx» — 54 пары колонок; строка 0 — секция, 1 — параметр,
  2 — единица, 3 — счётчик (отбрасывается), 4+ — данные.

Каждая пара «дата — значение» разбирается отдельно (требование плана);
исходники никогда не изменяются.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from source.contracts import ReliabilityFlag, Sample, SourceKind
from source.data.prepare import NS_AV, NS_GODT, resolve_unit

# Файлы телеметрии и их пространства имён
TELEMETRY_FILES: dict[str, str] = {
    "avt_tags.csv": NS_AV,
    "242000_tags.csv": NS_GODT,
}

# Префикс пространства имён лабораторных данных
NS_LIMS = "ЛИМС"

# Строки шапки ЛИМС
_ROW_SECTION = 0
_ROW_PARAMETER = 1
_ROW_UNIT = 2
_ROW_COUNT = 3
_ROW_DATA_START = 4


# --- Телеметрия (CSV) -----------------------------------------------


def _is_service_column(name: str) -> bool:
    """Служебные индексные колонки: 'Unnamed: …' (по ТЗ)."""
    return name.strip().startswith("Unnamed:")


def _rename(df: pd.DataFrame, namespace: str) -> pd.DataFrame:
    """Добавляет namespace ко всем колонкам, кроме временного ключа."""
    mapping = {c: f"{namespace}:{c}" for c in df.columns if c != "date"}
    return df.rename(columns=mapping)


def read_telemetry_csv(
    path: str | Path, namespace: str, chunksize: int | None = None
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Читает CSV телеметрии.

    Возвращает DataFrame с колонкой ``date`` (datetime) и тегами с префиксом
    namespace. При chunksize возвращает итератор по чанкам — большой файл
    (247 МБ) не загружается в память целиком.
    """
    header = pd.read_csv(path, nrows=0)
    service = [c for c in header.columns if _is_service_column(c)]
    usecols = [c for c in header.columns if c not in service]

    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, parse_dates=["date"])
    if chunksize is None:
        assert isinstance(reader, pd.DataFrame)
        return _rename(reader, namespace)
    return (_rename(chunk, namespace) for chunk in reader)


def iter_telemetry_chunks(
    path: str | Path, namespace: str, chunksize: int = 50_000
) -> Iterator[pd.DataFrame]:
    """Итератор по чанкам телеметрии (для потоковой обработки и кэша)."""
    header = pd.read_csv(path, nrows=0)
    service = [c for c in header.columns if _is_service_column(c)]
    usecols = [c for c in header.columns if c not in service]
    reader = pd.read_csv(path, usecols=usecols, chunksize=chunksize, parse_dates=["date"])
    return (_rename(chunk, namespace) for chunk in reader)


def telemetry_sample(path: str | Path, namespace: str, rows: int = 100) -> pd.DataFrame:
    """Быстрый preview первых строк телеметрии (для тестов и инспекции)."""
    header = pd.read_csv(path, nrows=0)
    service = [c for c in header.columns if _is_service_column(c)]
    usecols = [c for c in header.columns if c not in service]
    df = pd.read_csv(path, usecols=usecols, nrows=rows, parse_dates=["date"])
    return _rename(df, namespace)


# --- ПАК (Excel) ------------------------------------------------------


def read_pak(path: str | Path) -> list[Sample]:
    """Читает выгрузку ПАК и возвращает плоский список измерений.

    Непустые значения становятся Sample с reliability=RELIABLE; пустые
    ячейки пропускаются: значение без времени несинхронизируемо
    (ТЗ: синхронизация только по времени).
    """
    raw = pd.read_excel(path, header=None)

    samples: list[Sample] = []
    n_cols = raw.shape[1]

    # Пары определяем по факту: тег-колонка — это колонка, у которой
    # в строке 0 непустой тег; её значения — в следующей колонке.
    # (Пары не обязаны быть строго чётными: между ними бывают
    # пустые колонки-разделители.)
    col = 0
    while col < n_cols - 1:
        tag_raw = raw.iloc[0, col]
        if pd.isna(tag_raw):
            col += 1
            continue

        tag = str(tag_raw).strip()
        ts_col, val_col = col, col + 1
        unit = resolve_unit(raw.iloc[1, ts_col])

        # Данные начинаются со строки 2 (строки 0-1 — теги/единицы)
        data = raw.iloc[2:, [ts_col, val_col]].copy()
        data.columns = ["timestamp", "value"]
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data.dropna(subset=["timestamp", "value"])

        for ts, value in data.itertuples(index=False):
            samples.append(
                Sample(
                    tag=tag,
                    source=SourceKind.PAK,
                    unit=unit,
                    timestamp=ts.to_pydatetime(),
                    value=float(value),
                    namespace=NS_GODT,
                    reliability=ReliabilityFlag.RELIABLE,
                )
            )

        col += 2

    return samples


# --- ЛИМС (Excel) -----------------------------------------------------


def normalize_section(section: str) -> str:
    """Нормализует имя секции к короткому пространству имён.

    «Установка 'АВТ'. Точка отбора '2.1'. Продукт '...'» → «АВТ.2.1»,
    «Установка 'Гидроочистка'. Точка отбора '2'...» → «Гидроочистка.2».
    Двойные точки в имени секции (опечатка источника) сглаживаются.
    """
    section = section.strip().rstrip(".").replace("..", ".")
    unit_match = re.search(r"Установка\s+'([^']+)'", section)
    point_match = re.search(r"Точка отбора\s+'([^']+)'", section)
    if not unit_match:
        return section
    namespace = unit_match.group(1)
    if point_match:
        namespace = f"{namespace}.{point_match.group(1)}"
    return namespace


def read_lims(path: str | Path) -> list[Sample]:
    """Читает выгрузку ЛИМС и возвращает плоский список измерений.

    Секция (строка 0) распространяется на соседние колонки группы;
    строка-счётчик «Количество значений:» отбрасывается; тег строится
    как «ЛИМС:<секция>:<параметр>» — уникален благодаря namespace.
    """
    raw = pd.read_excel(path, header=None)

    samples: list[Sample] = []
    n_cols = raw.shape[1]

    col = 0
    current_section = ""
    while col < n_cols - 1:
        # Обновляем текущую секцию, если в этой колонке задано новое имя
        section_raw = raw.iloc[_ROW_SECTION, col]
        if pd.notna(section_raw):
            current_section = str(section_raw)

        param_raw = raw.iloc[_ROW_PARAMETER, col]
        if pd.isna(param_raw):
            col += 1
            continue

        parameter = str(param_raw).strip()
        namespace = normalize_section(current_section)
        tag = f"{NS_LIMS}:{namespace}:{parameter}"
        unit = resolve_unit(raw.iloc[_ROW_UNIT, col])

        ts_col, val_col = col, col + 1
        data = raw.iloc[_ROW_DATA_START:, [ts_col, val_col]].copy()
        data.columns = ["timestamp", "value"]
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
        data["value"] = pd.to_numeric(data["value"], errors="coerce")
        data = data.dropna(subset=["timestamp", "value"])

        for ts, value in data.itertuples(index=False):
            samples.append(
                Sample(
                    tag=tag,
                    source=SourceKind.LIMS,
                    unit=unit,
                    timestamp=ts.to_pydatetime(),
                    value=float(value),
                    namespace=namespace,
                    parameter=parameter,
                    reliability=ReliabilityFlag.RELIABLE,
                )
            )

        col += 2

    return samples


__all__ = [
    "NS_LIMS",
    "TELEMETRY_FILES",
    "iter_telemetry_chunks",
    "normalize_section",
    "read_lims",
    "read_pak",
    "read_telemetry_csv",
    "telemetry_sample",
]
