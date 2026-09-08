"""Чтение телеметрии АВТ и гидроочистки (CSV).

Файлы data/avt_tags.csv и data/242000_tags.csv содержат синхронные
10-минутные ряды (~189 тыс. точек с 01.01.2023). Колонки вида
«Unnamed: …» — служебные индексы (ТЗ), временной ключ — `date`.

Служебные колонки не загружаются вовсе (usecols), остальные теги
получают префикс пространства имён: «АВТ:T1», «24-2000:F1».
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from source.tags import NS_AV, NS_GODT

# Файлы телеметрии и их пространства имён
TELEMETRY_FILES: dict[str, str] = {
    "avt_tags.csv": NS_AV,
    "242000_tags.csv": NS_GODT,
}


def _is_service_column(name: str) -> bool:
    """Служебные индексные колонки: 'Unnamed: …' (по ТЗ)."""
    return name.strip().startswith("Unnamed:")


def read_telemetry_csv(path: str | Path, namespace: str, chunksize: int | None = None):
    """Читает CSV телеметрии.

    Возвращает DataFrame с колонкой `date` (datetime) и тегами с префиксом
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


def _rename(df: pd.DataFrame, namespace: str) -> pd.DataFrame:
    """Добавляет namespace ко всем колонкам, кроме временного ключа."""
    mapping = {c: f"{namespace}:{c}" for c in df.columns if c != "date"}
    return df.rename(columns=mapping)


def iter_telemetry_chunks(path: str | Path, namespace: str, chunksize: int = 50_000):
    """Итератор по чанкам телеметрии (для потоковой обработки и кэша)."""
    return read_telemetry_csv(path, namespace, chunksize=chunksize)


def telemetry_sample(path: str | Path, namespace: str, rows: int = 100) -> pd.DataFrame:
    """Быстрый preview первых строк телеметрии (для тестов и инспекции)."""
    header = pd.read_csv(path, nrows=0)
    service = [c for c in header.columns if _is_service_column(c)]
    usecols = [c for c in header.columns if c not in service]
    df = pd.read_csv(path, usecols=usecols, nrows=rows, parse_dates=["date"])
    return _rename(df, namespace)


__all__ = [
    "TELEMETRY_FILES",
    "iter_telemetry_chunks",
    "read_telemetry_csv",
    "telemetry_sample",
]
