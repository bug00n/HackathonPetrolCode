"""Контракты обмена данными системы Нефтекод (DESIGN.md §3).

Фиксируют контракт между чтением данных (backend) и их использованием
(ML, агенты). Все измерения проходят через :class:`Sample`, который хранит
происхождение значения, единицу и причину недостоверности — по требованиям
ТЗ и этапа 0 плана реализации.

По правилам импортов DESIGN модуль не импортирует другие модули проекта.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ReliabilityFlag(str, Enum):
    """Признак достоверности измерения."""

    RELIABLE = "reliable"  # значение пригодно к использованию
    STALE = "stale"  # значение устарело (возраст анализа велик)
    MISSING = "missing"  # значения нет
    CONFLICTING = "conflicting"  # источники противоречат друг другу
    UNVERIFIED_TAG = "unverified_tag"  # тег не подтверждён справочником
    SERVICE_VALUE = "service_value"  # служебное значение (не измерение)


class Unit(str, Enum):
    """Единицы измерения, встречающиеся в данных."""

    CELSIUS = "°С"
    DENSITY = "кг/м3"
    PERCENT_VOLUME = "% об."
    PERCENT_MASS = "% масс."
    MG_KG = "мг/кг"
    PPM = "ppm"
    CETANE = "ед.цет.ч."
    MM2_S = "мм2/с"
    KPA = "кПа"
    MPA = "МПа"
    TH_M3_H = "тыс.м3/ч"
    M3_H = "м3/ч"
    T_H = "т/ч"
    PERCENT = "%"
    DIMENSIONLESS = "безразмерная"
    UNKNOWN = "unknown"


class SourceKind(str, Enum):
    """Тип источника данных. Приоритет: ЛИМС → ПАК → ВАК → телеметрия."""

    LIMS = "ЛИМС"
    PAK = "ПАК"
    VAK = "ВАК"
    TELEMETRY = "телеметрия"


class Sample(BaseModel):
    """Единичное измерение показателя.

    Хранит всё необходимое для последующей синхронизации по времени
    и оценки свежести: момент измерения, значение, единицу, источник
    и признак достоверности с причиной.
    """

    tag: str = Field(description="Идентификатор тега с пространством имён, напр. 'АВТ:T1'")
    source: SourceKind = Field(description="Тип источника значения")
    unit: Unit = Field(description="Единица измерения")
    timestamp: datetime | None = Field(
        default=None, description="Момент измерения (None — измерения не было)"
    )
    value: float | None = Field(default=None, description="Значение (None — пропуск)")
    measured_at: datetime | None = Field(
        default=None,
        description="Момент, когда значение стало доступно системе (если отличается от timestamp)",
    )
    reliability: ReliabilityFlag = Field(default=ReliabilityFlag.RELIABLE)
    unreliability_reason: str | None = Field(
        default=None,
        description="Причина признания значения недостоверным (при reliability != RELIABLE)",
    )
    namespace: str | None = Field(
        default=None, description="Пространство имён: 'АВТ', '24-2000' или секция ЛИМС"
    )
    parameter: str | None = Field(
        default=None, description="Человекочитаемое название показателя из справочника"
    )

    @model_validator(mode="after")
    def _check_reason(self) -> "Sample":
        if self.reliability is not ReliabilityFlag.RELIABLE and not self.unreliability_reason:
            raise ValueError("unreliability_reason обязателен, когда reliability != RELIABLE")
        return self

    @property
    def is_usable(self) -> bool:
        """Значение пригодно к использованию в расчётах."""
        return (
            self.reliability is ReliabilityFlag.RELIABLE
            and self.value is not None
            and self.timestamp is not None
        )


class TagMeta(BaseModel):
    """Метаданные тега из справочника."""

    tag_id: str = Field(description="Идентификатор тега (как в источнике)")
    description: str = Field(default="", description="Человекочитаемое описание")
    unit: Unit = Field(default=Unit.UNKNOWN, description="Единица измерения")
    namespace: str = Field(description="Пространство имён: 'АВТ' или '24-2000'")
    controllable: bool = Field(
        default=False,
        description="Подтверждённая возможность управления параметром",
    )
    source_kind: SourceKind | None = Field(
        default=None, description="Тип источника, если тег является анализатором"
    )


__all__ = [
    "ReliabilityFlag",
    "Sample",
    "SourceKind",
    "TagMeta",
    "Unit",
]
