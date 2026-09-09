"""Точка входа для запуска полного цикла обработки данных."""

from __future__ import annotations

from source.data.state import DataProvider


def main() -> None:
    """Настроить pipeline, выполнить, завершить с очисткой ресурсов.

    Сейчас заглушка: создаёт DataProvider (будущий оркестратор цикла).
    В реализации добавятся: подготовка, обучение, оценка, replay.
    """
    data = DataProvider()
    data.clear_cache()
