"""Тесты для точки входа main.py.

Проверяют интерфейс и структуру загрузчика данных без реальных данных.
"""

from __future__ import annotations

import source.main as main_mod


def test_main_does_not_raise() -> None:
    """Заглушка main() не падает при вызове."""
    main_mod.main()
