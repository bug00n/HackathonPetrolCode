# HackathonPetrolCode

Сейчас реализован этап 0 из [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md): строгие DTO,
конфигурация, словарь сигналов, чтение и нормализация исходников, временная семантика,
manifest и малые контрактные fixtures. Цикл рекомендаций, модели, оптимизатор, журнал и UI
начинаются с этапа 1 и пока не реализованы.

## Проверка этапа 0

Нужны Python 3.11, Git LFS и `tar` с поддержкой RAR.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m source.main validate-stage0
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy source
```

Ожидаемый результат `validate-stage0`: четыре сценария, три model-demo fixture и полный
словарь известных входных тегов. Неизвестные единицы и управляющие параметры помечены
`ambiguous`, все реальные управляющие воздействия отключены.

## Данные

Канонический исходник телеметрии — `materials/data.rar` в Git LFS. Распакованные CSV,
подготовленные наборы, модели, отчёты и журналы в Git не добавляются. Внутри приложения
время хранится в UTC; наивные даты источников пока интерпретируются как `Europe/Moscow`,
а ЛИМС считается доступным через 6 часов. Оба значения — проверяемые допущения из
`config/runtime.toml`, не свойства установки.

Полный контракт и границы утверждений описаны в [DESIGN.md](DESIGN.md).
