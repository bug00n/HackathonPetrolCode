# HackathonPetrolCode

## Документация

- [План реализации](IMPLEMENTATION_PLAN.md) — этапы, сроки и разделение работы между backend и ML.
- [Технический дизайн](DESIGN.md) — архитектура, структура файлов, типы данных, взаимодействие компонентов и проверки.
- [Гайд по проекту](PROJECT_GUIDE.md) — объяснение с нуля: суть ТЗ, роли backend/ML, объекты, этапы и рабочий процесс.
- [Stage 1](STAGE1.md) — первый сквозной backend-цикл, demo-сценарии, журнал и ограничения этапа.
- [Stage 2](STAGE2.md) — как оригинальные материалы превращаются в prepared dataset и `ProcessState`.
- [Stage 3](STAGE3.md) — hard constraints, причины отбраковки кандидатов, materiality и cooldown.
- [Code walkthrough](CODE_WALKTHROUGH.md) — папки, файлы и хронология вызовов почти построчно.
- [ML system design](DESIGN.md#8-ml-неопределённость-и-модель-последствий) — обучение, метрики, анализ ошибок и жизненный цикл модели; общие контракты и данные описаны в том же документе.
- [Материалы задания](materials/README.md) — ТЗ, схемы и исходные данные.

## Текущее состояние проекта

Реализация находится на начальном backend-этапе. Структура и команды в дизайн-документе
описывают целевую систему.

Сейчас реализованы:

- этап 0: строгие DTO, конфигурация, словарь сигналов, чтение и нормализация исходников,
  временная семантика, manifest и малые контрактные fixtures;
- stage 1: первый backend-цикл для model-demo сценариев через `run-model-demo`;
- backend-срез stage 2: CLI-команды `prepare` и `build-state`, чтобы оригинальные
  `materials/` можно было превратить в `data/processed/<dataset_id>/` и собрать
  `ProcessState` для сценария `history`;
- backend-срез stage 3: более строгий выбор `hold`/`recommend`/`abstain`, единые
  hard checks, причины отбраковки кандидатов, повторная проверка selected и trace
  в журнале.

Полноценной ML-модели, промышленного управления реальными уставками и Streamlit UI пока
нет. Текущий backend уже умеет готовить данные и запускать безопасный demo-каркас с
guardrails, но не выдаёт промышленную рекомендацию на реальной истории.

## Проверка

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

## Demo Stage 1/3

```bash
python -m source.main run-model-demo blend_normal
python -m source.main run-model-demo blend_risk
python -m source.main run-model-demo blend_missing
```

Ожидаемые статусы:

- `blend_normal` -> `hold`;
- `blend_risk` -> `recommend`;
- `blend_missing` -> `abstain`.

Stage 3 не меняет публичные demo-команды. Он делает внутренний выбор строже:
infeasible-кандидаты не ранжируются, причины отказа сохраняются в журнале, selected
повторно проверяется через `check_constraints()`, а cooldown не скрывает нарушение
качества.

## Stage 2: подготовка реальных данных

Подготовить оригинальные материалы в локальный производный датасет:

```bash
python -m source.main prepare --materials materials --config config/runtime.toml
```

Команда выводит JSON с `dataset_id`, путём к `data/processed/<dataset_id>`, количеством
строк и числом issues. Производные данные не коммитятся.

Собрать состояние для исторического сценария:

```bash
python -m source.main build-state --dataset data/processed/<dataset_id> --scenario history --as-of 2025-01-15T10:00:00+03:00
```

`build-state` печатает валидный `ProcessState`: только данные, которые были измерены и
доступны к `as_of`. Если нужного сигнала нет или он устарел, это фиксируется в issues,
а не заменяется придуманным значением.

## Данные

Канонический исходник телеметрии — `materials/data.rar` в Git LFS. Распакованные CSV,
подготовленные наборы, модели, отчёты и журналы в Git не добавляются. Внутри приложения
время хранится в UTC; наивные даты источников пока интерпретируются как `Europe/Moscow`,
а ЛИМС считается доступным через 6 часов. Оба значения — проверяемые допущения из
`config/runtime.toml`, не свойства установки.

Полный контракт и границы утверждений описаны в [DESIGN.md](DESIGN.md).

Pytest настроен на локальную временную папку `.test_tmp` и не пишет
`.pytest_cache`, поэтому обычная команда `python -m pytest` не должна зависеть от старых
`.pytest_tmp`/`.pytest_cache` с некорректными правами Windows.
