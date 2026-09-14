# HackathonPetrolCode

## Документация

- [План реализации](IMPLEMENTATION_PLAN.md) — этапы, сроки и разделение работы между backend и ML.
- [Технический дизайн](DESIGN.md) — архитектура, структура файлов, типы данных, взаимодействие компонентов и проверки.
- [Уточнения Q&A 11.09](QA_CLARIFICATIONS.md) — подтверждённые ответы, открытые вопросы и последствия для backend/ML.
- [Гайд по проекту](PROJECT_GUIDE.md) — объяснение с нуля: суть ТЗ, роли backend/ML, объекты, этапы и рабочий процесс.
- [Stage 1](STAGE1.md) — первый сквозной backend-цикл, demo-сценарии, журнал и ограничения этапа.
- [Stage 2](STAGE2.md) — как оригинальные материалы превращаются в prepared dataset и `ProcessState`.
- [Stage 3](STAGE3.md) — hard constraints, причины отбраковки кандидатов, materiality и cooldown.
- [Stage 4](STAGE4.md) — hybrid chain, model blending и газовые теги как context-only сигналы.
- [Stage 5](STAGE5.md) — uncertainty, applicability, robustness и policy guardrails без action model.
- [Stage 6](STAGE6.md) — чистый запуск, frozen models, исторические метрики и приёмочная демонстрация.
- [Stage 7](STAGE7.md) — safety-first alarm, joint applicability и усиленный action gate.
- [Stage 8](STAGE8.md) — диагностика drift, режимов, ПАК–ЛИМС и устойчивости признаков.
- [Stage 9](STAGE9.md) — эпизодный multi-horizon shadow-прогноз и отложенный LIMS-контроль.
- [Code walkthrough](CODE_WALKTHROUGH.md) — папки, файлы и хронология вызовов почти построчно.
- [ML system design](DESIGN.md#8-ml-неопределённость-и-модель-последствий) — обучение, метрики, анализ ошибок и жизненный цикл модели; общие контракты и данные описаны в том же документе.
- [Материалы задания](materials/README.md) — ТЗ, схемы и исходные данные.

## Текущее состояние проекта

Реализация дошла до Stage 9 и содержит desktop UI, воспроизводимое обучение,
историческую оценку, replay и приёмочную демонстрацию. Промышленная action model не
заявлена; актуальный исполняемый контракт описан ниже.

Read-only диагностика ML:

```bash
python -m source.main diagnose-ml --dataset data/processed/aacc7c1ab3d9
```

Stage 9 добавляет schema-1.2 эпизодный multi-horizon прогноз только в shadow-режиме.
Ответы Q&A 11.09 не подтверждают физический смысл/единицы `P8/T11/F19`, поэтому эти
колонки не считаются разрешёнными действиями и требуют отдельной PAK-only ablation.

Сейчас реализованы:

- этап 0: строгие DTO, конфигурация, словарь сигналов, чтение и нормализация исходников,
  временная семантика, manifest и малые контрактные fixtures;
- stage 1: первый backend-цикл для model-demo сценариев через `run-model-demo`;
- backend-срез stage 2: CLI-команды `prepare` и `build-state`, чтобы оригинальные
  `materials/` можно было превратить в `data/processed/<dataset_id>/` и собрать
  `ProcessState` для сценария `history`;
- backend-срез stage 3: более строгий выбор `hold`/`recommend`/`abstain`, единые
  hard checks, причины отбраковки кандидатов, повторная проверка selected и trace
  в журнале;
- stage 4: модельная связка гидроочистка -> блендинг, массовый баланс рецептур,
  модельные контуры T95/цетанового числа, присадка до 3% и gas context для
  `ht:F9`, `ht:F22`, `ht:Q21` без включения реального управления газом.
- stage 5: empirical upper estimate для прогноза серы, applicability/OOD gate, robustness
  reporting и materiality/cooldown policy helpers без включения action model.
- stage 6: точные версии зависимостей, команды `train`/`evaluate`/`replay`, каталог
  демонстрационных эпизодов, проверка frozen models и экспорт полного журнала. Пять
  сценариев покрывают `hold`, рекомендации по сере/T95/цетановому числу и отказ при
  неполном паспорте компонента.

Полноценной промышленной ML-модели и управления реальными уставками пока нет. Доступен
локальный desktop UI на Python: он запускает существующие model-demo сценарии, показывает
проверки и журнал, но не выдаёт промышленную рекомендацию на реальной истории.

Stage 4 показывает связанную цепочку и модельный блендинг. Газовые теги видны как
технологический контекст, но не становятся action controls: нет подтверждённых единиц,
диапазонов и модели эффекта. В `model_demo` проверяются сера, T95 и цетановое число:
T95 и базовое цетановое число линейно смешиваются как явное допущение, а эффект присадки
берётся из настраиваемой сценарной кривой. Это полный паспорт внутри синтетической модели,
но не подтверждение промышленного соответствия товарному стандарту.

Stage 5 добавляет осторожность вокруг ML-прогноза: если artifact поддерживает uncertainty,
quality-agent сначала проверяет область применимости признаков, потом использует point и
upper sulfur. Missing/OOD/отсутствующий upper не превращаются в pass. Это не action model:
backend по-прежнему не рекомендует реальные setpoint-изменения и не управляет газом.

## Проверка

Нужны Python 3.11 или 3.12, Git LFS и `tar` с поддержкой RAR. Для приёмочного запуска
используется точный набор прямых зависимостей из `requirements.lock.txt`.

```bash
git lfs install
git lfs pull
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m source.main validate-stage0
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy source
```

Ожидаемый результат `validate-stage0`: семь сценариев, пять model-demo fixtures и полный
словарь известных входных тегов. Неизвестные единицы и управляющие параметры помечены
`ambiguous`, все реальные управляющие воздействия отключены.

## Demo Stage 1/3

```bash
python -m source.main run-model-demo blend_normal
python -m source.main run-model-demo blend_risk
python -m source.main run-model-demo blend_t95_risk
python -m source.main run-model-demo blend_cetane_risk
python -m source.main run-model-demo blend_missing
```

Ожидаемые статусы:

- `blend_normal` -> `hold`;
- `blend_risk`, `blend_t95_risk`, `blend_cetane_risk` -> `recommend` с одновременным
  прохождением верхних границ серы/T95 и нижней границы цетанового числа;
- `blend_missing` -> `abstain`: неизвестное качество не превращается в `pass`.

Stage 3 не меняет публичные demo-команды. Он делает внутренний выбор строже:
infeasible-кандидаты не ранжируются, причины отказа сохраняются в журнале, selected
повторно проверяется через `check_constraints()`, а cooldown не скрывает нарушение
качества.

## Desktop UI

Запустить приложение:

```bash
python -m source.ui
```

Можно сразу выбрать стартовый сценарий:

```bash
python -m source.ui --scenario blend_normal
python -m source.ui --scenario blend_risk
python -m source.ui --scenario blend_t95_risk
python -m source.ui --scenario blend_cetane_risk
python -m source.ui --scenario blend_missing
```

Интерфейс сохраняет текущие возможности системы: модельный расчёт рецептуры, доступные
ограничения, экспорт результата, журнал запусков, проверку конфигурации, подготовку данных
и сборку historical state. UI показывает рассчитанные верхние границы серы/T95, нижнюю
границу цетанового числа и долю присадки; модельный характер расчёта остаётся видимым.
Для проверки без дисплея:

```bash
python -m source.ui --smoke --scenario blend_risk
```

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

## Stage 6: обучение, оценка и приёмка

Обучение запускается только из чистого Git worktree; test не участвует в выборе модели:

```bash
python -m source.main train --dataset data/processed/<dataset_id> --with-uncertainty

# Обучить point + upper + calibrated exceedance alarm.
python -m source.main train --dataset data/processed/<dataset_id> --with-uncertainty --with-safety

# Эпизодный прогноз 10/20/30/60 минут; создаёт только shadow-артефакт schema 1.2.
python -m source.main train-v2-shadow --dataset data/processed/<dataset_id>
```

Контракт и ограничения ML v2 описаны в [STAGE9.md](STAGE9.md). Test 2026 служит
только audit-набором; production требует нового shadow-периода.

Сравнить frozen point model с persistence baseline на одинаковых timestamp:

```bash
python -m source.main evaluate --dataset data/processed/<dataset_id> --model artifacts/models/<model_id> --source pak --split test
python -m source.main evaluate --dataset data/processed/<dataset_id> --model artifacts/models/<model_id> --source lims --split test
```

Воспроизвести историческую точку и отдельно прогнать фиксированные модельные эпизоды:

```bash
python -m source.main replay --dataset data/processed/<dataset_id> --model artifacts/models/<model_id> --scenario history --at 2026-01-15T12:00:00+03:00
python -m source.main acceptance --output reports/full-quality-acceptance
python -m source.main verify-model-freeze
```

`config/model_freeze.json` относится к артефактам, обученным на указанном в нём
`training_git_commit`. Модели не коммитятся; для точного воспроизведения нужно обучить их
на этом commit, затем вернуться в финальную ветку и выполнить проверку хешей.

`acceptance` создаёт `summary.json`, каталоги полных запусков и `journals.zip`. Повторный
запуск требует нового output-каталога, поэтому ранее полученное доказательство не
перезаписывается. Один или несколько обычных журналов экспортируются отдельно:

```bash
python -m source.main export-journal --run <run_id> --output reports/journal-export.zip
```

## Данные

Канонический исходник телеметрии — `materials/data.rar` в Git LFS. Распакованные CSV,
подготовленные наборы, модели, отчёты и журналы в Git не добавляются. Внутри приложения
время хранится в UTC; наивные даты источников пока интерпретируются как `Europe/Moscow`,
а ЛИМС считается доступным через 4 часа. Оба значения — проверяемые допущения из
`config/runtime.toml`, не свойства установки.

Полный контракт и границы утверждений описаны в [DESIGN.md](DESIGN.md).

Pytest настроен на локальную временную папку `.test_tmp` и не пишет
`.pytest_cache`, поэтому обычная команда `python -m pytest` не должна зависеть от старых
`.pytest_tmp`/`.pytest_cache` с некорректными правами Windows.
