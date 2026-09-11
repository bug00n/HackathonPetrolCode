# Stage 2: Real Data Preparation

> Актуальный статус: pipeline `prepare`/`build-state` остаётся рабочей границей
> данных. После Stage 5 он может снабжать history-расчёт признаками, но public CLI
> всё ещё не обучает и не запускает historical ML-forecast.

Stage 2 делает первый практический мост от оригинальных материалов к backend-коду.

До этого проект в основном доказывал, что контракты, fixtures и базовые загрузчики
живые. Теперь появляется нормальный CLI-путь:

```text
materials/ -> data/processed/<dataset_id>/ -> ProcessState(history)
```

## Что реализовано

### `prepare`

Команда:

```bash
python -m source.main prepare --materials materials --config config/runtime.toml
```

Что делает:

1. Читает `materials/data.rar`.
2. Читает ПАК Excel.
3. Читает ЛИМС Excel.
4. Нормализует время в UTC.
5. Применяет задержку доступности ЛИМС из `config/runtime.toml`.
6. Собирает `telemetry`, `quality`, `issues`, `manifest`.
7. Пишет результат в `data/processed/<dataset_id>/`.

На выходе печатается JSON:

```json
{
  "dataset_id": "...",
  "dataset_path": "data/processed/...",
  "issues_count": 0,
  "row_counts": {},
  "time_ranges": {}
}
```

`dataset_id` считается из исходных файлов, словаря тегов и runtime config. Это нужно,
чтобы понимать, из каких именно данных получился результат.

### `build-state`

Команда:

```bash
python -m source.main build-state --dataset data/processed/<dataset_id> --scenario history --as-of 2025-01-15T10:00:00+03:00
```

Что делает:

1. Загружает prepared dataset с диска.
2. Загружает `config/scenarios/history.json`.
3. Берёт только наблюдения, которые уже были измерены и доступны к `as_of`.
4. Выбирает последнее валидное значение для обязательных сигналов сценария.
5. Возвращает валидный `ProcessState` JSON.

Если нужного сигнала нет, он конфликтный или устарел, это попадает в `issues`. Stage 2
не подставляет средние значения и не пытается угадать качество.

## Что не реализовано

Stage 2 не обучает ML-модель и не делает прогноз серы.

В этом этапе также нет:

- промышленной рекомендации оператору;
- подбора управляющих воздействий;
- historical ML-forecast в UI;
- оценки экономического эффекта;
- автоматического удаления или изменения оригинальных `materials/`.

Задача stage 2 проще и важнее для фундамента: сделать воспроизводимую границу данных,
на которую потом сможет опереться ML.

## Зачем это нужно

ML-человек не должен писать отдельный загрузчик Excel/CSV в ноутбуке, а backend не
должен получать модель, обученную на непонятно каких данных.

Правильный поток такой:

```text
backend prepare -> prepared dataset + manifest -> ML training -> model artifact -> backend run
```

`manifest.json` фиксирует происхождение данных: исходные файлы, хеши, timezone,
задержку ЛИМС, количество строк и диапазоны времени.

## Проверка

Базовые команды:

```bash
python -m source.main validate-stage0
python -m pytest
python -m ruff check .
python -m mypy source
```

Ручная проверка на оригинальных данных:

```bash
python -m source.main prepare --materials materials --config config/runtime.toml
python -m source.main build-state --dataset data/processed/<dataset_id> --scenario history --as-of <ISO datetime>
```

`<dataset_id>` нужно взять из JSON-ответа команды `prepare`.

## Что коммитить

Коммитить нужно код, тесты и документацию.

Не коммитить:

- `data/processed/`;
- `.test_tmp/`;
- `.pytest_tmp/`;
- `.pytest_cache/`;
- `runs/`;
- распакованные CSV из `materials/data.rar`;
- модели и отчёты, если они не являются маленькими специально подготовленными fixtures.
