# Code Walkthrough

Этот файл объясняет код проекта по папкам и в хронологическом порядке:
что запускается первым, какие файлы подключаются дальше, какие объекты создаются
и зачем вообще нужны отдельные строки.

Если ты пока не понимаешь проект, нормальный порядок чтения такой:

1. `README.md` - что проект умеет сейчас.
2. `PROJECT_GUIDE.md` - общая картина и обязанности backend.
3. `STAGE2.md` - что именно добавил stage 2.
4. `STAGE3.md` - как backend выбирает результат и отбраковывает кандидатов.
5. Этот файл - как код устроен по папкам и строкам.
6. `DESIGN.md` - строгая целевая архитектура.

## 1. Главная хронология программы

Сейчас есть три основные CLI-команды:

```bash
python -m source.main validate-stage0
python -m source.main run-model-demo blend_normal
python -m source.main prepare --materials materials --config config/runtime.toml
python -m source.main build-state --dataset data/processed/<dataset_id> --scenario history --as-of 2025-01-15T10:00:00+03:00
```

В хронологическом виде backend делает так:

```text
source.main
-> читает аргументы CLI
-> загружает config/runtime.toml
-> загружает config/scenarios/*.json
-> загружает config/tags.csv
-> prepare: читает materials и пишет data/processed/<dataset_id>
-> build-state: читает data/processed/<dataset_id>
-> собирает ProcessState на момент as_of
-> run-model-demo: генерирует кандидатов, проверяет constraints и пишет журнал
-> печатает JSON
```

Самая важная идея: код не должен "угадывать" качество. Если данных нет, они
устарели или не подтверждены, это превращается в `Issue`, а не в красивое
придуманное число.

## 2. Корень проекта

В корне лежат документы, настройки инструментов и основные папки.

### `README.md`

Короткая входная точка:

- какие документы читать;
- что реализовано сейчас;
- какие команды запускать;
- что такое stage 2;
- какие данные нельзя коммитить.

README не должен быть огромным учебником. Его задача - быстро сориентировать.

### `PROJECT_GUIDE.md`

Человеческое объяснение проекта:

- зачем проект нужен;
- что уже реализовано;
- что пока не реализовано;
- кто за что отвечает;
- какие правила разработки важны.

Это файл для погружения в смысл, а не для точного описания каждой функции.

### `STAGE2.md`

Документ конкретного этапа.

Он отвечает на вопросы:

- что сделал stage 2;
- какие команды появились;
- что входит в scope;
- чего stage 2 ещё не делает;
- как проверить этап.

### `CODE_WALKTHROUGH.md`

Этот файл. Он объясняет код по папкам и почти построчно.

### `DESIGN.md`

Самый строгий документ. Если `README.md`, `PROJECT_GUIDE.md` или этот файл
расходятся с `DESIGN.md`, правильным считать `DESIGN.md`.

### `IMPLEMENTATION_PLAN.md`

План этапов. Он отвечает не на "как работает эта строка", а на "что делать
дальше по проекту".

### `requirements.txt`

Список Python-зависимостей.

Основные:

- `pandas` - таблицы CSV/Excel;
- `numpy` - численные операции, нужен pandas/ML;
- `pydantic` - строгие DTO-контракты;
- `openpyxl` - чтение Excel;
- `tzdata` - timezone-данные на Windows.

Инструменты разработки:

- `pytest` - тесты;
- `ruff` - форматирование и линтер;
- `mypy` - проверка типов;
- `pre-commit` - хуки перед коммитом.

### `pyproject.toml`

Настройки инструментов.

Блок `[tool.ruff]`:

- `line-length = 100` - максимум 100 символов в строке;
- `target-version = "py311"` - код пишется под Python 3.11.

Блок `[tool.ruff.lint]`:

- `select = ["E", "W", "F", "I", "B"]` - включены базовые ошибки стиля,
  предупреждения, неиспользуемые импорты, сортировка импортов и bugbear-проверки.

Блок `[tool.pytest.ini_options]`:

- `testpaths = ["global_tests"]` - pytest ищет тесты в `global_tests`;
- `pythonpath = ["."]` - корень проекта добавляется в import path;
- `--import-mode=importlib` - более безопасный импорт тестов;
- `-p no:cacheprovider` - pytest не пишет `.pytest_cache`;
- `--basetemp=.test_tmp` - временные файлы тестов уходят в `.test_tmp`.

Блок `[tool.mypy]`:

- `python_version = "3.11"` - типизация под Python 3.11;
- `files = ["source"]` - mypy проверяет production-код;
- `disallow_untyped_defs = true` - функции должны иметь типы;
- `check_untyped_defs = true` - даже частично нетипизированные места проверяются;
- `plugins = ["pydantic.mypy"]` - mypy лучше понимает Pydantic-модели.

### `.gitignore`

Говорит Git, что не коммитить.

Для проекта особенно важны:

- `venv/`, `.venv/` - виртуальные окружения;
- `.test_tmp/`, `.pytest_tmp/`, `.pytest_cache/` - временные файлы тестов;
- `data/processed/` - подготовленные производные данные;
- `artifacts/`, `reports/`, `runs/` - модели, отчёты, журналы;
- `data/*.csv` - распакованная телеметрия из архива.

Оригинальный `materials/data.rar` остаётся источником, а распакованные и
подготовленные данные не должны попадать в Git.

## 3. Папка `config`

`config` хранит не код, а проверяемые настройки.

```text
config/
├── runtime.toml
├── tags.csv
└── scenarios/
    ├── history.json
    ├── blend_normal.json
    ├── blend_risk.json
    └── blend_missing.json
```

### `config/runtime.toml`

Это runtime-настройки. Их читает `load_runtime_config()`, после чего Pydantic
проверяет, что поля соответствуют `RuntimeConfig`.

Почти построчно:

```toml
source_timezone = "Europe/Moscow"
```

Если в исходных CSV/Excel время без timezone, код считает, что это московское
время.

```toml
lims_delay_hours = 6.0
```

Лабораторный анализ ЛИМС не доступен мгновенно. Если он измерен в 12:00, backend
считает его доступным только в 18:00 локального времени.

```toml
horizon_minutes = 60
```

Горизонт будущего прогноза. ML ещё не подключён, но контракт уже держит это
значение.

```toml
seed = 42
```

Фиксированный seed для воспроизводимых экспериментов.

```toml
max_candidates = 125
```

Будущий лимит числа вариантов действий, чтобы оптимизатор не перебирал бесконечно.

```toml
data_dir = "data/processed"
materials_dir = "materials"
tag_dictionary_path = "config/tags.csv"
models_dir = "artifacts/models"
reports_dir = "reports"
runs_dir = "runs"
```

Стандартные пути:

- где лежат подготовленные данные;
- где лежат оригинальные материалы;
- где словарь тегов;
- куда позже будут писаться модели, отчёты и журналы.

```toml
[freshness_minutes]
telemetry = 20
pak = 30
lims = 2880
```

Правила свежести:

- телеметрия считается свежей 20 минут;
- ПАК - 30 минут;
- ЛИМС - 2880 минут, то есть 48 часов.

Если последнее значение старше лимита, `build_state()` всё равно может выбрать
его как последнее известное, но пометит `STALE_REQUIRED_SIGNAL`.

### `config/tags.csv`

Словарь тегов.

Одна строка описывает один сигнал:

- `signal_id` - внутреннее имя, например `ht:2:Mg.Sulfur`;
- `raw_name` - имя в исходнике;
- `stage` - стадия процесса: `avt`, `ht`, `blend`;
- `meaning` - человеческое описание;
- `raw_unit` - единица в исходнике;
- `canonical_unit` - единица в контракте;
- `conversion` - правило конвертации, пока в основном `none`;
- `mapping_status` - подтверждён ли маппинг;
- `controllable` - можно ли считать сигнал управляющим;
- `evidence_ref` - откуда взято подтверждение.

Код не должен молча использовать неподтверждённые теги как надёжные. Если тег
`ambiguous` или `excluded`, это должно отражаться в issues или запрете управления.

### `config/scenarios/history.json`

Сценарий для реальной истории.

Главные поля:

```json
"id": "history"
```

Имя сценария.

```json
"mode": "history"
```

Режим означает: работаем с подготовленными историческими данными, не с
синтетическим блендингом.

```json
"required_signals": ["ht:2:Mg.Sulfur"]
```

Для этого сценария нужен сигнал серы гидроочистки.

```json
"constraints": [...]
```

Ограничения. Сейчас есть ограничение по сере:

- metric `sulfur`;
- stage `ht`;
- upper `10.0`;
- unit `mg/kg`;
- basis `tz`;
- evidence_ref на ТЗ.

```json
"controls": []
```

Управляющие воздействия отключены. Backend пока не советует менять реальные
уставки.

```json
"assumptions": [...]
```

Явное допущение: реальные controls отключены, пока нет валидированной модели
действий.

### `config/scenarios/blend_*.json`

Это синтетические model-demo сценарии.

Они нужны не для реального завода, а чтобы проверить будущий агентный контур на
понятных числах.

`blend_normal`:

- текущая смесь проходит ограничение;
- ожидаемый смысл: `hold`.

`blend_risk`:

- текущая смесь превышает серу;
- система должна найти допустимую рецептуру.

`blend_missing`:

- у компонента не хватает данных по сере;
- правильное поведение: `abstain`, а не угадывание.

## 4. Папка `materials`

Здесь лежат оригинальные материалы задания.

```text
materials/
├── data.rar
├── Выгрузка ПАК 01.01.2023 - н.в_.xlsx
├── ЛИМСы 01.01.2023 - н.в_ (2).xlsx
├── Теги_хакатон.xlsx
├── АВТ_схемы.pdf
├── ТЗ_нефтекод.docx
└── README.md
```

### `data.rar`

Архив телеметрии.

Код ожидает внутри:

```text
data/avt_tags.csv
data/242000_tags.csv
```

`prepare_dataset()` сначала проверяет список файлов в архиве, а потом
распаковывает его во временную папку.

### `Выгрузка ПАК ...xlsx`

Данные ПАК. В коде ПАК читается как пары колонок:

```text
tag/time column + value column
```

ПАК считается доступным сразу в момент измерения:

```text
available_at = measured_at
```

### `ЛИМСы ...xlsx`

Лабораторные анализы. Они тоже читаются парами колонок, но с секциями.

Для ЛИМС код добавляет задержку:

```text
available_at = measured_at + lims_delay_hours
```

Это защищает от утечки будущего: нельзя принимать решение на основании анализа,
который лаборатория ещё не опубликовала.

### `Теги_хакатон.xlsx`

Справочник из задания. Сейчас production-код напрямую в основном использует
`config/tags.csv`, а `source/ml/formulas.py` умеет читать из Excel формулы ВАК
и параметры ЛА.

## 5. Папка `data/processed`

Это не исходный код и не коммитится.

После команды:

```bash
python -m source.main prepare --materials materials --config config/runtime.toml
```

появляется папка:

```text
data/processed/<dataset_id>/
├── manifest.json
├── telemetry.csv.gz
├── quality.csv.gz
├── issues.csv.gz
└── feature_order.json
```

### `manifest.json`

Паспорт датасета:

- какие исходные файлы использовались;
- SHA-256 хеши источников;
- сколько строк получилось;
- какие диапазоны времени;
- какая timezone;
- какая задержка ЛИМС;
- какой `dataset_id`.

### `telemetry.csv.gz`

Широкая таблица телеметрии:

```text
timestamp, avt:..., ht:...
```

Одна строка - один timestamp.

### `quality.csv.gz`

Длинная таблица измерений качества:

```text
observation_id, signal_id, stage, source, measured_at, available_at, value, unit, validity, source_ref
```

Она длинная, потому разные лабораторные/ПАК показатели удобно хранить строками,
а не тысячей колонок.

### `issues.csv.gz`

Проблемы, найденные при подготовке:

- неподтверждённый тег;
- неизвестная единица;
- конфликт источников;
- битое значение;
- битое время.

### `feature_order.json`

Фиксированный порядок признаков. Он нужен, чтобы backend и ML не перепутали
колонки при обучении и применении модели.

## 6. Папка `source`

Это production-код.

```text
source/
├── main.py
├── config.py
├── contracts.py
├── data/
├── agents/
└── ml/
```

## 7. `source/main.py`

Это входная точка CLI. Когда ты пишешь:

```bash
python -m source.main prepare ...
```

Python запускает именно этот файл как модуль.

### Импорты

```python
from __future__ import annotations
```

Позволяет писать типы современным способом и откладывает вычисление аннотаций.

```python
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence
```

Стандартная библиотека:

- `argparse` разбирает CLI-аргументы;
- `json` печатает ответы команд;
- `sys` нужен для stderr и настройки UTF-8 stdout;
- `datetime` парсит `--as-of`;
- `Path` работает с путями;
- `Sequence` типизирует `argv`.

```python
from source.config import load_runtime_config, load_scenario, load_tag_dictionary
from source.contracts import ProcessState, Recommendation
from source.data import build_state, load_prepared_dataset, prepare_dataset, write_prepared_dataset
```

Импорты своего кода:

- config loaders;
- DTO-классы для валидации;
- data-функции stage 2.

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

Находит корень проекта от файла `source/main.py`.

Если `__file__` это `.../source/main.py`, то:

- `.parent` даёт `source/`;
- `.parent.parent` даёт корень проекта.

### `validate_stage0()`

Эта функция проверяет, что базовые контракты и конфиги живые.

```python
load_runtime_config(root / "config/runtime.toml")
```

Проверяет runtime config.

```python
tags = load_tag_dictionary(root / "config/tags.csv")
```

Загружает словарь тегов и потом возвращает `len(tags)`.

```python
scenarios = [load_scenario(path) for path in sorted((root / "config/scenarios").glob("*.json"))]
```

Находит все JSON-сценарии и валидирует каждый через `ScenarioConfig`.

```python
ProcessState.model_validate_json(...)
Recommendation.model_validate_json(...)
```

Проверяет, что контрактные примеры из fixtures соответствуют DTO.

```python
for path in demo_fixtures:
    payload = json.loads(...)
    ProcessState.model_validate(payload["state"])
```

Для каждого demo-fixture проверяет вложенный `state`.

```python
return {"tags": ..., "scenarios": ..., "model_demo_fixtures": ...}
```

Возвращает короткий JSON-отчёт.

### `_resolve_path()`

```python
value = Path(path)
return value if value.is_absolute() else root / value
```

Если пользователь передал абсолютный путь, код оставляет его как есть.
Если относительный, путь считается относительно корня проекта.

Это важно, чтобы команда работала одинаково из тестов и из терминала.

### `_parse_as_of()`

```python
return datetime.fromisoformat(value.replace("Z", "+00:00"))
```

Парсит ISO datetime. Python понимает `+00:00`, но не всегда удобно принимает
`Z`, поэтому `Z` заменяется на `+00:00`.

Если строка неправильная, функция бросает `argparse.ArgumentTypeError`, и CLI
показывает понятную ошибку.

### `prepare_command()`

Это программная часть CLI-команды `prepare`.

```python
config = load_runtime_config(_resolve_path(config_path, root))
```

Читает `config/runtime.toml`.

```python
data = prepare_dataset(_resolve_path(materials, root), config)
```

Запускает настоящую подготовку оригинальных материалов.

```python
output_root = _resolve_path(output if output is not None else config.data_dir, root)
```

Если пользователь передал `--output`, пишем туда. Если нет, используем
`data_dir` из runtime config.

```python
dataset_path = write_prepared_dataset(data, output_root)
```

Записывает prepared dataset на диск.

```python
return {...}
```

Возвращает короткий summary, а не весь огромный датасет.

### `build_state_command()`

Это программная часть CLI-команды `build-state`.

```python
config = load_runtime_config(...)
```

Берёт freshness-лимиты и другие настройки.

```python
scenario_path = Path(scenario)
if not scenario_path.suffix:
    scenario_path = Path("config/scenarios") / f"{scenario}.json"
```

Если пользователь написал `--scenario history`, код превращает это в
`config/scenarios/history.json`.

Если пользователь передал путь к JSON, путь используется напрямую.

```python
scenario_config = load_scenario(...)
data = load_prepared_dataset(...)
return build_state(data, as_of, scenario_config, config)
```

Загружает сценарий, загружает prepared dataset, строит `ProcessState`.

### `main()`

`main()` собирает весь CLI.

```python
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
```

На Windows это помогает нормально печатать русский JSON.

```python
parser = argparse.ArgumentParser(...)
subparsers = parser.add_subparsers(dest="command", required=True)
```

Создаёт CLI и говорит: команда обязательна.

```python
subparsers.add_parser("validate-stage0", ...)
```

Добавляет команду `validate-stage0`.

```python
prepare = subparsers.add_parser("prepare", ...)
prepare.add_argument("--materials", ...)
prepare.add_argument("--config", ...)
prepare.add_argument("--output", ...)
```

Добавляет команду подготовки данных и её параметры.

```python
state = subparsers.add_parser("build-state", ...)
state.add_argument("--dataset", required=True, ...)
state.add_argument("--scenario", default="history", ...)
state.add_argument("--as-of", required=True, type=_parse_as_of, ...)
```

Добавляет команду сборки состояния.

```python
args = parser.parse_args(argv)
```

Разбирает аргументы.

```python
try:
    if args.command == ...
```

Выбирает, какую команду выполнить.

```python
except (FileNotFoundError, ValueError) as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1
```

Если файл не найден или данные невалидны, CLI не падает traceback'ом, а печатает
понятную ошибку и возвращает exit code `1`.

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

Когда файл запускается как `python -m source.main`, вызывается `main()`, а её
числовой результат становится exit code процесса.

## 8. `source/config.py`

Этот файл отвечает только за загрузку конфигов.

### Константы путей

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MATERIALS_DIR = PROJECT_ROOT / "materials"
CONFIG_DIR = PROJECT_ROOT / "config"
```

Фиксирует стандартные пути относительно корня проекта.

### `DataPaths`

```python
@dataclass(frozen=True)
class DataPaths:
```

Небольшой immutable-контейнер с путями к raw-source файлам.

`frozen=True` значит: после создания объект нельзя менять.

### `DEFAULT_PATHS`

Показывает, где по умолчанию лежат:

- telemetry CSV;
- ПАК Excel;
- ЛИМС Excel;
- tags CSV.

Сейчас основная stage-2 команда использует `RuntimeConfig`, но `DEFAULT_PATHS`
сохраняет явное знание о стандартных файлах.

### `load_runtime_config()`

```python
with Path(path).open("rb") as stream:
    return RuntimeConfig.model_validate(tomllib.load(stream))
```

Открывает TOML в бинарном режиме, читает его через `tomllib`, потом отдаёт
словарь в Pydantic-модель `RuntimeConfig`.

Если в TOML лишнее поле или неправильный тип, Pydantic бросит ошибку.

### `load_scenario()`

```python
return ScenarioConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))
```

Читает JSON-сценарий и валидирует его как `ScenarioConfig`.

### `load_tag_dictionary()`

Читает `config/tags.csv`.

Важные строки:

```python
with Path(path).open(encoding="utf-8-sig", newline="") as stream:
```

`utf-8-sig` нужен, чтобы пережить BOM в CSV.

```python
for row in csv.DictReader(stream):
```

Читает CSV как словари: имя колонки -> значение.

```python
row["raw_unit"] = row["raw_unit"] or None
```

Пустая строка в CSV превращается в `None`, потому DTO так ожидает отсутствие
единицы.

```python
row["controllable"] = row["controllable"].strip().lower() == "true"
```

CSV хранит текст, а DTO ждёт boolean. Эта строка делает `"true"` -> `True`.

```python
tag = TagMeta.model_validate(row)
```

Каждая строка проходит строгую проверку.

```python
if tag.raw_name in result:
    raise ValueError(...)
```

Дубликаты raw-name запрещены, чтобы один исходный тег не маппился случайно на
два разных сигнала.

### `config_fingerprint()`

```python
json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
```

Делает стабильную JSON-строку из runtime config. Потом эта строка хешируется и
попадает в manifest. Если config изменился, prepared dataset получает другую
идентичность.

## 9. `source/contracts.py`

Это центральный файл DTO-контрактов. Он отвечает за форму данных между слоями.

### Общая политика

```python
class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, ...)
```

Все DTO:

- запрещают лишние поля;
- immutable после создания;
- валидируют значения по умолчанию.

Это нужно, чтобы ошибка формата всплывала сразу, а не через 10 функций.

### Типы чисел

```python
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
```

Backend запрещает `NaN`, `inf`, отрицательные массы и нулевые горизонты.

### Enum-классы

`Severity`:

- `warning`;
- `blocking`.

`Stage`:

- `avt`;
- `ht`;
- `blend`.

`SourceKind`:

- telemetry;
- lims;
- pak;
- vak;
- scenario.

`Validity`:

- valid;
- missing;
- invalid;
- conflict.

`OperationMode`:

- `history` - реальные исторические prepared data;
- `model_demo` - синтетические сценарии;
- `hybrid` - будущий режим, пока не готов.

Эти enum нужны, чтобы в коде не разъезжались строки вроде `"History"`,
`"history"` и `"hist"`.

### `Issue`

Проблема с данными или расчётом.

Поля:

- `code` - машинный код проблемы;
- `severity` - warning/blocking;
- `signal_id` - к какому сигналу относится;
- `detail` - человеческое описание;
- `source_ref` - ссылка на источник.

### `Observation`

Одно измерение.

Главные поля:

- `measured_at` - когда измерили;
- `available_at` - когда это стало доступно системе;
- `value` - число или `None`;
- `unit` - единица;
- `validity` - можно ли использовать.

Валидатор проверяет:

- `available_at` не раньше `measured_at`;
- `valid` не может иметь `value=None`;
- `missing` не может иметь число.

### `SignalSnapshot`

Состояние одного сигнала на момент `as_of`.

Поля:

- `selected` - выбранное наблюдение;
- `alternatives` - другие доступные наблюдения;
- `age_seconds` - возраст выбранного наблюдения;
- `fresh` - свежее ли оно;
- `issues` - проблемы именно этого сигнала.

### `ProcessState`

Главный объект stage 2.

Он отвечает на вопрос:

```text
что backend знает на момент as_of?
```

Поля:

- `state_id` - hash состояния;
- `as_of` - момент принятия решения;
- `dataset_id` - из какого prepared dataset оно собрано;
- `mode` - history/model_demo/hybrid;
- `signals` - snapshots по required signals;
- `issues` - общие проблемы состояния.

### `RuntimeConfig`, `ScenarioConfig`, `DatasetManifest`

`RuntimeConfig` - настройки запуска.

`ScenarioConfig` - что хотим посчитать и какие ограничения применить.

`DatasetManifest` - паспорт подготовленного датасета.

Эти модели связывают config, data и CLI.

## 10. `source/data/prepare.py`

Этот файл превращает оригинальные материалы в prepared dataset.

### `PREPARATION_VERSION`

```python
PREPARATION_VERSION = "1"
```

Версия алгоритма подготовки. Если логика подготовки поменяется несовместимо,
версию нужно будет поднять.

### `_ARCHIVE_MEMBERS`

```python
_ARCHIVE_MEMBERS = {"data/avt_tags.csv", "data/242000_tags.csv"}
```

Код ожидает строго эти файлы внутри `data.rar`.

Если архив внезапно содержит другое, `prepare_dataset()` не будет молча читать
непонятную структуру.

### `RAW_UNIT_MAP`

Маппинг исходных русских единиц в канонические единицы DTO:

- `°С` -> `degC`;
- `кг/м3` -> `kg/m3`;
- `мг/кг` -> `mg/kg`;
- и так далее.

Если единица неизвестна, она становится `unknown`.

### `PreparedData`

Локальный контейнер, не внешний JSON DTO.

Он хранит:

- telemetry DataFrame;
- quality DataFrame;
- issues DataFrame;
- manifest;
- feature_order.

### `resolve_unit()`

```python
if raw is None or pd.isna(raw):
    return Unit.UNKNOWN.value
return RAW_UNIT_MAP.get(str(raw).strip(), Unit.UNKNOWN).value
```

Если единица пустая или неизвестная, возвращает `unknown`. Код не пытается
угадать.

### `to_utc()`

```python
timestamp = pd.Timestamp(value)
```

Pandas парсит дату из Excel/CSV.

```python
if timestamp.tzinfo is None:
    timestamp = timestamp.tz_localize(ZoneInfo(source_timezone))
```

Если timezone нет, считаем время локальным временем источника.

```python
return timestamp.tz_convert(UTC).to_pydatetime()
```

Внутри проекта всё хранится в UTC.

### `canonical_column()`

Добавляет namespace к raw-тегу.

Пример:

```text
T1 + avt -> avt:T1
```

Если prefix уже есть, второй раз он не добавляется.

### `tag_stage()`

Переводит русские/короткие имена установок в enum `Stage`.

Например:

- `АВТ` или `avt` -> `Stage.AVT`;
- `24-2000`, `Гидроочистка`, `ht` -> `Stage.HYDROTREATMENT`.

### `issue_frame()`

Превращает список `Issue` в таблицу для записи на диск.

### `quality_frame()`

Создаёт таблицу качества в стабильном порядке колонок. Это важно для тестов и
для ML, чтобы формат не плавал.

### `known_feature_order()`

```python
return tuple(sorted({tag.signal_id for tag in tags.values()}))
```

Берёт все `signal_id` из словаря тегов, сортирует и фиксирует порядок.

### `_sha256()`

Читает файл кусками по 1 MB и считает SHA-256.

Это нужно для manifest: если исходник изменился, hash изменится.

### `_source_artifact()`

Создаёт DTO `SourceArtifact`:

- path;
- sha256;
- size_bytes.

Если файл лежит внутри общего root, path записывается относительным. Если нет,
записывается абсолютный path. Это нужно для тестов с временными materials.

### `_extract_telemetry()`

```python
tar -tf archive
```

Смотрит список файлов в архиве.

```python
if members != _ARCHIVE_MEMBERS | {"data"}:
    raise ValueError(...)
```

Проверяет, что архив имеет ожидаемую структуру.

```python
tar -xf archive -C destination
```

Распаковывает архив во временную папку.

### `_deduplicate_quality()`

Группирует observations по:

```text
source, signal_id, measured_at
```

Если дубликаты полностью одинаковые, оставляет один.

Если значения конфликтуют, создаёт observation с:

```text
value = None
validity = conflict
```

и добавляет `Issue` с кодом `SOURCE_CONFLICT`.

### `prepare_dataset()`

Главная функция подготовки.

Хронология внутри:

1. Получает `materials_dir` и `RuntimeConfig`.
2. Формирует пути к `data.rar`, ПАК, ЛИМС и `tags.csv`.
3. Проверяет, что все файлы существуют.
4. Загружает словарь тегов.
5. Создаёт список issues.
6. Распаковывает телеметрию во временную папку.
7. Читает АВТ CSV.
8. Читает гидроочистку CSV.
9. Склеивает telemetry по `timestamp`.
10. Читает ПАК.
11. Читает ЛИМС.
12. Дедуплицирует quality observations.
13. Собирает `quality` DataFrame.
14. Считает hashes исходников и config.
15. Создаёт `dataset_id`.
16. Считает диапазоны времени.
17. Создаёт `DatasetManifest`.
18. Возвращает `PreparedData`.

Самая важная строка по безопасности времени:

```python
read_lims(..., config.lims_delay_hours)
```

Именно тут ЛИМС получает задержку доступности.

### `write_prepared_dataset()`

Пишет prepared dataset на диск.

```python
target = output_root / data.manifest.dataset_id
```

Папка называется по `dataset_id`.

```python
if manifest_path.exists():
```

Если датасет уже был записан, код проверяет, что manifest тот же. Если manifest
другой, это collision и ошибка.

```python
frame.to_csv(temporary, index=False, compression="gzip")
temporary.replace(target / name)
```

Сначала пишет временный файл, потом атомарно заменяет целевой. Это снижает шанс
получить полузаписанный CSV.

```python
feature_order.json
```

Stage 2 дополнительно сохраняет порядок признаков.

### `load_prepared_dataset()`

Обратная операция к writer'у.

Проверяет, что есть:

- `manifest.json`;
- `telemetry.csv.gz`;
- `quality.csv.gz`;
- `issues.csv.gz`;
- `feature_order.json`.

Если чего-то нет, бросает `FileNotFoundError` с понятным списком missing files.

Потом читает CSV/JSON и возвращает `PreparedData`.

## 11. `source/data/ingest.py`

Этот файл читает конкретные форматы источников.

Он ничего не пишет на диск. Его задача - прочитать raw-файл и вернуть:

```text
данные + issues
```

### Константы ЛИМС

```python
NS_LIMS = "ЛИМС"
_ROW_SECTION = 0
_ROW_PARAMETER = 1
_ROW_UNIT = 2
_ROW_DATA_START = 4
```

Говорят, где в Excel находятся:

- строка секции;
- строка параметра;
- строка единицы;
- начало данных.

### `TelemetryRead` и `QualityRead`

Маленькие dataclass-контейнеры.

`TelemetryRead`:

- `frame`;
- `issues`.

`QualityRead`:

- `observations`;
- `issues`.

### `_issue()`

Создаёт warning issue одинаковой формы.

В ingest ошибки чаще warning, потому один плохой тег или значение не должен
ронять весь датасет.

### `_observation_id()`

```python
uuid5(NAMESPACE_URL, source_ref)
```

Создаёт детерминированный ID observation из ссылки на источник.

Один и тот же source_ref всегда даст один и тот же UUID.

### `_is_service_column()`

Определяет служебные колонки pandas, например `Unnamed: 0`.

Такие колонки выбрасываются из telemetry.

### `read_telemetry_csv()`

Хронология:

1. Читает CSV через `pd.read_csv`.
2. Удаляет служебные колонки.
3. Проверяет, что есть колонка `date`.
4. Парсит даты.
5. На плохие даты создаёт `INVALID_TIMESTAMP`.
6. Валидные даты переводит в UTC.
7. Удаляет исходную колонку `date`.
8. Переименовывает raw-колонки в canonical signal IDs.
9. Если тег отсутствует или не confirmed, добавляет `TAG_UNCONFIRMED`.
10. Приводит значения к числам.
11. На плохие числа создаёт `INVALID_VALUE`.
12. Сворачивает дубликаты timestamp.
13. Сортирует по timestamp.

### `_collapse_telemetry_duplicates()`

Если у telemetry есть несколько строк с одинаковым timestamp:

- одинаковые значения схлопываются;
- разные значения превращаются в `pd.NA`;
- создаётся `SOURCE_CONFLICT`.

### `normalize_section()`

Из длинного заголовка ЛИМС достаёт нормальную секцию.

Пример:

```text
Установка 'Гидроочистка'. Точка отбора '2'. Продукт 'ДТ'
-> Гидроочистка.2
```

### `_mapping()`

Проверяет связь raw source field -> internal signal.

Она проверяет:

- есть ли raw-name в `config/tags.csv`;
- confirmed ли mapping;
- совпадает ли stage;
- известна ли единица;
- совпадает ли canonical unit.

Если что-то не так, observation получает `Validity.INVALID`, а issues получают
коды вроде `TAG_UNCONFIRMED`, `TAG_STAGE_MISMATCH`, `UNIT_UNCONFIRMED`.

### `read_pak()`

Читает ПАК Excel.

Логика:

1. Excel читается без header, потому структура ручная.
2. Код идёт по колонкам парами.
3. В первой строке берёт raw tag.
4. Во второй строке берёт unit.
5. Через `_mapping()` проверяет тег и единицу.
6. Начиная с третьей строки читает пары timestamp/value.
7. Плохое время -> `INVALID_TIMESTAMP`.
8. Плохое значение -> `INVALID_VALUE`.
9. Для каждой строки создаёт `Observation`.
10. `available_at = measured_at`, потому ПАК считается доступным сразу.

### `read_lims()`

Читает ЛИМС Excel.

Отличие от ПАК:

- есть секции;
- raw-name собирается как `ЛИМС:<section>:<parameter>`;
- stage берётся из section;
- `available_at = measured_at + lims_delay_hours`.

Это одна из ключевых функций проекта, потому она предотвращает использование
будущих лабораторных данных.

## 12. `source/data/state.py`

Этот файл строит `ProcessState`.

### `_SOURCE_PRIORITY`

```python
_SOURCE_PRIORITY = {
    SourceKind.LIMS: 0,
    SourceKind.PAK: 1,
    ...
}
```

Используется при сортировке кандидатов наблюдений. Чем меньше число, тем выше
приоритет при одинаковом времени.

### `_observation()`

Преобразует строку таблицы `quality` обратно в DTO `Observation`.

Зачем это нужно: после чтения CSV всё является строками/числами pandas, а
дальше backend хочет строгий DTO.

### `build_state()`

Главная функция stage 2 после подготовки данных.

Хронология:

1. Проверяет, что `as_of` timezone-aware.
2. Переводит `as_of` в UTC.
3. Копирует `data.quality`.
4. Приводит `measured_at` и `available_at` к datetime UTC.
5. Оставляет только видимые наблюдения:

```python
visible = frame[(frame["measured_at"] <= as_of) & (frame["available_at"] <= as_of)]
```

Это строка защиты от утечки будущего.

6. Для каждого required signal из scenario ищет candidates.
7. Сортирует candidates по времени и приоритету источника.
8. Выбирает первое usable observation:

```text
validity == valid и value != None
```

9. Если ничего нет, создаёт `MISSING_REQUIRED_SIGNAL`.
10. Если выбранное есть, считает возраст.
11. Сравнивает возраст с `freshness_minutes`.
12. Если старое, создаёт `STALE_REQUIRED_SIGNAL`.
13. Создаёт `SignalSnapshot`.
14. Собирает payload будущего `ProcessState`.
15. Считает стабильный `state_id` как hash payload.
16. Возвращает `ProcessState`.

Важная мысль: `build_state()` не прогнозирует. Оно только честно говорит, что
известно на момент времени.

## 13. `source/data/__init__.py`

Это публичный фасад data-пакета.

Он импортирует функции из `ingest.py`, `prepare.py`, `state.py` и складывает их
в `__all__`.

Зачем это нужно:

```python
from source.data import build_state, prepare_dataset
```

вместо длинных импортов из конкретных файлов.

## 14. Папка `source/agents`

Agents - это будущий слой оценок качества, надёжности и вариантов действий.

После stage 2 основной data-поток идёт через `prepare` и `build_state`. После
stage 3 demo-cycle дополнительно прогоняет кандидатов через constraints, ranking
guardrails и журналирование. На реальной истории orchestrator всё ещё не меняет
промышленные уставки.

### `source/agents/optimizer.py`

Сейчас генерирует candidates.

`generate_candidates()`:

1. Проверяет, что `state.mode == scenario.mode`.
2. Всегда создаёт `hold` candidate.
3. Если режим не `model_demo`, возвращает только `hold`.
4. Если model-demo и есть два blend components, строит сетку рецептур.
5. Каждая рецептура проверяет, что доли суммируются в 1.

Для реальных history-данных это означает: пока никаких реальных управляющих
воздействий не предлагается.

### `source/agents/quality.py`

Оценивает качество.

Для `history`:

- смотрит `ht:2:Mg.Sulfur`;
- проверяет, что сигнал есть;
- проверяет unit;
- если значение старое, статус `DEGRADED`;
- возвращает `MetricEstimate` с `basis=measured`.

Для `model_demo`:

- вызывает `assess_blend_candidate()`;
- качество считается по синтетической формуле смешения.

Если передать model не `None`, функция сейчас бросит `ValueError`, потому stage 2
не подключает ML-модель.

### `source/agents/reliability.py`

Считает прозрачный индекс тяжести режима, но не вероятность аварии.

`SeverityFactor` описывает один фактор:

- signal_id;
- unit;
- normal_edge;
- model_boundary;
- adverse_direction;
- evidence_ref.

`factor_contribution()` превращает значение фактора в число от 0 до 1.

`assess_confirmed_factors()` берёт подтверждённые факторы из состояния. Если
факторов нет, честно возвращает `RELIABILITY_UNAVAILABLE`.

### `source/agents/effects.py`

Синтетические последствия для model-demo блендинга.

Главные функции:

- `_recipe()` проверяет рецептуру;
- `_weighted()` считает массовое среднее;
- `calculate_blend_metrics()` считает sulfur, risk_index, throughput, cost_proxy,
  change_size;
- `assess_blend_candidate()` возвращает три оценки: quality, reliability,
  optimizer.

Все числа здесь относятся к demo-сценариям, не к реальному заводу.

### `source/agents/__init__.py`

Фасад agent-пакета. Через него можно импортировать публичные функции агентов из
одного места.

## 15. Папка `source/ml`

ML делает другой человек, но в проекте есть небольшой ML-adjacent файл.

### `source/ml/formulas.py`

Он читает справочник `Теги_хакатон.xlsx`.

`load_vak_formulas()`:

- ищет лист `ВАК`;
- читает пары колонок tag/formula;
- возвращает словарь `tag -> formula`;
- формулы не исполняет.

`load_lab_parameters()`:

- ищет лист `ЛА`;
- читает список лабораторных параметров по секциям;
- возвращает `section -> list[str]`.

Почему формулы не исполняются: это зона ML и требует отдельной проверки. Backend
не должен запускать строковые формулы из Excel как код.

## 16. Папка `global_tests`

Тесты лежат отдельно от production-кода.

```text
global_tests/
├── test_stage0.py
├── test_stage1_ml.py
├── test_stage2_data.py
├── test_stage3_guardrails.py
├── test_formula_inventory.py
└── fixtures/
```

### `test_stage0.py`

Проверяет фундамент:

- configs загружаются;
- DTO reject extra fields, naive time, NaN;
- telemetry нормализуется в UTC;
- ПАК и ЛИМС сохраняют временную семантику;
- `build_state()` не видит задержанный ЛИМС раньше `available_at`.

### `test_stage2_data.py`

Проверяет новый stage 2.

`test_prepared_dataset_roundtrip()`:

- создаёт `PreparedData` из маленьких fixtures;
- пишет его на диск;
- читает обратно через `load_prepared_dataset()`;
- сравнивает manifest, feature_order и таблицы.

`test_prepare_and_build_state_cli_on_small_materials()`:

- создаёт временный маленький `materials/`;
- пишет туда mini `data.rar`, ПАК Excel и ЛИМС Excel;
- запускает `main(["prepare", ...])`;
- проверяет, что появился prepared dataset;
- запускает `main(["build-state", ...])`;
- проверяет валидный `ProcessState`.

`test_build_state_cli_reports_missing_dataset()`:

- запускает `build-state` с несуществующим dataset;
- ожидает exit code `1`;
- проверяет понятную ошибку.

### `test_stage3_guardrails.py`

Проверяет новый stage 3.

Главные сценарии:

- feasible baseline остаётся `hold`;
- materiality threshold не даёт рекомендовать слишком маленькое улучшение;
- cooldown подавляет повторную рекомендацию, когда baseline безопасен;
- cooldown не скрывает нарушение качества;
- journal сохраняет `selection_reason` и `rejection_summary`;
- selected повторно проверяется через `check_constraints()`.

### `fixtures`

Маленькие данные для тестов.

Они нужны, чтобы CI и локальные тесты не зависели от больших оригинальных
материалов.

## 17. Хронология команды `prepare`

Команда:

```bash
python -m source.main prepare --materials materials --config config/runtime.toml
```

Последовательность:

1. Python запускает `source/main.py`.
2. `main()` создаёт parser.
3. `argparse` видит команду `prepare`.
4. `main()` вызывает `prepare_command(args.materials, args.config, args.output)`.
5. `prepare_command()` загружает `RuntimeConfig`.
6. `prepare_command()` вызывает `prepare_dataset()`.
7. `prepare_dataset()` проверяет наличие исходных файлов.
8. `prepare_dataset()` загружает `config/tags.csv`.
9. `prepare_dataset()` распаковывает `materials/data.rar`.
10. `read_telemetry_csv()` читает АВТ.
11. `read_telemetry_csv()` читает гидроочистку.
12. `read_pak()` читает ПАК.
13. `read_lims()` читает ЛИМС с задержкой доступности.
14. `_deduplicate_quality()` убирает дубли и фиксирует конфликты.
15. `DatasetManifest` получает hashes и row counts.
16. `write_prepared_dataset()` пишет файлы в `data/processed/<dataset_id>`.
17. CLI печатает короткий JSON-summary.

## 18. Хронология команды `build-state`

Команда:

```bash
python -m source.main build-state --dataset data/processed/d175aedffaba --scenario history --as-of 2026-08-11T13:00:00Z
```

Последовательность:

1. Python запускает `source/main.py`.
2. `argparse` видит команду `build-state`.
3. `_parse_as_of()` превращает строку в timezone-aware `datetime`.
4. `build_state_command()` загружает runtime config.
5. `build_state_command()` превращает `history` в `config/scenarios/history.json`.
6. `load_scenario()` валидирует сценарий.
7. `load_prepared_dataset()` читает prepared dataset.
8. `build_state()` фильтрует quality observations по `measured_at <= as_of` и
   `available_at <= as_of`.
9. `build_state()` выбирает последнее валидное значение серы.
10. `build_state()` проверяет свежесть по `freshness_minutes`.
11. `ProcessState` валидируется Pydantic.
12. CLI печатает полный JSON состояния.

Почему JSON большой: `SignalSnapshot.alternatives` содержит другие доступные
наблюдения сигнала. Это честно по текущему DTO, но позже можно добавить CLI-режим
`--summary`, чтобы печатать короткий результат.

## 19. Где сейчас граница backend и ML

Backend сейчас отвечает за:

- чтение и нормализацию исходников;
- timestamps и `available_at`;
- DTO-контракты;
- manifest и воспроизводимость;
- сборку `ProcessState`;
- отказ от выдуманных данных.

ML сейчас не реализуется в этой ветке.

ML позже должен получить:

- `data/processed/<dataset_id>`;
- `feature_order.json`;
- `manifest.json`;
- понятный `ProcessState`;
- стабильные DTO из `source/contracts.py`.

## 20. Что делать, если ты потерялся

Если непонятно, куда идти по коду, начинай с команды:

```bash
python -m source.main prepare --materials materials --config config/runtime.toml
```

Потом открой:

1. `source/main.py`;
2. `source/data/prepare.py`;
3. `source/data/ingest.py`;
4. `source/data/state.py`;
5. `source/contracts.py`.

И держи в голове одну цепочку:

```text
raw materials -> prepared dataset -> ProcessState -> future ML/recommendation
```

На текущей стадии самая ценная работа backend - сделать эту цепочку честной,
проверяемой и воспроизводимой.

## 21. Что изменил Stage 3 в коде

Stage 3 касается не подготовки данных, а принятия решения в demo-cycle.

Хронология `run-model-demo` теперь такая:

```text
source.main
-> run_model_demo_command()
-> run_cycle()
-> _build_cycle_state()
-> generate_candidates()
-> evaluate_candidates()
-> check_constraints() для каждого кандидата
-> _select_result()
-> _recheck_selected()
-> build_explanation()
-> write_run_journal()
```

### `source/orchestrator.py`

`run_cycle()` - главный проводник одного запуска.

После Stage 3 он делает не только "сгенерировать и выбрать", а полный безопасный
контур:

1. Собирает `ProcessState`.
2. Создаёт кандидатов через `generate_candidates()`.
3. Оценивает кандидатов через `evaluate_candidates()`.
4. Находит baseline-кандидата `hold`.
5. Выбирает итог через `_select_result()`.
6. Повторно проверяет selected через `_recheck_selected()`.
7. Формирует `Recommendation`.
8. Пишет журнал.

`_select_result()` содержит правила:

- baseline feasible и улучшения нет -> `hold`;
- baseline feasible и улучшение меньше threshold -> `hold`;
- baseline feasible и cooldown активен -> `hold`;
- baseline feasible и улучшение существенное -> `recommend`;
- baseline infeasible и есть feasible альтернатива -> `recommend`;
- baseline infeasible и альтернатив нет -> `abstain`.

`_recheck_selected()` нужен как fail-closed защита. Если selected в момент выбора
считался feasible, но повторная проверка через тот же `check_constraints()` дала
другой результат, backend падает с `ValueError`. Это лучше, чем молча записать
сомнительную рекомендацию.

### `source/constraints.py`

Это единственное место hard checks.

Stage 3 специально держит это правило жёстким: `orchestrator` выбирает между уже
проверенными кандидатами, но не придумывает собственные ограничения.

### `source/journal.py`

В журнал добавлены:

- `selection_reason` - короткая причина выбора;
- `rejection_summary` - сколько кандидатов отлетело по каждому `reason_code`;
- trace-событие с теми же полями.

`candidates.jsonl` остаётся полным списком кандидатов и их checks.

### `global_tests/test_stage3_guardrails.py`

Тесты Stage 3 проверяют именно поведение guardrails:

- feasible baseline остаётся `hold`;
- cooldown подавляет повторную рекомендацию только когда baseline безопасен;
- cooldown не скрывает нарушение качества;
- journal пишет summary причин отбраковки;
- selected повторно проходит constraints.

## 22. Что добавил Stage 4

Stage 4 показывает связанную цепочку от прогноза гидроочистки к модельному
блендингу:

```text
Hydrotreater forecast
-> apply_hydrotreater_forecast()
-> calculate_mass_blend()
-> sulfur_constraint_status()
-> rank_feasible_blends()
```

### `source/ml/blending.py`

`HybridComponentForecast` описывает прогноз серы гидроочищенного компонента. В нём
есть point sulfur, upper sulfur, `source_state_id`, ссылка на входное качество и
диапазон транспортного лага.

`apply_hydrotreater_forecast()` заменяет только один компонент смеси. Сейчас это
компонент `A` в `hybrid_blend`.

`calculate_mass_blend()` считает серу смеси по массовым долям:

```text
S_mix = sum(w_i * S_i)
```

Эта же формула применяется к `upper`, но это не статистический доверительный
интервал смеси. Это консервативная сценарная граница.

`sulfur_constraint_status()` возвращает:

- `pass`, если upper sulfur доступна и не выше лимита;
- `fail`, если нарушена сера или запас компонента;
- `unknown`, если upper sulfur отсутствует.

`rank_feasible_blends()` сначала отбрасывает infeasible рецептуры, потом сортирует
оставшиеся по `(risk, -throughput, cost, change_size, id)`.

### Gas Context

`collect_gas_context()` берёт из `config/tags.csv` газовые теги `ht:F9`, `ht:F22`,
`ht:Q21` и возвращает их как `GasContextSignal`.

Важное ограничение: `GasContextSignal.action_enabled` всегда `False`. Эти сигналы
можно показывать как технологический контекст, но нельзя рекомендовать менять как
уставки, пока не подтверждены единицы, диапазоны и модель эффекта.

### `global_tests/test_stage4_blending.py`

Тесты Stage 4 проверяют:

- замену только нужного компонента прогнозом гидроочистки;
- массовый баланс point/upper sulfur;
- изменение feasible рецептур при изменении качества компонента;
- `UNKNOWN`, если нет upper sulfur;
- stock shortfall;
- запрет невалидных долей;
- context-only статус газовых тегов.

## 23. Что добавил Stage 5

Stage 5 добавляет слой неопределенности и применимости вокруг исторического ML-прогноза.
Он не меняет внешний CLI и не включает реальные управляющие воздействия.

Общий поток:

```text
prepared dataset
-> supervised features
-> point model artifact
-> fit_stage5_uncertainty()
-> calibrated upper predictor
-> save_stage5_model()
-> predict_quality()
-> check_applicability()
-> predict() + predict_upper()
-> check_constraints()
```

### `source/ml/uncertainty.py`

`fit_upper_calibrator()` получает реальные validation targets и несколько вариантов raw
upper-прогнозов. Первая половина validation выбирает модель по pinball loss, вторая
половина считает non-negative shift:

```text
shift = max(0, q95(y - upper_raw))
```

`evaluate_upper_bounds()` проверяет held-out качество upper-границы: coverage и среднюю
ширину `upper - point`.

`check_applicability()` не прогнозирует серу. Он только отвечает, можно ли применять
модель к текущей строке признаков:

- missing/NaN -> `FEATURES_UNAVAILABLE`;
- выход за bounds -> `OUT_OF_DOMAIN`;
- все признаки внутри bounds -> available.

`fit_stage5_uncertainty()` проверяет, что point artifact и dataset используют один и тот же
feature order и одни временные split boundaries. Потом обучает quantile-регрессоры,
калибрует upper, считает test metrics и robustness cases.

`save_stage5_model()` сохраняет artifact так, чтобы metadata явно говорила:

```text
supports_forecast = true
supports_uncertainty = true
supports_actions = false
```

### `source/ml/artifacts.py`

`ModelBundle.predict_upper()` нужен для serving. Он проверяет:

- artifact действительно заявил `supports_uncertainty`;
- feature order совпадает с metadata;
- predictor вернул одно конечное значение на строку;
- upper не ниже point.

`ModelBundle.check_applicability()` берет `feature_bounds` из metadata и вызывает
`source.ml.uncertainty.check_applicability()`.

### `source/agents/quality.py`

В `history` mode quality-agent теперь ведет себя так:

1. Проверяет совместимость target signal/unit/horizon.
2. Если artifact поддерживает uncertainty, вызывает applicability gate.
3. Если gate недоступен или не пройден, возвращает unavailable issue.
4. Считает point forecast.
5. Если доступен uncertainty, считает upper forecast.
6. Возвращает `MetricEstimate` с `interval_kind=EMPIRICAL` и `interval_level=0.95`.

Главная идея: если upper нужен, но его нет, backend не должен делать вид, что качество
прошло constraint.

### `source/ml/policy.py`

Policy helpers работают отдельно от модели качества. Они отвечают на вопрос:

```text
достаточно ли кандидат лучше hold, чтобы разрешить рекомендацию
```

`assess_change_policy()` смотрит на первый отличающийся active criterion. Если отличие
только в `change_size`, это не улучшение. Если есть существенное улучшение, проверяется
cooldown. Cooldown действует только при feasible hold.

`tune_policy()` выбирает параметры на validation replay: сначала минимизирует пропущенные
обязательные изменения, потом лишние действия, потом общее число действий.

### `source/ml/__init__.py`

Фасад лениво экспортирует Stage 5 helper API:

- `fit_stage5_uncertainty`;
- `fit_upper_calibrator`;
- `check_applicability`;
- `evaluate_upper_bounds`;
- `assess_change_policy`;
- `tune_policy`;
- `PolicyParameters`.

Это удобно для внешнего кода и не заставляет импортировать тяжелые ML-зависимости, пока
конкретная функция не запрошена.

### `global_tests/test_stage5_uncertainty_policy.py`

Тесты Stage 5 проверяют:

- раздельный selection/calibration split;
- held-out coverage и width;
- missing/OOD не превращаются в pass;
- robustness cases сохраняют unavailable;
- materiality threshold и first differing criterion;
- cooldown только при feasible hold;
- tuning policy на validation replay;
- интеграцию quality-agent с upper bound;
- доступность Stage 5 helper API через lazy `source.ml` facade.
