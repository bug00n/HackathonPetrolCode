# Stage 5: uncertainty, applicability, policy guardrails

## Зачем нужен этап

До Stage 5 backend мог работать с точечным прогнозом серы: модель говорит "ожидаю 9.2 mg/kg", а дальше constraints сравнивают это с лимитом. Проблема в том, что точечный прогноз сам по себе слишком смелый: он не говорит, насколько прогноз надежен, похож ли текущий режим на обучающие данные и можно ли вообще использовать модель сейчас.

Stage 5 добавляет честный слой осторожности:

- empirical upper estimate уровня `0.95`;
- проверку области применимости признаков;
- robustness-отчет по стрессовым случаям;
- materiality/cooldown policy для решения, стоит ли менять рекомендацию.

Это всё еще backend-only. Этап не включает action model, не разрешает управление газом и не делает промышленную оптимизацию уставок.

## Что реализовано

### Uncertainty

Файл `source/ml/uncertainty.py` содержит отдельный слой для верхней оценки прогноза:

- `fit_upper_calibrator()` выбирает upper-модель на первой половине validation и калибрует additive shift на второй половине validation;
- `evaluate_upper_bounds()` считает held-out coverage и среднюю ширину `upper - point`;
- `check_applicability()` отклоняет missing, invalid и out-of-domain features;
- `fit_stage5_uncertainty()` строит uncertainty-capable predictor поверх уже существующей point-модели;
- `save_stage5_model()` сохраняет artifact с `supports_uncertainty=true` и `supports_actions=false`.

Важно: test-часть используется только для финальной оценки качества. Она не участвует в подборе upper-модели или policy.

### Model Artifact

`source/ml/artifacts.py` теперь умеет обслуживать uncertainty-capable artifact:

- `ModelCapabilities.supports_uncertainty`;
- `ModelBundle.predict_upper(features)`;
- `ModelBundle.check_applicability(features)`.

`predict_upper()` проверяет порядок признаков, конечность значений и гарантирует, что `upper >= point`. Если artifact не заявляет uncertainty support, метод падает явно, а не возвращает фиктивную границу.

### Quality-Agent

`source/agents/quality.py` в `history` mode использует Stage 5 осторожно:

- если модель поддерживает uncertainty, сначала вызывается applicability gate;
- missing features дают `FEATURES_UNAVAILABLE`;
- выход за train-only bounds дает `OUT_OF_DOMAIN`;
- если upper недоступен, результат становится unavailable/degraded issue, а не pass;
- constraint по сере использует upper, когда scenario требует верхнюю границу.

### Policy

`source/ml/policy.py` содержит чистые helper-функции без DTO и без CLI:

- `PolicyParameters` задает пороги risk/throughput/cost и cooldown;
- `assess_change_policy()` решает, является ли отличие кандидата от hold существенным;
- `tune_policy()` выбирает policy по validation replay.

Правила:

- risk threshold абсолютный, по умолчанию `0.02`;
- throughput/cost thresholds относительные, по умолчанию `2%`;
- изменение только `change_size` не считается полезным улучшением;
- cooldown подавляет повторную рекомендацию только если baseline/hold feasible;
- если hold нарушает hard constraint, cooldown не скрывает проблему.

### Lazy Facade

`source/ml/__init__.py` экспортирует Stage 5 helper API лениво. Это нужно, чтобы пользователь мог импортировать `source.ml.fit_stage5_uncertainty` или `source.ml.assess_change_policy`, но обычный импорт `source.ml` не тянул тяжелые зависимости раньше времени.

## Поток работы

Логика Stage 5 выглядит так:

```text
prepared dataset
-> supervised features
-> point model artifact
-> fit_stage5_uncertainty()
-> calibrated upper predictor
-> save_stage5_model()
-> quality.predict_quality()
-> ModelBundle.check_applicability()
-> ModelBundle.predict() + predict_upper()
-> check_constraints()
```

Policy flow отдельно:

```text
hold candidate + best feasible candidate
-> active ranking criteria
-> assess_change_policy()
-> allow recommend / keep hold / cooldown
```

На текущем этапе эти helper-функции уже покрыты unit tests. Отдельной публичной CLI-команды Stage 5 нет: внешний контракт проекта пока остается прежним.

## Что не реализовано

- Нет causal/action model.
- Нет реального управления газом.
- Нет разрешенных промышленных setpoint-рекомендаций.
- Нет гарантии safety coverage: `0.95` является эмпирической исторической оценкой, а не промышленной гарантией.
- Applicability bounds являются marginal q0.001/q0.999 по train-признакам, а не полноценной многомерной OOD-моделью.
- `T95`, цетановое число и полный паспорт товарного дизеля остаются `not_assessed`.

## Как проверять

```bash
python -m pytest global_tests/test_stage5_uncertainty_policy.py
python -m pytest global_tests/test_stage4_blending.py
python -m pytest
python -m ruff check .
python -m mypy source
python -m source.main validate-stage0
```

Demo-команды должны сохранить прежние статусы:

```bash
python -m source.main run-model-demo blend_normal
python -m source.main run-model-demo blend_risk
python -m source.main run-model-demo blend_missing
```

Ожидаемо:

- `blend_normal` -> `hold`;
- `blend_risk` -> `recommend`;
- `blend_missing` -> `abstain`.
