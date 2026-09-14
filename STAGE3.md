# Stage 3: Constraints, Controls, Rejection Reasons

> Актуальный статус: guardrails используются и после Stage 5. Model-demo снова
> демонстрирует `hold`/`recommend`/`abstain`, но `recommend` относится только к
> явно синтетическому блендингу; реальные history controls остаются выключены.

Stage 3 состоит из двух связанных частей:

1. backend guardrails для выбора `hold`, `recommend` или `abstain`;
2. ML/control-исследование управляемых воздействий, которое пока не включает реальные
   действия в runtime backend.

Главная мысль этапа: система должна не просто посчитать красивый вариант, а доказать,
что вариант прошёл hard constraints, объяснить причины отбраковки остальных и честно
отказаться от рекомендации, если безопасного действия нет.

## Backend Guardrails

Основная backend-цепочка:

```text
run_cycle -> generate_candidates -> evaluate_candidates -> check_constraints -> Recommendation
```

Внутри этой цепочки теперь есть обязательные правила:

1. Каждый кандидат проходит единый hard-constraint фильтр через `check_constraints()`.
2. Infeasible кандидаты получают `rank_key=None` и не участвуют в выборе лучшего.
3. Причины отбраковки собираются в summary по `reason_code`.
4. Выбранный кандидат повторно проверяется тем же `check_constraints()`.
5. Если повторная проверка отличается от сохранённой оценки, запуск падает с
   технической ошибкой, а не выбирает что-то молча.
6. `hold`, `recommend`, `abstain` учитывают materiality threshold и cooldown.

## Как Выбирается Результат

### `hold`

`hold` означает: текущий режим допустим, и менять его сейчас не нужно.

Backend оставляет `hold`, если:

- baseline проходит hard constraints;
- нет feasible альтернатив;
- или лучшая альтернатива есть, но улучшение меньше materiality threshold;
- или cooldown активен и baseline всё ещё безопасен.

Важно: cooldown не должен скрывать нарушение качества. Если baseline уже infeasible,
система не имеет права сказать `hold` только потому, что недавно была рекомендация.

### `recommend`

`recommend` означает: выбран feasible non-hold кандидат.

Backend рекомендует альтернативу, если:

- baseline infeasible, но есть feasible альтернатива;
- или baseline feasible, но лучшая альтернатива даёт существенное улучшение и cooldown
  не активен.

### `abstain`

`abstain` означает: безопасной рекомендации нет.

Backend отказывается рекомендовать, если:

- baseline infeasible;
- feasible альтернатив нет;
- данные отсутствуют, устарели, конфликтуют или дают unknown checks.

## Где Это Реализовано В Backend

### `source/orchestrator.py`

Главный файл backend-cycle.

- `_select_result()` выбирает `hold`, `recommend` или `abstain`.
- `_has_material_improvement()` сравнивает baseline и кандидата по активным критериям.
- `_cooldown_active()` проверяет, не слишком ли рано повторять рекомендацию.
- `_rejection_summary()` собирает причины отбраковки кандидатов.
- `_recheck_selected()` повторно прогоняет selected через `check_constraints()`.

### `source/agents/optimizer.py`

`evaluate_candidates()` оценивает каждого кандидата и сохраняет:

- agent assessments;
- hard checks;
- `feasible`;
- `rank_key`.

Если кандидат infeasible, `rank_key` остаётся `None`. Такой кандидат не должен случайно
попасть в ranking.

### `source/constraints.py`

Это единственное место hard checks.

Если нужно добавить новое жёсткое ограничение, его нужно добавлять сюда, а не
размазывать проверки по `orchestrator`, `explain`, UI или тестам.

### `source/explain.py`

Explanation показывает:

- текущее значение серы;
- выбранное значение серы;
- статус checks;
- границу ограничения;
- `reason_code`;
- `evidence_ref`, то есть источник границы.

### `source/journal.py`

Журнал дополнительно пишет:

- `selection_reason`;
- `rejection_summary`;
- trace-событие с теми же полями.

`candidates.jsonl` остаётся главным источником всех проверенных кандидатов.

## ML/Controls Часть Stage 3

Отдельная часть stage 3 исследует управляемые воздействия и ранжирование setpoint
кандидатов. Она не должна незаметно превращаться в промышленное управление.

Экспертно выделены три потенциально доступные оператору переменные:

| Тег | Смысл | Статус реального управления |
| --- | --- | --- |
| `ht:P8` | температура ГСС на входе Р-202 | выключено: единица и пределы не подтверждены |
| `ht:T11` | массовый расход сырья | выключено: единица и пределы не подтверждены |
| `ht:F19` | давление на входе Р-202 | выключено: единица и пределы не подтверждены |

На train-периоде `2023-01-01` - `2024-12-31` получены диагностические статистики:

| Тег | q05 | медиана | q95 | медиана abs шага | q95 abs шага |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ht:P8` | 0.01976 | 0.15090 | 0.20537 | 0.000729 | 0.002614 |
| `ht:T11` | 234.12277 | 363.94243 | 380.67846 | 0.213501 | 0.964081 |
| `ht:F19` | 138.69654 | 210.96191 | 250.39071 | 1.876770 | 7.274223 |

Это наблюдавшиеся распределения, а не инженерные допустимые диапазоны. Их нельзя
использовать как разрешение на реальные controls.

## Что Реализовано В ML/Controls Части

- До трёх controls, не более пяти значений на каждый и не более 125 комбинаций плюс
  `hold`; переполнение завершается ошибкой, а не тихим усечением.
- Ограничения `lower`, `upper`, `max_step` проверяются до ранжирования.
- Совместная область проверяется robust Mahalanobis distance, обученной только на
  train-периоде; одних маргинальных min/max недостаточно.
- Линейная модель последствий существует только как прозрачный сценарий. Она считает
  sulfur, upper sulfur, severity proxy, throughput, cost proxy и размер изменения.
- Ранжирование строго использует `(risk, -throughput, cost, change_size, id)` и получает
  только feasible кандидатов.
- `supports_actions` выдаётся отдельным gate: нужны известные единицы и пределы,
  минимум наблюдений эпизодов изменения, лаг 0-180 минут и временной выигрыш над baseline.

## Что Пока Недоступно

- Реальные действия по `P8/T11/F19`: нет подтверждённых единиц и инженерных диапазонов.
- Причинная оценка эффекта: исторический прогноз stage 2 имеет `supports_actions=false`,
  отдельная temporal validation на эпизодах изменений ещё не прошла.
- Runtime backend пока не вызывает stage-3 setpoint API для реальных history controls.
- Stage 3 не подключает промышленную ML-модель, не обучает модель последствий и не
  оптимизирует реальные уставки на сценарии `history`.

Для `history` безопаснее показывать состояние, issues и предупреждения, чем делать вид,
что backend знает эффект реального управляющего действия.

## Как Проверить

Backend guardrails:

```bash
python -m source.main run-model-demo blend_normal
python -m source.main run-model-demo blend_risk
python -m source.main run-model-demo blend_missing
python -m pytest global_tests/test_stage3_guardrails.py
python -m pytest
```

Ожидаемые demo-статусы:

- `blend_normal` -> `hold`;
- `blend_risk` -> `recommend`;
- `blend_missing` -> `abstain`.

ML/control checks, если соответствующие файлы есть в ветке:

```powershell
.\.venv\Scripts\python.exe -m pytest global_tests\test_stage3_controls.py -q
.\.venv\Scripts\ruff.exe check source\ml\controls.py source\ml\action_effects.py global_tests\test_stage3_controls.py
.\.venv\Scripts\mypy.exe source\ml\controls.py source\ml\action_effects.py
```

Полный backend-check:

```bash
python -m ruff format .
python -m ruff check .
python -m mypy source
python -m pytest
python -m source.main validate-stage0
```
