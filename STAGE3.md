# Stage 3: Constraints, Rejection Reasons, Ranking Guardrails

Stage 3 усиливает уже существующий backend-cycle. Это всё ещё не ML-модель и не
промышленная оптимизация уставок. Цель этапа другая: сделать так, чтобы backend
честно и воспроизводимо объяснял, почему он оставил текущий режим, почему выбрал
альтернативу или почему отказался рекомендовать.

## Что появилось

Главная цепочка осталась прежней:

```text
run_cycle -> generate_candidates -> evaluate_candidates -> check_constraints -> Recommendation
```

Но теперь внутри этой цепочки появились guardrails:

1. Каждый кандидат проходит единый hard-constraint фильтр через `check_constraints()`.
2. Невозможные кандидаты получают `rank_key=None` и не участвуют в выборе лучшего.
3. Причины отбраковки собираются в summary по `reason_code`.
4. Выбранный кандидат повторно проверяется тем же `check_constraints()`.
5. Если повторная проверка отличается от сохранённой оценки, запуск падает с
   технической ошибкой, а не выбирает что-то молча.
6. `hold`, `recommend`, `abstain` учитывают materiality threshold и cooldown.

## Как выбирается результат

### `hold`

`hold` означает: текущий режим допустим, и менять его сейчас не нужно.

Backend оставляет `hold`, если:

- baseline проходит hard constraints;
- нет feasible альтернатив;
- или лучшая альтернатива есть, но улучшение меньше materiality threshold;
- или cooldown активен и baseline всё ещё безопасен.

Важно: cooldown не должен скрывать нарушение качества. Если baseline уже
infeasible, система не имеет права сказать `hold` только потому, что недавно была
рекомендация.

### `recommend`

`recommend` означает: выбран feasible non-hold кандидат.

Backend рекомендует альтернативу, если:

- baseline infeasible, но есть feasible альтернатива;
- или baseline feasible, но лучшая альтернатива даёт существенное улучшение и
  cooldown не активен.

### `abstain`

`abstain` означает: безопасной рекомендации нет.

Backend отказывается рекомендовать, если:

- baseline infeasible;
- feasible альтернатив нет;
- данные отсутствуют, устарели, конфликтуют или дают unknown checks.

## Где это реализовано

### `source/orchestrator.py`

Это главный файл Stage 3.

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

Если кандидат infeasible, `rank_key` остаётся `None`. Это важно: такой кандидат
не должен случайно попасть в ranking.

### `source/constraints.py`

Это единственное место hard checks.

Если нужно добавить новое жёсткое ограничение, его нужно добавлять сюда, а не
размазывать проверки по `orchestrator`, `explain`, UI или тестам.

### `source/explain.py`

Explanation теперь показывает:

- текущее значение серы;
- выбранное значение серы;
- статус checks;
- границу ограничения;
- `reason_code`;
- `evidence_ref`, то есть источник границы.

### `source/journal.py`

Журнал теперь дополнительно пишет:

- `selection_reason`;
- `rejection_summary`;
- trace-событие с теми же полями.

`candidates.jsonl` остаётся главным источником всех проверенных кандидатов.

## Что Stage 3 не делает

Stage 3 не подключает реальную ML-модель.

Stage 3 не обучает модель последствий.

Stage 3 не оптимизирует реальные промышленные уставки на сценарии `history`.

Для `history` всё ещё безопаснее показывать состояние, issues и предупреждения,
чем делать вид, что backend знает эффект реального управляющего действия.

## Как проверить

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

Полный backend-check:

```bash
python -m ruff format .
python -m ruff check .
python -m mypy source
python -m pytest
python -m source.main validate-stage0
```
