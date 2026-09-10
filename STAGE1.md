# Stage 1: первый сквозной backend-цикл

Этот этап нужен, чтобы превратить подготовленные данные и DTO из stage 0 в
проверяемый цикл принятия решения. ML-модель ещё не обязательна: backend
использует только явные значения из model-demo сценариев и сохраняет одинаковый
контракт для будущих ML-артефактов.

## Что реализовано

- `source/orchestrator.py` запускает один цикл: состояние, кандидаты, оценки,
  ограничения, выбор результата и журнал.
- `source/agents/*` содержит логические роли качества, надёжности и оптимизации.
  Сейчас они считают прозрачные demo-метрики, не обученную модель.
- `source/constraints.py` является единственным местом проверки жёстких
  ограничений: сера, доступность верхней оценки и запас компонентов.
- `source/journal.py` пишет `metadata.json`, `input.json`, `features.json`,
  `trace.jsonl`, `candidates.jsonl` и `result.json`.
- CLI получил команду `run-model-demo`.

## Зачем это нужно

Stage 1 даёт общий исполняемый контур для backend и ML. Backend больше не ждёт
готовую модель, а ML может подставлять реальные `predict_quality`,
`assess_reliability` и `evaluate_candidates` в уже существующий поток и видеть,
какой JSON должен возвращаться в UI и журнал.

## После rebase с dev

В `dev` уже появилась более доменная реализация agent-слоя. Поэтому после rebase
конфликт решён так:

- `effects.py`, `quality.py` и `reliability.py` оставлены в логике `dev`;
- `optimizer.py` дополнен stage-1 функцией `evaluate_candidates`;
- `orchestrator.py`, `constraints.py`, `journal.py`, `explain.py`, CLI и тесты
  сохраняют сквозной backend-цикл.

Текущий flow:

```text
run_cycle -> generate_candidates -> evaluate_candidates -> check_constraints -> Recommendation
```

В `model_demo` каждый кандидат оценивается через `assess_blend_candidate`.
В `history/hybrid` stage 1 не делает вид, что есть модель действий: реальные
управляющие воздействия остаются выключенными до ML-артефакта.

## Запуск

```bash
python -m source.main validate-stage0
python -m source.main run-model-demo blend_normal
python -m source.main run-model-demo blend_risk
python -m source.main run-model-demo blend_missing
```

Тесты stage 1:

```bash
python -m pytest global_tests/test_stage1_cycle.py
```

Полный набор тестов:

```bash
python -m pytest
```

Ожидаемые статусы:

- `blend_normal` -> `hold`;
- `blend_risk` -> `recommend`;
- `blend_missing` -> `abstain`.

## Ограничения этапа

Реальные управляющие воздействия и промышленная оптимизация не включены.
Блендинг является модельным сценарием, а расчёт серы использует массовые доли из
конфига. Экономика и надёжность представлены прозрачными proxy-метриками только
для ранжирования внутри одного сценария.
