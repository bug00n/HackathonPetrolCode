# Stage 7: trusted history serving и safety forecast

> Current status: Stage 7 connects a trusted local forecast artifact to the
> history backend path through CLI. It does not add an action model, train a new
> model, or enable industrial setpoint recommendations.

## Purpose

Stage 7 closes the gap between the prepared historical dataset and the model
artifact contract. A user can now run one history forecast cycle from a checked
prepared dataset and an explicitly trusted local model directory, then inspect
the same journal files used by the model-demo flow.

## What Is Implemented

`python -m source.main run-history`:

- loads a prepared dataset from `data/processed/<dataset_id>/` or another
  supplied directory;
- loads a local model artifact only when `--trusted-model` is present;
- verifies model metadata before `joblib` loading, including horizon,
  tag-dictionary hash, target signal and target unit;
- builds leakage-safe serving features in the exact order stored in artifact
  metadata;
- runs the existing `history` scenario through `run_cycle`;
- writes the normal journal files under `runs/` or `--run-dir`;
- prints the strict `Recommendation` JSON.

Example:

```bash
python -m source.main run-history \
  --dataset data/processed/<dataset_id> \
  --model artifacts/models/<model_id> \
  --trusted-model \
  --as-of 2026-01-15T09:00:00Z
```

The command is intentionally explicit about trust because local joblib artifacts
can execute code when loaded. A missing `--trusted-model` flag returns a CLI
error instead of silently loading the artifact.

## Limits

- Stage 7 serves a forecast artifact; it does not create or retrain one.
- A model with `supports_forecast=true` still does not imply
  `supports_actions=true`.
- If the artifact has no uncertainty support, sulfur upper remains unavailable
  and the history scenario can abstain.
- Reliability factors and real setpoint action candidates are still not enabled
  for the history path.
- Desktop UI exposes a forecast-only history screen; model-demo remains the only
  path that can show a synthetic recipe recommendation.

## How To Check

```bash
python -m pytest global_tests/test_stage7_history_serving.py
python -m pytest global_tests/test_stage6_acceptance.py
python -m pytest
python -m ruff check .
python -m mypy source
```
## Constraint-aware safety forecast

## Что реализовано

- отдельная бинарная цель `S(t+60) > 10 mg/kg` поверх прежнего point forecast;
- Logistic Regression, HistGradientBoostingClassifier и LightGBM с весами превышений
  `1, 2, 5, 10, 20` и purged temporal folds;
- для LightGBM зона 8--12 mg/kg получает дополнительный вес `2`;
- Platt calibration на первой половине validation;
- safety-first выбор alarm threshold на второй половине validation;
- gate: `FNR <= 10%` при `FPR <= 20%`;
- composite artifact schema 1.1: point, upper, probability, alarm и applicability;
- joint PCA/Mahalanobis applicability вместо пересечения всех marginal bounds;
- отдельный исследовательский PAK-to-LIMS correction helper;
- action capability требует не менее 100 эпизодов на control, 10% temporal gain,
  coverage 95%, устойчивый знак, shadow replay и одобрение технолога.

## Обучение

```bash
python -m source.main train \
  --dataset data/processed/<dataset_id> \
  --with-uncertainty \
  --with-safety
```

`--with-safety` без `--with-uncertainty` отклоняется. Если validation gate не пройден,
safety artifact не публикуется. Старые schema-1.0 point/upper artifacts остаются
совместимыми fallback-моделями.

## Граница доказательств

Safety alarm улучшает обнаружение риска качества, но не доказывает эффект изменения
уставок. `supports_actions` остаётся `false`, пока отсутствуют подтверждённые пределы
`P8/T11/F19`, достаточные change episodes, shadow replay и пилот с технологом. Hard
constraints продолжают проверяться после ML и не заменяются штрафом в loss.

## Результат на закреплённом dataset `aacc7c1ab3d9`

Лучший sklearn-кандидат -- HistGradientBoostingClassifier с весом превышения `2`.
На второй половине validation он получил `FNR=11.63%` при `FPR=19.93%`, поэтому
не прошёл обязательный gate. На test: `FNR=6.08%`, `FPR=31.90%`, transition recall
`79.11%`, joint applicability `98.78%`. Test не участвовал в выборе.

После этого проверен LightGBM. Он оказался хуже sklearn: validation `FNR=13.22%`
при `FPR=19.92%`, test `FNR=7.61%`, `FPR=32.04%`, transition recall `74.57%`.
LightGBM не продвигается.

Отдельная LIMS-коррекция также не проходит gate: исходный PAK point forecast имеет
test MAE `1.781 mg/kg`, скорректированный -- `2.094 mg/kg`, coverage верхней оценки
`84.98%`. Production LIMS capability не объявляется.

Итог: schema 1.1 и весь safety pipeline реализованы, но новый artifact намеренно
не публикуется. Runtime продолжает использовать frozen fallback. PyTorch и
decision-focused loss откладываются до появления достоверных action outcomes.
