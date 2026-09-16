# Экспериментальная complex-версия ML v2

Эта ветка (`ml-v2-experimental-complex`) содержит отдельный исследовательский
артефакт. Она не заменяет frozen/shadow v2 и не включает управление реальными
уставками.

## Что добавлено

- среднее ансамблирование HGB и LightGBM;
- две seed-модели на каждую голову (4 члена для point/upper/risk);
- более глубокий budget деревьев: 80 итераций HGB и 180 LightGBM;
- нелинейные признаки только из доступных на `as_of` значений: квадраты,
  произведения и устойчивые отношения PAK/`P8`/`F19`/`T11`;
- те же episode labels, temporal purge, inverse-episode/month weights и Platt
  calibration, что и в v2;
- два отчёта порога: строгий FPR budget 20% и исследовательский budget 22%;
- верхняя граница с калибровкой residual quantile 97.5% (рядом сохраняется
  строгая 95%-граница для сравнения).

Для запуска на CPU rolling/final fit ограничены 5 000 строками на fold с
сохранением положительных эпизодов. Это ограничение фиксируется в отчёте и не
меняет временной purge; при наличии GPU/большего CPU-бюджета его можно поднять
отдельным экспериментом.

Новые признаки не используют ЛИМС, будущие значения или неподтверждённые
сигналы. `P8/F19` остаются только историческим экраном эффекта; `supports_actions`
всегда `false`.

## Запуск

Из корня репозитория:

```powershell
.venv\Scripts\python.exe -m source.main train-v2-experimental `
  --dataset data/processed/a2fe8577752a `
  --false-alarm-budget 0.22 `
  --allow-dirty-experimental
```

Для одной исторической точки:

```powershell
.venv\Scripts\python.exe -m source.main replay-v2-experimental `
  --dataset data/processed/a2fe8577752a `
  --model artifacts/models/sulfur-v2-experimental-<id> `
  --at 2025-06-01T12:00:00+03:00
```

Артефакт получает `production_status=experimental_shadow_only`. UI распознаёт
его по `processing.experimental_variant` и вызывает отдельный replay-путь.

## Как читать результат

В отчёте одновременно сохраняются `threshold_strict`, `threshold_relaxed`,
`audit_2026` и `audit_2026_strict_threshold`. Расслабление бюджета — это
диагностический эксперимент, а не новый safety-gate. Даже если relaxed-метрики
лучше, `promotion_eligible=false`, `supports_actions=false`, а hard feasibility
фильтр проекта не меняется.

Action-effect и LIMS-коррекция не объявлены «почти прошедшими»: текущая action
модель хуже hold, а LIMS-коррекция не достигает MAE/coverage gate. Они остаются
отдельными research-only слоями.
