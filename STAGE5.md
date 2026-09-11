# Stage 5: Неопределённость, область применимости и устойчивость

## Результат

Исторический прогноз серы дополнен эмпирической верхней оценкой уровня `0.95`.
Quality-agent использует её для ограничения `sulfur <= 10 mg/kg`; если upper или
область применимости недоступны, результат становится `unknown/unavailable`, а не
`pass`.

## Обучение без утечки

1. Point-модель и фиксированные временные границы берутся из строгого Stage-2
   artifact.
2. HGB quantile-кандидаты обучаются только на train с `loss="quantile"`,
   `quantile=0.95` и отключённым внутренним случайным early stopping.
3. Первая половина validation выбирает конфигурацию по pinball loss.
4. Вторая половина validation вычисляет
   `shift=max(0, q95(y - upper_raw))`.
5. Test используется один раз только для coverage и средней ширины.

Artifact сохраняет calibration split, shift, train-only feature bounds,
`supports_uncertainty=true`, `supports_actions=false` и поведение OOD.

## Проверка на полном наборе

Источник: prepared dataset `aacc7c1ab3d9`, PAK sulfur, 189 649 supervised строк,
54 признака, горизонт 60 минут.

| Показатель | Значение |
| --- | ---: |
| Выбранная upper-модель | `hgb_quantile_1` |
| Selection / calibration | 26 277 / 26 277 строк |
| Начало calibration | `2025-07-02T08:30:00Z` |
| Calibration shift | 0.0 mg/kg |
| Test | 31 819 строк |
| Test coverage | 0.96342 |
| Средняя ширина upper − point | 1.94352 mg/kg |
| Доля test внутри feature domain | 0.63362 |
| Coverage внутри feature domain | 0.95987 |
| Средняя ширина внутри domain | 1.96250 mg/kg |
| Coverage при +0.5 mg/kg measurement error | 0.84327 |
| Coverage на последней четверти test | 0.97800 |

Локальный проверенный artifact: `artifacts/models/sulfur-upper-aacc7c1ab3d9`.
Он не коммитится: модели и полные prepared data исключены из Git.

## Admission, OOD и robustness

- `ModelBundle.predict_upper` проверяет feature order, конечность и `upper >= point`.
- Train-only q0.001/q0.999 по каждому признаку задают консервативный marginal
  applicability gate. Пропуск даёт `FEATURES_UNAVAILABLE`, выход — `OUT_OF_DOMAIN`.
- В отчёте отдельно видны nominal test, ошибка измерения `+0.5 mg/kg`, последняя
  четверть test как простой regime-shift slice и факт 4-часовой задержки ЛИМС.
- Резкое падение coverage до 0.84327 при систематической ошибке +0.5 показывает,
  что 0.95 — эмпирическая характеристика истории, не гарантия безопасности.

## Materiality и cooldown

- Сравнивается первый различающийся активный критерий.
- Risk использует абсолютный порог 0.02; throughput и cost — относительные 2%.
- Различие только в `change_size` не считается улучшением.
- Cooldown 60 минут подавляет повторное изменение только при feasible hold.
- Если hold нарушает обязательное ограничение, cooldown и экономический порог не
  блокируют поиск безопасного действия.
- `tune_policy` выбирает параметры только по validation replay, сначала минимизируя
  пропущенные обязательные изменения, затем лишние и общее число действий.

## Что остаётся ограничением

- Upper coverage зависит от исторического распределения и ухудшается при bias или
  новом режиме; для production нужен drift monitor и периодическая recalibration.
- Marginal feature bounds не заменяют полноценную многомерную OOD-модель.
- Текущий строгий gate пропускает только 63.36% test-строк: это безопасное
  воздержание, но слишком высокая доля отказов для production. Нужна отдельная
  калибровка applicability threshold на validation без ослабления missing-data gate.
- Реальные setpoint-рекомендации всё ещё отключены: Stage 5 не создаёт причинную
  action-модель и не исправляет неизвестные единицы/пределы `P8/T11/F19`.
- T95 и цетановое число не имеют валидированных blend/effect моделей, поэтому
  полное соответствие товарного дизеля не заявляется.

## Проверка

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check source global_tests
.\.venv\Scripts\ruff.exe format --check source global_tests
.\.venv\Scripts\mypy.exe source
.\.venv\Scripts\python.exe -m source.main validate-stage0
```
