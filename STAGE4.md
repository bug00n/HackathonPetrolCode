# Stage 4: Hybrid Chain, Blending, Gas Context

## Результат

Stage 4 демонстрирует связанную цепочку:

```text
history / hydrotreater forecast -> blend component A -> mass-balance recipes -> ranked feasible blends
```

Прогноз серы гидроочищенного компонента заменяет компонент `A` в hybrid/blend
сценарии. После этого рецептуры пересчитываются по массовому балансу. Изменение
point/upper sulfur входного прогноза меняет множество допустимых рецептур.

Газовые теги гидроочистки теперь явно фиксируются как технологический контекст. Это
важно для понимания процесса, но Stage 4 не включает реальное управление газом:
нет подтверждённых единиц, инженерных диапазонов и модели эффекта действия.

## Реализовано

- `HybridComponentForecast` хранит point/upper sulfur, `source_state_id`, ссылку на
  входное качество и явный диапазон транспортного лага.
- `apply_hydrotreater_forecast()` заменяет только соответствующий компонент смеси.
- Сера смеси считается как `sum(w_i * S_i)`.
- Верхние оценки серы смешиваются тем же массовым балансом, но помечаются как
  консервативная сценарная граница без заявления о совместном статистическом покрытии.
- Доли рецептуры неотрицательные, содержат все компоненты и суммируются в единицу.
- Расход компонента проверяется против доступного запаса.
- Недопустимые по сере, uncertainty или запасу рецептуры отбрасываются до ranking.
- Ranking использует только feasible варианты и стабильный ключ
  `(risk, -throughput, cost, change_size, id)`.
- Grid рецептур детерминированный, с явным пределом размера; silent truncation запрещён.
- `GasContextSignal` и `collect_gas_context()` возвращают газовые теги как context-only,
  а не как action controls.

## Gas Context

Stage 4 отслеживает следующие газовые сигналы из `config/tags.csv`:

| Signal | Meaning | Stage 4 status |
| --- | --- | --- |
| `ht:F9` | газ поддува / массовый расход | context only |
| `ht:F22` | газ поддува / связанный технологический сигнал | context only |
| `ht:Q21` | газ поддува | context only |

Для всех этих сигналов:

- `action_enabled=false`;
- причина фиксируется как `context_only`;
- они не попадают в `ScenarioConfig.controls`;
- backend не рекомендует менять их значения.

Если позже эксперт подтвердит единицы, диапазоны, max step и модель эффекта, газ можно
будет рассматривать как отдельный action/control этап. Сейчас это только наблюдаемый
контекст процесса.

## Честные Границы

- Связь качества АВТ с подачей на гидроочистку и лаг `0-180` минут остаются
  модельным предположением: подтверждённого flow mapping нет.
- Реальная история рецептур и запасов не предоставлена, поэтому blending остаётся
  модельным сценарием.
- Проверяются только сера и запас компонента.
- `T95`, цетановое число и полная товарная спецификация не оценены:
  `full_specification_status="not_assessed"`.
- Отсутствующая верхняя граница серы даёт `UNKNOWN`, а не `PASS`.
- Газовые сигналы не являются безопасными уставками.

## Проверка

```powershell
.\.venv\Scripts\python.exe -m pytest global_tests\test_stage4_blending.py -q
.\.venv\Scripts\python.exe -m pytest global_tests\test_stage3_guardrails.py -q
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy source
.\.venv\Scripts\python.exe -m source.main validate-stage0
```

Acceptance-тесты покрывают provenance, массовый баланс point/upper, изменение
допустимого множества рецептур, `UNKNOWN` при отсутствующей upper-границе, запасы,
невалидные доли, ranking только feasible вариантов и context-only статус газовых тегов.
