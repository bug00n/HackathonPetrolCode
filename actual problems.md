# Актуальные проблемы проекта

Последнее обновление: 2026-09-21.

Это постоянный журнал найденных ошибок, расхождений с ТЗ и планом,
существенных ограничений демонстрации и незавершённых проверок.
Запись в журнале сама по себе не означает, что проблема исправлена.

## Правила ведения

- Добавлять новые существенные находки по мере работы, а не оставлять их только в чате.
- Сохранять идентификаторы записей; дополнять существующую запись вместо создания дубля.
- Разделять подтверждённые ошибки, ограничения и предположения, требующие проверки.
- Для ошибки указывать условие воспроизведения, фактическое и ожидаемое поведение,
  затронутый код и критерий закрытия.
- После исправления обновлять статус, дату и результат проверки. Закрытые записи
  сохранять, чтобы оставалась история решений.
- Прохождение общих тестов не считать подтверждением исправления конкретной ошибки
  без проверки её исходного случая.

Приоритеты: `P1` - исправить до демонстрации; `P2` - существенная проблема или пробел,
который нужно устранить либо явно учесть при защите.

## Реестр

| ID | Приоритет | Тип | Статус | Проблема |
| --- | --- | --- | --- | --- |
| AP-001 | P1 | Ошибка | Закрыта 21.09 | Отсутствующее значение T95/цетанового числа превращается в ноль |
| AP-002 | P1 | Пробел проверок | Закрыта 21.09 | T95 и цетановое число не ограничивают допустимость рецептуры |
| AP-003 | P2 | Расхождение с планом | Закрыта 21.09 | History UI не показывает полноценный результат прогноза |
| AP-004 | P2 | Ошибка UI | Исправлена, UI-проверка ожидается | Завершение history-расчёта после смены вкладки вызывает TclError |
| AP-005 | P2 | Ошибка оценки | Открыта | Validation-отчёт оценивает artifact на данных его финального обучения |
| AP-006 | P2 | Ошибка объяснения | Закрыта 21.09 | Recommend ошибочно объясняется нарушением ограничений |
| AP-007 | P2 | Ограничение демо | Открыта | Сценарии слабо демонстрируют конфликт критериев оптимизации |
| AP-008 | P2 | Пробел проверки | Требует проверки | Качество ML на полном реальном датасете не подтверждено этой ревизией |

## AP-001. Пропуски свойств превращаются в ноль

Обнаружено и воспроизведено: 2026-09-14.

Код: [source/agents/effects.py](source/agents/effects.py), `_weighted_metric()`;
для сравнения: [source/ml/blending.py](source/ml/blending.py), `calculate_mass_blend()`.

Условие воспроизведения: взять `blend_risk` и у обоих компонентов оставить объект
`t95`, но задать `value=None`, `lower=None`, `upper=None`, `interval_kind="none"`.
Аналогичный случай воспроизводится для `cetane_number`.

Фактический результат: выражение `estimate.value or 0.0` заменяет отсутствующее
значение нулём. Цикл возвращает `recommend`, quality-agent имеет статус `ok`,
обязательное свойство равно `0.0`, blocking issues отсутствуют.
Второй расчёт смеси, `calculate_mass_blend()`, на тех же данных сохраняет `None`.

Ожидается: неизвестное значение свойства компонента с положительной долей остаётся
неизвестным. Если свойство отсутствует у всех доступных компонентов, цикл возвращает
`abstain`, а причина объясняет нехватку обязательных данных.

Критерий закрытия: проверить отдельно отсутствие объекта свойства и `value=None`
в существующем объекте для T95 и цетанового числа; результаты обоих расчётов смеси
должны совпадать. Нулевая доля компонента с отсутствующим свойством не должна
ошибочно делать расчёт неизвестным.

Исправлено 21.09: неиспользуемая `_weighted_metric()` удалена; активный расчёт
сохраняет неизвестное значение. Параметризованный тест проверяет оба вида пропуска
для T95/CN, согласованность двух расчётов и `abstain` цикла.

## AP-002. T95 и цетановое число не участвуют в проверке пределов

Обнаружено и воспроизведено: 2026-09-14.

Код: [blend_risk.json](config/scenarios/blend_risk.json), а также
[blend_normal.json](config/scenarios/blend_normal.json) и
[blend_missing.json](config/scenarios/blend_missing.json);
[source/constraints.py](source/constraints.py), `check_constraints()`;
[source/ui.py](source/ui.py), `_constraint_rows()`.

Условие воспроизведения: в `blend_risk` задать обоим компонентам `T95=500`
с верхней оценкой `505` и цетановое число `20` с верхней оценкой `20`.

Фактический результат: цикл возвращает `recommend`. В выбранном варианте есть
проверки только серы и запасов компонентов. Для T95 и цетанового числа вычисляются
значения, но ограничения для них в сценариях отсутствуют; UI показывает их как
«Оценено» зелёным цветом.

Это не утверждение о конкретных промышленных пределах. Проблема в том, что система
вообще не может проверить допустимость этих обязательных свойств в рамках демо.

Критерий закрытия: явно задать модельные пределы выбранных сценариев, подписать их
как допущения и включить в общий фильтр допустимости. Проверить выход за каждый
предел и случай отсутствующей оценки. UI должен отличать наличие рассчитанного
значения от успешной проверки ограничения.

Исправлено 21.09: сценарные ограничения T95/CN включены в единый hard-filter;
они обозначены как модельные допущения. Проверки Stage 1/4/UI покрывают выход за
пределы и неизвестное качество; промышленной сертификацией это не является.

## AP-003. History UI не показывает полноценный прогноз

Обнаружено и воспроизведено без открытия Tk: 2026-09-14.

Код: [source/ui.py](source/ui.py), `_run_history_from_ui()`,
`_render_history()` и `history_smoke_snapshot()`;
тест: [global_tests/test_ui.py](global_tests/test_ui.py).

Условие воспроизведения: передать fixture dataset и point-model из
[global_tests/test_stage7_history_serving.py](global_tests/test_stage7_history_serving.py)
с `as_of=2026-01-15T09:00:00Z`.

Фактический результат: backend вычисляет серу `8.8 мг/кг`, верхняя оценка отсутствует,
а quality-agent возвращает issue `UNCERTAINTY_UNAVAILABLE`. Экран сохраняет только
`status`, `model_id`, `reason_codes`, `explanation` и `run_id`. Сам прогноз серы,
подробности issues и ссылка на журнал не выводятся. Для этого fixture статус
`abstain` и текст объяснения также не показывают численное значение прогноза.

Критерий закрытия: выводить point/upper sulfur, состояние uncertainty, issues,
reason codes, model ID и доступный путь к журналу. Проверять эту проекцию тем же
кодом, который использует настоящий экран. Отсутствие upper должно быть показано
явно; history остаётся forecast-only.

Исправлено 21.09: экран использует `format_history_replay_view()` для полной
проекции, включая point/upper, issues, причины и журнал; тесты проверяют тот же
formatter и forecast-only статус.

## AP-004. Смена вкладки ломает завершение history-расчёта

Обнаружено и воспроизведено на настоящем Tk: 2026-09-14.

Код: [source/ui.py](source/ui.py), `show_page()`, `_clear_body()`,
`_run_history_from_ui()` и `_history_done()`.

Условие воспроизведения: открыть «Историю», запустить расчёт, перейти в «Журнал»
до прихода результата. В проверке был воспроизведён тот же порядок событий:
создание history-виджета, смена страницы, вызов обработчика завершения.

Фактический результат: смена страницы уничтожает поле вывода, но callback сохраняет
ссылку на него. `_history_done()` вызывает `delete()` у удалённого виджета и получает
`TclError: invalid command name`.

Критерий закрытия: сохранять состояние и результат запроса независимо от виджетов;
обновлять только существующий актуальный экран. Проверить переход на другую вкладку
во время расчёта, возврат на «Историю» и повторные запуски с разным порядком завершения.

Реализация исправлена 21.09: удалён перекрывавший новый экран старый обработчик;
результат хранится отдельно от виджетов, устаревшие request ID игнорируются,
удалённый виджет не обновляется. Интерактивный Tk-сценарий после интеграции
ещё не повторён, поэтому запись не закрыта.

## AP-005. Validation-отчёт не является независимой оценкой artifact

Обнаружено и подтверждено трассировкой обучения: 2026-09-14.

Код: [source/ml/train.py](source/ml/train.py), `train_model()`;
[source/main.py](source/main.py), `evaluate_command()`;
[source/ml/evaluate.py](source/ml/evaluate.py), `evaluate_model()`.

Условие воспроизведения: выполнить `train`, затем `evaluate --split validation`
для созданного artifact и того же prepared dataset.

Фактический результат: после выбора модели финальный predictor обучается на
`train + validation`. Команда evaluation затем использует те же validation-строки.
В проверке все 26 строк validation-отчёта входили в финальную обучающую выборку.

Такие метрики нельзя трактовать как независимую проверку качества. Это замечание
не означает, что test использовался для выбора модели: такой утечки в данном
случае не установлено.

Критерий закрытия: различать validation-метрики выбора модели до финального refit
и оценку сохранённого artifact. Для artifact, обученного на train + validation,
либо запрещать выдачу validation как независимой оценки, либо явно подписывать
пересечение с обучением. Held-out оценка должна использовать отдельные данные.

## AP-006. Объяснение recommend может противоречить проверкам

Обнаружено и воспроизведено: 2026-09-14.

Код: [source/explain.py](source/explain.py), `build_explanation()`;
[source/ui.py](source/ui.py), `recommendation_to_view()`.

Условие воспроизведения: взять `blend_normal`, установить текущую рецептуру
`A=1.0, B=0.0` и порог существенности `cost_proxy=0.005`.

Фактический результат: baseline допустим, статус равен `recommend`, причина равна
`MATERIAL_IMPROVEMENT`. При этом объяснение пишет «текущий режим не проходит
ограничения», а UI пишет «Текущая рецептура нарушает ограничение по сере».
В тех же результатах проверка серы baseline имеет статус `pass`.

Критерий закрытия: строить текст из фактической причины выбора и результатов checks.
Отдельно проверить рекомендацию из-за нарушения качества и рекомендацию из-за
существенного улучшения при допустимом baseline. Во втором случае объяснять,
какой критерий улучшился.

Исправлено 21.09: объяснение и UI различают нарушение baseline и улучшение
допустимого baseline. Исходный случай закреплён регрессионным тестом.

## AP-007. Демо слабо показывает конфликт критериев

Обнаружено по конфигурации: 2026-09-14. Это ограничение демонстрации, а не ошибка
арифметики или нарушение запрета на реальные управляющие воздействия.

Код: [blend_risk.json](config/scenarios/blend_risk.json);
[source/agents/effects.py](source/agents/effects.py), `calculate_blend_metrics()`.

У обоих компонентов индекс риска равен `0.0`, а выпуск задан одной и той же массой
партии для всех рецептур. Поэтому сценарий не показывает содержательный компромисс
по риску и производительности: выбор фактически определяется серой, запасами,
стоимостным прокси и политикой существенности изменения.

Критерий закрытия: добавить явно синтетический пример с различающимися критериями
и показать, почему выбранный допустимый вариант предпочтительнее альтернатив.
Жёсткие ограничения должны сохранять приоритет. Не выдавать модельные прокси
за измеренную надёжность оборудования или реальную экономию.

## AP-008. Нет подтверждения качества на полном реальном датасете

Зафиксировано как граница проведённой проверки: 2026-09-14.

В этой ревизии обучение, evaluation и history-serving проверены на fixtures.
Качество ML на полном реальном датасете не оценивалось. Успешные CLI-команды
и unit tests подтверждают работоспособность пути, но не точность реального прогноза.

Критерий закрытия: выполнить воспроизводимый прогон подготовки данных, обучения
и оценки на отложенном реальном периоде; сохранить model/dataset IDs, версии
окружения, отчёт и residuals. Сопоставить модель с baseline, показать ошибки около
предела серы и пропущенные превышения. Зафиксировать результаты и ограничения
в этом журнале со ссылками на полученные материалы.

## Что уже проверено

Результаты повторной проверки 2026-09-14:

- `pytest -q`: 109 тестов прошли.
- `ruff check .`, `ruff format --check .`, `mypy source`: успешно.
- `validate-stage0`: 5 сценариев, 3 model-demo fixtures, 170 тегов.
- `accept-stage6`: успешно; `blend_normal=hold`, `blend_risk=recommend`,
  `blend_missing=abstain`; обязательные journal-файлы присутствуют.
- Train/evaluate и history-serving работают на проверенных fixtures.
- Настоящий Tk запускается вне песочницы. Ошибка `Can't find a usable init.tcl`
  внутри песочницы оказалась ограничением проверки и не считается открытой
  неисправностью установки Python. AP-004 подтверждена отдельно после успешного запуска Tk.

Это архивная проверка от 14.09; актуальные статусы указаны в реестре выше.

Проверка интеграции `dev` от 21.09: `pytest -q` — 163 passed;
`ruff check .`, `ruff format --check .`, `mypy source` — успешно;
`validate-stage0` — 7 сценариев, 5 fixtures, 170 тегов;
`verify-model-freeze` — 2 модели и 2 отчёта;
`acceptance` — 5 эпизодов (`hold`, 3 `recommend`, `abstain`).
Это не снимает AP-005/AP-007/AP-008 и не доказывает промышленную пригодность.
## Дополнение от 20.09.2026

Ниже сохранены результаты отдельной проверки ветки. Их статусы требуют сверки
с текущим `dev`; пункты не считаются автоматически исправленными.

## Active blockers

- `actual problems.md` был восстановлен в `dev` при интеграции 21.09.
- Frozen model artifacts и отчёты сейчас присутствуют локально: `verify-model-freeze`
  успешно проверяет две модели и два отчёта. На другом checkout требуются Git LFS
  и проверка доступности тех же файлов.
- Архивная запись про единственный `data/processed/d175aedffaba` более неактуальна:
  локально присутствуют `a2fe8577752a` и `aacc7c1ab3d9`. Совместимость каждого
  подготовленного набора с текущими конфигурациями надо проверять перед использованием.
- Диагностика на устаревшем dataset должна возвращать ошибку совместимости до ML-анализа;
  текущие CLI-команды выполняют такую проверку. Отдельный реальный прогон не проведён.
- Для атомарной публикации prepared dataset добавлен ограниченный retry на Windows
  `PermissionError`; интеграционный тест Stage 2 прошёл.
- Current P8/F19 historical action artifacts remain shadow-only. `supports_actions=true` is blocked until engineering bounds, step/unit evidence, temporal gates, shadow replay, and technologist approval all pass.

## Current implementation status

- Model-demo acceptance is working: `blend_normal=hold`, risk scenarios recommend, and missing quality abstains.
- History forecast is read-only unless a verified action artifact is supplied.
- Tkinter UI is now stage-aware: `avt`, `hydrotreating`, `blend`, `history`, and `journal` have real screens or read-only projections instead of disabled placeholders.
- Observed telemetry quantiles are research context only and must not become engineering limits.
- `ht:P8` and `ht:F19` are the only first production-action candidates; `ht:T11` and `ht:F26` remain context-only.

## Fixed in current working tree

- Prepared dataset publication now retries transient Windows `PermissionError` during atomic directory replace.
- CLI commands that consume prepared data now reject stale config/tag/rules hashes before ML diagnostics/training/replay.
- `verify-model-freeze` now understands production action artifacts and checks controls, horizons, fingerprints and gate report hashes.
- `replay` has an optional `--action-model` path. Without it, history remains forecast-only and abstains with `ACTION_MODEL_UNAVAILABLE`.
- Tkinter history replay has a separate optional verified action artifact field and a dedicated `История/ML` screen.
- Tkinter AVT and hydrotreatment buttons now open read-only stage dashboards with value/source/age/freshness instead of placeholders.
- Tkinter blend screen now has a hybrid sulfur-only panel that refuses to invent component A when history forecast is unavailable.

## Required evidence before full action capability

- Fresh canonical prepared dataset from current materials/runtime/tags/rules.
- Point, upper, and safety forecast artifacts trained from that canonical dataset.
- PAK and LIMS test reports written under `reports/`.
- Updated `config/model_freeze.json` after successful freeze verification.
- Action-effect gate report for `ht:P8` and `ht:F19`:
  - at least 100 change episodes per enabled control;
  - validation MAE at least 10% better than hold;
  - validation and audit upper coverage at least 95%;
  - stable effect sign across 3 temporal folds;
  - confirmed engineering units, bounds, step, and max_step;
  - shadow replay passed;
  - technologist pilot approved.
