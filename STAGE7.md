# Stage 7: constraint-aware safety forecast

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
