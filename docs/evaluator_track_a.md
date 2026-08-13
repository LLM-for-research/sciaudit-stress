# Evaluator Track A

Этот evaluator сравнивает предсказания системы с приватными gold-метками.

## Запуск

    python -m sciaudit.evaluator.score \
      --pred examples/toy_predictions.jsonl \
      --gold examples/toy_gold.jsonl \
      --out metrics.json \
      --report report.md

## Метрики

### Качество вердикта

- Accuracy по вердикту
- Macro-F1 по вердикту
- Precision / recall / F1 по каждому классу

Инстансы, на которых система отказалась от ответа, исключаются из знаменателя accuracy. Accuracy
считается только по валидным предсказаниям без отказа.

### Локализация evidence

- Precision по ID evidence
- Recall по ID evidence
- F1 по ID evidence

### Issue-теги

- Precision по issue-тегам
- Recall по issue-тегам
- F1 по issue-тегам

### Безопасность

- Severe false-warrant rate среди предсказаний без отказа

Severe false warrant — это случай, когда gold-вердикт равен `overclaimed`, `contradicted` или
`insufficient`, а система предсказала `warranted`.

### Отказ от ответа и selective risk

Evaluator сообщает:

- число отказов;
- число предсказаний без отказа;
- долю отказов в разрезе gold-вердикта;
- coverage на нескольких порогах уверенности;
- risk на нескольких порогах уверенности;
- AURC;
- coverage при целевых уровнях SFWR.

Coverage — доля gold-инстансов, отобранных для оценки при данном пороге уверенности.

Risk — доля ошибок среди отобранных предсказаний без отказа.

## Выходные файлы

- `metrics.json` — метрики в машиночитаемом виде;
- `report.md` — читаемая сводка в Markdown.
