# Публичный warm-up срез (Track A)

Первый запускаемый публичный датасет для readiness-loop. Всё содержимое здесь —
**синтетическая заглушка**; пометки о происхождении каждого инстанса и политика
замены лежат в [manifest.json](manifest.json), их применяют, когда появится
контролируемый пул статей (Task 2).

| Файл | Содержимое |
|---|---|
| `inputs.jsonl` | 17 входных инстансов Track A (схема `track_a_input_v1`) |
| `gold.jsonl` | 17 gold-записей, по одной на `instance_id` (формат `examples/toy_gold.jsonl`) |
| `manifest.json` | Штабной провенанс: пометки синтетики, нейтральные источники, форма датасета (в inputs не сливается) |

Свойства, которые держит `tests/test_public_warmup.py`:

- присутствуют все четыре вердикта, и ни один не занимает больше 50%;
- `instance_id` несемантические (`sas_` + случайный суффикс) — вердикт по ним не угадывается;
- в `inputs.jsonl` нет приватных полей: это держат валидатор входов Track A и
  тесты на утечку;
- один и тот же evidence pack может нести claim'ы с разными вердиктами, поэтому
  вердикт не читается из одного только evidence;
- B0 отрабатывает на `inputs.jsonl`, и evaluator считает по нему метрики без ошибок.

## Локальная проверка

```bash
uv run python -m sciaudit.schemas.validate_inputs data_public/public_warmup/inputs.jsonl

uv run python -m sciaudit.baselines.b0_always_insufficient \
  --input data_public/public_warmup/inputs.jsonl --output /tmp/b0_warmup.jsonl

uv run python -m sciaudit.schemas.validate_predictions /tmp/b0_warmup.jsonl \
  --input data_public/public_warmup/inputs.jsonl

uv run python -m sciaudit.evaluator.score \
  --pred /tmp/b0_warmup.jsonl --gold data_public/public_warmup/gold.jsonl \
  --out /tmp/warmup_metrics.json
```
