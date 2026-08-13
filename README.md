# SciAudit-Stress — центральный публичный репозиторий

Публичная «рельса» курсового проекта SciAudit-Stress (benchmark-and-systems):
схемы, публичные данные, бейзлайны, evaluator, инструменты против утечек,
переиспользуемый шаблон системы, документация и тесты.

Собрано по принципу **rails-before-trains** (мануал §1.3): пайплайн проходит от
входа до метрик с системой-заглушкой ещё *до* того, как появится настоящая
модель.

> **Это штабной (админский) репозиторий.** Warm-up и dev-сплиты лежат здесь
> вместе с их gold-метками намеренно — участники разрабатываются против них.
> Чего здесь быть не должно никогда — материала *скрытых сплитов*: скрытых
> входов и их gold, стресс-метаданных, провенанса и записей об арбитраже. Всё
> это живёт в штабном хранилище с ограниченным доступом, а экспорт
> студенческого бандла его вырезает.
>
> Правило держат два автоматических ворот, а `.gitignore` — только страховка за
> ними: `sciaudit.leakage.forbidden_key_scan` (§10.1) и
> `sciaudit.leakage.split_overlap_check` (§10.4). Оба гоняются в CI.

## Контракт запуска

Любая система — бейзлайн или командная — подчиняется одному интерфейсу:

```bash
python system_template/run.py --input inputs.jsonl --output predictions.jsonl
```

или в контейнере:

```bash
docker build -t sciaudit-system system_template/
docker run --rm -v $PWD/input:/input:ro -v $PWD/output:/output \
  sciaudit-system python run.py --input /input/inputs.jsonl --output /output/predictions.jsonl
```

Что бы ни стояло внутри системы (BM25, LLM, численный чекер, отказ от ответа,
LoRA), внешний контракт одинаков.

## Контракт данных

Три слоя, строго разделённых (см. [docs/schemas.md](docs/schemas.md)):
публичный вход Track A (`schemas/track_a_input.schema.json`), предсказание
(`schemas/prediction.schema.json`) и приватный черновик внутренней аннотации
(`schemas/internal_annotation.schema.json`). Проверить файлы:

```bash
uv run python -m sciaudit.schemas.validate_inputs examples/sample_inputs.jsonl
uv run python -m sciaudit.schemas.validate_predictions examples/sample_predictions.jsonl \
  --input examples/sample_inputs.jsonl
```

## Readiness-loop

Та же цепочка, что гоняет CI (мануал §8.1) — входы → валидация → бейзлайн →
валидация предсказаний → ворота утечки → evaluator → метрики. Прогоните её
локально до того, как открывать PR:

```bash
uv run pytest -q

# 1. публичные входы валидны по схеме и не несут приватных полей
uv run python -m sciaudit.schemas.validate_inputs data_public/public_warmup/inputs.jsonl

# 2. ворота утечки: запрещённые ключи (§10.1) и пересечение сплитов (§10.4)
uv run python -m sciaudit.leakage.forbidden_key_scan data_public/ examples/ --verbose
uv run python -m sciaudit.leakage.split_overlap_check examples data_public/public_warmup

# 3. бейзлайн выдаёт по одному предсказанию на строку входа
uv run python -m sciaudit.baselines.b0_always_insufficient \
  --input data_public/public_warmup/inputs.jsonl --output /tmp/b0_warmup.jsonl

# 4. эти предсказания валидны по схеме и ссылаются только на разрешённые evidence
uv run python -m sciaudit.schemas.validate_predictions /tmp/b0_warmup.jsonl \
  --input data_public/public_warmup/inputs.jsonl

# 5. evaluator их оценивает
uv run python -m sciaudit.evaluator.score \
  --pred /tmp/b0_warmup.jsonl --gold data_public/public_warmup/gold.jsonl \
  --out /tmp/warmup_metrics.json --report /tmp/warmup_report.md
```

B0 всегда отвечает `insufficient`, поэтому его accuracy равна доле этого класса
в gold, а severe false-warrant rate по построению нулевая. В этом и смысл
тривиального контроля: любая настоящая система обязана обойти его по обоим
показателям.
