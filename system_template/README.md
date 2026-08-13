# Шаблон системы

Форкните или скопируйте эту директорию, чтобы собрать систему SciAudit-Stress.
Контракт запуска не трогайте; меняйте только то, что *внутри* `audit_instance`.

## Интерфейс (не менять)

```bash
python run.py --input inputs.jsonl --output predictions.jsonl
```

- `--input`: JSONL с входными объектами Track A (`../schemas/track_a_input.schema.json`)
- `--output`: JSONL с объектами-предсказаниями (`../schemas/prediction.schema.json`)
- по одной строке предсказания на строку входа, `instance_id` сохраняется.

## Docker

Контекст сборки — эта директория, поэтому обе команды запускаются из
`system_template/`:

```bash
docker build -t my-team-system .
docker run --rm -v $PWD/input:/input:ro -v $PWD/output:/output \
  my-team-system python run.py --input /input/inputs.jsonl --output /output/predictions.jsonl
```

Версия uv в образе закреплена, так что сборка стабильна — но **ваши зависимости
зафиксированы только после того, как вы закоммитите локфайл**. Пока его нет,
сборка печатает предупреждение и резолвит зависимости на месте. Чинится так:

```bash
uv lock          # в system_template/, затем закоммитить uv.lock
```

## Зависимости (uv)

```bash
uv add transformers==4.44.2 rank-bm25==0.2.2   # пример закреплённых зависимостей
uv lock                                         # обновить uv.lock (закоммитить его)
uv run python run.py --input examples/in.jsonl --output preds.jsonl
```

## Что править

Замените функцию `audit_instance` в `run.py` своим пайплайном (retrieval, вызов
LLM, численный чекер, отказ от ответа, …). Зависимости добавляйте через
`uv add` — версии фиксируются в `uv.lock`, его нужно коммитить.

## Доступ к модели

Модель дёргается по **OpenAI-совместимому API с ключом**, а не грузится весами
(issue #12). Выберите профиль из `../configs/models.yaml`; ключ берётся из
переменной окружения, названной в профиле, и никогда из файла:

```bash
export SCIAUDIT_API_KEY=...        # не в репозиторий — .env в .gitignore
uv run python run.py --input inputs.jsonl --output predictions.jsonl
```

Политика §7.4 мануала ограничивает **саму модель**, а не способ доступа к ней:
сабмишен обязан декларировать профиль из списка `approved_profiles`. Всё
остальное — только для разработки. В `system_info.model` кладите идентификатор
модели — не команду запуска и тем более не строку, содержащую ключ.
