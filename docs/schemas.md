# Схемы данных: публичный вход, предсказание, приватный gold

В Track A действует **жёсткое разделение на три слоя**. Система видит только слой 1 и порождает
только слой 2. Слой 3 существует исключительно в приватном репозитории
`sciaudit-stress-private`.

| Слой | Схема | Кто видит | Что содержит |
|---|---|---|---|
| 1. Публичный вход | [`schemas/track_a_input.schema.json`](../schemas/track_a_input.schema.json) | участники и системы | абстрагированный `paper_id`, `instance_id`, нормализованный claim, замороженный evidence pack, `allowed_evidence_ids` |
| 2. Предсказание | [`schemas/prediction.schema.json`](../schemas/prediction.schema.json) | порождается системами, потребляется evaluator'ом | вердикт, ID evidence, issue-теги, уверенность, опциональные отказ, время и стоимость |
| 3. Внутренняя аннотация (приватный gold) | [`schemas/internal_annotation.schema.json`](../schemas/internal_annotation.schema.json) *(черновик)* | только штаб/TA | всё перечисленное выше **плюс** провенанс, стресс-метаданные, gold-вердикт, severity, записи ревью, приватное обоснование, сплит |

Почему разделение жёсткое: стресс-метаданные тривиально сливают метки (увидев
`stress_type=evidence_removal`, система бесплатно угадывает `insufficient`), а gold-метки и
провенанс не должны попадать ни к системам, ни во внешние LLM. См.
[рамку бенчмарка](benchmark_framing.md).

---

## 1. Публичный вход (`track_a_input.schema.json`)

Одна строка JSONL на инстанс. Везде `additionalProperties: false` — любой лишний ключ (например,
просочившийся блок `gold`) делает файл невалидным.

```json
{
  "schema_version": "track_a_input_v1",
  "paper_id": "P001",
  "instance_id": "sas_8f3kq2m9",
  "claim": {
    "text": "The method consistently outperforms the compared baselines across all reported robustness benchmarks.",
    "claim_type": "baseline_superiority",
    "scope": "Judge only from the supplied evidence pack."
  },
  "evidence_pack": [
    {"eid": "e01", "source_kind": "table", "modality": "table_text", "text": "..."},
    {"eid": "e02", "source_kind": "caption", "modality": "caption_text", "text": "..."}
  ],
  "allowed_evidence_ids": ["e01", "e02"]
}
```

- `paper_id` **абстрагирован** (никаких названий, авторов, площадок, URL); соответствие реальным
  статьям живёт в приватной карте провенанса.
- `allowed_evidence_ids` — единственные ID, на которые вправе сослаться предсказание; валидатор
  проверяет, что это подмножество `eid` из пака.
- `claim_type` — один из документированных типов claim (см.
  [`configs/allowed_labels.yaml`](../configs/allowed_labels.yaml)):
  `numerical_performance`, `baseline_superiority`, `ablation`, `robustness`,
  `efficiency`, `bounded_generalization`.
- **Никаких приватных меток и метаданных, никогда.** Валидатор явно отвергает
  `gold`, `verdict`, `severity`, `stress`/`stress_type`, `private_rationale`,
  `provenance`/`provenance_map`, `review_note`, `split` и подобные ключи на любой глубине
  вложенности.

Примеры: [`examples/sample_inputs.jsonl`](../examples/sample_inputs.jsonl)
(6 инстансов, покрывающих все типы claim).

## 2. Предсказание (`prediction.schema.json`)

Одна строка JSONL на строку входа, `instance_id` сохраняется.

**Обязательные:** `instance_id`, `verdict`, `confidence`, `predicted_eids`, `issue_tags`.
**Опциональные (рекомендуются; бейзлайны всегда их пишут):** `abstain`,
`rationale_short`, `runtime_seconds`, `estimated_cost`, `system_info`.

- **Вердикты фиксированы:** `warranted` | `overclaimed` | `contradicted` |
  `insufficient`. Ничто другое не проходит валидацию.
- **Issue-теги фиксированы** списком из
  [`configs/allowed_labels.yaml`](../configs/allowed_labels.yaml):
  `numerical_inconsistency`, `claim_stronger_than_evidence`,
  `unsupported_generalization`, `missing_ablation_support`,
  `non_comparable_baseline`, `weak_statistical_support`,
  `evidence_missing_or_incomplete`, `caption_chart_mismatch`.
- `predicted_eids` («evidence_ids» предсказания) обязаны браться из `allowed_evidence_ids`
  соответствующего инстанса; валидатор сверяет это, когда ему передан `--input`.
- `confidence` ∈ [0, 1]. Gold-метка `insufficient` и действие системы `abstain` остаются разными
  понятиями.

Примеры: [`examples/sample_predictions.jsonl`](../examples/sample_predictions.jsonl)
(6 предсказаний: все четыре вердикта, один отказ, один минимальный объект только с обязательными
полями).

## 3. Внутренняя аннотация (`internal_annotation.schema.json`, черновик)

Приватная запись, доступная только штабу и TA, из которой публичные входы получают
**вычёркиванием** провенанса, стресс-метаданных, gold, полей ревью и сплита. Она добавляет:

- `paper.provenance_ref` — ключ в приватную карту провенанса;
- `evidence_pack[].source_ref`, `is_distractor` — приватные указатели и пометки;
- `stress` — `is_stress_case`, `stress_type` (10 документированных трансформаций),
  `seed_instance_id`;
- `gold` — вердикт, `supporting_eids`, issue-теги, `severity`
  (`minor`/`moderate`/`severe`), `private_rationale`;
- `review` — `validation_level` (`auto_unchecked` → `team_verified` →
  `ta_validated` → `ta_adjudicated`), рецензент, `review_note`,
  `human_verification_note`;
- `split` — уровень бенчмарка.

**Настоящие внутренние аннотации никогда не должны попадать в этот репозиторий.**
[`examples/sample_internal_annotation.synthetic.jsonl`](../examples/sample_internal_annotation.synthetic.jsonl)
содержит две явно помеченные *синтетические* записи исключительно как документацию схемы.

---

## Валидаторы

```bash
# публичные входы: схема + утечки + согласованность eid
uv run python -m sciaudit.schemas.validate_inputs examples/sample_inputs.jsonl

# предсказания: схема + дубликаты (+ покрытие и сверка eid при --input)
uv run python -m sciaudit.schemas.validate_predictions examples/sample_predictions.jsonl \
  --input examples/sample_inputs.jsonl
```

Код возврата 0 — валидно; 1 — есть ошибки (печатаются с координатами `файл:строка`). Оба гоняются
в CI. Тесты: `tests/test_schema_validation.py`.

`configs/allowed_labels.yaml` — единственный источник истины для вердиктов, issue-тегов и типов
claim; отдельный тест следит, чтобы enum'ы в JSON Schema с ним не разъезжались.
