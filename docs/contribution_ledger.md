# Реестр вкладов

Текущая запись того, **кто что сделал**, со ссылкой на артефакт и статусом ревью. Реестр нужен,
чтобы вклады были видимы и засчитываемы, а решения об авторстве позже опирались на
задокументированную работу, а не на память. **Авторство не автоматическое** — оно следует из
существенных, задокументированных и проверенных вкладов, записанных здесь.

Добавляйте строку, когда вашу работу смёржили (или когда артефакт принят). Держите реестр в этом
файле или в связанной таблице, если курсу так удобнее, — контрактом являются колонки.

| Дата | Участник | Вклад (что) | Тип | Ссылка на артефакт | Статус ревью | Проверил | Роль в стиле CRediT |
|---|---|---|---|---|---|---|---|
| 2026-07-12 | Rodion Krainov | Реализовал модуль бейзлайна B0 | code | #3 / [commit](https://github.com/LLM-for-research/sciaudit-stress/commit/5344fa852e12fade7191797d9ed2cfc554c49ac1) | merged | - | Software |
| 2026-07-12 | Abdullo Muminov | Реализовал базовый evaluator для B0 и отчёт лидерборда | code | #5 / [PR](https://github.com/LLM-for-research/sciaudit-stress/pull/7) | merged | Rodion Krainov | Software |
| 2026-07-28 | Ge-os | Схемы бенчмарка и валидаторы Track A | code | #2 / [PR](https://github.com/LLM-for-research/sciaudit-stress/pull/8) | merged | - | Software, Data curation |
| 2026-07-30 | Abdullo Muminov | Методология стресс-трансформаций и шаблоны | docs | #4 / [PR](https://github.com/LLM-for-research/sciaudit-stress/pull/17) | merged | - | Writing |
| 2026-08-10 | Abdullo Muminov | Evaluator v2: отказ и selective-risk метрики | evaluation | #16 / [PR](https://github.com/LLM-for-research/sciaudit-stress/pull/19) | merged | - | Software |
| 2026-08-10 | Rodion Krainov | Скан запрещённых ключей (§10.1) и ворота в CI | validation | #10 / [PR](https://github.com/LLM-for-research/sciaudit-stress/pull/20) | merged | - | Software, Validation |
| 2026-08-13 | Ge-os | Публичный warm-up: 15 синтетических инстансов и gold | data | #9 / [PR](https://github.com/LLM-for-research/sciaudit-stress/pull/18) | merged | - | Data curation |
| 2026-08-13 | Rodion Krainov | Детерминированный численный/табличный чекер | code | #15 / [PR](https://github.com/LLM-for-research/sciaudit-stress/pull/22) | merged | - | Software |
| 2026-08-20 | Abdullo Muminov | Бейзлайн B1: BM25-ретрив и структурированный аудит | code | #13 / [PR](https://github.com/LLM-for-research/sciaudit-stress/pull/21) | merged | - | Software |
| 2026-08-20 | Rodion Krainov | Бейзлайн B2 и починка метрик evaluator'а | code | #14 / [PR](https://github.com/LLM-for-research/sciaudit-stress/pull/23) | merged | - | Software |
| 2026-08-20 | Rodion Krainov | Датасет из двух статей: 24 инстанса public_dev с приватными аннотациями | data | [commit](https://github.com/LLM-for-research/sciaudit-stress/commit/e5336c2) | merged | - | Data curation |
| 2026-08-20 | Rodion Krainov | Бейзлайны B3 и B4, повторы в сравнении, ворота §10.2/§10.3/§5.5, скрытый контур, полный §12 | code | не смёржено — строка ждёт PR | proposed | - | Software |


> **Колонка «Проверил» не заполнена ни в одной строке.** Правило двух подписей из
> [гида первой недели](week1_guide.md) требует её заполнять, и заполнить её может
> только человек, который работу проверял, — не автор строки. Это первое, что
> стоит закрыть: авторство распределяется по этим строкам, а строка без подписи
> слабее строки с подписью.

## Что означают колонки

- **Вклад (что)** — одна конкретная строка; не «помогал по мелочи».
- **Тип** — одно из: `data`, `stress-case`, `code`, `evaluation`, `docs`, `validation`,
  `analysis`, `writing`, `release`.
- **Ссылка на артефакт** — номер PR, путь к файлу или ID эксперимента. **Обязательна** — вклад без
  ссылки засчитать нельзя.
- **Статус ревью** — `proposed` / `in-review` / `merged` / `rejected`.
- **Проверил** — TA или коллега, подписавший работу (связано с правилом двух подписей из
  [гида первой недели](week1_guide.md)).
- **Роль в стиле CRediT** — например Software, Data curation, Validation, Analysis, Writing,
  Visualization.

## Как это связано с остальным

- Каждая строка должна прослеживаться до PR, который нёс [еженедельный апдейт](weekly_update_template.md).
- У настоящих систем, упомянутых здесь, должна быть [карточка системы](system_card_template.md).
- Полезная работа, не дотягивающая до авторства, всё равно записывается и **отмечается в
  благодарностях** — фиксируйте всё; кредит распределяется позже и прозрачно, по этим строкам.
