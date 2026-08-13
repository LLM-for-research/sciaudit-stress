"""Бейзлайны B0–B4 (Task 4 — реализуются отдельными issue).

Каждый бейзлайн — модуль, запускаемый так:
    python -m sciaudit.baselines.b0_always_insufficient --input <in> --output <out>

Запланировано (мануал §11):
- b0_always_insufficient   : проверка работоспособности схем и evaluator'а
- b1_bm25_llm              : BM25-retrieval + структурированный аудит открытой моделью
- b2_fullpack_llm          : структурированный аудит по полному evidence pack
- b3_numeric_checker_llm   : детерминированный численный/табличный чекер + LLM
- b4_selective_threshold   : бейзлайн с избирательным отказом от ответа
"""
