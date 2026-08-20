#!/usr/bin/env python3
"""Детерминированная заглушка вместо модели. ЭТО НЕ МОДЕЛЬ.

Читает промпт из stdin, печатает в stdout валидный JSON предсказания по
нескольким жёстко заданным правилам. Никакого обучения, вывода и сети: одна и
та же строка на входе всегда даёт один и тот же ответ.

Зачем она есть:

* smoke-шаги CI прогоняют B1 и B2 целиком, включая точку входа, не имея
  доступа к модели (issue #12 ещё не закрыт);
* харнесс сравнения :mod:`sciaudit.baselines.compare_b1_b2` может собрать
  таблицу и показать, что различие B1 и B2 берётся именно из ретрива, а не из
  случайности модели.

Чего она НЕ делает: не измеряет качество ни одной системы. Любые числа,
полученные с этой заглушкой, характеризуют харнесс, а не аудит. В лидерборд,
в отчёты и в сравнения систем они не идут.

Запуск::

    python -m sciaudit.baselines.stub_model < prompt.txt
"""
from __future__ import annotations

import json
import re
import sys

UNIVERSAL = re.compile(r"\b(all|every|consistently|always|any)\b", re.I)


def parse_prompt(prompt):
    """Достать текст claim и список eid из промпта, собранного model_audit."""
    claim_match = re.search(r"^Claim:\n(.*?)\n\nEvidence:\n", prompt,
                            flags=re.S | re.M)
    claim = claim_match.group(1).strip() if claim_match else ""

    evidence_block = prompt.split("Evidence:\n", 1)[1] if "Evidence:\n" in prompt else ""
    eids = re.findall(r"^- ([^:]+):", evidence_block, flags=re.M)
    return claim, [eid.strip() for eid in eids]


def decide(claim, eids):
    """Правила заглушки. Меняются только вместе с docs/b1_vs_b2.md."""
    if not eids:
        return {
            "verdict": "insufficient", "confidence": 0.2, "predicted_eids": [],
            "issue_tags": ["evidence_missing_or_incomplete"], "abstain": True,
            "rationale_short": "stub: no evidence supplied",
        }

    if UNIVERSAL.search(claim):
        if len(eids) >= 2:
            return {
                "verdict": "overclaimed", "confidence": 0.7,
                "predicted_eids": eids[:2],
                "issue_tags": ["claim_stronger_than_evidence"], "abstain": False,
                "rationale_short": "stub: universal claim against several evidence units",
            }
        return {
            "verdict": "insufficient", "confidence": 0.3, "predicted_eids": eids[:1],
            "issue_tags": ["evidence_missing_or_incomplete"], "abstain": False,
            "rationale_short": "stub: universal claim against a single evidence unit",
        }

    return {
        "verdict": "warranted", "confidence": 0.6, "predicted_eids": eids[:1],
        "issue_tags": [], "abstain": False,
        "rationale_short": "stub: bounded claim with evidence present",
    }


def main(argv: list[str] | None = None) -> int:
    prompt = sys.stdin.read()
    claim, eids = parse_prompt(prompt)
    print(json.dumps(decide(claim, eids), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
