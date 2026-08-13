"""Инструменты против утечек (владелец — Leakage/security TA).

Готово:

* ``forbidden_key_scan`` — рекурсивный поиск приватных ключей и строковых
  значений в student-visible файлах (§10.1);
* ``split_overlap_check`` — аудит пересечения сплитов по тексту claim (§10.4).

Ещё нет: ``metadata_probe`` (§10.2) и ``id_randomness_check`` (§5.5).
"""
