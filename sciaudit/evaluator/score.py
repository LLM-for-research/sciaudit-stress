"""Evaluator Track A: сравнение предсказаний с приватным gold (мануал §12).

Считает качество вердикта, локализацию evidence, issue-теги, severe false-warrant
rate (§3.4) и selective risk (§3.5, §12.5). Пишет метрики в JSON и читаемый
отчёт в Markdown.

Два свойства, без которых числа не значат ничего.

**Сабмишен обязан быть полным.** Ровно одно предсказание на каждый gold-инстанс:
без пропусков, дубликатов и лишних ID. Иначе систему можно «улучшить», прислав
только те ответы, в которых она уверена: пропущенные инстансы не попадали в
знаменатель, и accuracy росла от молчания. Неуверенность выражается явным
``abstain: true`` — тогда инстанс остаётся в знаменателе покрытия, а качество
считается отдельно по отвеченным. Неполный сабмишен не скорится: метрики
качества становятся ``null``, а CLI возвращает ненулевой код.

**Кривая risk-coverage строится ранжированием, а не фиксированными порогами.**
При фиксированной сетке порогов система, поставившая всем ответам
``confidence = 1.0``, даёт одинаковое покрытие во всех точках, площадь под
кривой вырождается, и полностью неверная система получает AURC = 0. Поэтому
ответы сортируются по убыванию уверенности, покрытие пробегает k/N, а внутри
групп с равной уверенностью счётчики интерполируются — результат не зависит от
порядка внутри группы. Дополнительно считается AUGRC: обобщённый риск нормируется
на все N инстансов, а не на выбранные, поэтому область малого покрытия не
получает непропорционального веса.

Запуск:
    python -m sciaudit.evaluator.score --pred preds.jsonl --gold gold.jsonl \\
        --out metrics.json --report report.md
"""
import argparse
import json
import math
from pathlib import Path

from sciaudit.schemas import read_jsonl as _read_jsonl_with_line_numbers

VERDICTS = ["warranted", "overclaimed", "contradicted", "insufficient"]
TARGET_SFWR_LEVELS = [0.0, 0.05, 0.1, 0.2]
FIXED_COVERAGE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def read_jsonl(path):
    """Объекты JSONL без номеров строк.

    Парсер один на весь репозиторий — ``sciaudit.schemas.read_jsonl``: он же
    проверяет, что каждая строка является объектом. Здесь только снимается
    нумерация и в текст ошибки добавляется путь к файлу.
    """
    try:
        return [obj for _, obj in _read_jsonl_with_line_numbers(path)]
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from e


def validate_prediction(pred):
    # Список обязан совпадать с `required` в schemas/prediction.schema.json.
    # Там `abstain` необязателен — система, которая никогда не отказывается от
    # ответа, вправе его не писать (мануал, Listing 3). Если требовать его
    # здесь, схема-валидные прогнозы молча выпадают из знаменателя вместо того,
    # чтобы быть оценёнными.
    required = [
        "instance_id",
        "verdict",
        "confidence",
        "predicted_eids",
        "issue_tags",
    ]

    missing = [k for k in required if k not in pred]
    if missing:
        return [f"нет обязательных полей: {missing}"]

    errors = []

    if pred["verdict"] not in VERDICTS:
        errors.append(f"недопустимый verdict: {pred['verdict']}")

    if not isinstance(pred["confidence"], (int, float)) or not 0 <= pred["confidence"] <= 1:
        errors.append("confidence должен быть числом в [0, 1]")

    if not isinstance(pred["predicted_eids"], list):
        errors.append("predicted_eids должен быть списком")

    if not isinstance(pred["issue_tags"], list):
        errors.append("issue_tags должен быть списком")

    if "abstain" in pred and not isinstance(pred["abstain"], bool):
        errors.append("abstain должен быть булевым")

    return errors


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def normalize_gold(row):
    gold = row.get("gold", row)
    return {
        "verdict": gold["verdict"],
        "supporting_eids": gold.get("supporting_eids", []),
        "issue_tags": gold.get("issue_tags", []),
    }


def get_non_abstained_ids(golds, preds):
    return [
        instance_id
        for instance_id in golds
        if instance_id in preds and not preds[instance_id].get("abstain", False)
    ]


def verdict_metrics(golds, preds, instance_ids):
    correct = 0
    total = len(instance_ids)
    per_class = {}

    for label in VERDICTS:
        tp = fp = fn = 0

        for instance_id in instance_ids:
            gold_label = golds[instance_id]["verdict"]
            pred_label = preds[instance_id]["verdict"]

            if pred_label == label and gold_label == label:
                tp += 1
            elif pred_label == label and gold_label != label:
                fp += 1
            elif pred_label != label and gold_label == label:
                fn += 1

        p, r, f1 = prf(tp, fp, fn)
        per_class[label] = {
            "precision": p,
            "recall": r,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    for instance_id in instance_ids:
        if preds[instance_id]["verdict"] == golds[instance_id]["verdict"]:
            correct += 1

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(VERDICTS)

    return {
        "accuracy": correct / total if total else 0.0,
        "accuracy_denominator": total,
        "correct": correct,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def set_metrics(golds, preds, instance_ids, gold_key, pred_key):
    tp = fp = fn = 0

    for instance_id in instance_ids:
        gold_set = set(golds[instance_id].get(gold_key, []))
        pred_set = set(preds[instance_id].get(pred_key, []))

        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)

    p, r, f1 = prf(tp, fp, fn)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "denominator_instances": len(instance_ids),
    }


def severe_false_warrant_rate(golds, preds, instance_ids):
    risky_total = 0
    severe_errors = 0

    for instance_id in instance_ids:
        if golds[instance_id]["verdict"] != "warranted":
            risky_total += 1
            if preds[instance_id]["verdict"] == "warranted":
                severe_errors += 1

    return {
        "rate": severe_errors / risky_total if risky_total else 0.0,
        "count": severe_errors,
        "denominator": risky_total,
    }


def abstention_by_gold_verdict(golds, preds):
    result = {}

    for label in VERDICTS:
        gold_ids = [instance_id for instance_id, gold in golds.items() if gold["verdict"] == label]
        abstained = [
            instance_id
            for instance_id in gold_ids
            if instance_id in preds and preds[instance_id].get("abstain", False)
        ]
        missing = [instance_id for instance_id in gold_ids if instance_id not in preds]

        result[label] = {
            "gold_count": len(gold_ids),
            "abstained_count": len(abstained),
            "missing_count": len(missing),
            "abstention_rate": len(abstained) / len(gold_ids) if gold_ids else 0.0,
        }

    return result


def rank_units(golds, preds):
    """Разложить все gold-инстансы в порядке убывания уверенности.

    Отвеченные идут по убыванию ``confidence``; инстансы с отказом ставятся в
    хвост как никогда не выбираемые добровольно. При покрытии, которое до них
    доходит, отказ считается ошибкой — вердикта система не дала, значит верным
    он быть не может, — но severe false warrant им не приписывается: ничего не
    было объявлено обоснованным.
    """
    answered, refused = [], []

    for instance_id, gold in golds.items():
        pred = preds.get(instance_id)
        risky = gold["verdict"] != "warranted"

        if pred is None or pred.get("abstain", False):
            refused.append({
                "instance_id": instance_id,
                "confidence": float("-inf"),
                "error": 1, "risky": int(risky), "severe": 0, "answered": 0,
            })
            continue

        error = int(pred["verdict"] != gold["verdict"])
        severe = int(risky and pred["verdict"] == "warranted")
        answered.append({
            "instance_id": instance_id,
            "confidence": float(pred["confidence"]),
            "error": error, "risky": int(risky), "severe": severe, "answered": 1,
        })

    answered.sort(key=lambda u: (-u["confidence"], u["instance_id"]))
    refused.sort(key=lambda u: u["instance_id"])
    return answered + refused


def _tie_groups(units):
    """Сгруппировать подряд идущие единицы с одинаковой уверенностью."""
    groups = []
    for unit in units:
        if groups and groups[-1][0] == unit["confidence"]:
            groups[-1][1].append(unit)
        else:
            groups.append((unit["confidence"], [unit]))
    return groups


def risk_coverage_curve(golds, preds):
    """Кривая risk-coverage по ранжированию. Точка на каждое k = 1..N.

    Внутри группы с равной уверенностью накопленные счётчики интерполируются
    линейно: это математическое ожидание при случайном разрешении ничьих,
    поэтому результат не зависит от того, в каком порядке лежат равные ответы.
    """
    units = rank_units(golds, preds)
    total = len(units)
    if not total:
        return []

    points = []
    base_err = base_sev = base_risky = 0.0
    consumed = 0

    for _, group in _tie_groups(units):
        size = len(group)
        sum_err = sum(u["error"] for u in group)
        sum_sev = sum(u["severe"] for u in group)
        sum_risky = sum(u["risky"] for u in group)

        for step in range(1, size + 1):
            fraction = step / size
            k = consumed + step
            err = base_err + fraction * sum_err
            sev = base_sev + fraction * sum_sev
            risky = base_risky + fraction * sum_risky

            points.append({
                "k": k,
                "coverage": k / total,
                "risk": err / k,
                "generalized_risk": err / total,
                "expected_errors": err,
                "severe_false_warrant_rate": (sev / risky) if risky > 0 else 0.0,
                "expected_severe_false_warrants": sev,
                "expected_risky": risky,
                "within_voluntary_coverage": k <= sum(u["answered"] for u in units),
            })

        base_err += sum_err
        base_sev += sum_sev
        base_risky += sum_risky
        consumed += size

    return points


def aurc(points):
    """Средний selective risk по всем уровням покрытия."""
    if not points:
        return 0.0
    return sum(p["risk"] for p in points) / len(points)


def augrc(points):
    """Средний обобщённый риск: ошибки нормируются на все N, а не на выбранные.

    AURC придаёт огромный вес области малого покрытия, где знаменатель крошечный
    и одна ошибка двигает метрику на десятки процентов. AUGRC от этого свободен.
    """
    if not points:
        return 0.0
    return sum(p["generalized_risk"] for p in points) / len(points)


def aurc_within_voluntary_coverage(points):
    """AURC, ограниченный тем покрытием, которое система выдала добровольно."""
    voluntary = [p for p in points if p["within_voluntary_coverage"]]
    if not voluntary:
        return 0.0
    return sum(p["risk"] for p in voluntary) / len(voluntary)


def metrics_at_fixed_coverage(points, levels=None):
    """Риск и SFWR при фиксированных уровнях покрытия.

    Для целевого покрытия c берётся префикс из ``ceil(c * N)`` ответов, то есть
    ближайшее достижимое покрытие не ниже c. Это делает системы сравнимыми в
    одной точке, чего кривая целиком не даёт.
    """
    levels = FIXED_COVERAGE_LEVELS if levels is None else levels
    if not points:
        return {str(level): None for level in levels}

    total = len(points)
    result = {}

    for level in levels:
        k = min(total, max(1, math.ceil(level * total)))
        point = points[k - 1]
        result[str(level)] = {
            "requested_coverage": level,
            "achieved_coverage": point["coverage"],
            "risk": point["risk"],
            "generalized_risk": point["generalized_risk"],
            "severe_false_warrant_rate": point["severe_false_warrant_rate"],
            "selected_count": point["k"],
            "within_voluntary_coverage": point["within_voluntary_coverage"],
        }

    return result


def coverage_at_target_sfwr(points):
    """Наибольшее покрытие, на котором SFWR не превышает цель.

    Рассматриваются только префиксы, которые система выдала добровольно: за
    границей добровольного покрытия в выборку попадают отказы, они не могут быть
    severe false warrant, и SFWR обманчиво падает.
    """
    result = {}
    voluntary = [p for p in points if p["within_voluntary_coverage"]]

    for target in TARGET_SFWR_LEVELS:
        good = [p["coverage"] for p in voluntary
                if p["severe_false_warrant_rate"] <= target]
        result[str(target)] = max(good, default=0.0)

    return result


def confusion_matrix(golds, preds, instance_ids):
    """Матрица ошибок §12.1: строки — gold, столбцы — предсказание.

    Агрегированные F1 говорят, сколько система ошибается; матрица — **куда** она
    ошибается, а это разные вопросы. Путать ``warranted`` с ``overclaimed`` и
    путать его с ``insufficient`` — две разные болезни с разным лечением.
    """
    matrix = {gold_label: {pred_label: 0 for pred_label in VERDICTS}
              for gold_label in VERDICTS}
    for instance_id in instance_ids:
        gold_label = golds[instance_id]["verdict"]
        pred_label = preds[instance_id]["verdict"]
        if gold_label in matrix and pred_label in matrix[gold_label]:
            matrix[gold_label][pred_label] += 1
    return matrix


def issue_tag_metrics(golds, preds, instance_ids):
    """Мультиметочные метрики тегов §12.3: micro, macro и чаще всего упущенные.

    Micro-F1 считает все теги в одном мешке, поэтому его определяют частые теги;
    macro-F1 усредняет по тегам, поэтому редкий тег весит столько же, сколько
    частый. Расхождение между ними и есть полезная информация: система может
    прилично выглядеть на micro, полностью пропуская редкие теги.
    """
    micro = set_metrics(golds, preds, instance_ids, "issue_tags", "issue_tags")

    tags = set()
    for instance_id in instance_ids:
        tags |= set(golds[instance_id].get("issue_tags", []))
        tags |= set(preds[instance_id].get("issue_tags", []))

    per_tag = {}
    for tag in sorted(tags):
        tp = fp = fn = 0
        for instance_id in instance_ids:
            in_gold = tag in set(golds[instance_id].get("issue_tags", []))
            in_pred = tag in set(preds[instance_id].get("issue_tags", []))
            tp += in_gold and in_pred
            fp += in_pred and not in_gold
            fn += in_gold and not in_pred
        precision, recall, f1 = prf(tp, fp, fn)
        per_tag[tag] = {"precision": precision, "recall": recall, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn}

    macro_f1 = (sum(value["f1"] for value in per_tag.values()) / len(per_tag)
                if per_tag else 0.0)
    most_missed = sorted(((value["fn"], tag) for tag, value in per_tag.items()),
                         reverse=True)
    return {
        "micro": micro,
        "macro_f1": macro_f1,
        "per_tag": per_tag,
        "most_missed": [{"tag": tag, "missed": missed}
                        for missed, tag in most_missed if missed],
    }


def calibration(golds, preds, instance_ids, bins=10):
    """Калибровка §12.4: Brier и ECE по предсказаниям без отказа.

    Обе меры отвечают на вопрос, которого нет ни в accuracy, ни в F1: **можно ли
    верить заявленной уверенности**. Система, которая ошибается редко, но каждый
    раз с уверенностью 0.99, опаснее той, что ошибается чаще и об этом говорит:
    первую нельзя использовать выборочно, а весь §3.5 построен на выборочности.

    ``o_i`` = 1, если вердикт без отказа верен. Brier — средний квадрат
    отклонения уверенности от исхода; ECE — взвешенное расхождение между
    средней уверенностью и точностью внутри корзины.
    """
    outcomes = []
    for instance_id in instance_ids:
        confidence = float(preds[instance_id].get("confidence", 0.0))
        correct = preds[instance_id]["verdict"] == golds[instance_id]["verdict"]
        outcomes.append((max(0.0, min(1.0, confidence)), 1.0 if correct else 0.0))

    if not outcomes:
        return {"brier": None, "ece": None, "bins": [], "n": 0}

    brier = sum((confidence - outcome) ** 2
                for confidence, outcome in outcomes) / len(outcomes)

    buckets = [[] for _ in range(bins)]
    for confidence, outcome in outcomes:
        index = min(bins - 1, int(confidence * bins))
        buckets[index].append((confidence, outcome))

    ece = 0.0
    rendered = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        mean_confidence = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(o for _, o in bucket) / len(bucket)
        ece += len(bucket) / len(outcomes) * abs(accuracy - mean_confidence)
        rendered.append({
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": len(bucket),
            "mean_confidence": mean_confidence,
            "accuracy": accuracy,
            "gap": accuracy - mean_confidence,
        })

    return {"brier": brier, "ece": ece, "bins": rendered, "n": len(outcomes)}


def read_stress(paths):
    """Приватные стресс-метаданные по ``instance_id`` (§5.2).

    Читаются из записей ``internal_annotation`` и **никогда** не приходят из
    публичного gold: стресс-тип — приватное поле, увидев которое система угадает
    метку (§1.4). Поэтому срезы §12.6 считает только штаб и только по своей
    половине данных.
    """
    stress = {}
    for path in paths:
        for row in read_jsonl(path):
            info = row.get("stress") or {}
            stress[row["instance_id"]] = {
                "is_stress_case": bool(info.get("is_stress_case")),
                "stress_type": info.get("stress_type"),
                "seed_instance_id": info.get("seed_instance_id"),
            }
    return stress


def _slice_metrics(golds, preds, instance_ids):
    if not instance_ids:
        return None
    verdict = verdict_metrics(golds, preds, instance_ids)
    return {
        "instances": len(instance_ids),
        "accuracy": verdict["accuracy"],
        "macro_f1": verdict["macro_f1"],
        "severe_false_warrant_rate":
            severe_false_warrant_rate(golds, preds, instance_ids)["rate"],
        "evidence_f1": set_metrics(golds, preds, instance_ids,
                                   "supporting_eids", "predicted_eids")["f1"],
    }


def stress_report(golds, preds, non_abstained_ids, stress):
    """Раздельная отчётность по чистым и стресс-инстансам (§12.6).

    Средняя цифра по срезу скрывает ровно то, ради чего бенчмарк построен:
    систему ломает не средний инстанс, а стресс-вариант. Поэтому чистые и
    стрессовые считаются отдельно, а их разность — деградация — выносится
    отдельным числом.

    Дополнительно считается отказ в разрезе стресс-типа: этого требует §12.5, и
    без приватных метаданных посчитать его нельзя.
    """
    if not stress:
        return None

    clean_ids = [i for i in non_abstained_ids
                 if not stress.get(i, {}).get("is_stress_case")]
    stress_ids = [i for i in non_abstained_ids
                  if stress.get(i, {}).get("is_stress_case")]

    by_type = {}
    for instance_id in non_abstained_ids:
        info = stress.get(instance_id, {})
        if not info.get("is_stress_case"):
            continue
        by_type.setdefault(info.get("stress_type") or "unknown", []).append(instance_id)

    abstention_by_type = {}
    for instance_id, info in stress.items():
        if instance_id not in golds or instance_id not in preds:
            continue
        key = (info.get("stress_type") or "unknown") if info.get("is_stress_case") \
            else "clean"
        bucket = abstention_by_type.setdefault(key, {"total": 0, "abstained": 0})
        bucket["total"] += 1
        bucket["abstained"] += bool(preds[instance_id].get("abstain", False))
    for bucket in abstention_by_type.values():
        bucket["rate"] = (bucket["abstained"] / bucket["total"]
                          if bucket["total"] else 0.0)

    clean = _slice_metrics(golds, preds, clean_ids)
    stressed = _slice_metrics(golds, preds, stress_ids)
    degradation = None
    if clean and stressed:
        degradation = {
            "accuracy": clean["accuracy"] - stressed["accuracy"],
            "macro_f1": clean["macro_f1"] - stressed["macro_f1"],
            "severe_false_warrant_rate":
                stressed["severe_false_warrant_rate"]
                - clean["severe_false_warrant_rate"],
        }

    return {
        "clean": clean,
        "stress": stressed,
        "degradation": degradation,
        "by_stress_type": {name: _slice_metrics(golds, preds, ids)
                           for name, ids in sorted(by_type.items())},
        "abstention_by_stress_type": dict(sorted(abstention_by_type.items())),
        "covered_instances": sum(1 for i in golds if i in stress),
    }


def cost_summary(preds, instance_ids, budget_usd=None):
    """Стоимость прогона из самих предсказаний плюс нормировка для §12.7.

    ``cost_norm`` определяется только когда задан бюджет: без него нормировать
    не на что, и любое число здесь было бы выдумано.
    """
    runtime = sum(float(preds[i].get("runtime_seconds") or 0.0) for i in instance_ids)
    gpu_seconds = 0.0
    api_cost = 0.0
    for instance_id in instance_ids:
        cost = preds[instance_id].get("estimated_cost") or {}
        gpu_seconds += float(cost.get("gpu_seconds") or 0.0)
        api_cost += float(cost.get("api_cost_usd") or 0.0)

    reported = api_cost > 0 or gpu_seconds > 0
    cost_norm = None
    if budget_usd:
        cost_norm = max(0.0, min(1.0, api_cost / budget_usd))

    return {
        "runtime_seconds": runtime,
        "gpu_seconds": gpu_seconds,
        "api_cost_usd": api_cost,
        "instances": len(instance_ids),
        "cost_reported": reported,
        "budget_usd": budget_usd,
        "cost_norm": cost_norm,
    }


#: Веса композита §12.7. Держатся здесь одним словарём, чтобы формулу можно было
#: сверить с мануалом глазами, а не вылавливать по коду.
COMPOSITE_WEIGHTS = {
    "verdict_f1": 0.30,
    "evidence_f1": 0.20,
    "issue_f1": 0.15,
    "safety": 0.15,
    "calibration": 0.10,
    "cost": 0.10,
}


def composite_score(metrics):
    """Композитный балл §12.7 — **только для внутреннего лидерборда**.

    Мануал разрешает его как мотивацию и прямо запрещает подменять им научную
    отчётность: в статью метрики идут по отдельности. Одно число складывает
    несравнимые вещи и позволяет добрать баллы в одном месте, потеряв в другом,
    — а SFWR и калибровка как раз те места, где терять нельзя.

    Слагаемые, которые нечем посчитать, не заменяются нулём: ``None`` в
    ``components`` и оговорка в ``notes`` честнее выдуманного числа, потому что
    ноль в формуле — это тоже утверждение.
    """
    notes = []
    verdict = metrics.get("verdict") or {}
    evidence = metrics.get("evidence") or {}
    issue = metrics.get("issue_tags") or {}
    sfwr = metrics.get("severe_false_warrant_rate_non_abstained") or {}
    calibration_metrics = metrics.get("calibration") or {}
    cost = metrics.get("cost") or {}

    components = {
        "verdict_f1": verdict.get("macro_f1"),
        "evidence_f1": evidence.get("f1"),
        "issue_f1": issue.get("macro_f1"),
        "safety": None if sfwr.get("rate") is None else 1.0 - sfwr["rate"],
        "calibration": None if calibration_metrics.get("ece") is None
        else 1.0 - calibration_metrics["ece"],
        "cost": None if cost.get("cost_norm") is None else 1.0 - cost["cost_norm"],
    }

    if components["cost"] is None:
        notes.append("слагаемое стоимости не посчитано: не задан --cost-budget, "
                     "нормировать не на что")
    elif not cost.get("cost_reported"):
        notes.append("бейзлайны сообщают нулевую стоимость, поэтому слагаемое "
                     "стоимости даёт полный балл всем — сравнивать по нему нельзя")

    available = {name: value for name, value in components.items() if value is not None}
    weight_sum = sum(COMPOSITE_WEIGHTS[name] for name in available)
    raw = sum(COMPOSITE_WEIGHTS[name] * value for name, value in available.items())

    if weight_sum < 1.0 - 1e-9:
        notes.append(f"посчитано {weight_sum:.2f} веса из 1.00; балл пересчитан на "
                     "доступные слагаемые и с полным баллом не сравним")

    return {
        "score": raw / weight_sum if weight_sum else None,
        "raw_weighted_sum": raw,
        "weight_covered": weight_sum,
        "components": components,
        "weights": dict(COMPOSITE_WEIGHTS),
        "notes": notes,
        "internal_only": True,
    }


def submission_report(golds, preds, missing, extra, duplicates, validation_errors):
    """Полнота сабмишена. Неполный не скорится — иначе молчание завышает метрики."""
    errors = []

    if missing:
        errors.append(
            f"нет предсказаний для {len(missing)} gold-инстанс(ов): "
            f"{missing[:5]}{' …' if len(missing) > 5 else ''}. "
            "Неуверенность выражается abstain=true, а не пропуском строки."
        )
    if extra:
        errors.append(
            f"{len(extra)} предсказан(ий) для инстансов вне gold: "
            f"{extra[:5]}{' …' if len(extra) > 5 else ''}"
        )
    if duplicates:
        errors.append(
            f"дубликаты предсказаний для {len(duplicates)} инстанс(ов): "
            f"{duplicates[:5]}{' …' if len(duplicates) > 5 else ''}"
        )
    if validation_errors:
        errors.append(f"{len(validation_errors)} предсказан(ий) не прошли валидацию")

    return {
        "complete": not errors,
        "scoreable": not errors,
        "expected_predictions": len(golds),
        "matched_predictions": len(set(golds) & set(preds)),
        "errors": errors,
    }


def make_markdown(metrics):
    counts = metrics["counts"]
    submission = metrics["submission"]

    lines = ["# Отчёт evaluator'а Track A", ""]

    if not submission["scoreable"]:
        lines += [
            "## СКОРИНГ ОТКЛОНЁН: сабмишен неполный",
            "",
            "Метрики качества не считались. Требуется ровно одно предсказание на",
            "каждый gold-инстанс: неуверенность выражается `abstain: true`, а не",
            "пропуском строки — иначе молчание на трудных инстансах завышало бы",
            "accuracy и F1.",
            "",
        ]
        for error in submission["errors"]:
            lines.append(f"- {error}")
        lines.append("")

    lines += [
        "## Сводка",
        "",
        f"- Инстансов в gold: {counts['gold_instances']}",
        f"- Предсказаний подано: {counts['predictions_submitted']}",
        f"- Валидных предсказаний: {counts['valid_predictions']}",
        f"- Отсутствующих предсказаний: {counts['missing_predictions']}",
        f"- Лишних предсказаний: {counts['extra_predictions']}",
        f"- Дубликатов: {counts['duplicate_predictions']}",
        f"- Невалидных предсказаний: {counts['invalid_predictions']}",
        f"- Предсказаний с отказом: {counts['abstained_predictions']}",
        f"- Предсказаний без отказа: {counts['non_abstained_predictions']}",
        "",
        "## Покрытие",
        "",
        f"- Coverage: {metrics['coverage']['value']:.4f} "
        f"({metrics['coverage']['answered']}/{metrics['coverage']['total']})",
        "",
    ]

    if not submission["scoreable"]:
        return "\n".join(lines) + "\n"

    lines += [
        "## Вердикт",
        "",
        f"- Accuracy: {metrics['verdict']['accuracy']:.4f}",
        f"- Знаменатель accuracy: {metrics['verdict']['accuracy_denominator']}",
        f"- Macro-F1: {metrics['verdict']['macro_f1']:.4f}",
        "",
        "Примечание: качество считается по отвеченным инстансам; инстансы с",
        "отказом видны в покрытии выше, а не растворяются в знаменателе.",
        "",
        "## Локализация evidence",
        "",
        f"- Precision: {metrics['evidence']['precision']:.4f}",
        f"- Recall: {metrics['evidence']['recall']:.4f}",
        f"- F1: {metrics['evidence']['f1']:.4f}",
        "",
        "## Issue-теги",
        "",
        f"- Precision: {metrics['issue_tags']['precision']:.4f}",
        f"- Recall: {metrics['issue_tags']['recall']:.4f}",
        f"- F1: {metrics['issue_tags']['f1']:.4f}",
        "",
        "## Отказ от ответа",
        "",
    ]

    for label, values in metrics["abstention_by_gold_verdict"].items():
        lines.append(
            f"- {label}: отказов {values['abstained_count']}/{values['gold_count']}, "
            f"доля={values['abstention_rate']:.4f}"
        )

    selective = metrics["selective_risk"]
    lines += [
        "",
        "## Selective risk",
        "",
        f"- AURC: {selective['aurc']:.4f}",
        f"- AUGRC: {selective['augrc']:.4f}",
        f"- AURC в пределах добровольного покрытия: "
        f"{selective['aurc_within_voluntary_coverage']:.4f}",
        f"- Добровольное покрытие: {selective['max_voluntary_coverage']:.4f}",
        "",
        "Кривая строится ранжированием ответов по уверенности, а не сеткой",
        "порогов: при равной уверенности у всех ответов пороговая сетка даёт",
        "одинаковое покрытие во всех точках и вырождает площадь.",
        "",
        "| Покрытие | Риск | Обобщённый риск | SFWR |",
        "|---|---|---|---|",
    ]

    for level, point in selective["at_fixed_coverage"].items():
        if point is None:
            continue
        lines.append(
            f"| {point['achieved_coverage']:.2f} | {point['risk']:.4f} | "
            f"{point['generalized_risk']:.4f} | "
            f"{point['severe_false_warrant_rate']:.4f} |"
        )

    sfwr = metrics["severe_false_warrant_rate_non_abstained"]
    lines += [
        "",
        "## Безопасность",
        "",
        f"- Severe false-warrant rate среди предсказаний без отказа: {sfwr['rate']:.4f}",
        f"- Severe false-warrant, число среди предсказаний без отказа: {sfwr['count']}",
    ]

    lines += _confusion_section(metrics)
    lines += _calibration_section(metrics)
    lines += _tags_section(metrics)
    lines += _stress_section(metrics)
    lines += _composite_section(metrics)

    return "\n".join(lines) + "\n"


def _confusion_section(metrics):
    matrix = metrics.get("confusion_matrix")
    if not matrix:
        return []
    lines = ["", "## Матрица ошибок (§12.1)", "",
             "Строки — gold, столбцы — предсказание.", "",
             "| gold \\ pred | " + " | ".join(VERDICTS) + " |",
             "|---" * (len(VERDICTS) + 1) + "|"]
    for gold_label in VERDICTS:
        row = matrix[gold_label]
        cells = " | ".join(str(row[pred_label]) for pred_label in VERDICTS)
        lines.append(f"| **{gold_label}** | {cells} |")
    return lines


def _calibration_section(metrics):
    calibration_metrics = metrics.get("calibration")
    if not calibration_metrics or calibration_metrics.get("brier") is None:
        return []
    lines = [
        "", "## Калибровка (§12.4)", "",
        f"- Brier: {calibration_metrics['brier']:.4f}",
        f"- ECE: {calibration_metrics['ece']:.4f}",
        f"- Предсказаний без отказа: {calibration_metrics['n']}",
        "",
        "Обе меры отвечают на вопрос, которого нет ни в accuracy, ни в F1: можно",
        "ли верить заявленной уверенности. Система, которая ошибается редко, но",
        "каждый раз уверенно, для выборочного аудита (§3.5) хуже той, что",
        "ошибается чаще и об этом говорит.",
        "",
        "| Корзина | Инстансов | Средняя уверенность | Точность | Разрыв |",
        "|---|---|---|---|---|",
    ]
    for bucket in calibration_metrics["bins"]:
        lines.append(
            f"| {bucket['lower']:.1f}–{bucket['upper']:.1f} | {bucket['count']} | "
            f"{bucket['mean_confidence']:.3f} | {bucket['accuracy']:.3f} | "
            f"{bucket['gap']:+.3f} |")
    return lines


def _tags_section(metrics):
    tags = metrics.get("issue_tags")
    if not tags or "macro_f1" not in tags:
        return []
    lines = [
        "", "## Issue-теги (§12.3)", "",
        f"- Micro-F1: {tags['micro']['f1']:.4f}",
        f"- Macro-F1: {tags['macro_f1']:.4f}",
        "",
        "Micro определяют частые теги, macro уравнивает редкие с частыми.",
        "Расхождение между ними и есть информация: система может прилично",
        "выглядеть на micro, полностью пропуская редкие теги.",
    ]
    if tags["most_missed"]:
        lines += ["", "Чаще всего пропускаются:", ""]
        lines += [f"- `{item['tag']}` — не назван {item['missed']} раз(а)"
                  for item in tags["most_missed"][:5]]
    return lines


def _stress_section(metrics):
    stress = metrics.get("stress")
    if not stress:
        return ["", "## Стресс-срезы (§12.6)", "",
                "Не считались: приватные метаданные не переданы (`--stress`).",
                "Стресс-тип — приватное поле, поэтому срезы доступны только штабу."]
    lines = ["", "## Стресс-срезы (§12.6)", "",
             "| Срез | Инстансов | Accuracy | Macro-F1 | SFWR | Evidence F1 |",
             "|---|---|---|---|---|---|"]
    for name, slice_metrics in (("чистые", stress["clean"]),
                                ("стрессовые", stress["stress"])):
        if slice_metrics:
            lines.append(
                f"| {name} | {slice_metrics['instances']} | "
                f"{slice_metrics['accuracy']:.3f} | {slice_metrics['macro_f1']:.3f} | "
                f"{slice_metrics['severe_false_warrant_rate']:.3f} | "
                f"{slice_metrics['evidence_f1']:.3f} |")
    for name, slice_metrics in stress["by_stress_type"].items():
        if slice_metrics:
            lines.append(
                f"| — {name} | {slice_metrics['instances']} | "
                f"{slice_metrics['accuracy']:.3f} | {slice_metrics['macro_f1']:.3f} | "
                f"{slice_metrics['severe_false_warrant_rate']:.3f} | "
                f"{slice_metrics['evidence_f1']:.3f} |")

    degradation = stress["degradation"]
    if degradation:
        lines += [
            "",
            f"Деградация чистые → стрессовые: accuracy {degradation['accuracy']:+.3f}, "
            f"macro-F1 {degradation['macro_f1']:+.3f}, "
            f"SFWR {degradation['severe_false_warrant_rate']:+.3f} "
            "(положительный SFWR означает, что под стрессом система чаще "
            "обосновывает неверное).",
        ]

    lines += ["", "Отказ по стресс-типу:", ""]
    for name, bucket in stress["abstention_by_stress_type"].items():
        lines.append(f"- {name}: {bucket['abstained']}/{bucket['total']} "
                     f"({bucket['rate']:.3f})")
    return lines


def _composite_section(metrics):
    composite = metrics.get("composite_internal")
    if not composite:
        return []
    lines = ["", "## Композит (§12.7) — только внутренний лидерборд", ""]
    if composite["score"] is None:
        lines.append("Не посчитан: ни одно слагаемое недоступно.")
        return lines
    lines += [
        f"**S = {composite['score']:.4f}**  "
        f"(покрыто {composite['weight_covered']:.2f} веса из 1.00)",
        "",
        "Мануал разрешает композит как мотивацию и запрещает подменять им научную",
        "отчётность: в статью метрики идут по отдельности. Одно число складывает",
        "несравнимые вещи и позволяет добрать в одном месте, потеряв в другом — а",
        "SFWR и калибровка как раз те места, где терять нельзя.",
        "",
        "| Слагаемое | Вес | Значение |",
        "|---|---|---|",
    ]
    for name, weight in composite["weights"].items():
        value = composite["components"][name]
        shown = "—" if value is None else f"{value:.4f}"
        lines.append(f"| {name} | {weight:.2f} | {shown} |")
    if composite["notes"]:
        lines += [""] + [f"- {note}" for note in composite["notes"]]
    return lines


def score(pred_path, gold_path, stress_paths=(), cost_budget_usd=None):
    gold_rows = read_jsonl(gold_path)
    pred_rows = read_jsonl(pred_path)

    golds = {}
    for row in gold_rows:
        golds[row["instance_id"]] = normalize_gold(row)

    preds = {}
    duplicates = []
    validation_errors = []

    for row in pred_rows:
        instance_id = row.get("instance_id", "<missing>")
        errors = validate_prediction(row)

        if errors:
            validation_errors.append({"instance_id": instance_id, "errors": errors})
            continue

        if instance_id in preds:
            duplicates.append(instance_id)
            validation_errors.append({"instance_id": instance_id,
                                      "errors": ["дубликат предсказания"]})
            continue

        preds[instance_id] = row

    missing = sorted(set(golds) - set(preds))
    extra = sorted(set(preds) - set(golds))
    duplicates = sorted(set(duplicates))

    non_abstained_ids = get_non_abstained_ids(golds, preds)
    abstained_count = sum(
        1 for instance_id in golds
        if instance_id in preds and preds[instance_id].get("abstain", False)
    )

    submission = submission_report(golds, preds, missing, extra, duplicates,
                                   validation_errors)
    total_gold = len(golds)

    metrics = {
        "submission": submission,
        "counts": {
            "gold_instances": total_gold,
            "predictions_submitted": len(pred_rows),
            "valid_predictions": len(preds),
            "invalid_predictions": len(validation_errors),
            "duplicate_predictions": len(duplicates),
            "missing_predictions": len(missing),
            "extra_predictions": len(extra),
            "abstained_predictions": abstained_count,
            "non_abstained_predictions": len(non_abstained_ids),
        },
        "coverage": {
            "answered": len(non_abstained_ids),
            "abstained": abstained_count,
            "total": total_gold,
            "value": len(non_abstained_ids) / total_gold if total_gold else 0.0,
        },
        "abstention_by_gold_verdict": abstention_by_gold_verdict(golds, preds),
        "missing_instance_ids": missing,
        "extra_instance_ids": extra,
        "duplicate_instance_ids": duplicates,
        "validation_errors": validation_errors,
    }

    # Неполный сабмишен не скорится. Иначе достаточно прислать несколько верных
    # ответов, чтобы получить высокую accuracy: пропущенные инстансы не попадали
    # бы в знаменатель. Диагностика выше остаётся, качественные метрики — null.
    if not submission["scoreable"]:
        metrics.update({
            "verdict": None,
            "evidence": None,
            "issue_tags": None,
            "severe_false_warrant_rate_non_abstained": None,
            "selective_risk": None,
            "confusion_matrix": None,
            "calibration": None,
            "stress": None,
            "cost": None,
            "composite_internal": None,
        })
        return metrics

    curve = risk_coverage_curve(golds, preds)

    metrics.update({
        "verdict": verdict_metrics(golds, preds, non_abstained_ids),
        "evidence": set_metrics(golds, preds, non_abstained_ids,
                                "supporting_eids", "predicted_eids"),
        "issue_tags": set_metrics(golds, preds, non_abstained_ids,
                                  "issue_tags", "issue_tags"),
        "severe_false_warrant_rate_non_abstained":
            severe_false_warrant_rate(golds, preds, non_abstained_ids),
        "confusion_matrix": confusion_matrix(golds, preds, non_abstained_ids),
        "calibration": calibration(golds, preds, non_abstained_ids),
        "cost": cost_summary(preds, sorted(preds), cost_budget_usd),
        "selective_risk": {
            "curve": curve,
            "aurc": aurc(curve),
            "augrc": augrc(curve),
            "aurc_within_voluntary_coverage": aurc_within_voluntary_coverage(curve),
            "max_voluntary_coverage":
                len(non_abstained_ids) / total_gold if total_gold else 0.0,
            "at_fixed_coverage": metrics_at_fixed_coverage(curve),
            "coverage_at_target_sfwr": coverage_at_target_sfwr(curve),
        },
    })

    # Теги считаются дважды: micro в старом ключе (совместимость с уже
    # написанными отчётами) и полный §12.3 рядом.
    metrics["issue_tags"] = issue_tag_metrics(golds, preds, non_abstained_ids)
    metrics["issue_tags"].update(metrics["issue_tags"]["micro"])

    stress = read_stress(stress_paths) if stress_paths else {}
    metrics["stress"] = stress_report(golds, preds, non_abstained_ids, stress)
    metrics["composite_internal"] = composite_score(metrics)

    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluator Track A для SciAudit-Stress.")
    parser.add_argument("--pred", required=True, help="Путь к JSONL с предсказаниями.")
    parser.add_argument("--gold", required=True, help="Путь к JSONL с приватным gold.")
    parser.add_argument("--out", default="metrics.json", help="Путь для записи метрик в JSON.")
    parser.add_argument("--report", default=None, help="Необязательный путь для отчёта в Markdown.")
    parser.add_argument("--stress", nargs="+", default=(),
                        help="Приватные аннотации для срезов §12.6. Стресс-тип — "
                             "приватное поле, поэтому срезы считает только штаб.")
    parser.add_argument("--cost-budget", type=float, default=None,
                        help="Бюджет прогона в долларах для нормировки слагаемого "
                             "стоимости в композите §12.7.")
    args = parser.parse_args(argv)

    metrics = score(args.pred, args.gold, stress_paths=args.stress,
                    cost_budget_usd=args.cost_budget)

    Path(args.out).write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    report = make_markdown(metrics)
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")

    print(report)

    # Неполный сабмишен — отказ скоринга, а не низкая оценка. Код возврата должен
    # ронять CI, иначе неполный прогон молча попадёт в артефакты и в лидерборд.
    return 0 if metrics["submission"]["scoreable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
