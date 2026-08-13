import argparse
import json
import math
import re
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path


VERDICTS = {"warranted", "overclaimed", "contradicted", "insufficient"}
SAFE_FALLBACK_VERDICT = "insufficient"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def bm25_rank(query, evidence_pack, top_k=3, k1=1.5, b=0.75):
    if not evidence_pack:
        return []

    docs = [tokenize(ev.get("text", "")) for ev in evidence_pack]
    query_terms = tokenize(query)

    doc_lens = [len(doc) for doc in docs]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0.0

    df = defaultdict(int)
    for doc in docs:
        for term in set(doc):
            df[term] += 1

    scores = []
    n_docs = len(docs)

    for idx, doc in enumerate(docs):
        tf = Counter(doc)
        score = 0.0

        for term in query_terms:
            if term not in tf:
                continue

            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * doc_lens[idx] / avgdl) if avgdl else 1.0
            score += idf * (tf[term] * (k1 + 1)) / denom

        scores.append((score, evidence_pack[idx]))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [ev for _, ev in scores[:top_k]]


def build_prompt(instance, retrieved_evidence):
    claim = instance.get("claim", {})
    claim_text = claim.get("text") or claim.get("normalized_claim") or ""

    evidence_text = "\n".join(
        f"- {ev.get('eid')}: {ev.get('text', '')}"
        for ev in retrieved_evidence
    )

    return f"""You are auditing whether a claim is supported by the supplied evidence only.

Return strict JSON only with these fields:
- verdict: one of warranted, overclaimed, contradicted, insufficient
- confidence: number from 0 to 1
- predicted_eids: list of evidence IDs
- issue_tags: list of strings
- abstain: boolean
- rationale_short: short explanation

Claim:
{claim_text}

Evidence:
{evidence_text}
"""


def call_model(prompt, model_command=None, timeout_seconds=30):
    if not model_command:
        return ""

    completed = subprocess.run(
        model_command,
        input=prompt,
        text=True,
        capture_output=True,
        shell=True,
        timeout=timeout_seconds,
        check=False,
    )
    return completed.stdout.strip()


def parse_model_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")

    return json.loads(match.group(0))


def safe_prediction(instance_id, model_name, runtime_seconds, rationale):
    return {
        "instance_id": instance_id,
        "verdict": SAFE_FALLBACK_VERDICT,
        "confidence": 0.0,
        "predicted_eids": [],
        "issue_tags": ["model_parse_failure"],
        "abstain": True,
        "rationale_short": rationale,
        "runtime_seconds": runtime_seconds,
        "estimated_cost": {
            "gpu_seconds": 0.0,
            "api_cost_usd": 0.0,
        },
        "system_info": {
            "model": model_name,
            "uses_numeric_checker": False,
            "uses_lora": False,
            "uses_vlm": False,
        },
    }


def normalize_prediction(raw, instance_id, allowed_eids, model_name, runtime_seconds):
    verdict = raw.get("verdict", SAFE_FALLBACK_VERDICT)
    if verdict not in VERDICTS:
        verdict = SAFE_FALLBACK_VERDICT

    confidence = raw.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    predicted_eids = raw.get("predicted_eids", [])
    if not isinstance(predicted_eids, list):
        predicted_eids = []

    predicted_eids = [
        eid for eid in predicted_eids
        if isinstance(eid, str) and eid in allowed_eids
    ]

    issue_tags = raw.get("issue_tags", [])
    if not isinstance(issue_tags, list):
        issue_tags = []
    issue_tags = [tag for tag in issue_tags if isinstance(tag, str)]

    abstain = raw.get("abstain", False)
    if not isinstance(abstain, bool):
        abstain = False

    rationale_short = raw.get("rationale_short", "")
    if not isinstance(rationale_short, str) or not rationale_short.strip():
        rationale_short = "B1 produced a normalized structured prediction."

    return {
        "instance_id": instance_id,
        "verdict": verdict,
        "confidence": confidence,
        "predicted_eids": predicted_eids,
        "issue_tags": issue_tags,
        "abstain": abstain,
        "rationale_short": rationale_short[:500],
        "runtime_seconds": runtime_seconds,
        "estimated_cost": {
            "gpu_seconds": 0.0,
            "api_cost_usd": 0.0,
        },
        "system_info": {
            "model": model_name,
            "uses_numeric_checker": False,
            "uses_lora": False,
            "uses_vlm": False,
        },
    }


def audit_instance(instance, top_k=3, model_command=None, retries=1, model_fn=None):
    start = time.perf_counter()

    instance_id = instance["instance_id"]
    claim = instance.get("claim", {})
    claim_text = claim.get("text") or claim.get("normalized_claim") or ""
    evidence_pack = instance.get("evidence_pack", [])
    allowed_eids = {ev.get("eid") for ev in evidence_pack if ev.get("eid")}

    retrieved = bm25_rank(claim_text, evidence_pack, top_k=top_k)
    prompt = build_prompt(instance, retrieved)

    model_name = model_command if model_command else "mocked_or_safe_fallback"

    for _ in range(retries + 1):
        try:
            if model_fn is not None:
                model_output = model_fn(prompt)
                model_name = getattr(model_fn, "__name__", "mocked_model")
            else:
                model_output = call_model(prompt, model_command=model_command)

            raw = parse_model_json(model_output)
            runtime_seconds = time.perf_counter() - start
            return normalize_prediction(
                raw=raw,
                instance_id=instance_id,
                allowed_eids=allowed_eids,
                model_name=model_name,
                runtime_seconds=runtime_seconds,
            )
        except Exception:
            continue

    runtime_seconds = time.perf_counter() - start
    return safe_prediction(
        instance_id=instance_id,
        model_name=model_name,
        runtime_seconds=runtime_seconds,
        rationale="Model returned invalid JSON after retries; safe fallback used.",
    )


def run(input_path, output_path, top_k=3, model_command=None, retries=1, model_fn=None):
    instances = read_jsonl(input_path)
    predictions = [
        audit_instance(
            instance,
            top_k=top_k,
            model_command=model_command,
            retries=retries,
            model_fn=model_fn,
        )
        for instance in instances
    ]
    write_jsonl(output_path, predictions)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="B1 BM25 retrieval + structured model audit baseline.")
    parser.add_argument("--input", required=True, help="Path to input JSONL.")
    parser.add_argument("--output", required=True, help="Path to output predictions JSONL.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of evidence units retrieved by BM25.")
    parser.add_argument("--model-command", default=None, help="Shell command for open model inference. Prompt is passed to stdin.")
    parser.add_argument("--retries", type=int, default=1, help="Retries after invalid model JSON.")
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
        model_command=args.model_command,
        retries=args.retries,
    )


if __name__ == "__main__":
    main()
