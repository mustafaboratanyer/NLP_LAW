import csv
import json
import re
from pathlib import Path
from typing import Any


QUESTION_KEYS = ("question", "soru", "query")
ANSWER_KEYS = (
    "expected_answer",
    "reference_answer",
    "gold_answer",
    "answer",
    "cevap",
)
DOCUMENT_KEYS = (
    "expected_document_id",
    "corpus_row_id",
    "gold_parent_id",
    "parent_id",
    "doc_id",
    "source_id",
)
TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def first_present(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def load_json_rows(path: Path) -> list[dict]:
    data: Any = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        for key in ("benchmark", "questions", "examples", "data"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError("Benchmark JSON must contain an object or a list of objects.")
    return [dict(row) for row in data if isinstance(row, dict)]


def load_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at line {line_no}.")
            rows.append(row)
    return rows


def load_benchmark_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            raw_rows = [dict(row) for row in csv.DictReader(file)]
    elif suffix == ".jsonl":
        raw_rows = load_jsonl_rows(path)
    elif suffix == ".json":
        raw_rows = load_json_rows(path)
    else:
        raise ValueError("Benchmark format must be CSV, JSON, or JSONL.")

    rows = []
    for index, row in enumerate(raw_rows, start=1):
        question = first_present(row, QUESTION_KEYS)
        if not question:
            continue

        gold_document_ids = []
        explicit_id = first_present(row, DOCUMENT_KEYS)
        if explicit_id:
            gold_document_ids.extend(
                item.strip() for item in re.split(r"[|;,]", explicit_id) if item.strip()
            )
        for source in row.get("gold_sources", []) or []:
            if isinstance(source, dict):
                source_id = first_present(source, DOCUMENT_KEYS)
                if source_id:
                    gold_document_ids.append(source_id)

        rows.append(
            {
                "question_id": first_present(
                    row,
                    ("question_id", "row_id", "id"),
                )
                or f"custom_{index:04d}",
                "question": question,
                "expected_answer": first_present(row, ANSWER_KEYS),
                "gold_document_ids": sorted(set(gold_document_ids)),
            }
        )

    if not rows:
        raise ValueError(
            "No benchmark questions found. Use a 'question', 'soru', or 'query' column."
        )
    return rows


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall(str(text).casefold()))


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    prediction_counts = {}
    reference_counts = {}
    for token in prediction_tokens:
        prediction_counts[token] = prediction_counts.get(token, 0) + 1
    for token in reference_tokens:
        reference_counts[token] = reference_counts.get(token, 0) + 1

    overlap = sum(
        min(count, reference_counts.get(token, 0))
        for token, count in prediction_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    previous = [0] * (len(reference_tokens) + 1)
    for prediction_token in prediction_tokens:
        current = [0]
        for index, reference_token in enumerate(reference_tokens, start=1):
            if prediction_token == reference_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current

    lcs_length = previous[-1]
    if lcs_length == 0:
        return 0.0
    precision = lcs_length / len(prediction_tokens)
    recall = lcs_length / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer(prediction: str, reference: str) -> dict[str, float | None]:
    if not str(reference).strip():
        return {"em": None, "f1": None, "rouge_l": None}
    return {
        "em": exact_match(prediction, reference),
        "f1": token_f1(prediction, reference),
        "rouge_l": rouge_l_f1(prediction, reference),
    }


def retrieval_metrics(contexts: list[dict], gold_document_ids: list[str]) -> dict:
    if not gold_document_ids:
        return {"retrieval_hit": None, "reciprocal_rank": None}

    gold = set(gold_document_ids)
    for rank, context in enumerate(contexts, start=1):
        if str(context.get("parent_id", "")) in gold:
            return {"retrieval_hit": 1.0, "reciprocal_rank": 1.0 / rank}
    return {"retrieval_hit": 0.0, "reciprocal_rank": 0.0}


def average_available(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None

