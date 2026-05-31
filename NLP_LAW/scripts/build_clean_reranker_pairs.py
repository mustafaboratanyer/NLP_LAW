import argparse
import csv
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = str(text or "").casefold().replace("\u0131", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def char_ngrams(token: str, n: int = 4) -> list[str]:
    if len(token) < n:
        return []
    return [f"char:{token[index:index+n]}" for index in range(len(token) - n + 1)]


def bm25_tokens(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_RE.findall(normalize_text(text)):
        if len(token) < 2:
            continue
        tokens.append(token)
        tokens.extend(char_ngrams(token))
    return tokens


def citation(item: dict) -> str:
    title = str(item.get("article_title", "")).strip()
    base = f"{item.get('law_name', '')} ({item.get('law_no', '')}), {item.get('article_no', '')}"
    return f"{base} - {title}" if title else base


def chunk_doc_text(chunk: dict) -> str:
    return " ".join(
        str(chunk.get(key, ""))
        for key in ("law_name", "law_no", "article_no", "article_title", "text")
    )


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_freqs: Counter[str] = Counter()

        for doc_id, document in enumerate(documents):
            counts = Counter(bm25_tokens(document))
            self.doc_lengths.append(sum(counts.values()))
            for term, freq in counts.items():
                self.postings[term].append((doc_id, freq))
                self.doc_freqs[term] += 1

        self.doc_count = len(documents)
        self.avg_doc_length = sum(self.doc_lengths) / max(1, self.doc_count)

    def search(self, query: str, top_k: int) -> list[tuple[float, int]]:
        query_terms = Counter(bm25_tokens(query))
        scores: dict[int, float] = defaultdict(float)

        for term, query_freq in query_terms.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            df = self.doc_freqs[term]
            idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            for doc_id, term_freq in postings:
                doc_length = self.doc_lengths[doc_id] or 1
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                scores[doc_id] += idf * (term_freq * (self.k1 + 1) / denominator) * query_freq

        return sorted(((score, doc_id) for doc_id, score in scores.items()), reverse=True)[:top_k]


def load_questions(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def make_pair(
    row_id: str,
    query_id: str,
    question_row: dict,
    chunk: dict,
    label: int,
    positive_parent_id: str,
    negative_type: str | None,
    bm25_score: float | None = None,
) -> dict:
    row = {
        "id": row_id,
        "query_id": query_id,
        "query": question_row["question"],
        "candidate_passage": chunk["text"],
        "label": int(label),
        "candidate_id": chunk["id"],
        "parent_id": chunk.get("parent_id"),
        "positive_parent_id": positive_parent_id,
        "citation_label": citation(chunk),
        "source": "clean_1000_question_set",
        "question_source_dataset": question_row.get("source_dataset", ""),
        "negative_type": negative_type,
        "audit_status": "clean_question_parent_id_hard_negative",
    }
    if bm25_score is not None:
        row["bm25_score"] = round(float(bm25_score), 6)
    return row


def build_pairs(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rng = random.Random(args.seed)
    questions = load_questions(Path(args.questions))
    chunks = load_json(Path(args.chunks))
    parent_to_chunks: dict[str, list[int]] = defaultdict(list)
    for index, chunk in enumerate(chunks):
        parent_to_chunks[str(chunk.get("parent_id") or chunk.get("id"))].append(index)

    index = BM25Index([chunk_doc_text(chunk) for chunk in chunks])
    pairs = []
    stats = Counter()
    seen = set()

    for question_index, question_row in enumerate(questions):
        question = question_row["question"]
        positive_parent_id = question_row["positive_parent_id"]
        positive_chunk_ids = parent_to_chunks.get(positive_parent_id, [])
        if not positive_chunk_ids:
            stats["skipped_no_positive_chunk"] += 1
            continue

        query_id = f"cleanq_{question_index:05d}"
        positive_added = 0
        for positive_chunk_id in positive_chunk_ids[: args.max_positive_chunks]:
            chunk = chunks[positive_chunk_id]
            signature = (query_id, chunk["id"], 1)
            if signature in seen:
                continue
            seen.add(signature)
            pairs.append(
                make_pair(
                    row_id=f"clean_pos_{len(pairs):07d}",
                    query_id=query_id,
                    question_row=question_row,
                    chunk=chunk,
                    label=1,
                    positive_parent_id=positive_parent_id,
                    negative_type=None,
                )
            )
            positive_added += 1

        if not positive_added:
            stats["skipped_no_positive_added"] += 1
            continue

        negatives_added = 0
        for bm25_score, chunk_id in index.search(question, args.hard_negative_pool):
            chunk = chunks[chunk_id]
            if str(chunk.get("parent_id") or chunk.get("id")) == positive_parent_id:
                stats["negative_skipped_same_parent"] += 1
                continue
            signature = (query_id, chunk["id"], 0)
            if signature in seen:
                continue
            seen.add(signature)
            pairs.append(
                make_pair(
                    row_id=f"clean_neg_{len(pairs):07d}",
                    query_id=query_id,
                    question_row=question_row,
                    chunk=chunk,
                    label=0,
                    positive_parent_id=positive_parent_id,
                    negative_type="bm25_hard_negative",
                    bm25_score=bm25_score,
                )
            )
            negatives_added += 1
            if negatives_added >= args.negatives_per_query:
                break

        stats["queries_used"] += 1
        stats["positive_rows"] += positive_added
        stats["negative_rows"] += negatives_added
        if negatives_added < args.negatives_per_query:
            stats["queries_with_few_negatives"] += 1

    rng.shuffle(pairs)
    stats["input_questions"] = len(questions)
    stats["output_rows"] = len(pairs)
    stats["unique_query_ids"] = len({row["query_id"] for row in pairs})
    stats["output_positive_rows"] = sum(1 for row in pairs if row["label"] == 1)
    stats["output_negative_rows"] = sum(1 for row in pairs if row["label"] == 0)
    stats["negatives_per_query"] = args.negatives_per_query
    stats["max_positive_chunks"] = args.max_positive_chunks
    return pairs, dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reranker pair JSONL from clean question-parent CSV.")
    parser.add_argument("--questions", default="data/reranker/clean_reranker_train_questions.csv")
    parser.add_argument("--chunks", default="data/processed/retrieval_chunks.json")
    parser.add_argument("--output", default="data/reranker/clean_reranker_pairs.jsonl")
    parser.add_argument("--stats-out", default="data/reranker/clean_reranker_pairs.stats.json")
    parser.add_argument("--negatives-per-query", type=int, default=5)
    parser.add_argument("--hard-negative-pool", type=int, default=80)
    parser.add_argument("--max-positive-chunks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=493)
    args = parser.parse_args()

    pairs, stats = build_pairs(args)
    write_jsonl(Path(args.output), pairs)
    write_json(Path(args.stats_out), stats)

    print(f"Saved reranker pairs: {args.output} ({len(pairs)})")
    print(f"Saved stats: {args.stats_out}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
