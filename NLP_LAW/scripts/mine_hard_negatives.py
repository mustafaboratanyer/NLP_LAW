import argparse
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = {
    "acaba",
    "altinda",
    "altina",
    "bir",
    "biri",
    "birini",
    "bu",
    "da",
    "de",
    "gore",
    "hangi",
    "icin",
    "ile",
    "ise",
    "kanun",
    "kanunda",
    "madde",
    "maddesi",
    "mi",
    "mu",
    "mudur",
    "nasil",
    "ne",
    "nedir",
    "olur",
    "olarak",
    "ve",
    "veya",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.casefold().replace("\u0131", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def char_ngrams(token: str, n: int = 4) -> list[str]:
    if len(token) < n:
        return []
    return [f"char:{token[index:index + n]}" for index in range(len(token) - n + 1)]


def bm25_tokens(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_RE.findall(normalize_text(text)):
        if len(token) < 2:
            continue
        tokens.append(token)
        tokens.extend(char_ngrams(token))
    return tokens


def content_tokens(text: str) -> set[str]:
    tokens = set()
    for token in TOKEN_RE.findall(normalize_text(text)):
        if len(token) < 3 or token.isdigit() or token in STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def containment_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def passage_text(row: dict) -> str:
    for key in ("passage", "candidate_passage", "positive_passage", "context"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def group_key(row: dict) -> str:
    return str(row.get("query_id") or row.get("query"))


def citation(item: dict) -> str:
    title = item.get("article_title", "").strip()
    base = f"{item.get('law_name', '')} ({item.get('law_no', '')}), {item.get('article_no', '')}"
    return f"{base} - {title}" if title else base


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


def minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    min_value = min(values.values())
    max_value = max(values.values())
    if min_value == max_value:
        return {key: 1.0 for key in values}
    return {key: (value - min_value) / (max_value - min_value) for key, value in values.items()}


def dense_search(
    queries: list[str],
    index_path: Path | None,
    config_path: Path | None,
    dense_candidates: int,
    batch_size: int,
    device: str,
) -> list[dict[int, float]]:
    if not index_path or not config_path:
        return [{} for _ in queries]

    import faiss
    from sentence_transformers import SentenceTransformer

    config = load_json(config_path)
    model_name = config["model"]
    print(f"Loading embedding model for hard-negative mining: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    index = faiss.read_index(str(index_path))
    embeddings = model.encode(
        [f"query: {query}" for query in queries],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    scores, indices = index.search(embeddings, min(dense_candidates, index.ntotal))

    dense_results = []
    for query_scores, query_indices in zip(scores, indices):
        dense_results.append(
            {
                int(index_id): float(score)
                for score, index_id in zip(query_scores, query_indices)
                if index_id >= 0
            }
        )
    return dense_results


def hybrid_candidates(
    query: str,
    bm25_index: BM25Index,
    dense_raw: dict[int, float],
    bm25_candidates: int,
    preliminary_top_k: int,
    alpha: float,
) -> list[tuple[float, float, float, int]]:
    bm25_raw = {doc_id: score for score, doc_id in bm25_index.search(query, bm25_candidates)}
    candidate_ids = set(dense_raw) | set(bm25_raw)
    dense_norm = minmax({doc_id: dense_raw.get(doc_id, 0.0) for doc_id in candidate_ids})
    bm25_norm = minmax({doc_id: bm25_raw.get(doc_id, 0.0) for doc_id in candidate_ids})

    ranked = []
    for doc_id in candidate_ids:
        dense_score = dense_norm.get(doc_id, 0.0)
        bm25_score = bm25_norm.get(doc_id, 0.0)
        preliminary_score = alpha * dense_score + (1 - alpha) * bm25_score
        ranked.append((preliminary_score, dense_score, bm25_score, doc_id))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:preliminary_top_k]


def make_pair(
    prefix: str,
    serial: int,
    query_id: str,
    query: str,
    item: dict,
    label: int,
    source: str,
    negative_type: str | None,
    scores: dict[str, float] | None = None,
) -> dict:
    row = {
        "id": f"{prefix}_{serial:06d}",
        "query_id": query_id,
        "query": query,
        "candidate_passage": item["text"],
        "label": int(label),
        "candidate_id": item["id"],
        "parent_id": item.get("parent_id"),
        "citation_label": citation(item),
        "source": source,
        "negative_type": negative_type,
        "audit_status": "auto_mined_current_retrieval_corpus",
    }
    if scores:
        row.update(scores)
    return row


def row_signature(row: dict) -> tuple[str, int, str]:
    return (
        normalize_text(str(row.get("query", ""))),
        int(row.get("label", 0)),
        str(row.get("candidate_id") or normalize_text(passage_text(row))[:240]),
    )


def align_positive_chunks(
    query_id: str,
    query: str,
    positive_rows: list[dict],
    chunks: list[dict],
    chunk_token_sets: list[set[str]],
    bm25_index: BM25Index,
    max_per_query: int,
    pool_size: int,
    min_overlap: float,
    serial_start: int,
    existing_signatures: set[tuple[str, int, str]],
) -> tuple[list[dict], set[str], int, Counter]:
    added = []
    protected_parents = set()
    stats = Counter()
    serial = serial_start

    for positive_row in positive_rows:
        positive_text = passage_text(positive_row)
        positive_tokens = content_tokens(positive_text)
        if not positive_tokens:
            continue

        best: tuple[float, float, int] | None = None
        for bm25_score, doc_id in bm25_index.search(positive_text, pool_size):
            overlap = containment_overlap(positive_tokens, chunk_token_sets[doc_id])
            if best is None or (overlap, bm25_score) > (best[0], best[1]):
                best = (overlap, bm25_score, doc_id)

        if not best or best[0] < min_overlap:
            stats["positive_alignment_skipped_low_overlap"] += 1
            continue

        _, bm25_score, doc_id = best
        item = chunks[doc_id]
        protected_parents.add(str(item.get("parent_id") or item["id"]))
        row = make_pair(
            "aligned_positive",
            serial,
            query_id,
            query,
            item,
            1,
            "CURRENT_CORPUS_ALIGNED_POSITIVE",
            None,
            {"alignment_bm25_score": float(bm25_score), "alignment_overlap": float(best[0])},
        )
        signature = row_signature(row)
        if signature in existing_signatures:
            stats["positive_alignment_skipped_duplicate"] += 1
            continue
        existing_signatures.add(signature)
        added.append(row)
        serial += 1
        stats["aligned_positives_added"] += 1
        if len(added) >= max_per_query:
            break

    return added, protected_parents, serial, stats


def load_seed_queries(path: Path | None) -> list[dict]:
    if not path:
        return []
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    data = load_json(path)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    raise ValueError("Seed query file must be JSONL or a JSON list.")


def seed_positive_rows(
    seed_rows: list[dict],
    chunks_by_parent: dict[str, list[dict]],
    max_chunks_per_seed: int,
    serial_start: int,
    existing_signatures: set[tuple[str, int, str]],
) -> tuple[list[dict], dict[str, set[str]], int, Counter]:
    added = []
    protected_by_query: dict[str, set[str]] = defaultdict(set)
    stats = Counter()
    serial = serial_start

    for index, seed in enumerate(seed_rows, start=1):
        query = str(seed.get("query", "")).strip()
        parent_id = str(seed.get("positive_parent_id") or seed.get("parent_id") or "").strip()
        if not query or not parent_id:
            stats["seed_skipped_missing_field"] += 1
            continue

        chunks = chunks_by_parent.get(parent_id, [])
        if not chunks:
            stats["seed_skipped_parent_not_found"] += 1
            continue

        query_id = str(seed.get("query_id") or f"curated_seed_{index:04d}")
        protected_by_query[query_id].add(parent_id)
        for item in chunks[:max_chunks_per_seed]:
            row = make_pair(
                "curated_positive",
                serial,
                query_id,
                query,
                item,
                1,
                "CURATED_CURRENT_CORPUS_POSITIVE",
                None,
                {"curated_parent_id": parent_id},
            )
            signature = row_signature(row)
            if signature in existing_signatures:
                stats["seed_skipped_duplicate"] += 1
                continue
            existing_signatures.add(signature)
            added.append(row)
            serial += 1
            stats["seed_positives_added"] += 1

    return added, protected_by_query, serial, stats


def mine_for_query(
    query_id: str,
    query: str,
    candidates: list[tuple[float, float, float, int]],
    chunks: list[dict],
    chunk_token_sets: list[set[str]],
    positive_token_sets: list[set[str]],
    protected_parent_ids: set[str],
    negatives_per_query: int,
    positive_overlap_skip: float,
    serial_start: int,
    existing_signatures: set[tuple[str, int, str]],
) -> tuple[list[dict], int, Counter]:
    added = []
    stats = Counter()
    serial = serial_start
    seen_parent_ids = set(protected_parent_ids)

    for preliminary_score, dense_score, bm25_score, doc_id in candidates:
        item = chunks[doc_id]
        parent_id = str(item.get("parent_id") or item["id"])
        if parent_id in seen_parent_ids:
            stats["negative_skipped_protected_parent"] += 1
            continue

        if any(
            containment_overlap(positive_tokens, chunk_token_sets[doc_id]) >= positive_overlap_skip
            for positive_tokens in positive_token_sets
        ):
            stats["negative_skipped_positive_overlap"] += 1
            continue

        row = make_pair(
            "hard_negative",
            serial,
            query_id,
            query,
            item,
            0,
            "CURRENT_CORPUS_HARD_NEGATIVE",
            "hybrid_retrieval_hard_negative",
            {
                "preliminary_score": float(preliminary_score),
                "dense_score": float(dense_score),
                "bm25_score": float(bm25_score),
            },
        )
        signature = row_signature(row)
        if signature in existing_signatures:
            stats["negative_skipped_duplicate"] += 1
            continue

        existing_signatures.add(signature)
        seen_parent_ids.add(parent_id)
        added.append(row)
        serial += 1
        stats["hard_negatives_added"] += 1
        if len(added) >= negatives_per_query:
            break

    if len(added) < negatives_per_query:
        stats["queries_with_few_negatives"] += 1
    return added, serial, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment reranker data with current-corpus hard negatives.")
    parser.add_argument("--input", required=True, help="Base reranker JSONL.")
    parser.add_argument("--output", required=True, help="Augmented reranker JSONL.")
    parser.add_argument("--chunks", default="data/processed/retrieval_chunks.json")
    parser.add_argument("--index", default=None, help="Optional FAISS index for dense mining.")
    parser.add_argument("--config", default=None, help="Index config containing the embedding model name.")
    parser.add_argument("--seed-queries", default=None, help="Optional JSONL/JSON with query + positive_parent_id.")
    parser.add_argument("--stats-out", default=None)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--dense-candidates", type=int, default=150)
    parser.add_argument("--bm25-candidates", type=int, default=250)
    parser.add_argument("--preliminary-top-k", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--negatives-per-query", type=int, default=3)
    parser.add_argument("--positive-overlap-skip", type=float, default=0.55)
    parser.add_argument("--add-aligned-positives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-aligned-positives-per-query", type=int, default=1)
    parser.add_argument("--positive-alignment-pool", type=int, default=20)
    parser.add_argument("--min-positive-overlap", type=float, default=0.55)
    parser.add_argument("--max-seed-positive-chunks", type=int, default=2)
    parser.add_argument("--include-original", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=493)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    chunks = load_json(Path(args.chunks))

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    print(f"Loaded base rows: {len(rows)}")
    print(f"Loaded query groups: {len(groups)}")
    print(f"Loaded current-corpus chunks: {len(chunks)}")
    print("Building reusable BM25 index...")

    documents = [chunk["text"] for chunk in chunks]
    chunk_token_sets = [content_tokens(document) for document in documents]
    bm25_index = BM25Index(documents)

    output_rows = list(rows) if args.include_original else []
    existing_signatures = {row_signature(row) for row in output_rows}
    stats = Counter(
        {
            "input_rows": len(rows),
            "input_query_groups": len(groups),
        }
    )

    chunks_by_parent: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_parent[str(chunk.get("parent_id") or chunk["id"])].append(chunk)

    serial = 1
    seed_rows = load_seed_queries(Path(args.seed_queries) if args.seed_queries else None)
    seed_added, seed_protected, serial, seed_stats = seed_positive_rows(
        seed_rows,
        chunks_by_parent,
        args.max_seed_positive_chunks,
        serial,
        existing_signatures,
    )
    output_rows.extend(seed_added)
    stats.update(seed_stats)

    seed_groups: dict[str, list[dict]] = defaultdict(list)
    for row in seed_added:
        seed_groups[row["query_id"]].append(row)

    # Curated seeds are mined together with the external queries.
    for seed_query_id, seed_group in seed_groups.items():
        if seed_query_id not in groups:
            groups[seed_query_id] = seed_group

    group_items = list(groups.items())
    if args.max_queries:
        group_items = group_items[: args.max_queries]
    query_ids = [key for key, _ in group_items]
    queries = [str(group[0].get("query", "")) for _, group in group_items]
    stats["mined_query_groups"] = len(group_items)
    print(f"Mining query groups: {len(group_items)}")

    dense_results = dense_search(
        queries,
        Path(args.index) if args.index else None,
        Path(args.config) if args.config else None,
        args.dense_candidates,
        args.embedding_batch_size,
        args.embedding_device,
    )

    for index, query_id in enumerate(query_ids, start=0):
        group = groups[query_id]
        query = str(group[0].get("query", "")).strip()
        positive_rows = [row for row in group if int(row.get("label", 0)) == 1]
        if not query or not positive_rows:
            stats["queries_skipped_no_positive"] += 1
            continue

        protected_parent_ids = set(seed_protected.get(query_id, set()))
        positive_token_sets = [content_tokens(passage_text(row)) for row in positive_rows if passage_text(row)]

        if args.add_aligned_positives and query_id not in seed_groups:
            aligned, aligned_protected, serial, aligned_stats = align_positive_chunks(
                query_id,
                query,
                positive_rows,
                chunks,
                chunk_token_sets,
                bm25_index,
                args.max_aligned_positives_per_query,
                args.positive_alignment_pool,
                args.min_positive_overlap,
                serial,
                existing_signatures,
            )
            output_rows.extend(aligned)
            positive_token_sets.extend(content_tokens(row["candidate_passage"]) for row in aligned)
            protected_parent_ids.update(aligned_protected)
            stats.update(aligned_stats)

        candidates = hybrid_candidates(
            query,
            bm25_index,
            dense_results[index],
            args.bm25_candidates,
            args.preliminary_top_k,
            args.alpha,
        )
        negatives, serial, negative_stats = mine_for_query(
            query_id,
            query,
            candidates,
            chunks,
            chunk_token_sets,
            positive_token_sets,
            protected_parent_ids,
            args.negatives_per_query,
            args.positive_overlap_skip,
            serial,
            existing_signatures,
        )
        output_rows.extend(negatives)
        stats.update(negative_stats)

    stats["output_rows"] = len(output_rows)
    stats["output_positive_rows"] = sum(1 for row in output_rows if int(row.get("label", 0)) == 1)
    stats["output_negative_rows"] = sum(1 for row in output_rows if int(row.get("label", 0)) == 0)

    write_jsonl(Path(args.output), output_rows)
    stats_path = Path(args.stats_out) if args.stats_out else Path(args.output).with_suffix(".stats.json")
    write_json(stats_path, dict(stats))

    print(f"Saved augmented rows: {args.output} ({len(output_rows)})")
    print(f"Saved stats: {stats_path}")
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
