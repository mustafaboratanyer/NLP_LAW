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
STOPWORDS = {
    "acaba",
    "altinda",
    "altina",
    "ama",
    "ancak",
    "bir",
    "biri",
    "birini",
    "bu",
    "da",
    "de",
    "diye",
    "gore",
    "hangi",
    "icin",
    "ile",
    "ise",
    "kanun",
    "kanuna",
    "kanunda",
    "kapsaminda",
    "madde",
    "maddede",
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


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def canonical_key(key: str) -> str:
    return normalize_text(key).replace(" ", "_")


def row_get(row: dict, *names: str) -> str:
    normalized = {canonical_key(key): value for key, value in row.items()}
    for name in names:
        value = normalized.get(canonical_key(name))
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


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


def law_tokens(text: str) -> set[str]:
    ignored = {"turkiye", "cumhuriyeti", "kanunu", "kanun", "turk", "sayili", "hakkinda", "dair"}
    return {
        token
        for token in TOKEN_RE.findall(normalize_text(text))
        if len(token) > 2 and token not in ignored
    }


def law_match_score(source: str, law_name: str) -> float:
    source_norm = normalize_text(source)
    law_norm = normalize_text(law_name)
    if not source_norm or not law_norm:
        return 0.0
    if source_norm in law_norm or law_norm in source_norm:
        return 1.0
    source_tokens = law_tokens(source)
    law_name_tokens = law_tokens(law_name)
    return containment_overlap(source_tokens, law_name_tokens)


def citation(item: dict) -> str:
    title = clean_space(item.get("article_title", ""))
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

    def search(self, query: str, top_k: int, allowed_doc_ids: set[int] | None = None) -> list[tuple[float, int]]:
        query_terms = Counter(bm25_tokens(query))
        scores: dict[int, float] = defaultdict(float)

        for term, query_freq in query_terms.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            df = self.doc_freqs[term]
            idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            for doc_id, term_freq in postings:
                if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                    continue
                doc_length = self.doc_lengths[doc_id] or 1
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                scores[doc_id] += idf * (term_freq * (self.k1 + 1) / denominator) * query_freq

        return sorted(((score, doc_id) for doc_id, score in scores.items()), reverse=True)[:top_k]


def read_kaggle_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_score(value: str) -> float:
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return -1.0


def build_law_candidates(chunks: list[dict]) -> dict[str, set[int]]:
    sources: dict[str, set[int]] = defaultdict(set)
    law_names = sorted({clean_space(chunk.get("law_name", "")) for chunk in chunks if chunk.get("law_name")})
    for law_name in law_names:
        matching_ids = {index for index, chunk in enumerate(chunks) if chunk.get("law_name") == law_name}
        sources[normalize_text(law_name)] |= matching_ids
    return sources


def allowed_ids_for_source(source: str, chunks: list[dict], min_law_overlap: float) -> set[int]:
    source = clean_space(source)
    if not source:
        return set()
    allowed = set()
    for index, chunk in enumerate(chunks):
        if law_match_score(source, chunk.get("law_name", "")) >= min_law_overlap:
            allowed.add(index)
    return allowed


def chunk_search_text(chunk: dict) -> str:
    return " ".join(
        clean_space(chunk.get(key, ""))
        for key in ("law_name", "law_no", "article_no", "article_title", "text")
    )


def make_pair(
    row_id: str,
    query_id: str,
    query: str,
    candidate: dict,
    label: int,
    positive_parent_id: str,
    source: str,
    negative_type: str | None,
    extras: dict[str, Any],
) -> dict:
    return {
        "id": row_id,
        "query_id": query_id,
        "query": query,
        "candidate_passage": candidate["text"],
        "label": int(label),
        "candidate_id": candidate["id"],
        "parent_id": candidate.get("parent_id"),
        "positive_parent_id": positive_parent_id,
        "citation_label": citation(candidate),
        "source": source,
        "negative_type": negative_type,
        "audit_status": "auto_aligned_high_confidence",
        **extras,
    }


def build_dataset(args: argparse.Namespace) -> tuple[list[dict], dict, list[dict]]:
    rng = random.Random(args.seed)
    raw_rows = read_kaggle_rows(Path(args.csv))
    chunks = load_json(Path(args.chunks))
    documents = [chunk_search_text(chunk) for chunk in chunks]
    chunk_token_sets = [content_tokens(chunk["text"]) for chunk in chunks]
    index = BM25Index(documents)

    stats: Counter[str] = Counter()
    output_rows: list[dict] = []
    audit_rows: list[dict] = []
    seen_pairs: set[tuple[str, str, int]] = set()
    allowed_cache: dict[str, set[int]] = {}
    positive_cache: dict[tuple[str, str], list[tuple[float, int]]] = {}

    for input_index, row in enumerate(raw_rows):
        stats["input_rows"] += 1
        if args.max_rows and input_index >= args.max_rows:
            stats["stopped_by_max_rows"] += 1
            break

        query = clean_space(row_get(row, "soru", "question", "query"))
        answer = clean_space(row_get(row, "cevap", "answer", "response"))
        context = clean_space(row_get(row, "context", "passage", "source_text"))
        source_name = clean_space(row_get(row, "kaynak", "source"))
        score = parse_score(row_get(row, "Score", "score"))

        if not query or not answer or not context:
            stats["skipped_missing_fields"] += 1
            continue
        if score < args.min_score:
            stats["skipped_low_dataset_score"] += 1
            continue

        source_key = normalize_text(source_name)
        if source_key not in allowed_cache:
            allowed_cache[source_key] = allowed_ids_for_source(source_name, chunks, args.min_law_overlap)
        allowed_doc_ids = allowed_cache[source_key]
        if args.require_law_match and not allowed_doc_ids:
            stats["skipped_no_law_match_in_corpus"] += 1
            continue

        context_token_set = content_tokens(context)
        answer_token_set = content_tokens(answer)

        cache_key = (source_key, normalize_text(context))
        matches = positive_cache.get(cache_key)
        if matches is None:
            candidate_doc_ids = allowed_doc_ids if allowed_doc_ids else set(range(len(chunks)))
            scored_matches = []
            for doc_id in candidate_doc_ids:
                context_overlap = containment_overlap(context_token_set, chunk_token_sets[doc_id])
                if context_overlap < args.min_context_overlap:
                    continue
                scored_matches.append((context_overlap, doc_id))
            scored_matches.sort(key=lambda item: item[0], reverse=True)
            matches = scored_matches[: args.positive_pool]
            positive_cache[cache_key] = matches
        if not matches:
            stats["skipped_no_positive_match"] += 1
            continue

        accepted_positive_ids = []
        best_score = matches[0][0]
        second_score = matches[1][0] if len(matches) > 1 else 0.0
        best_margin = (best_score - second_score) / max(best_score, 1e-9)

        for match_score, doc_id in matches:
            candidate = chunks[doc_id]
            context_overlap = match_score
            answer_overlap = containment_overlap(answer_token_set, chunk_token_sets[doc_id])
            if answer_overlap < args.min_answer_overlap:
                continue
            accepted_positive_ids.append((doc_id, match_score, context_overlap, answer_overlap))
            if len(accepted_positive_ids) >= args.max_positives_per_query:
                break

        if not accepted_positive_ids:
            stats["skipped_low_overlap"] += 1
            if len(audit_rows) < args.max_audit_rows:
                audit_rows.append(
                    {
                        "status": "skipped_low_overlap",
                        "query": query,
                        "source": source_name,
                        "score": score,
                        "top_citation": citation(chunks[matches[0][1]]),
                        "top_context_overlap": round(
                            containment_overlap(context_token_set, content_tokens(chunks[matches[0][1]]["text"])),
                            4,
                        ),
                        "top_answer_overlap": round(
                            containment_overlap(answer_token_set, content_tokens(chunks[matches[0][1]]["text"])),
                            4,
                        ),
                    }
                )
            continue

        positive_parent_ids = {chunks[doc_id].get("parent_id") for doc_id, _, _, _ in accepted_positive_ids}
        query_id = f"kaggle_law_{input_index:06d}"
        extras_base = {
            "original_source": source_name,
            "dataset_score": score,
            "positive_match_margin": round(best_margin, 6),
        }

        positive_added = 0
        for doc_id, match_score, context_overlap, answer_overlap in accepted_positive_ids:
            candidate = chunks[doc_id]
            signature = (normalize_text(query), candidate["id"], 1)
            if signature in seen_pairs:
                stats["skipped_duplicate_positive"] += 1
                continue
            seen_pairs.add(signature)
            output_rows.append(
                make_pair(
                    row_id=f"kagglelaw_pos_{len(output_rows):07d}",
                    query_id=query_id,
                    query=query,
                    candidate=candidate,
                    label=1,
                    positive_parent_id=str(candidate.get("parent_id") or candidate["id"]),
                    source="kaggle_turkish_law_dataset",
                    negative_type=None,
                    extras={
                        **extras_base,
                        "match_score": round(match_score, 6),
                        "context_overlap": round(context_overlap, 6),
                        "answer_overlap": round(answer_overlap, 6),
                    },
                )
            )
            positive_added += 1

        if not positive_added:
            continue

        negative_query = " ".join([query, answer])
        negative_candidates = index.search(negative_query, top_k=args.hard_negative_pool)
        negatives_added = 0
        for neg_score, neg_doc_id in negative_candidates:
            candidate = chunks[neg_doc_id]
            if candidate.get("parent_id") in positive_parent_ids:
                stats["negative_skipped_same_parent"] += 1
                continue
            negative_context_overlap = containment_overlap(context_token_set, chunk_token_sets[neg_doc_id])
            if negative_context_overlap >= args.protect_context_overlap:
                stats["negative_skipped_context_overlap"] += 1
                continue
            signature = (normalize_text(query), candidate["id"], 0)
            if signature in seen_pairs:
                stats["negative_skipped_duplicate"] += 1
                continue
            seen_pairs.add(signature)
            output_rows.append(
                make_pair(
                    row_id=f"kagglelaw_neg_{len(output_rows):07d}",
                    query_id=query_id,
                    query=query,
                    candidate=candidate,
                    label=0,
                    positive_parent_id="|".join(sorted(str(parent_id) for parent_id in positive_parent_ids)),
                    source="kaggle_turkish_law_dataset",
                    negative_type="bm25_hard_negative",
                    extras={
                        **extras_base,
                        "negative_match_score": round(neg_score, 6),
                        "negative_context_overlap": round(negative_context_overlap, 6),
                    },
                )
            )
            negatives_added += 1
            if negatives_added >= args.negatives_per_query:
                break

        stats["accepted_queries"] += 1
        stats["positive_rows"] += positive_added
        stats["negative_rows"] += negatives_added
        if negatives_added < args.negatives_per_query:
            stats["queries_with_few_negatives"] += 1

        if len(audit_rows) < args.max_audit_rows:
            first_positive = chunks[accepted_positive_ids[0][0]]
            audit_rows.append(
                {
                    "status": "accepted",
                    "query": query,
                    "source": source_name,
                    "score": score,
                    "positive_citation": citation(first_positive),
                    "positive_id": first_positive["id"],
                    "context_overlap": round(accepted_positive_ids[0][2], 4),
                    "answer_overlap": round(accepted_positive_ids[0][3], 4),
                    "negatives_added": negatives_added,
                }
            )

    rng.shuffle(output_rows)
    stats["output_rows"] = len(output_rows)
    stats["unique_query_ids"] = len({row["query_id"] for row in output_rows})
    stats["unique_queries"] = len({normalize_text(row["query"]) for row in output_rows})
    stats["output_positive_rows"] = sum(1 for row in output_rows if row["label"] == 1)
    stats["output_negative_rows"] = sum(1 for row in output_rows if row["label"] == 0)
    stats["min_score"] = args.min_score
    stats["min_context_overlap"] = args.min_context_overlap
    stats["min_answer_overlap"] = args.min_answer_overlap
    stats["require_law_match"] = int(args.require_law_match)
    return output_rows, dict(stats), audit_rows


def write_audit(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the Kaggle Turkish law QA dataset into high-confidence reranker pairs."
    )
    parser.add_argument("--csv", default="data/external/kaggle_law_dataset/turkish_law_dataset.csv")
    parser.add_argument("--chunks", default="data/processed/retrieval_chunks.json")
    parser.add_argument("--output", default="data/reranker/kaggle_law_reranker.jsonl")
    parser.add_argument("--stats-out", default="data/reranker/kaggle_law_reranker.stats.json")
    parser.add_argument("--audit-out", default="data/reranker/kaggle_law_reranker.audit.csv")
    parser.add_argument("--min-score", type=float, default=9.0)
    parser.add_argument("--min-context-overlap", type=float, default=0.55)
    parser.add_argument("--min-answer-overlap", type=float, default=0.05)
    parser.add_argument("--protect-context-overlap", type=float, default=0.65)
    parser.add_argument("--min-law-overlap", type=float, default=0.6)
    parser.add_argument("--positive-pool", type=int, default=8)
    parser.add_argument("--max-positives-per-query", type=int, default=2)
    parser.add_argument("--negatives-per-query", type=int, default=3)
    parser.add_argument("--hard-negative-pool", type=int, default=80)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-audit-rows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=493)
    parser.add_argument("--allow-unmatched-law", action="store_true")
    args = parser.parse_args()
    args.require_law_match = not args.allow_unmatched_law

    rows, stats, audit_rows = build_dataset(args)
    write_jsonl(Path(args.output), rows)
    write_json(Path(args.stats_out), stats)
    write_audit(Path(args.audit_out), audit_rows)

    print(f"Saved reranker rows: {args.output} ({len(rows)})")
    print(f"Saved stats: {args.stats_out}")
    print(f"Saved audit sample: {args.audit_out}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
