import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
ARTICLE_RE = re.compile(r"(?:madde|m\.?)\s*([0-9]+(?:/[a-z])?)", re.IGNORECASE)
SOURCE_ID_RE = re.compile(r"_(?P<law_no>[0-9]{3,4})_[a-z0-9_]+?_m(?P<article_no>[0-9]+(?:/[a-z])?)$", re.IGNORECASE)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text: str) -> str:
    text = str(text).casefold().replace("\u0131", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def normalize_article_no(text: str) -> str:
    value = normalize_text(text)
    value = re.sub(r"\([^)]*\)", " ", value)
    value = value.replace("geçici", "gecici").replace("mükerrer", "mukerrer")
    value = re.sub(r"\b(madde|mad|m|article)\b\.?", " ", value)
    value = value.replace("ek madde", "ek")
    value = re.sub(r"[^a-z0-9/]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -")
    value = value.replace(" ", "_")
    value = re.sub(r"_+$", "", value)
    return value


def id_article_key(article_id: str, law_no: str) -> str:
    prefix = f"{law_no}_"
    value = article_id[len(prefix) :] if article_id.startswith(prefix) else article_id
    value = value.replace("madde_", "")
    value = value.replace("ek_", "ek_").replace("gecici_", "gecici_").replace("mukerrer_", "mukerrer_")
    return normalize_article_no(value)


def article_key_variants(article_no: str) -> set[str]:
    key = normalize_article_no(article_no)
    variants = {key}
    if key.startswith("ek_"):
        variants.add(key.replace("ek_", "", 1))
    if key.startswith("gecici_"):
        variants.add(key.replace("gecici_", "", 1))
    if key.startswith("mukerrer_"):
        variants.add(key.replace("mukerrer_", "", 1))
    return {variant for variant in variants if variant}


def citation(item: dict) -> str:
    title = item.get("article_title", "").strip()
    base = f"{item.get('law_name', '')} ({item.get('law_no', '')}), {item.get('article_no', '')}"
    return f"{base} - {title}" if title else base


def build_corpus_lookups(corpus_path: Path) -> tuple[dict[str, dict], dict[tuple[str, str], set[str]], dict[str, str]]:
    corpus = load_json(corpus_path)
    by_id = {row["id"]: row for row in corpus}
    by_law_article: dict[tuple[str, str], set[str]] = defaultdict(set)
    law_name_to_no: dict[str, str] = {}

    for row in corpus:
        law_no = str(row.get("law_no", "")).strip()
        if not law_no:
            continue
        law_name_to_no[normalize_text(row.get("law_name", ""))] = law_no
        for key in {
            normalize_article_no(row.get("article_no", "")),
            id_article_key(row["id"], law_no),
        }:
            if key:
                by_law_article[(law_no, key)].add(row["id"])

    # Common short or alternate names in our benchmark files.
    aliases = {
        "bilgi edinme kanunu": "4982",
        "bilgi edinme hakki kanunu": "4982",
        "bilgi edinme hakkı kanunu": "4982",
        "turkiye cumhuriyeti is kanunu": "4857",
        "türkiye cumhuriyeti iş kanunu": "4857",
        "is kanunu": "4857",
        "iş kanunu": "4857",
        "turk medeni kanunu": "4721",
        "türk medeni kanunu": "4721",
        "turk medeni\u0307 kanunu": "4721",
        "turk ceza kanunu": "5237",
        "türk ceza kanunu": "5237",
        "ceza muhakemesi kanunu": "5271",
        "turk borclar kanunu": "6098",
        "türk borçlar kanunu": "6098",
        "turkiye cumhuriyeti anayasasi": "2709",
        "türkiye cumhuriyeti anayasası": "2709",
        "anayasa": "2709",
    }
    for name, law_no in aliases.items():
        law_name_to_no[normalize_text(name)] = law_no

    return by_id, by_law_article, law_name_to_no


def resolve_gold_ids(
    law_no: str,
    article_no: str,
    by_law_article: dict[tuple[str, str], set[str]],
) -> set[str]:
    resolved = set()
    law_no = str(law_no).strip()
    if not law_no:
        return resolved
    for variant in article_key_variants(article_no):
        resolved.update(by_law_article.get((law_no, variant), set()))
    return resolved


def parse_source_id(text: str) -> tuple[str, str] | None:
    match = SOURCE_ID_RE.search(str(text))
    if not match:
        return None
    return match.group("law_no"), match.group("article_no")


def extract_article_no_from_context(context: str) -> str:
    match = ARTICLE_RE.search(context or "")
    return match.group(1) if match else ""


def split_article_numbers(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[,;|]+", str(text))
    cleaned = []
    for part in parts:
        value = part.strip()
        if value:
            cleaned.append(value)
    return cleaned


def infer_law_no(row: dict, law_name_to_no: dict[str, str]) -> str:
    fields = [
        row.get("kanun_adi", ""),
        row.get("kaynak", ""),
        row.get("question", ""),
        row.get("soru", ""),
        row.get("context", ""),
    ]
    for field in fields:
        normalized = normalize_text(field)
        if normalized in law_name_to_no:
            return law_name_to_no[normalized]
    joined = normalize_text(" ".join(str(field) for field in fields))
    # Prefer longer aliases first so "Anayasa" does not beat a full law name.
    for law_name, law_no in sorted(law_name_to_no.items(), key=lambda item: len(item[0]), reverse=True):
        if law_name and law_name in joined:
            return law_no
    return ""


def query_law_hints(question: str, law_name_to_no: dict[str, str]) -> set[str]:
    normalized = normalize_text(question)
    hints = set()
    for law_name, law_no in sorted(law_name_to_no.items(), key=lambda item: len(item[0]), reverse=True):
        if law_name and law_name in normalized:
            hints.add(law_no)
    return hints


def query_article_hints(question: str) -> set[str]:
    return {
        variant
        for match in ARTICLE_RE.findall(question or "")
        for variant in article_key_variants(match)
    }


def chunk_article_keys(chunk: dict) -> set[str]:
    law_no = str(chunk.get("law_no", "")).strip()
    keys = set(article_key_variants(chunk.get("article_no", "")))
    parent_id = str(chunk.get("parent_id") or chunk.get("id") or "")
    if law_no and parent_id:
        keys.add(id_article_key(parent_id, law_no))
    return {key for key in keys if key}


def apply_metadata_boost(
    ranked: list[tuple],
    metadata: list[dict],
    question: str,
    law_name_to_no: dict[str, str],
    law_boost: float,
    article_boost: float,
    exact_citation_boost: float,
) -> list[tuple]:
    law_hints = query_law_hints(question, law_name_to_no)
    article_hints = query_article_hints(question)
    if not law_hints and not article_hints:
        return ranked

    boosted = []
    for item in ranked:
        score = float(item[0])
        doc_id = item[-1]
        chunk = metadata[doc_id]
        chunk_law_no = str(chunk.get("law_no", "")).strip()
        law_match = bool(law_hints and chunk_law_no in law_hints)
        article_match = bool(article_hints and chunk_article_keys(chunk) & article_hints)

        bonus = 0.0
        if law_match:
            bonus += law_boost
        if article_match:
            # Article numbers are ambiguous across laws, so use a smaller bonus
            # unless the law name also matches.
            bonus += article_boost if law_match else article_boost * 0.5
        if law_match and article_match:
            bonus += exact_citation_boost

        boosted.append((score + bonus, *item[1:]))

    boosted.sort(key=lambda row: row[0], reverse=True)
    return boosted


def benchmark_from_json(
    path: Path,
    by_law_article: dict[tuple[str, str], set[str]],
) -> tuple[list[dict], Counter]:
    rows = load_json(path)
    examples = []
    stats = Counter(total_rows=len(rows))
    known_parent_ids = {item for values in by_law_article.values() for item in values}
    for row in rows:
        gold_ids = set()
        raw_gold = []
        for source in row.get("gold_sources", []):
            law_no = str(source.get("law_no", "")).strip()
            article_no = str(source.get("article_no", "")).strip()
            explicit_id = (
                str(source.get("corpus_row_id", "")).strip()
                or str(source.get("parent_id", "")).strip()
                or str(source.get("doc_id", "")).strip()
                or str(source.get("source_id", "")).strip()
            )
            if explicit_id and explicit_id in known_parent_ids:
                gold_ids.add(explicit_id)
                raw_gold.append({"corpus_row_id": explicit_id})
                continue
            if not law_no or not article_no:
                parsed = (
                    parse_source_id(source.get("source_id", ""))
                    or parse_source_id(source.get("corpus_row_id", ""))
                    or parse_source_id(source.get("citation_label", ""))
                )
                if parsed:
                    law_no, article_no = parsed
            raw_gold.append({"law_no": law_no, "article_no": article_no})
            gold_ids.update(resolve_gold_ids(law_no, article_no, by_law_article))

        if not gold_ids:
            stats["skipped_unresolved_gold"] += 1
            continue
        examples.append(
            {
                "question_id": row.get("question_id") or f"json_{len(examples) + 1:04d}",
                "question": row.get("question", ""),
                "gold_parent_ids": sorted(gold_ids),
                "raw_gold": raw_gold,
            }
        )
    stats["resolved_examples"] = len(examples)
    return examples, stats


def benchmark_from_csv(
    path: Path,
    by_law_article: dict[tuple[str, str], set[str]],
    law_name_to_no: dict[str, str],
    active_only: bool,
) -> tuple[list[dict], Counter]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    examples = []
    stats = Counter(total_rows=len(rows))
    for index, row in enumerate(rows, start=1):
        if active_only and str(row.get("is_active", "")).casefold() != "true":
            stats["skipped_inactive"] += 1
            continue

        law_no = infer_law_no(row, law_name_to_no)
        article_numbers = (
            split_article_numbers(row.get("madde_no", ""))
            or split_article_numbers(row.get("madde_nolari_context", ""))
            or [extract_article_no_from_context(row.get("context", ""))]
        )

        gold_ids = set()
        for article_no in article_numbers:
            gold_ids.update(resolve_gold_ids(law_no, article_no, by_law_article))

        if not law_no:
            stats["skipped_unresolved_law"] += 1
            continue
        if not gold_ids:
            stats["skipped_unresolved_article"] += 1
            continue

        examples.append(
            {
                "question_id": row.get("row_id") or f"csv_{index:04d}",
                "question": row.get("soru", ""),
                "gold_parent_ids": sorted(gold_ids),
                "raw_gold": [{"law_no": law_no, "article_no": article_no} for article_no in article_numbers],
            }
        )
    stats["resolved_examples"] = len(examples)
    return examples, stats


def load_benchmark(
    path: Path,
    by_law_article: dict[tuple[str, str], set[str]],
    law_name_to_no: dict[str, str],
    active_only: bool,
) -> tuple[list[dict], Counter]:
    if path.suffix.lower() == ".csv":
        return benchmark_from_csv(path, by_law_article, law_name_to_no, active_only)
    return benchmark_from_json(path, by_law_article)


def minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    min_value = min(values.values())
    max_value = max(values.values())
    if min_value == max_value:
        return {key: 1.0 for key in values}
    return {key: (value - min_value) / (max_value - min_value) for key, value in values.items()}


def char_ngrams(token: str, n: int = 4) -> list[str]:
    if len(token) < n:
        return []
    return [f"char:{token[index:index + n]}" for index in range(len(token) - n + 1)]


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_RE.findall(normalize_text(text)):
        if len(token) < 2:
            continue
        tokens.append(token)
        tokens.extend(char_ngrams(token))
    return tokens


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_freqs: Counter[str] = Counter()

        for doc_id, document in enumerate(documents):
            counts = Counter(tokenize(document))
            self.doc_lengths.append(sum(counts.values()))
            for term, freq in counts.items():
                self.postings[term].append((doc_id, freq))
                self.doc_freqs[term] += 1

        self.doc_count = len(documents)
        self.avg_doc_length = sum(self.doc_lengths) / max(1, self.doc_count)

    def search(self, query: str, top_k: int) -> list[tuple[float, int]]:
        query_terms = Counter(tokenize(query))
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


def dense_search_all(
    questions: list[str],
    index_path: Path,
    config_path: Path,
    top_k: int,
    batch_size: int,
    device: str,
) -> tuple[list[dict[int, float]], list[dict[int, float]], str]:
    import faiss
    from sentence_transformers import SentenceTransformer

    config = load_json(config_path)
    model_name = config["model"]
    print(f"Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)
    if config.get("max_seq_length"):
        model.max_seq_length = int(config["max_seq_length"])
    index = faiss.read_index(str(index_path))
    encode_kwargs = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": True,
    }
    query_prompt_name = config.get("query_prompt_name")
    query_prompt = config.get("query_prompt")
    if query_prompt_name:
        encode_kwargs["prompt_name"] = query_prompt_name
        query_texts = questions
    elif query_prompt:
        encode_kwargs["prompt"] = query_prompt
        query_texts = questions
    else:
        query_prefix = config.get("query_prefix", "query: ")
        query_texts = [f"{query_prefix}{question}" for question in questions]
    embeddings = model.encode(query_texts, **encode_kwargs).astype("float32")
    scores, indices = index.search(embeddings, min(top_k, index.ntotal))

    ranked = []
    raw = []
    for query_scores, query_indices in zip(scores, indices):
        raw_scores = {
            int(index_id): float(score)
            for score, index_id in zip(query_scores, query_indices)
            if index_id >= 0
        }
        raw.append(raw_scores)
        ranked.append(raw_scores)
    return ranked, raw, model_name


def hybrid_rank(
    query: str,
    dense_raw: dict[int, float],
    bm25_index: BM25Index,
    dense_candidates: int,
    bm25_candidates: int,
    top_k: int,
    alpha: float,
) -> list[tuple[float, float, float, int]]:
    dense_trimmed = dict(sorted(dense_raw.items(), key=lambda item: item[1], reverse=True)[:dense_candidates])
    bm25_raw = {doc_id: score for score, doc_id in bm25_index.search(query, bm25_candidates)}
    candidate_ids = set(dense_trimmed) | set(bm25_raw)
    dense_norm = minmax({doc_id: dense_trimmed.get(doc_id, 0.0) for doc_id in candidate_ids})
    bm25_norm = minmax({doc_id: bm25_raw.get(doc_id, 0.0) for doc_id in candidate_ids})

    ranked = []
    for doc_id in candidate_ids:
        dense_score = dense_norm.get(doc_id, 0.0)
        bm25_score = bm25_norm.get(doc_id, 0.0)
        score = alpha * dense_score + (1 - alpha) * bm25_score
        ranked.append((score, dense_score, bm25_score, doc_id))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:top_k]


def rerank(
    query: str,
    candidates: list[tuple[float, float, float, int]],
    metadata: list[dict],
    reranker,
    batch_size: int,
) -> list[tuple[float, float, float, float, int]]:
    pairs = [(query, metadata[doc_id]["text"]) for _, _, _, doc_id in candidates]
    scores = reranker.predict(pairs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    ranked = []
    for rerank_score, (prelim, dense_score, bm25_score, doc_id) in zip(scores, candidates):
        ranked.append((float(rerank_score), prelim, dense_score, bm25_score, doc_id))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def fuse_rerank_scores(
    reranked: list[tuple[float, float, float, float, int]],
    reranker_weight: float,
    hybrid_weight: float,
) -> list[tuple[float, float, float, float, int]]:
    rerank_norm = minmax({index: item[0] for index, item in enumerate(reranked)})
    fused = []
    for index, (rerank_score, preliminary_score, dense_score, bm25_score, doc_id) in enumerate(reranked):
        final_score = reranker_weight * rerank_norm[index] + hybrid_weight * preliminary_score
        fused.append((float(final_score), preliminary_score, dense_score, bm25_score, doc_id))
    fused.sort(key=lambda item: item[0], reverse=True)
    return fused


def dedupe_parent(ranked: list[tuple], metadata: list[dict], top_k: int) -> list[tuple]:
    results = []
    seen = set()
    for item in ranked:
        doc_id = item[-1]
        parent_id = metadata[doc_id].get("parent_id", metadata[doc_id].get("id"))
        if parent_id in seen:
            continue
        seen.add(parent_id)
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def first_hit_rank(results: list[dict], gold_parent_ids: set[str]) -> int | None:
    for index, result in enumerate(results, start=1):
        if result["parent_id"] in gold_parent_ids:
            return index
    return None


def dcg_at_k(results: list[dict], gold_parent_ids: set[str], k: int) -> float:
    score = 0.0
    for rank, result in enumerate(results[:k], start=1):
        if result["parent_id"] in gold_parent_ids:
            score += 1.0 / math.log2(rank + 1)
    return score


def ideal_dcg(relevant_count: int, k: int) -> float:
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, min(relevant_count, k) + 1))


def summarize(details: list[dict], ks: list[int]) -> dict:
    total = len(details)
    summary = {"queries": total}
    for k in ks:
        summary[f"recall@{k}"] = sum(1 for row in details if row["hit_rank"] and row["hit_rank"] <= k) / max(1, total)
    summary["top1_accuracy"] = sum(1 for row in details if row["hit_rank"] == 1) / max(1, total)
    summary["mrr"] = sum((1 / row["hit_rank"]) if row["hit_rank"] else 0.0 for row in details) / max(1, total)
    summary["ndcg@10"] = sum(row["ndcg@10"] for row in details) / max(1, total)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval/reranking on a Turkish legal gold benchmark.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--corpus", default="data/processed/retrieval_corpus.json")
    parser.add_argument("--chunks", default="data/processed/retrieval_chunks.json")
    parser.add_argument("--index", default="data/index/faiss_bge_m3.index")
    parser.add_argument("--metadata", default="data/index/metadata_bge_m3.json")
    parser.add_argument("--config", default="data/index/index_config_bge_m3.json")
    parser.add_argument("--mode", choices=["dense", "hybrid", "rerank", "rerank_fusion"], default="hybrid")
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dense-candidates", type=int, default=150)
    parser.add_argument("--bm25-candidates", type=int, default=250)
    parser.add_argument("--preliminary-top-k", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--reranker-weight", type=float, default=0.35)
    parser.add_argument("--hybrid-weight", type=float, default=0.65)
    parser.add_argument("--metadata-boost", action="store_true")
    parser.add_argument("--law-boost", type=float, default=0.05)
    parser.add_argument("--article-boost", type=float, default=0.25)
    parser.add_argument("--exact-citation-boost", type=float, default=0.50)
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--reranker-device", default="cuda")
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--active-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resolve-only", action="store_true")
    parser.add_argument("--output", default="data/eval/retrieval_eval.json")
    args = parser.parse_args()

    corpus_by_id, by_law_article, law_name_to_no = build_corpus_lookups(Path(args.corpus))
    examples, coverage_stats = load_benchmark(Path(args.benchmark), by_law_article, law_name_to_no, args.active_only)
    if args.limit:
        examples = examples[: args.limit]

    print("Benchmark coverage:")
    print(json.dumps(dict(coverage_stats), ensure_ascii=False, indent=2))
    print(f"Evaluation examples: {len(examples)}")

    if args.resolve_only:
        write_json(
            Path(args.output),
            {
                "coverage": dict(coverage_stats),
                "examples": examples,
            },
        )
        return

    if not examples:
        raise ValueError("No benchmark examples could be resolved to the current corpus.")

    metadata = load_json(Path(args.metadata))
    questions = [example["question"] for example in examples]
    dense_top = max(args.top_k, args.dense_candidates, args.preliminary_top_k)
    _, dense_raw_list, model_name = dense_search_all(
        questions,
        Path(args.index),
        Path(args.config),
        dense_top,
        args.embedding_batch_size,
        args.embedding_device,
    )

    bm25_index = None
    if args.mode in {"hybrid", "rerank", "rerank_fusion"}:
        print("Building BM25 index...")
        bm25_index = BM25Index([row["text"] for row in metadata])

    reranker_model = None
    if args.mode in {"rerank", "rerank_fusion"}:
        if not args.reranker_model:
            raise ValueError("--reranker-model is required for rerank modes")
        from sentence_transformers import CrossEncoder

        print(f"Loading reranker: {args.reranker_model} on {args.reranker_device}")
        reranker_model = CrossEncoder(args.reranker_model, device=args.reranker_device, max_length=args.max_length)

    details = []
    for example, dense_raw in zip(examples, dense_raw_list):
        if args.mode == "dense":
            ranked = [(score, score, 0.0, doc_id) for doc_id, score in sorted(dense_raw.items(), key=lambda item: item[1], reverse=True)]
        else:
            ranked = hybrid_rank(
                example["question"],
                dense_raw,
                bm25_index,
                args.dense_candidates,
                args.bm25_candidates,
                args.preliminary_top_k,
                args.alpha,
            )
            if args.mode in {"rerank", "rerank_fusion"}:
                ranked = rerank(
                    example["question"],
                    ranked,
                    metadata,
                    reranker_model,
                    args.rerank_batch_size,
                )
                if args.mode == "rerank_fusion":
                    ranked = fuse_rerank_scores(
                        ranked,
                        reranker_weight=args.reranker_weight,
                        hybrid_weight=args.hybrid_weight,
                    )

        if args.metadata_boost:
            ranked = apply_metadata_boost(
                ranked,
                metadata,
                example["question"],
                law_name_to_no,
                law_boost=args.law_boost,
                article_boost=args.article_boost,
                exact_citation_boost=args.exact_citation_boost,
            )

        ranked = dedupe_parent(ranked, metadata, args.top_k)
        result_rows = []
        for rank, item in enumerate(ranked, start=1):
            doc_id = item[-1]
            chunk = metadata[doc_id]
            parent_id = chunk.get("parent_id", chunk.get("id"))
            article = corpus_by_id.get(parent_id, {})
            result_rows.append(
                {
                    "rank": rank,
                    "chunk_id": chunk.get("id"),
                    "parent_id": parent_id,
                    "score": float(item[0]),
                    "citation": citation(article or chunk),
                }
            )

        gold_parent_ids = set(example["gold_parent_ids"])
        hit_rank = first_hit_rank(result_rows, gold_parent_ids)
        ndcg10_idcg = ideal_dcg(len(gold_parent_ids), 10)
        details.append(
            {
                "question_id": example["question_id"],
                "question": example["question"],
                "gold_parent_ids": example["gold_parent_ids"],
                "raw_gold": example["raw_gold"],
                "hit_rank": hit_rank,
                "ndcg@10": dcg_at_k(result_rows, gold_parent_ids, 10) / ndcg10_idcg if ndcg10_idcg else 0.0,
                "results": result_rows,
            }
        )

    summary = summarize(details, ks=[5, 10])
    output = {
        "mode": args.mode,
        "embedding_model": model_name,
        "reranker_model": args.reranker_model,
        "benchmark": str(args.benchmark),
        "coverage": dict(coverage_stats),
        "summary": summary,
        "details": details,
    }
    write_json(Path(args.output), output)

    print("Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved details to: {args.output}")


if __name__ == "__main__":
    main()
