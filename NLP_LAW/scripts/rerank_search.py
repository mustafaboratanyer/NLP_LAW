import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
LEGAL_STOPWORDS = {
    "acaba",
    "bir",
    "biri",
    "birini",
    "birine",
    "birinin",
    "bu",
    "hangi",
    "icin",
    "iş",
    "is",
    "işçi",
    "isci",
    "işçinin",
    "iscinin",
    "işçiye",
    "isciye",
    "işçiler",
    "isciler",
    "işveren",
    "isveren",
    "işverene",
    "isverene",
    "işverenin",
    "isverenin",
    "ile",
    "ise",
    "kanun",
    "kanuna",
    "kanunda",
    "madde",
    "maddesi",
    "mi",
    "mu",
    "mudur",
    "mudur",
    "mı",
    "nasıl",
    "ne",
    "nedir",
    "olur",
    "olarak",
    "suç",
    "suc",
    "şart",
    "şartlar",
    "sart",
    "sartlar",
    "var",
    "ve",
    "veya",
}
TURKISH_NUMBERS = {
    "0": "sıfır",
    "1": "bir",
    "2": "iki",
    "3": "üç",
    "4": "dört",
    "5": "beş",
    "6": "altı",
    "7": "yedi",
    "8": "sekiz",
    "9": "dokuz",
    "10": "on",
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold().replace("ı", "i"))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def char_ngrams(token: str, n: int = 4) -> list[str]:
    if len(token) < n:
        return []
    return [f"char:{token[index:index+n]}" for index in range(len(token) - n + 1)]


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(normalize_text(text))
    expanded = []

    for token in tokens:
        expanded.append(token)
        if token in TURKISH_NUMBERS:
            expanded.append(TURKISH_NUMBERS[token])
        expanded.extend(char_ngrams(token))

    return expanded


def lexical_terms(text: str) -> list[str]:
    terms = []
    for token in TOKEN_RE.findall(normalize_text(text)):
        if len(token) < 3 or token.isdigit() or token in LEGAL_STOPWORDS:
            continue
        terms.append(token)
    return terms


def token_char_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0

    left_grams = set(char_ngrams(left))
    right_grams = set(char_ngrams(right))
    if not left_grams or not right_grams:
        return 0.0

    overlap = len(left_grams & right_grams)
    return (2 * overlap) / (len(left_grams) + len(right_grams))


def max_term_similarity(left_terms: list[str], right_terms: list[str]) -> float:
    if not left_terms or not right_terms:
        return 0.0

    best = 0.0
    for left in left_terms:
        for right in right_terms:
            best = max(best, token_char_similarity(left, right))
    return best


def title_relevance_score(question: str, item: dict) -> float:
    """Score whether the article title directly names the concept in the question."""
    title = item.get("article_title", "")
    if not title:
        return 0.0

    question_terms = lexical_terms(question)
    title_terms = lexical_terms(title)
    if not question_terms or not title_terms:
        return 0.0

    matched_question_terms = set()
    matched_title_terms = set()
    for question_term in question_terms:
        for title_term in title_terms:
            if token_char_similarity(question_term, title_term) >= 0.82:
                matched_question_terms.add(question_term)
                matched_title_terms.add(title_term)

    if not matched_question_terms:
        return 0.0

    query_coverage = len(matched_question_terms) / max(1, len(set(question_terms)))
    title_coverage = len(matched_title_terms) / max(1, len(set(title_terms)))

    # A single shared generic term should not dominate the final score.
    if len(matched_question_terms) == 1 and len(matched_title_terms) == 1:
        return min(0.45, title_coverage * 0.45)

    return min(1.0, 0.35 * query_coverage + 0.65 * title_coverage)


def build_article_lookup(article_corpus_path: Path) -> dict[str, dict]:
    if not article_corpus_path.exists():
        return {}
    articles = load_json(article_corpus_path)
    return {article["id"]: article for article in articles}


def encode_query(model: SentenceTransformer, question: str, config: dict):
    encode_kwargs = {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    query_prompt_name = config.get("query_prompt_name")
    query_prompt = config.get("query_prompt")
    if query_prompt_name:
        encode_kwargs["prompt_name"] = query_prompt_name
        texts = [question]
    elif query_prompt:
        encode_kwargs["prompt"] = query_prompt
        texts = [question]
    else:
        query_prefix = config.get("query_prefix", "query: ")
        texts = [f"{query_prefix}{question}"]
    return model.encode(texts, **encode_kwargs).astype("float32")


def bm25_scores(query: str, documents: list[dict], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    tokenized_docs = [tokenize(document["text"]) for document in documents]
    doc_count = len(tokenized_docs)
    doc_lengths = np.array([len(tokens) for tokens in tokenized_docs], dtype=np.float32)
    avg_doc_length = float(doc_lengths.mean()) if doc_count else 1.0

    query_terms = list(Counter(tokenize(query)))
    document_frequencies = Counter()
    term_frequencies = []

    for tokens in tokenized_docs:
        counts = Counter(tokens)
        term_frequencies.append(counts)
        for term in query_terms:
            if term in counts:
                document_frequencies[term] += 1

    scores = np.zeros(doc_count, dtype=np.float32)
    for term in query_terms:
        df = document_frequencies[term]
        if df == 0:
            continue

        idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
        for index, counts in enumerate(term_frequencies):
            freq = counts.get(term, 0)
            if freq == 0:
                continue

            denominator = freq + k1 * (1 - b + b * doc_lengths[index] / avg_doc_length)
            scores[index] += idf * (freq * (k1 + 1) / denominator)

    return scores


def minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}

    min_value = min(values.values())
    max_value = max(values.values())
    if max_value == min_value:
        return {key: 1.0 for key in values}

    return {key: (value - min_value) / (max_value - min_value) for key, value in values.items()}


def preliminary_candidates(
    question: str,
    metadata: list[dict],
    index,
    model: SentenceTransformer,
    config: dict,
    dense_candidates: int,
    bm25_candidates: int,
    preliminary_top_k: int,
    alpha: float,
) -> list[tuple[float, float, float, dict]]:
    query_embedding = encode_query(model, question, config)
    dense_scores, dense_indices = index.search(query_embedding, min(dense_candidates, len(metadata)))
    dense_raw = {
        int(index_id): float(score)
        for score, index_id in zip(dense_scores[0], dense_indices[0])
        if index_id >= 0
    }

    bm25_raw_array = bm25_scores(question, metadata)
    bm25_top_indices = np.argsort(-bm25_raw_array)[: min(bm25_candidates, len(metadata))]
    bm25_raw = {int(index_id): float(bm25_raw_array[index_id]) for index_id in bm25_top_indices}

    candidate_indices = set(dense_raw) | set(bm25_raw)
    dense_norm = minmax({index_id: dense_raw.get(index_id, 0.0) for index_id in candidate_indices})
    bm25_norm = minmax({index_id: bm25_raw.get(index_id, 0.0) for index_id in candidate_indices})

    ranked = []
    for index_id in candidate_indices:
        dense_score = dense_norm.get(index_id, 0.0)
        bm25_score = bm25_norm.get(index_id, 0.0)
        combined = alpha * dense_score + (1 - alpha) * bm25_score
        ranked.append((combined, dense_score, bm25_score, metadata[index_id]))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[:preliminary_top_k]


def rerank_candidates(
    question: str,
    candidates: list[tuple[float, float, float, dict]],
    reranker: CrossEncoder,
    batch_size: int,
) -> list[tuple[float, float, float, float, dict]]:
    pairs = [(question, item["text"]) for _, _, _, item in candidates]
    rerank_scores = reranker.predict(
        pairs,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    reranked = []
    for rerank_score, (prelim_score, dense_score, bm25_score, item) in zip(rerank_scores, candidates):
        reranked.append((float(rerank_score), prelim_score, dense_score, bm25_score, item))

    reranked.sort(key=lambda item: item[0], reverse=True)
    return reranked


def fuse_scores(
    question: str,
    reranked: list[tuple[float, float, float, float, dict]],
    reranker_weight: float,
    preliminary_weight: float,
    title_weight: float,
    direct_title_bonus: float,
) -> list[tuple[float, float, float, float, float, float, dict]]:
    rerank_values = {index: item[0] for index, item in enumerate(reranked)}
    rerank_norm = minmax(rerank_values)

    fused = []
    for index, (rerank_score, preliminary_score, dense_score, bm25_score, item) in enumerate(reranked):
        title_score = title_relevance_score(question, item)
        direct_bonus = direct_title_bonus if title_score >= 0.75 else 0.0
        final_score = (
            reranker_weight * rerank_norm[index]
            + preliminary_weight * preliminary_score
            + title_weight * title_score
            + direct_bonus
        )
        fused.append(
            (
                float(final_score),
                rerank_score,
                preliminary_score,
                dense_score,
                bm25_score,
                title_score,
                item,
            )
        )

    fused.sort(key=lambda item: item[0], reverse=True)
    return fused


def dedupe_parent(
    reranked: list[tuple],
    top_k: int,
) -> list[tuple]:
    results = []
    seen_parents = set()

    for item in reranked:
        parent_id = item[-1]["parent_id"]
        if parent_id in seen_parents:
            continue

        seen_parents.add(parent_id)
        results.append(item)
        if len(results) >= top_k:
            break

    return results


def format_result(
    rank: int,
    final_score: float | None,
    rerank_score: float,
    preliminary_score: float,
    dense_score: float,
    bm25_score: float,
    title_score: float | None,
    item: dict,
    article_lookup: dict[str, dict],
    show_text: bool,
) -> str:
    title = item.get("article_title", "").strip()
    citation = f"{item['law_name']} ({item['law_no']}), {item['article_no']}"
    if title:
        citation = f"{citation} - {title}"

    score_line = (
        f"[{rank}] rerank={rerank_score:.4f} "
        f"prelim={preliminary_score:.4f} dense={dense_score:.4f} bm25={bm25_score:.4f}"
    )
    if final_score is not None and title_score is not None:
        score_line = (
            f"[{rank}] final={final_score:.4f} rerank={rerank_score:.4f} "
            f"title={title_score:.4f} prelim={preliminary_score:.4f} "
            f"dense={dense_score:.4f} bm25={bm25_score:.4f}"
        )

    lines = [
        score_line,
        f"chunk_id: {item['id']}",
        f"parent_id: {item['parent_id']}",
        f"citation: {citation}",
        f"chunk: {item['chunk_index'] + 1}/{item['chunk_count']}, words={item['chunk_word_count']}",
    ]

    article = article_lookup.get(item["parent_id"])
    if article:
        lines.append(f"parent_article_chars: {len(article['text'])}")

    if show_text:
        text = article["text"] if article else item["text"]
        lines.append(f"text: {text.replace(chr(10), ' ')[:1600]}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid retrieval + cross-encoder reranking.")
    parser.add_argument("question", nargs="?", help="Turkish legal question.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-candidates", type=int, default=150)
    parser.add_argument("--bm25-candidates", type=int, default=250)
    parser.add_argument("--preliminary-top-k", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.55, help="Dense score weight before reranking.")
    parser.add_argument("--index", default="data/index/faiss.index")
    parser.add_argument("--metadata", default="data/index/metadata.json")
    parser.add_argument("--config", default="data/index/index_config.json")
    parser.add_argument("--articles", default="data/processed/retrieval_corpus.json")
    parser.add_argument("--device", default=None, help="Device for the reranker model, for example cuda or cpu.")
    parser.add_argument(
        "--embedding-device",
        default="cpu",
        help="Device for the embedding model used only for query encoding. CPU is safer on Kaggle T4.",
    )
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    parser.add_argument("--rerank-batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--ranking-mode",
        choices=["fusion", "rerank"],
        default="fusion",
        help="fusion combines reranker score with legal title/directness signals.",
    )
    parser.add_argument("--reranker-weight", type=float, default=0.55)
    parser.add_argument("--preliminary-weight", type=float, default=0.10)
    parser.add_argument("--title-weight", type=float, default=0.35)
    parser.add_argument("--direct-title-bonus", type=float, default=0.30)
    parser.add_argument("--no-dedupe-parent", action="store_true")
    parser.add_argument("--show-text", action="store_true")
    args = parser.parse_args()

    question = args.question
    if not question:
        question = input("Soru: ").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    config = load_json(Path(args.config))
    metadata = load_json(Path(args.metadata))
    article_lookup = build_article_lookup(Path(args.articles))

    print(f"Loading embedding model: {config['model']} on {args.embedding_device}")
    embedding_model = SentenceTransformer(config["model"], device=args.embedding_device)
    if config.get("max_seq_length"):
        embedding_model.max_seq_length = int(config["max_seq_length"])
    index = faiss.read_index(str(Path(args.index)))

    candidates = preliminary_candidates(
        question=question,
        metadata=metadata,
        index=index,
        model=embedding_model,
        config=config,
        dense_candidates=args.dense_candidates,
        bm25_candidates=args.bm25_candidates,
        preliminary_top_k=args.preliminary_top_k,
        alpha=args.alpha,
    )

    # The embedding model is only needed to encode the query. Releasing it before
    # loading the cross-encoder avoids unnecessary GPU/CPU memory pressure.
    del embedding_model

    print(f"Loading reranker model: {args.reranker_model} on {args.device or 'auto'}")
    reranker = CrossEncoder(
        args.reranker_model,
        device=args.device,
        max_length=args.max_length,
    )
    reranked = rerank_candidates(
        question=question,
        candidates=candidates,
        reranker=reranker,
        batch_size=args.rerank_batch_size,
    )

    if args.ranking_mode == "fusion":
        ranked_results = fuse_scores(
            question=question,
            reranked=reranked,
            reranker_weight=args.reranker_weight,
            preliminary_weight=args.preliminary_weight,
            title_weight=args.title_weight,
            direct_title_bonus=args.direct_title_bonus,
        )
    else:
        ranked_results = [(None, *item[:4], None, item[4]) for item in reranked]

    results = ranked_results[: args.top_k] if args.no_dedupe_parent else dedupe_parent(ranked_results, args.top_k)

    print(f"\nQuestion: {question}")
    print(f"Candidates reranked: {len(candidates)}")
    print(f"Ranking mode: {args.ranking_mode}")
    print(f"Top-{args.top_k} reranked results\n")

    for rank, (final_score, rerank_score, preliminary_score, dense_score, bm25_score, title_score, item) in enumerate(
        results,
        start=1,
    ):
        print(
            format_result(
                rank=rank,
                final_score=final_score,
                rerank_score=rerank_score,
                preliminary_score=preliminary_score,
                dense_score=dense_score,
                bm25_score=bm25_score,
                title_score=title_score,
                item=item,
                article_lookup=article_lookup,
                show_text=args.show_text,
            )
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
