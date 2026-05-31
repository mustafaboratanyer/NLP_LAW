import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


TOKEN_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]+")
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
    normalized = unicodedata.normalize("NFKD", str(text).casefold().replace("ı", "i"))
    return "".join(char for char in normalized if not unicodedata.combining(char))


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens = TOKEN_RE.findall(normalized)
    expanded = []
    for token in tokens:
        expanded.append(token)
        if token in TURKISH_NUMBERS:
            expanded.append(TURKISH_NUMBERS[token])
    return expanded


def expand_query(question: str) -> str:
    normalized = normalize_text(question)
    expansions = []

    if any(number in normalized.split() for number in ("2", "iki")):
        expansions.append("iki iki gün iki işgünü ardı ardına")

    if "isci" in normalized and (
        "ise gel" in normalized
        or "gelmez" in normalized
        or "gelmedi" in normalized
        or "devamsiz" in normalized
        or "devamsizlik" in normalized
    ):
        expansions.append(
            "işçinin işverenden izin almaksızın haklı bir sebebe dayanmaksızın "
            "ardı ardına iki işgünü işine devam etmemesi devamsızlık "
            "işverenin haklı nedenle derhal fesih hakkı"
        )

    if "öldür" in normalized or "oldur" in normalized:
        expansions.append(
            "kasten öldürme insan öldürme bir insanı kasten öldüren kişi "
            "müebbet hapis Türk Ceza Kanunu hayata karşı suçlar"
        )

    if not expansions:
        return question
    return f"{question}\n" + "\n".join(expansions)


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
    avg_doc_length = float(doc_lengths.mean()) if doc_count else 0.0

    query_tokens = tokenize(query)
    query_counts = Counter(query_tokens)
    query_terms = list(query_counts)

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


def hybrid_search(
    question: str,
    metadata: list[dict],
    index,
    model: SentenceTransformer,
    config: dict,
    top_k: int,
    dense_candidates: int,
    bm25_candidates: int,
    alpha: float,
    use_query_expansion: bool,
    dedupe_parent: bool,
) -> list[tuple[float, float, float, dict]]:
    search_query = expand_query(question) if use_query_expansion else question

    query_embedding = encode_query(model, search_query, config)
    dense_scores, dense_indices = index.search(query_embedding, min(dense_candidates, len(metadata)))
    dense_raw = {
        int(index_id): float(score)
        for score, index_id in zip(dense_scores[0], dense_indices[0])
        if index_id >= 0
    }

    bm25_raw_array = bm25_scores(search_query, metadata)
    bm25_top_indices = np.argsort(-bm25_raw_array)[: min(bm25_candidates, len(metadata))]
    bm25_raw = {int(index_id): float(bm25_raw_array[index_id]) for index_id in bm25_top_indices}

    candidate_indices = set(dense_raw) | set(bm25_raw)
    dense_norm = minmax({index_id: dense_raw.get(index_id, 0.0) for index_id in candidate_indices})
    bm25_norm = minmax({index_id: bm25_raw.get(index_id, 0.0) for index_id in candidate_indices})

    ranked = []
    for index_id in candidate_indices:
        dense_score = dense_norm.get(index_id, 0.0)
        lexical_score = bm25_norm.get(index_id, 0.0)
        combined = alpha * dense_score + (1 - alpha) * lexical_score
        ranked.append((combined, dense_score, lexical_score, metadata[index_id]))

    ranked.sort(key=lambda item: item[0], reverse=True)

    if not dedupe_parent:
        return ranked[:top_k]

    deduped = []
    seen_parents = set()
    for item in ranked:
        parent_id = item[3]["parent_id"]
        if parent_id in seen_parents:
            continue
        seen_parents.add(parent_id)
        deduped.append(item)
        if len(deduped) >= top_k:
            break
    return deduped


def format_result(
    rank: int,
    combined_score: float,
    dense_score: float,
    bm25_score: float,
    item: dict,
    article_lookup: dict[str, dict],
    show_text: bool,
) -> str:
    title = item.get("article_title", "").strip()
    citation = f"{item['law_name']} ({item['law_no']}), {item['article_no']}"
    if title:
        citation = f"{citation} - {title}"

    lines = [
        f"[{rank}] score={combined_score:.4f} dense={dense_score:.4f} bm25={bm25_score:.4f}",
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
        preview = text.replace("\n", " ")
        lines.append(f"text: {preview[:1400]}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Turkish legal search with dense FAISS + BM25.")
    parser.add_argument("question", nargs="?", help="Turkish legal question.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-candidates", type=int, default=100)
    parser.add_argument("--bm25-candidates", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.55, help="Dense score weight. BM25 weight is 1-alpha.")
    parser.add_argument("--index", default="data/index/faiss.index")
    parser.add_argument("--metadata", default="data/index/metadata.json")
    parser.add_argument("--config", default="data/index/index_config.json")
    parser.add_argument("--articles", default="data/processed/retrieval_corpus.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-query-expansion", action="store_true")
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

    print(f"Loading model: {config['model']}")
    model = SentenceTransformer(config["model"], device=args.device or config.get("device"))
    if config.get("max_seq_length"):
        model.max_seq_length = int(config["max_seq_length"])
    index = faiss.read_index(str(Path(args.index)))

    results = hybrid_search(
        question=question,
        metadata=metadata,
        index=index,
        model=model,
        config=config,
        top_k=args.top_k,
        dense_candidates=args.dense_candidates,
        bm25_candidates=args.bm25_candidates,
        alpha=args.alpha,
        use_query_expansion=args.use_query_expansion,
        dedupe_parent=not args.no_dedupe_parent,
    )

    print(f"\nQuestion: {question}")
    if args.use_query_expansion:
        expanded = expand_query(question)
        if expanded != question:
            print(f"Expanded query: {expanded.replace(chr(10), ' | ')}")
    print(f"Top-{args.top_k} hybrid results\n")

    for rank, (combined, dense_score, bm25_score, item) in enumerate(results, start=1):
        print(format_result(rank, combined, dense_score, bm25_score, item, article_lookup, args.show_text))
        print("-" * 100)


if __name__ == "__main__":
    main()
