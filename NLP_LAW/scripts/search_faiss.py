import argparse
import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_article_lookup(article_corpus_path: Path) -> dict[str, dict]:
    if not article_corpus_path.exists():
        return {}
    articles = load_json(article_corpus_path)
    return {article["id"]: article for article in articles}


def encode_query(model: SentenceTransformer, question: str):
    return model.encode(
        [f"query: {question}"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")


def encode_query_with_config(model: SentenceTransformer, question: str, config: dict):
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


def format_result(rank: int, score: float, item: dict, article_lookup: dict[str, dict], show_text: bool) -> str:
    title = item.get("article_title", "").strip()
    citation = f"{item['law_name']} ({item['law_no']}), {item['article_no']}"
    if title:
        citation = f"{citation} - {title}"

    lines = [
        f"[{rank}] score={score:.4f}",
        f"chunk_id: {item['id']}",
        f"parent_id: {item['parent_id']}",
        f"citation: {citation}",
        f"chunk: {item['chunk_index'] + 1}/{item['chunk_count']}, words={item['chunk_word_count']}",
    ]

    article = article_lookup.get(item["parent_id"])
    if article:
        lines.append(f"parent_article_chars: {len(article['text'])}")

    if show_text:
        preview = item["text"].replace("\n", " ")
        lines.append(f"text: {preview[:1200]}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Turkish legal FAISS index.")
    parser.add_argument("question", nargs="?", help="Turkish legal question.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--index", default="data/index/faiss.index")
    parser.add_argument("--metadata", default="data/index/metadata.json")
    parser.add_argument("--config", default="data/index/index_config.json")
    parser.add_argument("--articles", default="data/processed/retrieval_corpus.json")
    parser.add_argument(
        "--device",
        default=None,
        help="Device for SentenceTransformer, e.g. cuda or cpu. Defaults to config/auto-detection.",
    )
    parser.add_argument("--show-text", action="store_true", help="Print retrieved chunk text previews.")
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

    query_embedding = encode_query_with_config(model, question, config)
    scores, indices = index.search(query_embedding, args.top_k)

    print(f"\nQuestion: {question}")
    print(f"Top-{args.top_k} results\n")

    for rank, (score, index_id) in enumerate(zip(scores[0], indices[0]), start=1):
        if index_id < 0:
            continue
        item = metadata[int(index_id)]
        print(format_result(rank, float(score), item, article_lookup, args.show_text))
        print("-" * 100)


if __name__ == "__main__":
    main()
