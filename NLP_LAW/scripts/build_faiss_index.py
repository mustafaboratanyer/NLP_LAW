import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def load_chunks(path: Path) -> list[dict]:
    chunks = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a list in {path}")
    return chunks


def passage_text(chunk: dict, passage_prefix: str) -> str:
    return f"{passage_prefix}{chunk['text']}"


def encode_passages(
    model: SentenceTransformer,
    chunks: list[dict],
    batch_size: int,
    passage_prefix: str,
) -> np.ndarray:
    embeddings = []

    for start in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks"):
        batch = chunks[start : start + batch_size]
        texts = [passage_text(chunk, passage_prefix) for chunk in batch]
        batch_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        embeddings.append(batch_embeddings.astype("float32"))

    return np.vstack(embeddings)


def metadata_for_chunk(chunk: dict) -> dict:
    return {
        "id": chunk["id"],
        "parent_id": chunk["parent_id"],
        "law_name": chunk["law_name"],
        "law_no": chunk["law_no"],
        "article_no": chunk["article_no"],
        "article_title": chunk.get("article_title", ""),
        "chunk_index": chunk["chunk_index"],
        "chunk_count": chunk["chunk_count"],
        "is_full_article": chunk["is_full_article"],
        "parent_word_count": chunk["parent_word_count"],
        "chunk_word_count": chunk["chunk_word_count"],
        "source_url": chunk["source_url"],
        "text": chunk["text"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index for Turkish legal retrieval chunks.")
    parser.add_argument(
        "--chunks",
        default="data/processed/retrieval_chunks.json",
        help="Chunk-level retrieval JSON file.",
    )
    parser.add_argument(
        "--index-out",
        default="data/index/faiss.index",
        help="Output FAISS index path.",
    )
    parser.add_argument(
        "--metadata-out",
        default="data/index/metadata.json",
        help="Output metadata JSON path.",
    )
    parser.add_argument(
        "--config-out",
        default="data/index/index_config.json",
        help="Output index config JSON path.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer embedding model.")
    parser.add_argument(
        "--query-prefix",
        default="query: ",
        help="Prefix added to questions at search/evaluation time when no query prompt is used.",
    )
    parser.add_argument(
        "--passage-prefix",
        default="passage: ",
        help="Prefix added to corpus chunks before embedding. Use an empty string for models such as Qwen3 embeddings.",
    )
    parser.add_argument(
        "--query-prompt-name",
        default=None,
        help="SentenceTransformers prompt_name used for query encoding, e.g. 'query' for Qwen3 embeddings.",
    )
    parser.add_argument(
        "--query-prompt",
        default=None,
        help="Explicit SentenceTransformers query prompt. If set, this overrides --query-prefix.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Optional max sequence length for the embedding model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for SentenceTransformer, e.g. cuda or cpu. Defaults to auto-detection.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    index_path = Path(args.index_out)
    metadata_path = Path(args.metadata_out)
    config_path = Path(args.config_out)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    print(f"Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model, device=args.device)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length
        print(f"Set model.max_seq_length={model.max_seq_length}")

    embeddings = encode_passages(
        model,
        chunks,
        batch_size=args.batch_size,
        passage_prefix=args.passage_prefix,
    )
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(index_path))

    metadata = [metadata_for_chunk(chunk) for chunk in chunks]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    config = {
        "model": args.model,
        "chunk_count": len(chunks),
        "embedding_dimension": dimension,
        "index_type": "faiss.IndexFlatIP",
        "similarity": "cosine_via_normalized_inner_product",
        "chunks_file": str(chunks_path),
        "index_file": str(index_path),
        "metadata_file": str(metadata_path),
        "query_prefix": args.query_prefix,
        "passage_prefix": args.passage_prefix,
        "query_prompt_name": args.query_prompt_name,
        "query_prompt": args.query_prompt,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved FAISS index to {index_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Saved config to {config_path}")
    print(f"Index vectors: {index.ntotal}, dimension: {dimension}")


if __name__ == "__main__":
    main()
