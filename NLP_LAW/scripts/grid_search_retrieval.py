import argparse
import csv
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_retrieval import (  # noqa: E402
    BM25Index,
    apply_metadata_boost,
    build_corpus_lookups,
    dcg_at_k,
    dedupe_parent,
    dense_search_all,
    first_hit_rank,
    hybrid_rank,
    ideal_dcg,
    load_benchmark,
    load_json,
    minmax,
    summarize,
    write_json,
)


METADATA_PROFILES = {
    "none": None,
    "default": {"law_boost": 0.05, "article_boost": 0.25, "exact_citation_boost": 0.50},
    "strong": {"law_boost": 0.10, "article_boost": 0.40, "exact_citation_boost": 0.80},
}


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_metadata_profiles(value: str) -> list[str]:
    profiles = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [profile for profile in profiles if profile not in METADATA_PROFILES]
    if unknown:
        raise ValueError(f"Unknown metadata profile(s): {unknown}. Valid: {sorted(METADATA_PROFILES)}")
    return profiles


def load_dense_raw(path: Path) -> tuple[list[dict[int, float]], str]:
    data = load_json(path)
    if isinstance(data, dict):
        rows = data.get("dense_raw_list", [])
        model_name = data.get("embedding_model", "precomputed")
    else:
        rows = data
        model_name = "precomputed"

    dense_raw_list = []
    for row in rows:
        dense_raw_list.append({int(doc_id): float(score) for doc_id, score in row})
    return dense_raw_list, model_name


def rank_to_result_rows(ranked: list[tuple], metadata: list[dict]) -> list[dict]:
    rows = []
    for rank, item in enumerate(ranked, start=1):
        doc_id = item[-1]
        chunk = metadata[doc_id]
        rows.append(
            {
                "rank": rank,
                "chunk_id": chunk.get("id"),
                "parent_id": chunk.get("parent_id", chunk.get("id")),
                "score": float(item[0]),
            }
        )
    return rows


def evaluate_hybrid_combo(
    examples: list[dict],
    dense_raw_list: list[dict[int, float]],
    bm25_raw_list: list[dict[int, float]],
    metadata: list[dict],
    law_name_to_no: dict[str, str],
    top_k: int,
    dense_candidates: int,
    bm25_candidates: int,
    preliminary_top_k: int,
    alpha: float,
    metadata_profile: str,
) -> dict:
    profile = METADATA_PROFILES[metadata_profile]
    details = []

    for example, dense_raw, bm25_raw_max in zip(examples, dense_raw_list, bm25_raw_list):
        dense_trimmed = dict(sorted(dense_raw.items(), key=lambda item: item[1], reverse=True)[:dense_candidates])
        bm25_raw = dict(sorted(bm25_raw_max.items(), key=lambda item: item[1], reverse=True)[:bm25_candidates])
        candidate_ids = set(dense_trimmed) | set(bm25_raw)
        dense_norm = {
            doc_id: dense_trimmed.get(doc_id, 0.0)
            for doc_id in candidate_ids
        }
        bm25_norm = {
            doc_id: bm25_raw.get(doc_id, 0.0)
            for doc_id in candidate_ids
        }
        dense_norm = minmax(dense_norm)
        bm25_norm = minmax(bm25_norm)
        ranked = []
        for doc_id in candidate_ids:
            dense_score = dense_norm.get(doc_id, 0.0)
            bm25_score = bm25_norm.get(doc_id, 0.0)
            score = alpha * dense_score + (1 - alpha) * bm25_score
            ranked.append((score, dense_score, bm25_score, doc_id))
        ranked.sort(key=lambda item: item[0], reverse=True)
        ranked = ranked[:preliminary_top_k]

        if profile:
            ranked = apply_metadata_boost(
                ranked,
                metadata,
                example["question"],
                law_name_to_no,
                law_boost=profile["law_boost"],
                article_boost=profile["article_boost"],
                exact_citation_boost=profile["exact_citation_boost"],
            )

        ranked = dedupe_parent(ranked, metadata, top_k)
        result_rows = rank_to_result_rows(ranked, metadata)
        gold_parent_ids = set(example["gold_parent_ids"])
        hit_rank = first_hit_rank(result_rows, gold_parent_ids)
        ndcg10_idcg = ideal_dcg(len(gold_parent_ids), 10)
        details.append(
            {
                "hit_rank": hit_rank,
                "ndcg@10": dcg_at_k(result_rows, gold_parent_ids, 10) / ndcg10_idcg if ndcg10_idcg else 0.0,
            }
        )

    summary = summarize(details, ks=[5, 10])
    # A practical selection score for legal retrieval: prefer source coverage,
    # then rank quality. The raw metrics remain the official reported values.
    summary["selection_score"] = (
        0.45 * summary["recall@5"]
        + 0.25 * summary["recall@10"]
        + 0.20 * summary["mrr"]
        + 0.10 * summary["ndcg@10"]
    )
    return summary


def sort_key(row: dict) -> tuple:
    summary = row["summary"]
    return (
        summary["selection_score"],
        summary["recall@5"],
        summary["recall@10"],
        summary["ndcg@10"],
        summary["mrr"],
        summary["top1_accuracy"],
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "selection_score",
        "recall@5",
        "recall@10",
        "top1_accuracy",
        "mrr",
        "ndcg@10",
        "alpha",
        "dense_candidates",
        "bm25_candidates",
        "preliminary_top_k",
        "metadata_profile",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            summary = row["summary"]
            params = row["params"]
            writer.writerow(
                {
                    "rank": rank,
                    "selection_score": summary["selection_score"],
                    "recall@5": summary["recall@5"],
                    "recall@10": summary["recall@10"],
                    "top1_accuracy": summary["top1_accuracy"],
                    "mrr": summary["mrr"],
                    "ndcg@10": summary["ndcg@10"],
                    **params,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search dense+BM25 hybrid retrieval parameters.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--corpus", default="data/processed/retrieval_corpus.json")
    parser.add_argument("--chunks", default="data/processed/retrieval_chunks.json")
    parser.add_argument("--index", default="data/index/faiss_bge_m3.index")
    parser.add_argument("--metadata", default="data/index/metadata_bge_m3.json")
    parser.add_argument("--config", default="data/index/index_config_bge_m3.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--alphas", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--dense-candidates-list", default="80,150,300")
    parser.add_argument("--bm25-candidates-list", default="100,250,500")
    parser.add_argument("--preliminary-top-k-list", default="50,80,120")
    parser.add_argument("--metadata-profiles", default="none,default,strong")
    parser.add_argument("--embedding-device", default="cuda")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument(
        "--dense-raw-input",
        default=None,
        help="Optional precomputed dense scores JSON. If set, the script skips loading the embedding model.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--active-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", default="data/eval/grid_search_hybrid.json")
    parser.add_argument("--output-csv", default="data/eval/grid_search_hybrid.csv")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    alphas = parse_float_list(args.alphas)
    dense_candidates_values = parse_int_list(args.dense_candidates_list)
    bm25_candidates_values = parse_int_list(args.bm25_candidates_list)
    preliminary_top_k_values = parse_int_list(args.preliminary_top_k_list)
    metadata_profiles = parse_metadata_profiles(args.metadata_profiles)

    max_dense_candidates = max(max(dense_candidates_values), args.top_k)
    max_dense_top = max(max_dense_candidates, max(preliminary_top_k_values), args.top_k)

    corpus_by_id, by_law_article, law_name_to_no = build_corpus_lookups(Path(args.corpus))
    # corpus_by_id is built to keep the same resolution behavior as evaluate_retrieval.
    _ = corpus_by_id

    examples, coverage_stats = load_benchmark(Path(args.benchmark), by_law_article, law_name_to_no, args.active_only)
    if args.limit:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("No benchmark examples could be resolved to the current corpus.")

    metadata = load_json(Path(args.metadata))
    questions = [example["question"] for example in examples]

    print("Benchmark coverage:", flush=True)
    print(json.dumps(dict(coverage_stats), ensure_ascii=False, indent=2), flush=True)
    print(f"Evaluation examples: {len(examples)}", flush=True)
    if args.dense_raw_input:
        print(f"Loading precomputed dense scores from {args.dense_raw_input}", flush=True)
        dense_raw_list, model_name = load_dense_raw(Path(args.dense_raw_input))
        if len(dense_raw_list) != len(examples):
            raise ValueError(
                f"Dense score row count mismatch: {len(dense_raw_list)} dense rows "
                f"for {len(examples)} benchmark examples."
            )
    else:
        print(f"Loading dense scores once with top_k={max_dense_top}", flush=True)
        _, dense_raw_list, model_name = dense_search_all(
            questions,
            Path(args.index),
            Path(args.config),
            top_k=max_dense_top,
            batch_size=args.embedding_batch_size,
            device=args.embedding_device,
        )

    print("Building reusable BM25 index...", flush=True)
    bm25_index = BM25Index([row["text"] for row in metadata])
    max_bm25_candidates = max(max(bm25_candidates_values), args.top_k)
    print(f"Precomputing BM25 scores once with top_k={max_bm25_candidates}...", flush=True)
    bm25_raw_list = []
    for index, question in enumerate(questions, start=1):
        bm25_raw_list.append({doc_id: score for score, doc_id in bm25_index.search(question, max_bm25_candidates)})
        if index == 1 or index % 50 == 0 or index == len(questions):
            print(f"  BM25 precomputed {index}/{len(questions)}", flush=True)

    total = (
        len(alphas)
        * len(dense_candidates_values)
        * len(bm25_candidates_values)
        * len(preliminary_top_k_values)
        * len(metadata_profiles)
    )
    print(f"Running {total} grid combinations...", flush=True)

    rows = []
    current = 0
    for metadata_profile in metadata_profiles:
        for preliminary_top_k in preliminary_top_k_values:
            for dense_candidates in dense_candidates_values:
                for bm25_candidates in bm25_candidates_values:
                    for alpha in alphas:
                        current += 1
                        summary = evaluate_hybrid_combo(
                            examples=examples,
                            dense_raw_list=dense_raw_list,
                            bm25_raw_list=bm25_raw_list,
                            metadata=metadata,
                            law_name_to_no=law_name_to_no,
                            top_k=args.top_k,
                            dense_candidates=dense_candidates,
                            bm25_candidates=bm25_candidates,
                            preliminary_top_k=preliminary_top_k,
                            alpha=alpha,
                            metadata_profile=metadata_profile,
                        )
                        rows.append(
                            {
                                "params": {
                                    "alpha": alpha,
                                    "dense_candidates": dense_candidates,
                                    "bm25_candidates": bm25_candidates,
                                    "preliminary_top_k": preliminary_top_k,
                                    "metadata_profile": metadata_profile,
                                },
                                "summary": summary,
                            }
                        )
                        if current == 1 or current % 50 == 0 or current == total:
                            best = max(rows, key=sort_key)
                            print(
                                f"[{current}/{total}] best selection={best['summary']['selection_score']:.4f} "
                                f"R@5={best['summary']['recall@5']:.4f} "
                                f"R@10={best['summary']['recall@10']:.4f} "
                                f"MRR={best['summary']['mrr']:.4f} params={best['params']}"
                                ,
                                flush=True,
                            )

    rows.sort(key=sort_key, reverse=True)
    output = {
        "embedding_model": model_name,
        "benchmark": str(args.benchmark),
        "coverage": dict(coverage_stats),
        "grid": {
            "alphas": alphas,
            "dense_candidates_list": dense_candidates_values,
            "bm25_candidates_list": bm25_candidates_values,
            "preliminary_top_k_list": preliminary_top_k_values,
            "metadata_profiles": metadata_profiles,
            "top_k": args.top_k,
        },
        "best": rows[0],
        "top_results": rows[: args.top_n],
        "all_results": rows,
    }
    write_json(Path(args.output_json), output)
    write_csv(Path(args.output_csv), rows)

    print("\nBest result:")
    print(json.dumps(rows[0], ensure_ascii=False, indent=2))
    print(f"\nSaved JSON to: {args.output_json}")
    print(f"Saved CSV to:  {args.output_csv}")


if __name__ == "__main__":
    main()
