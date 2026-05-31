import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

# Keep training on one visible GPU in Kaggle. This avoids accidental DataParallel
# behavior in libraries that are not always tested for multi-GPU notebooks.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def passage_text(row: dict) -> str:
    for key in ("candidate_passage", "passage", "positive_passage", "context"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError("Expected passage text in candidate_passage/passage/positive_passage/context")


def grouped_triplets(rows: list[dict], max_triplets_per_query: int | None = None) -> list[InputExample]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"positive": [], "negative": [], "query": []})
    for row in rows:
        query_id = str(row.get("query_id") or row.get("query"))
        grouped[query_id]["query"].append(row["query"])
        if int(row["label"]) == 1:
            grouped[query_id]["positive"].append(passage_text(row))
        else:
            grouped[query_id]["negative"].append(passage_text(row))

    examples = []
    for group in grouped.values():
        if not group["positive"] or not group["negative"]:
            continue
        query = group["query"][0]
        created = 0
        for positive in group["positive"]:
            for negative in group["negative"]:
                examples.append(
                    InputExample(
                        texts=[
                            f"query: {query}",
                            f"passage: {positive}",
                            f"passage: {negative}",
                        ]
                    )
                )
                created += 1
                if max_triplets_per_query and created >= max_triplets_per_query:
                    break
            if max_triplets_per_query and created >= max_triplets_per_query:
                break
    return examples


def maybe_triplet_evaluator(dev_examples: list[InputExample]):
    try:
        from sentence_transformers.evaluation import TripletEvaluator

        if not dev_examples:
            return None
        return TripletEvaluator.from_input_examples(dev_examples, name="legal-embedding-dev")
    except Exception as exc:  # pragma: no cover - evaluator API differs by version
        print(f"TripletEvaluator unavailable, training without evaluator. Reason: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder embedding model with legal triplets.")
    parser.add_argument("--train", default="data/reranker/clean_train.jsonl")
    parser.add_argument("--dev", default="data/reranker/clean_dev.jsonl")
    parser.add_argument("--base-model", default="BAAI/bge-m3")
    parser.add_argument("--output-dir", default="data/models/legal-bge-m3-embedding")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--triplet-margin", type=float, default=0.25)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-triplets-per-query", type=int, default=3)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--evaluation-steps", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-best-model", action="store_true")
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train), args.train_limit)
    dev_rows = load_jsonl(Path(args.dev), args.dev_limit)

    train_examples = grouped_triplets(train_rows, args.max_triplets_per_query)
    dev_examples = grouped_triplets(dev_rows, args.max_triplets_per_query)
    if not train_examples:
        raise ValueError("No train triplets were created. Check labels and query_id groups.")

    print(f"Loaded train rows: {len(train_rows)}")
    print(f"Loaded dev rows:   {len(dev_rows)}")
    print(f"Train triplets:    {len(train_examples)}")
    print(f"Dev triplets:      {len(dev_examples)}")
    print(f"Base model:        {args.base_model}")

    model = SentenceTransformer(args.base_model, device=args.device)
    model.max_seq_length = args.max_seq_length
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=args.triplet_margin,
    )
    evaluator = maybe_triplet_evaluator(dev_examples)

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = math.ceil(total_steps * args.warmup_ratio)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_kwargs = {
        "train_objectives": [(train_dataloader, train_loss)],
        "epochs": args.epochs,
        "warmup_steps": warmup_steps,
        "optimizer_params": {"lr": args.learning_rate},
        "output_path": str(output_dir),
        "save_best_model": args.save_best_model,
    }
    if evaluator is not None:
        fit_kwargs["evaluator"] = evaluator
        fit_kwargs["evaluation_steps"] = args.evaluation_steps

    model.fit(**fit_kwargs)
    if not args.save_best_model or not (output_dir / "modules.json").exists():
        model.save(str(output_dir))

    config = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "triplet_margin": args.triplet_margin,
        "max_seq_length": args.max_seq_length,
        "max_triplets_per_query": args.max_triplets_per_query,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "train_triplets": len(train_examples),
        "dev_triplets": len(dev_examples),
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "training_objective": "TripletLoss(query, positive_passage, hard_negative_passage)",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved fine-tuned embedding model to: {output_dir}")


if __name__ == "__main__":
    main()
