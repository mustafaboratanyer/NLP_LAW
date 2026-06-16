import argparse
import json
import math
import os
from pathlib import Path

# Kaggle can expose multiple GPUs and Sentence-Transformers may wrap the model
# with DataParallel. CrossEncoder's BCE loss expects model.device, which
# DataParallel does not expose, so we keep training on a single visible GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
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
    for key in ("passage", "candidate_passage", "positive_passage", "context"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError("Expected one of: passage, candidate_passage, positive_passage, context")


def to_input_examples(rows: list[dict]) -> list[InputExample]:
    return [
        InputExample(texts=[row["query"], passage_text(row)], label=float(row["label"]))
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a Turkish legal cross-encoder reranker.")
    parser.add_argument("--train", default="data/reranker/train_pairs.jsonl")
    parser.add_argument("--dev", default="data/reranker/dev_pairs.jsonl")
    parser.add_argument("--base-model", default="dbmdz/bert-base-turkish-cased")
    parser.add_argument("--output-dir", default="data/models/legal-berturk-reranker")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--evaluation-steps", type=int, default=1000)
    parser.add_argument("--save-best-model", action="store_true")
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train), args.train_limit)
    dev_rows = load_jsonl(Path(args.dev), args.dev_limit)
    if not train_rows:
        raise ValueError("Training data is empty.")
    if not dev_rows:
        raise ValueError("Dev data is empty.")

    print(f"Loaded train pairs: {len(train_rows)}")
    print(f"Loaded dev pairs:   {len(dev_rows)}")
    print(f"Base model:         {args.base_model}")

    model = CrossEncoder(
        args.base_model,
        num_labels=1,
        max_length=args.max_length,
        device=args.device,
    )

    train_examples = to_input_examples(train_rows)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)

    dev_sentence_pairs = [[row["query"], passage_text(row)] for row in dev_rows]
    dev_labels = [int(row["label"]) for row in dev_rows]
    evaluator = CrossEncoderClassificationEvaluator(
        dev_sentence_pairs,
        dev_labels,
        name="legal-reranker-dev",
    )

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = math.ceil(total_steps * 0.1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=args.evaluation_steps,
        optimizer_params={"lr": args.learning_rate},
        output_path=str(output_dir),
        save_best_model=args.save_best_model,
    )

    if not args.save_best_model:
        model.save(str(output_dir))

    config = {
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "train_pairs": len(train_rows),
        "dev_pairs": len(dev_rows),
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved fine-tuned reranker to: {output_dir}")


if __name__ == "__main__":
    main()
