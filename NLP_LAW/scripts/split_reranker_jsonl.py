import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_key(row: dict) -> str:
    return str(row.get("query_id") or row.get("query"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Split reranker JSONL by query_id to avoid query leakage.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--dev-out", required=True)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=493)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    rng = random.Random(args.seed)
    keys = list(groups)
    rng.shuffle(keys)
    dev_count = max(1, int(len(keys) * args.dev_ratio))
    dev_keys = set(keys[:dev_count])

    train_rows = [row for key in keys if key not in dev_keys for row in groups[key]]
    dev_rows = [row for key in keys if key in dev_keys for row in groups[key]]

    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.dev_out), dev_rows)

    print(f"Input rows:  {len(rows)}")
    print(f"Query groups: {len(groups)}")
    print(f"Train rows:  {len(train_rows)}")
    print(f"Dev rows:    {len(dev_rows)}")
    print(f"Dev groups:  {len(dev_keys)}")


if __name__ == "__main__":
    main()
