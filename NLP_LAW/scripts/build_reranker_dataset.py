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
DEFAULT_QUESTION_KEYS = ("question", "soru", "query", "instruction", "prompt", "input")
DEFAULT_ANSWER_KEYS = ("answer", "cevap", "response", "output", "completion", "text")
DEFAULT_CONTEXT_KEYS = ("context", "passage", "document", "source_text", "article_text")
GENERIC_TITLES = {
    "amac",
    "amaç",
    "kapsam",
    "tanimlar",
    "tanımlar",
    "yururluk",
    "yürürlük",
    "yurutme",
    "yürütme",
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
    text = text.casefold().replace("ı", "i")
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def char_ngrams(token: str, n: int = 4) -> list[str]:
    if len(token) < n:
        return []
    return [f"char:{token[index:index+n]}" for index in range(len(token) - n + 1)]


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(normalize_text(text))
    expanded = []
    for token in tokens:
        if len(token) < 2:
            continue
        expanded.append(token)
        expanded.extend(char_ngrams(token))
    return expanded


def short_text(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def article_passage(article: dict) -> str:
    title = article.get("article_title", "")
    header = f"{article['law_name']} ({article['law_no']}), {article['article_no']}"
    if title:
        header = f"{header} - {title}"
    return f"{header}\n{article['text']}"


def citation(article: dict) -> str:
    title = article.get("article_title", "")
    base = f"{article['law_name']} ({article['law_no']}), {article['article_no']}"
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
            counts = Counter(tokenize(document))
            self.doc_lengths.append(sum(counts.values()))
            for term, freq in counts.items():
                self.postings[term].append((doc_id, freq))
                self.doc_freqs[term] += 1

        self.doc_count = len(documents)
        self.avg_doc_length = sum(self.doc_lengths) / max(1, self.doc_count)

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, int]]:
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


def iter_json_records(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        records = []
        for value in data.values():
            records.extend(iter_json_records(value))
        return records
    return []


def load_local_qa_files(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                rows.extend(dict(row) for row in csv.DictReader(file))
        elif suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as file:
                rows.extend(json.loads(line) for line in file if line.strip())
        else:
            rows.extend(iter_json_records(load_json(path)))
    return rows


def load_hf_rows(dataset_names: list[str], max_rows_per_dataset: int | None) -> list[dict]:
    if not dataset_names:
        return []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets first: pip install datasets") from exc

    rows = []
    for dataset_name in dataset_names:
        dataset = load_dataset(dataset_name)
        for split_name, split in dataset.items():
            for row in split:
                item = dict(row)
                item["_source_dataset"] = dataset_name
                item["_source_split"] = split_name
                rows.append(item)
                if max_rows_per_dataset and len(rows) >= max_rows_per_dataset:
                    break
            if max_rows_per_dataset and len(rows) >= max_rows_per_dataset:
                break
    return rows


def first_nonempty(row: dict, keys: tuple[str, ...]) -> str:
    lowered_keys = {str(key).casefold(): key for key in row}
    for key in keys:
        actual_key = lowered_keys.get(key.casefold(), key)
        value = row.get(actual_key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def qa_from_messages(row: dict) -> tuple[str, str]:
    messages = row.get("messages") or row.get("conversations")
    if not isinstance(messages, list):
        return "", ""

    question = ""
    answer = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or message.get("from") or "").lower()
        content = str(message.get("content") or message.get("value") or "").strip()
        if not content:
            continue
        if not question and role in {"user", "human"}:
            question = content
        elif question and not answer and role in {"assistant", "gpt", "bot"}:
            answer = content
            break
    return question, answer


def extract_qa(row: dict) -> dict | None:
    question, answer = qa_from_messages(row)
    if not question:
        question = first_nonempty(row, DEFAULT_QUESTION_KEYS)
    if not answer:
        answer = first_nonempty(row, DEFAULT_ANSWER_KEYS)

    context = first_nonempty(row, DEFAULT_CONTEXT_KEYS)
    if not question or len(question) < 8:
        return None

    # Avoid rows where the same field was accidentally used as both question and answer.
    if normalize_text(question) == normalize_text(answer):
        answer = ""

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "source_dataset": row.get("_source_dataset") or row.get("source") or "local",
        "source_split": row.get("_source_split") or "",
    }


def positive_query_text(example: dict) -> str:
    # Context/answer carry more legal wording than the user question, so they help
    # map public QA datasets back to our article-level corpus.
    return " ".join(
        value
        for value in [example.get("context", ""), example.get("answer", ""), example.get("question", "")]
        if value
    )


def synthetic_questions(article: dict, per_article: int) -> list[str]:
    title = article.get("article_title", "").strip().strip(":;-")
    law_name = article.get("law_name", "").strip()
    if not title or normalize_text(title) in GENERIC_TITLES:
        return []

    templates = [
        f"{law_name} kapsamında {title} nedir?",
    ]

    verb_phrase = title_to_infinitive_phrase(title)
    if article.get("law_no") == "5237":
        if verb_phrase:
            templates.extend(
                [
                    f"Birini {verb_phrase} suç mudur?",
                    f"{verb_phrase.capitalize()} suç mudur?",
                ]
            )
        templates.extend(
            [
                f"{title} suç mudur?",
                f"{title} cezası nedir?",
                f"{law_name} kapsamında {title} cezası nedir?",
            ]
        )

    templates.extend(
        [
            f"{law_name} kapsamında {title} nasıl düzenlenir?",
            f"{title} nedir?",
            f"{title} nasıl düzenlenir?",
            f"{title} hakkında hangi hükümler vardır?",
        ]
    )
    return templates[:per_article]


def title_to_infinitive_phrase(title: str) -> str:
    words = title.strip().split()
    if not words:
        return ""

    last = words[-1]
    normalized_last = normalize_text(last)
    if normalized_last.endswith("me"):
        words[-1] = last[:-2] + "mek"
    elif normalized_last.endswith("ma"):
        words[-1] = last[:-2] + "mak"
    else:
        return ""

    return " ".join(words).casefold()


def add_pair(
    rows: list[dict],
    query: str,
    article: dict,
    label: int,
    positive_article_id: str,
    source: str,
    match_score: float | None = None,
) -> None:
    rows.append(
        {
            "query": query,
            "passage": article_passage(article),
            "label": int(label),
            "positive_article_id": positive_article_id,
            "candidate_article_id": article["id"],
            "candidate_citation": citation(article),
            "source": source,
            "match_score": match_score,
        }
    )


def hard_negative_ids(
    query: str,
    index: BM25Index,
    positive_id: str,
    articles: list[dict],
    count: int,
    pool_size: int,
    rng: random.Random,
) -> list[int]:
    negatives = []
    seen = {positive_id}
    for _, doc_id in index.search(query, top_k=pool_size):
        article_id = articles[doc_id]["id"]
        if article_id in seen:
            continue
        seen.add(article_id)
        negatives.append(doc_id)
        if len(negatives) >= count:
            return negatives

    # Fallback random negatives if BM25 did not produce enough.
    while len(negatives) < count:
        doc_id = rng.randrange(len(articles))
        article_id = articles[doc_id]["id"]
        if article_id in seen:
            continue
        seen.add(article_id)
        negatives.append(doc_id)
    return negatives


def build_pairs(
    articles: list[dict],
    index: BM25Index,
    qa_examples: list[dict],
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    rng = random.Random(args.seed)
    pairs = []
    stats = Counter()

    for article_id, article in enumerate(articles[: args.synthetic_limit or len(articles)]):
        for question in synthetic_questions(article, args.synthetic_per_article):
            for repeat_index in range(args.positive_repeat):
                source = "synthetic_title" if repeat_index == 0 else "synthetic_title:repeat"
                add_pair(pairs, question, article, 1, article["id"], source, None)
            for negative_id in hard_negative_ids(
                question,
                index,
                article["id"],
                articles,
                args.negatives_per_query,
                args.hard_negative_pool,
                rng,
            ):
                add_pair(pairs, question, articles[negative_id], 0, article["id"], "synthetic_title", None)
            stats["synthetic_queries"] += 1

    for raw_example in qa_examples[: args.max_qa_examples or len(qa_examples)]:
        example = extract_qa(raw_example)
        if not example:
            stats["qa_skipped_no_question"] += 1
            continue

        match_results = index.search(positive_query_text(example), top_k=1)
        if not match_results:
            stats["qa_skipped_no_match"] += 1
            continue

        match_score, positive_doc_id = match_results[0]
        if match_score < args.min_positive_score:
            stats["qa_skipped_low_score"] += 1
            continue

        positive_article = articles[positive_doc_id]
        query = example["question"]
        for repeat_index in range(args.positive_repeat):
            source = f"qa:{example['source_dataset']}:{example['source_split']}"
            if repeat_index > 0:
                source = f"{source}:repeat"
            add_pair(
                pairs,
                query,
                positive_article,
                1,
                positive_article["id"],
                source,
                match_score,
            )
        for negative_id in hard_negative_ids(
            query,
            index,
            positive_article["id"],
            articles,
            args.negatives_per_query,
            args.hard_negative_pool,
            rng,
        ):
            add_pair(
                pairs,
                query,
                articles[negative_id],
                0,
                positive_article["id"],
                f"qa:{example['source_dataset']}:{example['source_split']}",
                match_score,
            )
        stats["qa_queries"] += 1

    rng.shuffle(pairs)
    stats["total_pairs"] = len(pairs)
    stats["positive_pairs"] = sum(1 for row in pairs if row["label"] == 1)
    stats["negative_pairs"] = sum(1 for row in pairs if row["label"] == 0)
    return pairs, dict(stats)


def make_rerank_eval_samples(dev_pairs: list[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for row in dev_pairs:
        key = row["query"]
        sample = grouped.setdefault(key, {"query": row["query"], "positive": [], "negative": []})
        if row["label"] == 1:
            sample["positive"].append(row["passage"])
        else:
            sample["negative"].append(row["passage"])
    return [sample for sample in grouped.values() if sample["positive"] and sample["negative"]]


def split_pairs(pairs: list[dict], dev_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    queries = sorted({row["query"] for row in pairs})
    rng.shuffle(queries)
    dev_query_count = max(1, int(len(queries) * dev_ratio))
    dev_queries = set(queries[:dev_query_count])

    train_rows = [row for row in pairs if row["query"] not in dev_queries]
    dev_rows = [row for row in pairs if row["query"] in dev_queries]
    return train_rows, dev_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build positive/negative pairs for Turkish legal reranker fine-tuning.")
    parser.add_argument("--corpus", default="data/processed/retrieval_corpus.json")
    parser.add_argument("--output-dir", default="data/reranker")
    parser.add_argument("--qa-files", nargs="*", default=[], help="Optional local QA JSON/JSONL files.")
    parser.add_argument("--hf-datasets", nargs="*", default=[], help="Optional HF datasets, e.g. Renicames/turkish-law-chatbot.")
    parser.add_argument("--max-hf-rows", type=int, default=None)
    parser.add_argument("--max-qa-examples", type=int, default=None)
    parser.add_argument("--min-positive-score", type=float, default=5.0)
    parser.add_argument("--synthetic-per-article", type=int, default=1)
    parser.add_argument("--synthetic-limit", type=int, default=None)
    parser.add_argument("--negatives-per-query", type=int, default=4)
    parser.add_argument("--positive-repeat", type=int, default=1)
    parser.add_argument("--hard-negative-pool", type=int, default=30)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=493)
    args = parser.parse_args()

    articles = load_json(Path(args.corpus))
    article_docs = [
        " ".join(
            [
                article.get("law_name", ""),
                article.get("law_no", ""),
                article.get("article_no", ""),
                article.get("article_title", ""),
                article.get("text", ""),
            ]
        )
        for article in articles
    ]

    print(f"Loaded {len(articles)} article-level documents.")
    print("Building BM25 index over the retrieval corpus...")
    index = BM25Index(article_docs)

    qa_rows = []
    qa_rows.extend(load_local_qa_files([Path(path) for path in args.qa_files]))
    qa_rows.extend(load_hf_rows(args.hf_datasets, args.max_hf_rows))
    print(f"Loaded {len(qa_rows)} raw QA rows.")

    pairs, stats = build_pairs(articles, index, qa_rows, args)
    train_pairs, dev_pairs = split_pairs(pairs, args.dev_ratio, args.seed)
    eval_samples = make_rerank_eval_samples(dev_pairs)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train_pairs.jsonl", train_pairs)
    write_jsonl(output_dir / "dev_pairs.jsonl", dev_pairs)
    write_json(output_dir / "dev_rerank_samples.json", eval_samples)
    write_json(
        output_dir / "dataset_stats.json",
        {
            **stats,
            "train_pairs": len(train_pairs),
            "dev_pairs": len(dev_pairs),
            "dev_rerank_samples": len(eval_samples),
            "negatives_per_query": args.negatives_per_query,
            "positive_repeat": args.positive_repeat,
            "min_positive_score": args.min_positive_score,
        },
    )

    print(f"Saved train pairs: {output_dir / 'train_pairs.jsonl'} ({len(train_pairs)})")
    print(f"Saved dev pairs:   {output_dir / 'dev_pairs.jsonl'} ({len(dev_pairs)})")
    print(f"Saved stats:       {output_dir / 'dataset_stats.json'}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
