import argparse
import csv
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
BAD_QUERY_PATTERNS = [
    r"\boricon\b",
    r"kayna[gğ][aı]na g[oö]re",
    r"kaynak.*g[oö]re",
    r"metne g[oö]re",
    r"par[cç]aya g[oö]re",
    r"yukar[ıi]daki",
    r"a[sş]a[gğ][ıi]daki",
    r"\bkay[ıi]t\s*\d+",
    r"sizce neden",
    r"olay[ıi]n [oö]z[üu]",
    r"karar sonucu",
]
STOPWORDS = {
    "acaba",
    "ama",
    "ancak",
    "bir",
    "biri",
    "birini",
    "bu",
    "da",
    "de",
    "diye",
    "gore",
    "hangi",
    "icin",
    "ile",
    "ise",
    "kanun",
    "kanuna",
    "kanunda",
    "madde",
    "maddede",
    "maddesi",
    "mi",
    "mu",
    "mudur",
    "nasil",
    "ne",
    "nedir",
    "olur",
    "olarak",
    "ve",
    "veya",
}
GENERIC_TITLE_TOKENS = {
    "amac",
    "ana",
    "basvuru",
    "bildirim",
    "diger",
    "esas",
    "genel",
    "hukum",
    "hukumler",
    "izin",
    "karar",
    "kapsam",
    "kosul",
    "olarak",
    "sure",
    "tanim",
    "tanimlar",
    "uygulama",
    "usul",
    "verilmesi",
    "yapilmasi",
}
LAW_ALIASES = {
    "anayasa": "2709",
    "anayasasi": "2709",
    "turkiye cumhuriyeti anayasasi": "2709",
    "tck": "5237",
    "turk ceza kanunu": "5237",
    "ceza kanunu": "5237",
    "cmk": "5271",
    "ceza muhakemesi kanunu": "5271",
    "tmk": "4721",
    "turk medeni kanunu": "4721",
    "medeni kanun": "4721",
    "tbk": "6098",
    "turk borclar kanunu": "6098",
    "borclar kanunu": "6098",
    "is kanunu": "4857",
    "turkiye cumhuriyeti is kanunu": "4857",
    "kvkk": "6698",
    "kisisel verilerin korunmasi kanunu": "6698",
    "hukuk muhakemeleri kanunu": "6100",
    "hmk": "6100",
    "icra ve iflas kanunu": "2004",
    "iik": "2004",
    "turk ticaret kanunu": "6102",
    "ttk": "6102",
    "tuketici kanunu": "6502",
    "tuketicinin korunmasi hakkinda kanun": "6502",
    "bilgi edinme kanunu": "4982",
    "bilgi edinme hakki kanunu": "4982",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text: str) -> str:
    text = str(text or "").casefold().replace("\u0131", "i")
    text = unicodedata.normalize("NFKD", text)
    return "".join(char for char in text if not unicodedata.combining(char))


def clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def content_tokens(text: str) -> set[str]:
    tokens = set()
    for token in TOKEN_RE.findall(normalize_text(text)):
        if len(token) < 3 or token.isdigit() or token in STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def containment_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def parse_float(value: str) -> float:
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return -1.0


def is_bad_question(question: str) -> bool:
    question_norm = normalize_text(question)
    if len(question) < 18 or len(question) > 240:
        return True
    if not question.endswith("?"):
        question = question + "?"
    for pattern in BAD_QUERY_PATTERNS:
        if re.search(pattern, question_norm):
            return True
    return False


def readable_question(question: str) -> str:
    question = clean_space(question)
    if question and not question.endswith("?"):
        question += "?"
    return question


def clean_title_for_question(title: str) -> str:
    title = clean_space(title).strip(":;-")
    title = re.sub(r"^[IVXLCDM]+\s*[-.)]\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^\d+\s*[-.)]\s*", "", title)
    title = re.sub(r"^[a-zçğıöşü]\s*[-.)]\s*", "", title, flags=re.IGNORECASE)
    title = clean_space(title).strip(":;-")
    return title


def is_useful_title(title: str) -> bool:
    title = clean_title_for_question(title)
    tokens = [
        token
        for token in TOKEN_RE.findall(normalize_text(title))
        if len(token) > 2 and token not in GENERIC_TITLE_TOKENS
    ]
    if len(tokens) < 2:
        return False
    if len(title) < 12:
        return False
    return True


def article_sort_key(article_no: str) -> str:
    return normalize_text(article_no).replace("madde", "").replace(" ", "")


def build_parent_lookup(corpus: list[dict]) -> tuple[dict[str, dict], dict[tuple[str, str], str]]:
    parent_lookup = {row["id"]: row for row in corpus}
    by_law_article = {}
    for row in corpus:
        law_no = str(row.get("law_no", ""))
        article_key = article_sort_key(str(row.get("article_no", "")))
        if article_key:
            by_law_article[(law_no, article_key)] = row["id"]
            by_law_article[(law_no, article_key.replace("/", ""))] = row["id"]
    return parent_lookup, by_law_article


def resolve_law_no(text: str, source: str = "") -> str:
    text_norm = normalize_text(" ".join([text, source]))
    for alias, law_no in sorted(LAW_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in text_norm:
            return law_no
    return ""


def parse_article_no(question: str) -> str:
    question_norm = normalize_text(question)
    patterns = [
        r"madde\s*(\d+)\s*/\s*([a-z])",
        r"madde\s*(\d+)\s*[-]?\s*([a-z])\b",
        r"(\d+)\s*[/]\s*([a-z])\s*(?:maddesi|madde)",
        r"madde\s*(\d+)",
        r"(\d+)\.\s*madd",
    ]
    for pattern in patterns:
        match = re.search(pattern, question_norm)
        if not match:
            continue
        if len(match.groups()) >= 2 and match.group(2):
            return f"{match.group(1)}/{match.group(2)}"
        return match.group(1)
    return ""


def law_match_score(source: str, law_name: str) -> float:
    source_norm = normalize_text(source)
    law_norm = normalize_text(law_name)
    if source_norm and (source_norm in law_norm or law_norm in source_norm):
        return 1.0
    source_law_no = resolve_law_no("", source)
    return 1.0 if source_law_no and source_law_no == str(law_name) else 0.0


def chunks_by_law(chunks: list[dict]) -> dict[str, list[int]]:
    by_law = defaultdict(list)
    for index, chunk in enumerate(chunks):
        by_law[str(chunk.get("law_no", ""))].append(index)
    return dict(by_law)


def find_best_chunk_for_context(
    source: str,
    context: str,
    chunks: list[dict],
    chunk_token_sets: list[set[str]],
    by_law: dict[str, list[int]],
    min_context_overlap: float,
) -> tuple[int | None, float]:
    law_no = resolve_law_no("", source)
    if not law_no or law_no not in by_law:
        return None, 0.0
    candidate_ids = by_law[law_no]
    context_tokens = content_tokens(context)
    best_doc_id = None
    best_overlap = 0.0
    for doc_id in candidate_ids:
        overlap = containment_overlap(context_tokens, chunk_token_sets[doc_id])
        if overlap > best_overlap:
            best_overlap = overlap
            best_doc_id = doc_id
    if best_doc_id is None or best_overlap < min_context_overlap:
        return None, best_overlap
    return best_doc_id, best_overlap


def add_candidate(
    candidates: list[dict],
    seen_questions: set[str],
    parent_lookup: dict[str, dict],
    question: str,
    parent_id: str,
    source_dataset: str,
    source_row: str,
    quality_reason: str,
    score: float,
    answer_overlap: float = 0.0,
    context_overlap: float = 0.0,
) -> bool:
    question = readable_question(question)
    question_key = normalize_text(question)
    if question_key in seen_questions or is_bad_question(question):
        return False
    article = parent_lookup.get(parent_id)
    if not article:
        return False
    seen_questions.add(question_key)
    candidates.append(
        {
            "question": question,
            "positive_parent_id": parent_id,
            "law_name": article.get("law_name", ""),
            "law_no": article.get("law_no", ""),
            "article_no": article.get("article_no", ""),
            "article_title": article.get("article_title", ""),
            "source_dataset": source_dataset,
            "source_row": source_row,
            "quality_reason": quality_reason,
            "score": round(score, 4),
            "answer_overlap": round(answer_overlap, 4),
            "context_overlap": round(context_overlap, 4),
        }
    )
    return True


def collect_hf_explicit(
    paths: list[Path],
    parent_lookup: dict[str, dict],
    by_law_article: dict[tuple[str, str], str],
    seen_questions: set[str],
    min_answer_overlap: float,
) -> tuple[list[dict], Counter]:
    stats = Counter()
    candidates = []
    for path in paths:
        if not path.exists():
            continue
        rows = load_json(path)
        for index, row in enumerate(rows):
            stats["hf_rows"] += 1
            question = row.get("Soru") or row.get("soru") or ""
            answer = row.get("Cevap") or row.get("cevap") or ""
            if is_bad_question(question):
                stats["hf_skipped_bad_question"] += 1
                continue
            law_no = resolve_law_no(question, "")
            article_no = parse_article_no(question)
            if not law_no or not article_no:
                stats["hf_skipped_no_explicit_article"] += 1
                continue
            parent_id = by_law_article.get((law_no, article_no))
            if not parent_id:
                stats["hf_skipped_article_not_in_corpus"] += 1
                continue
            article = parent_lookup[parent_id]
            answer_overlap = containment_overlap(content_tokens(answer), content_tokens(article.get("text", "")))
            if answer_overlap < min_answer_overlap:
                stats["hf_skipped_low_answer_overlap"] += 1
                continue
            if add_candidate(
                candidates,
                seen_questions,
                parent_lookup,
                question,
                parent_id,
                "huggingface_lawchatbot_explicit_article",
                f"{path.name}:{index}",
                "explicit_law_article_in_question",
                score=answer_overlap,
                answer_overlap=answer_overlap,
            ):
                stats["hf_added"] += 1
    return candidates, stats


def collect_kaggle_context(
    csv_path: Path,
    chunks: list[dict],
    parent_lookup: dict[str, dict],
    seen_questions: set[str],
    min_dataset_score: float,
    min_context_overlap: float,
    min_answer_overlap: float,
) -> tuple[list[dict], Counter]:
    stats = Counter()
    if not csv_path.exists():
        return [], stats

    chunk_token_sets = [content_tokens(chunk.get("text", "")) for chunk in chunks]
    by_law = chunks_by_law(chunks)
    context_cache: dict[tuple[str, str], tuple[int | None, float]] = {}
    candidates = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader):
            stats["kaggle_rows"] += 1
            question = row.get("soru", "")
            answer = row.get("cevap", "")
            context = row.get("context", "")
            source = row.get("kaynak", "")
            dataset_score = parse_float(row.get("Score", ""))
            if dataset_score < min_dataset_score:
                stats["kaggle_skipped_low_score"] += 1
                continue
            if is_bad_question(question):
                stats["kaggle_skipped_bad_question"] += 1
                continue
            cache_key = (normalize_text(source), normalize_text(context))
            if cache_key not in context_cache:
                context_cache[cache_key] = find_best_chunk_for_context(
                    source,
                    context,
                    chunks,
                    chunk_token_sets,
                    by_law,
                    min_context_overlap,
                )
            chunk_id, context_overlap = context_cache[cache_key]
            if chunk_id is None:
                stats["kaggle_skipped_no_context_match"] += 1
                continue
            chunk = chunks[chunk_id]
            parent_id = chunk.get("parent_id")
            if not parent_id:
                stats["kaggle_skipped_no_parent"] += 1
                continue
            answer_overlap = containment_overlap(content_tokens(answer), content_tokens(chunk.get("text", "")))
            if answer_overlap < min_answer_overlap:
                stats["kaggle_skipped_low_answer_overlap"] += 1
                continue
            if add_candidate(
                candidates,
                seen_questions,
                parent_lookup,
                question,
                parent_id,
                "kaggle_turkish_law_dataset_context_matched",
                f"{csv_path.name}:{index}",
                "high_score_context_matched",
                score=dataset_score,
                answer_overlap=answer_overlap,
                context_overlap=context_overlap,
            ):
                stats["kaggle_added"] += 1
    return candidates, stats


def collect_corpus_title_fallback(
    corpus: list[dict],
    parent_lookup: dict[str, dict],
    seen_questions: set[str],
    per_article: int,
) -> tuple[list[dict], Counter]:
    stats = Counter()
    candidates = []
    templates = [
        "{law_name} kapsamında {title} nasıl düzenlenir?",
        "{law_name} kapsamında {title} hakkında hangi hükümler vardır?",
        "{title} konusunda {law_name} ne düzenler?",
    ]
    for article in corpus:
        raw_title = clean_space(article.get("article_title", "")).strip(":;-")
        title = clean_title_for_question(raw_title)
        if not title or len(title) < 5:
            stats["fallback_skipped_no_title"] += 1
            continue
        if normalize_text(title) in {"amac", "kapsam", "tanimlar", "yururluk", "yurutme"}:
            stats["fallback_skipped_generic_title"] += 1
            continue
        if not is_useful_title(title):
            stats["fallback_skipped_weak_title"] += 1
            continue
        for template in templates[:per_article]:
            question = template.format(title=title, law_name=article.get("law_name", ""))
            if add_candidate(
                candidates,
                seen_questions,
                parent_lookup,
                question,
                article["id"],
                "corpus_article_title_fallback",
                article["id"],
                "exact_parent_from_corpus_title",
                score=1.0,
            ):
                stats["fallback_added"] += 1
    return candidates, stats


def balanced_select(
    rows: list[dict],
    target: int,
    max_per_law: int,
    max_per_parent: int,
    seed: int,
) -> list[dict]:
    priority = {
        "huggingface_lawchatbot_explicit_article": 0,
        "kaggle_turkish_law_dataset_context_matched": 1,
        "corpus_article_title_fallback": 2,
    }
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda row: (priority.get(row["source_dataset"], 99), -float(row.get("score", 0))))

    selected = []
    law_counts = Counter()
    parent_counts = Counter()
    for row in shuffled:
        law_no = row["law_no"]
        parent_id = row["positive_parent_id"]
        if law_counts[law_no] >= max_per_law:
            continue
        if parent_counts[parent_id] >= max_per_parent:
            continue
        selected.append(row)
        law_counts[law_no] += 1
        parent_counts[parent_id] += 1
        if len(selected) >= target:
            break

    if len(selected) < target:
        selected_keys = {row["question"] for row in selected}
        for row in shuffled:
            if row["question"] in selected_keys:
                continue
            law_no = row["law_no"]
            parent_id = row["positive_parent_id"]
            if law_counts[law_no] >= max_per_law:
                continue
            if parent_counts[parent_id] >= max_per_parent + 1:
                continue
            selected.append(row)
            law_counts[law_no] += 1
            parent_counts[parent_id] += 1
            selected_keys.add(row["question"])
            if len(selected) >= target:
                break

    if len(selected) < target:
        selected_keys = {row["question"] for row in selected}
        for row in shuffled:
            if row["question"] in selected_keys:
                continue
            parent_id = row["positive_parent_id"]
            if parent_counts[parent_id] >= max_per_parent + 1:
                continue
            selected.append(row)
            parent_counts[parent_id] += 1
            selected_keys.add(row["question"])
            if len(selected) >= target:
                break
    return selected


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question",
        "positive_parent_id",
        "law_name",
        "law_no",
        "article_no",
        "article_title",
        "source_dataset",
        "source_row",
        "quality_reason",
        "score",
        "answer_overlap",
        "context_overlap",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cleaner question-to-parent-id reranker training set.")
    parser.add_argument("--corpus", default="data/processed/retrieval_corpus.json")
    parser.add_argument("--chunks", default="data/processed/retrieval_chunks.json")
    parser.add_argument("--kaggle-csv", default="data/external/kaggle_law_dataset/turkish_law_dataset.csv")
    parser.add_argument("--hf-json", nargs="*", default=["huggingface_dataset/train.json", "huggingface_dataset/test.json"])
    parser.add_argument("--output", default="data/reranker/clean_reranker_train_questions.csv")
    parser.add_argument("--stats-out", default="data/reranker/clean_reranker_train_questions.stats.json")
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--min-kaggle-score", type=float, default=9.0)
    parser.add_argument("--min-context-overlap", type=float, default=0.55)
    parser.add_argument("--min-answer-overlap", type=float, default=0.08)
    parser.add_argument("--max-per-law", type=int, default=180)
    parser.add_argument("--max-per-parent", type=int, default=3)
    parser.add_argument("--include-title-fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=493)
    args = parser.parse_args()

    corpus = load_json(Path(args.corpus))
    chunks = load_json(Path(args.chunks))
    parent_lookup, by_law_article = build_parent_lookup(corpus)
    seen_questions: set[str] = set()
    all_candidates = []
    stats = Counter()

    hf_candidates, hf_stats = collect_hf_explicit(
        [Path(path) for path in args.hf_json],
        parent_lookup,
        by_law_article,
        seen_questions,
        args.min_answer_overlap,
    )
    all_candidates.extend(hf_candidates)
    stats.update(hf_stats)

    kaggle_candidates, kaggle_stats = collect_kaggle_context(
        Path(args.kaggle_csv),
        chunks,
        parent_lookup,
        seen_questions,
        args.min_kaggle_score,
        args.min_context_overlap,
        args.min_answer_overlap,
    )
    all_candidates.extend(kaggle_candidates)
    stats.update(kaggle_stats)

    if args.include_title_fallback:
        fallback_candidates, fallback_stats = collect_corpus_title_fallback(
            corpus,
            parent_lookup,
            seen_questions,
            per_article=1,
        )
        all_candidates.extend(fallback_candidates)
        stats.update(fallback_stats)

    selected = balanced_select(
        all_candidates,
        args.target,
        args.max_per_law,
        args.max_per_parent,
        args.seed,
    )
    write_csv(Path(args.output), selected)

    output_stats = {
        **dict(stats),
        "all_candidates": len(all_candidates),
        "selected": len(selected),
        "selected_by_source": Counter(row["source_dataset"] for row in selected),
        "selected_by_law": Counter(f"{row['law_no']} {row['law_name']}" for row in selected),
        "selected_unique_parents": len({row["positive_parent_id"] for row in selected}),
        "target": args.target,
        "max_per_law": args.max_per_law,
        "max_per_parent": args.max_per_parent,
        "min_kaggle_score": args.min_kaggle_score,
        "min_context_overlap": args.min_context_overlap,
        "min_answer_overlap": args.min_answer_overlap,
    }
    write_json(Path(args.stats_out), output_stats)

    print(f"Saved selected questions: {args.output} ({len(selected)})")
    print(f"Saved stats: {args.stats_out}")
    print(json.dumps(output_stats, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
