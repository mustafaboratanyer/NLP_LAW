import argparse
import json
import re
from pathlib import Path


DEFAULT_MAX_WORDS = 450
DEFAULT_MIN_WORDS = 120
DEFAULT_OVERLAP_WORDS = 80

WORD_RE = re.compile(r"\S+")
CLAUSE_START_RE = re.compile(
    r"(?m)^\s*(?:\(?\d+\)|\d+[\).]|[a-zçğıöşü][\).]|[A-ZÇĞİÖŞÜ][\).]|[IVXLCDM]+[\).])\s+"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+(?=\(?[A-ZÇĞİÖŞÜ0-9])")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_paragraphs(text: str) -> list[str]:
    paragraphs = []
    for match in re.finditer(r"\S[\s\S]*?(?=(?:\n\s*\n)+|\Z)", text):
        paragraph = match.group(0).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs or [text.strip()]


def split_by_clause_markers(text: str) -> list[str]:
    matches = list(CLAUSE_START_RE.finditer(text))
    if len(matches) <= 1:
        return [text.strip()]

    starts = [0]
    starts.extend(match.start() for match in matches if match.start() > 0)
    starts.append(len(text))

    parts = []
    for index in range(len(starts) - 1):
        part = text[starts[index] : starts[index + 1]].strip()
        if part:
            parts.append(part)
    return parts or [text.strip()]


def split_by_sentences(text: str) -> list[str]:
    parts = []
    start = 0
    for match in SENTENCE_SPLIT_RE.finditer(text):
        part = text[start : match.start()].strip()
        if part:
            parts.append(part)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts if len(parts) > 1 else [text.strip()]


def hard_split_words(text: str, max_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[index : index + max_words]) for index in range(0, len(words), max_words)]


def split_oversized_unit(text: str, max_words: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if word_count(text) <= max_words:
        return [text]

    for splitter in (split_by_clause_markers, split_by_sentences):
        parts = splitter(text)
        if len(parts) <= 1:
            continue

        result = []
        for part in parts:
            result.extend(split_oversized_unit(part, max_words))
        return result

    return hard_split_words(text, max_words)


def pack_units(units: list[str], max_words: int, min_words: int) -> list[str]:
    chunks = []
    current = []
    current_words = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        unit_words = word_count(unit)
        should_flush = current and current_words + unit_words > max_words and current_words >= min_words
        if should_flush:
            chunks.append("\n\n".join(current).strip())
            current = [unit]
            current_words = unit_words
        else:
            current.append(unit)
            current_words += unit_words

    if current:
        chunks.append("\n\n".join(current).strip())

    if len(chunks) > 1 and word_count(chunks[-1]) < min_words:
        merged_word_count = word_count(chunks[-2]) + word_count(chunks[-1])
        if merged_word_count > max_words + min_words:
            return chunks
        chunks[-2] = f"{chunks[-2].rstrip()}\n\n{chunks[-1].lstrip()}"
        chunks.pop()

    return chunks


def chunk_body_text(text: str, max_words: int, min_words: int) -> list[str]:
    if word_count(text) <= max_words:
        return [text.strip()]

    units = []
    for paragraph in split_paragraphs(text):
        units.extend(split_oversized_unit(paragraph, max_words))

    return pack_units(units, max_words, min_words)


def tail_words(text: str, overlap_words: int) -> str:
    if overlap_words <= 0:
        return ""

    words = text.split()
    if len(words) <= overlap_words:
        return text.strip()
    return " ".join(words[-overlap_words:])


def build_chunk_header(article: dict[str, str]) -> str:
    parts = [f"{article['law_name']} ({article['law_no']})", article["article_no"]]
    article_title = article.get("article_title", "").strip()
    if article_title:
        parts.append(article_title)
    return " - ".join(parts)


def build_chunks(
    corpus: list[dict[str, str]],
    max_words: int,
    min_words: int,
    overlap_words: int,
) -> list[dict[str, object]]:
    chunks = []

    for article in corpus:
        parent_text = article["text"].strip()
        body_chunks = chunk_body_text(parent_text, max_words=max_words, min_words=min_words)
        chunk_count = len(body_chunks)
        parent_word_count = word_count(parent_text)
        header = build_chunk_header(article)

        for index, raw_body in enumerate(body_chunks):
            overlap = tail_words(body_chunks[index - 1], overlap_words) if index > 0 else ""
            chunk_body = f"{overlap}\n\n{raw_body}".strip() if overlap else raw_body
            chunk_text = f"{header}\n{chunk_body}".strip()

            chunks.append(
                {
                    "id": f"{article['id']}_chunk_{index + 1:03d}",
                    "parent_id": article["id"],
                    "law_name": article["law_name"],
                    "law_no": article["law_no"],
                    "article_no": article["article_no"],
                    "article_title": article.get("article_title", ""),
                    "chunk_index": index,
                    "chunk_count": chunk_count,
                    "is_full_article": chunk_count == 1,
                    "parent_word_count": parent_word_count,
                    "chunk_word_count": word_count(chunk_body),
                    "overlap_word_count": word_count(overlap),
                    "text": chunk_text,
                    "source_url": article["source_url"],
                }
            )

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunk-level retrieval corpus from article-level corpus.")
    parser.add_argument(
        "--in",
        dest="input_path",
        default="data/processed/retrieval_corpus.json",
        help="Input article-level corpus JSON.",
    )
    parser.add_argument(
        "--out",
        default="data/processed/retrieval_chunks.json",
        help="Output chunk-level corpus JSON.",
    )
    parser.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    parser.add_argument("--overlap-words", type=int, default=DEFAULT_OVERLAP_WORDS)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = json.loads(input_path.read_text(encoding="utf-8"))
    chunks = build_chunks(
        corpus,
        max_words=args.max_words,
        min_words=args.min_words,
        overlap_words=args.overlap_words,
    )

    with out_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    split_articles = sum(1 for chunk in chunks if not chunk["is_full_article"] and chunk["chunk_index"] == 0)
    print(f"Loaded {len(corpus)} articles from {input_path}")
    print(f"Saved {len(chunks)} retrieval chunks to {out_path}")
    print(f"Split {split_articles} long articles; kept {len(corpus) - split_articles} articles as single chunks")
    print(
        "Chunk settings: "
        f"max_words={args.max_words}, min_words={args.min_words}, overlap_words={args.overlap_words}"
    )


if __name__ == "__main__":
    main()
