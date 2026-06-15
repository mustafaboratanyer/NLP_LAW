import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


TEXT_EXTENSIONS = {".txt", ".md"}
STRUCTURED_EXTENSIONS = {".json", ".jsonl", ".csv"}


def slugify(text: str) -> str:
    text = str(text).strip().lower()
    replacements = {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "doc"


def stable_id(text: str, prefix: str = "custom") -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def read_json(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        if "documents" in data and isinstance(data["documents"], list):
            data = data["documents"]
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON object/list or {{'documents': [...]}} in {path}")
    return [dict(row) for row in data]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def read_text_file(path: Path) -> dict:
    return {
        "id": stable_id(str(path), prefix="custom"),
        "title": path.stem,
        "text": path.read_text(encoding="utf-8", errors="ignore"),
        "source_url": str(path),
    }


def read_pdf_file(path: Path) -> dict:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PDF input requires PyMuPDF. Install dependencies from requirements.txt, "
            "or convert PDFs to .txt first."
        ) from exc

    with fitz.open(path) as document:
        text = "\n\n".join(page.get_text("text") or "" for page in document)
    return {
        "id": stable_id(str(path), prefix="custom"),
        "title": path.stem,
        "text": text,
        "source_url": str(path),
    }


def load_rows(input_path: Path) -> list[dict]:
    if input_path.is_file():
        suffix = input_path.suffix.lower()
        if suffix == ".json":
            return read_json(input_path)
        if suffix == ".jsonl":
            return read_jsonl(input_path)
        if suffix == ".csv":
            return read_csv(input_path)
        if suffix in TEXT_EXTENSIONS:
            return [read_text_file(input_path)]
        if suffix == ".pdf":
            return [read_pdf_file(input_path)]
        raise ValueError(f"Unsupported input file type: {input_path}")

    rows = []
    for path in sorted(input_path.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in STRUCTURED_EXTENSIONS:
            rows.extend(load_rows(path))
        elif suffix in TEXT_EXTENSIONS:
            rows.append(read_text_file(path))
        elif suffix == ".pdf":
            rows.append(read_pdf_file(path))
    if not rows:
        raise ValueError(f"No supported documents found under {input_path}")
    return rows


def first_present(row: dict, keys: list[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def normalize_document(row: dict, index: int, default_collection_name: str) -> dict | None:
    text = first_present(row, ["text", "content", "context", "body", "metin"])
    if not text:
        return None

    law_name = first_present(
        row,
        ["law_name", "kanun_adi", "kaynak", "source", "collection", "title"],
        default_collection_name,
    )
    title = first_present(row, ["article_title", "title", "heading", "baslik"], "")
    law_no = first_present(row, ["law_no", "kanun_no", "doc_no"], "CUSTOM")
    article_no = first_present(row, ["article_no", "madde_no", "section", "doc_no"], f"Doc {index:04d}")
    source_url = first_present(row, ["source_url", "url", "path", "source"], "")

    raw_id = first_present(row, ["id", "doc_id", "parent_id", "context_key"], "")
    if raw_id:
        doc_id = slugify(raw_id)
    else:
        doc_id = f"{slugify(law_no)}_{slugify(article_no)}_{stable_id(text, prefix='')}".strip("_")

    return {
        "id": doc_id,
        "law_name": law_name,
        "law_no": law_no,
        "article_no": article_no,
        "article_title": title,
        "text": text.strip(),
        "source_url": source_url,
    }


def split_words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_document(doc: dict, max_words: int, overlap_words: int) -> list[dict]:
    words = split_words(doc["text"])
    if not words:
        return []

    if len(words) <= max_words:
        windows = [(0, len(words))]
    else:
        windows = []
        start = 0
        step = max(1, max_words - overlap_words)
        while start < len(words):
            end = min(len(words), start + max_words)
            windows.append((start, end))
            if end >= len(words):
                break
            start += step

    chunk_count = len(windows)
    chunks = []
    title = f" - {doc['article_title']}" if doc.get("article_title") else ""
    header = f"{doc['law_name']} ({doc['law_no']}) - {doc['article_no']}{title}".strip()

    for chunk_index, (start, end) in enumerate(windows):
        chunk_words = words[start:end]
        body = " ".join(chunk_words)
        chunk_text = f"{header}\n{body}" if header else body
        chunks.append(
            {
                "id": f"{doc['id']}_chunk_{chunk_index + 1:03d}",
                "parent_id": doc["id"],
                "law_name": doc["law_name"],
                "law_no": doc["law_no"],
                "article_no": doc["article_no"],
                "article_title": doc.get("article_title", ""),
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "is_full_article": chunk_count == 1,
                "parent_word_count": len(words),
                "chunk_word_count": len(chunk_words),
                "overlap_word_count": overlap_words if chunk_count > 1 else 0,
                "text": chunk_text,
                "source_url": doc.get("source_url", ""),
            }
        )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare custom documents for the Turkish legal RAG pipeline.")
    parser.add_argument("--input", required=True, help="Input file or directory: json/jsonl/csv/txt/md/pdf.")
    parser.add_argument("--corpus-out", required=True, help="Output article/document-level retrieval corpus JSON.")
    parser.add_argument("--chunks-out", required=True, help="Output chunk-level retrieval JSON.")
    parser.add_argument("--collection-name", default="Custom Documents")
    parser.add_argument("--max-words", type=int, default=450)
    parser.add_argument("--overlap-words", type=int, default=64)
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    corpus = []
    seen_ids = set()
    skipped = 0

    for index, row in enumerate(rows, start=1):
        doc = normalize_document(row, index, default_collection_name=args.collection_name)
        if not doc:
            skipped += 1
            continue
        base_id = doc["id"]
        suffix = 2
        while doc["id"] in seen_ids:
            doc["id"] = f"{base_id}_{suffix}"
            suffix += 1
        seen_ids.add(doc["id"])
        corpus.append(doc)

    chunks = []
    for doc in corpus:
        chunks.extend(chunk_document(doc, max_words=args.max_words, overlap_words=args.overlap_words))

    corpus_path = Path(args.corpus_out)
    chunks_path = Path(args.chunks_out)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "input_rows": len(rows),
            "documents": len(corpus),
            "chunks": len(chunks),
            "skipped_empty_text": skipped,
            "corpus_out": str(corpus_path),
            "chunks_out": str(chunks_path),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
