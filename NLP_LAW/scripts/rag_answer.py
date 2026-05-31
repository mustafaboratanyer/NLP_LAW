import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def normalize_text(text: str) -> str:
    return (
        text.casefold()
        .replace("\u0307", "")
        .replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )


def expand_query(question: str) -> str:
    """Add narrow legal hints for common colloquial Turkish legal questions."""
    normalized = normalize_text(question)
    normalized_tokens = set(normalized.split())
    expansions: list[str] = []

    if {"2", "iki"} & normalized_tokens:
        expansions.append("iki iki gün iki işgünü ardı ardına")

    if "isci" in normalized and (
        "ise gel" in normalized
        or "ise gelmez" in normalized
        or "gelmez" in normalized
        or "gelmedi" in normalized
        or "devamsiz" in normalized
        or "devamsizlik" in normalized
    ):
        expansions.append(
            "işçinin işverenden izin almaksızın haklı bir sebebe dayanmaksızın "
            "ardı ardına iki işgünü işine devam etmemesi devamsızlık "
            "işverenin haklı nedenle derhal fesih hakkı"
        )

    if not expansions:
        return question
    return question + "\n" + "\n".join(expansions)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def article_lookup(path: Path) -> dict[str, dict]:
    articles = load_json(path)
    return {row["id"]: row for row in articles}


def format_citation(row: dict) -> str:
    title = str(row.get("article_title", "")).strip()
    citation = f"{row.get('law_name', '')} ({row.get('law_no', '')}), {row.get('article_no', '')}"
    return f"{citation} - {title}" if title else citation


class LegalRAG:
    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        config_path: str | Path,
        corpus_path: str | Path,
        alpha: float = 0.70,
        dense_candidates: int = 300,
        bm25_candidates: int = 100,
        preliminary_top_k: int = 50,
        use_query_expansion: bool = False,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.config_path = Path(config_path)
        self.corpus_path = Path(corpus_path)
        self.alpha = alpha
        self.dense_candidates = dense_candidates
        self.bm25_candidates = bm25_candidates
        self.preliminary_top_k = preliminary_top_k
        self.use_query_expansion = use_query_expansion

        self.metadata: list[dict] | None = None
        self.config: dict | None = None
        self.articles: dict[str, dict] | None = None
        self.index = None
        self.embedding_model = None
        self.bm25_index = None
        self.tokenizer = None
        self.model = None

    def load_retriever(self, embedding_device: str = "cpu") -> None:
        import faiss
        from sentence_transformers import SentenceTransformer
        from evaluate_retrieval import BM25Index

        self.metadata = load_json(self.metadata_path)
        self.config = load_json(self.config_path)
        self.articles = article_lookup(self.corpus_path)

        model_name = self.config["model"]
        print(f"Loading embedding model: {model_name} on {embedding_device}")
        self.embedding_model = SentenceTransformer(model_name, device=embedding_device)
        if self.config.get("max_seq_length"):
            self.embedding_model.max_seq_length = int(self.config["max_seq_length"])

        print(f"Loading FAISS index: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        print("Building BM25 index...")
        self.bm25_index = BM25Index([row["text"] for row in self.metadata])
        print("Retriever ready.")

    def load_llm(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        adapter_path: str | Path | None = None,
        load_in_4bit: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer_path = str(adapter_path) if adapter_path else base_model
        print(f"Loading tokenizer: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        print(f"Loading base LLM: {base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        if adapter_path:
            from peft import PeftModel

            print(f"Loading LoRA adapter: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))

        self.model.eval()
        print("LLM ready.")

    def _encode_query(self, question: str):
        if self.embedding_model is None or self.config is None:
            raise RuntimeError("Retriever is not loaded. Call load_retriever() first.")

        encode_kwargs = {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if self.config.get("query_prompt_name"):
            encode_kwargs["prompt_name"] = self.config["query_prompt_name"]
            texts = [question]
        elif self.config.get("query_prompt"):
            encode_kwargs["prompt"] = self.config["query_prompt"]
            texts = [question]
        else:
            query_prefix = self.config.get("query_prefix", "query: ")
            texts = [f"{query_prefix}{question}"]

        return self.embedding_model.encode(texts, **encode_kwargs).astype("float32")

    def retrieve(self, question: str, top_k: int = 5) -> list[dict]:
        if any(value is None for value in [self.metadata, self.articles, self.index, self.bm25_index]):
            raise RuntimeError("Retriever is not loaded. Call load_retriever() first.")

        from evaluate_retrieval import dedupe_parent, hybrid_rank

        search_query = expand_query(question) if self.use_query_expansion else question
        dense_top_k = max(self.dense_candidates, self.preliminary_top_k, top_k)
        query_embedding = self._encode_query(search_query)
        scores, indices = self.index.search(query_embedding, min(dense_top_k, self.index.ntotal))
        dense_raw = {
            int(index_id): float(score)
            for score, index_id in zip(scores[0], indices[0])
            if int(index_id) >= 0
        }

        ranked = hybrid_rank(
            search_query,
            dense_raw,
            self.bm25_index,
            dense_candidates=self.dense_candidates,
            bm25_candidates=self.bm25_candidates,
            top_k=self.preliminary_top_k,
            alpha=self.alpha,
        )
        ranked = dedupe_parent(ranked, self.metadata, top_k)

        contexts = []
        for rank, (score, dense_score, bm25_score, doc_id) in enumerate(ranked, start=1):
            chunk = self.metadata[doc_id]
            parent_id = chunk.get("parent_id", chunk.get("id"))
            article = self.articles.get(parent_id, chunk)
            contexts.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "dense_score": float(dense_score),
                    "bm25_score": float(bm25_score),
                    "chunk_id": chunk.get("id"),
                    "parent_id": parent_id,
                    "citation": format_citation(article),
                    "text": article.get("text", chunk.get("text", "")),
                    "source_url": article.get("source_url", chunk.get("source_url", "")),
                }
            )
        return contexts

    def build_prompt(self, question: str, contexts: list[dict], max_chars_per_doc: int = 1800) -> list[dict]:
        blocks = []
        for context in contexts:
            text = str(context["text"]).strip().replace("\n\n\n", "\n\n")
            blocks.append(
                "\n".join(
                    [
                        f"[Kaynak {context['rank']}]",
                        f"Citation: {context['citation']}",
                        f"Doc ID: {context['parent_id']}",
                        f"Metin: {text[:max_chars_per_doc]}",
                    ]
                )
            )

        context_text = "\n\n".join(blocks)
        system_prompt = (
            "Sen bir Türk hukuku RAG asistanısın. "
            "Yalnızca verilen kaynak metinlere dayanarak cevap ver. "
            "Kaynakta olmayan bilgiyi üretme. "
            "Cevabın sonunda kullandığın kanun ve madde citation bilgisini belirt. "
            "Kaynaklarda cevap yoksa 'Verilen kaynaklarda bu sorunun cevabı yoktur.' de."
        )
        user_prompt = f"[Kaynaklar]\n{context_text}\n\n[Soru]\n{question}"
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def generate(
        self,
        question: str,
        contexts: list[dict],
        max_new_tokens: int = 384,
        do_sample: bool = False,
        temperature: float = 0.2,
        repetition_penalty: float = 1.1,
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM is not loaded. Call load_llm() first.")

        import torch

        messages = self.build_prompt(question, contexts)
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        answer_ids = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    def answer(self, question: str, top_k: int = 5, **generation_kwargs) -> dict:
        contexts = self.retrieve(question, top_k=top_k)
        answer = self.generate(question, contexts, **generation_kwargs)
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Turkish legal RAG answer generation.")
    parser.add_argument("question")
    parser.add_argument("--index", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--show-contexts", action="store_true")
    parser.add_argument("--use-query-expansion", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    rag = LegalRAG(
        index_path=args.index,
        metadata_path=args.metadata,
        config_path=args.config,
        corpus_path=args.corpus,
        use_query_expansion=args.use_query_expansion,
    )
    rag.load_retriever(embedding_device=args.embedding_device)
    rag.load_llm(base_model=args.base_model, adapter_path=args.adapter_path)
    result = rag.answer(args.question, top_k=args.top_k, max_new_tokens=args.max_new_tokens)

    print("\nANSWER")
    print("=" * 80)
    print(result["answer"])

    if args.show_contexts:
        print("\nCONTEXTS")
        print("=" * 80)
        for context in result["contexts"]:
            print(f"[{context['rank']}] score={context['score']:.4f} {context['citation']}")
            print(f"parent_id={context['parent_id']}")
            print("-" * 80)


if __name__ == "__main__":
    main()
