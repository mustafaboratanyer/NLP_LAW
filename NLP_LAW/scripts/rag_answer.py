import argparse
import hashlib
import json
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def portable_cache_root(name: str) -> Path:
    if os.environ.get("PUBLIC"):
        return Path(os.environ["PUBLIC"]) / "NLP_LAW" / name
    return Path.home() / ".cache" / "nlp_law" / name


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


def faiss_readable_path(path: Path) -> Path:
    """Stage a FAISS index at an ASCII path on Windows when needed."""
    resolved = path.resolve()
    if os.name != "nt" or str(resolved).isascii():
        return resolved

    cache_dir = portable_cache_root("faiss_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_stat = resolved.stat()
    fingerprint = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    cached_path = cache_dir / f"{fingerprint}_{resolved.name}"
    cache_is_current = (
        cached_path.exists()
        and cached_path.stat().st_size == source_stat.st_size
        and cached_path.stat().st_mtime_ns >= source_stat.st_mtime_ns
    )
    if not cache_is_current:
        print(f"Copying FAISS index to ASCII cache path: {cached_path}")
        shutil.copy2(resolved, cached_path)
    return cached_path


def model_readable_path(path: str | Path) -> str:
    """Stage a local model directory at an ASCII path on Windows when needed."""
    candidate = Path(path)
    if not candidate.exists() or os.name != "nt" or str(candidate.resolve()).isascii():
        return str(path)

    resolved = candidate.resolve()
    fingerprint = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
    cached_path = portable_cache_root("model_cache") / f"{fingerprint}_{resolved.name}"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(resolved, cached_path, dirs_exist_ok=True)
    return str(cached_path)


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
        self.adapter_loaded = False
        self.generation_backend = "transformers"
        self.ollama_host = ""
        self.ollama_base_model = ""
        self.ollama_fine_tuned_model = ""

    def load_retriever(self, embedding_device: str = "cpu") -> None:
        import faiss
        from sentence_transformers import SentenceTransformer
        from evaluate_retrieval import BM25Index

        self.metadata = load_json(self.metadata_path)
        self.config = load_json(self.config_path)
        self.articles = article_lookup(self.corpus_path)

        model_name = self.config.get("model") or self.config.get("embedding_model")
        if not model_name:
            raise ValueError(
                f"Embedding model is missing from config: {self.config_path}. "
                "Expected 'model' or 'embedding_model'."
            )
        print(f"Loading embedding model: {model_name} on {embedding_device}")
        self.embedding_model = SentenceTransformer(model_name, device=embedding_device)
        if self.config.get("max_seq_length"):
            self.embedding_model.max_seq_length = int(self.config["max_seq_length"])

        readable_index_path = faiss_readable_path(self.index_path)
        print(f"Loading FAISS index: {readable_index_path}")
        self.index = faiss.read_index(str(readable_index_path))

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

        if load_in_4bit and not torch.cuda.is_available():
            print("CUDA is not available; loading the LLM without 4-bit quantization.")
            load_in_4bit = False

        readable_adapter_path = model_readable_path(adapter_path) if adapter_path else None
        tokenizer_path = base_model
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
                llm_int8_enable_fp32_cpu_offload=True,
            )

        print(f"Loading base LLM: {base_model}")
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["max_memory"] = {
                0: "3200MiB",
                "cpu": "24GiB",
            }
        elif torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

        if adapter_path:
            from peft import PeftModel

            print(f"Loading LoRA adapter: {readable_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, readable_adapter_path)
            self.adapter_loaded = True
        else:
            self.adapter_loaded = False

        self.model.eval()
        print("LLM ready.")

    def load_ollama(
        self,
        base_model: str = "qwen2.5:7b-instruct-q4_K_M",
        fine_tuned_model: str = "nlp-law-finetuned",
        host: str = "http://127.0.0.1:11434",
    ) -> None:
        self.ollama_host = host.rstrip("/")
        self.ollama_base_model = base_model
        self.ollama_fine_tuned_model = fine_tuned_model

        try:
            with urlopen(f"{self.ollama_host}/api/tags", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError) as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self.ollama_host}. Start Ollama first."
            ) from exc

        installed_models = {
            item.get("name", "").removesuffix(":latest")
            for item in payload.get("models", [])
        }
        missing = [
            model
            for model in [base_model, fine_tuned_model]
            if model.removesuffix(":latest") not in installed_models
        ]
        if missing:
            raise RuntimeError(f"Missing Ollama model(s): {', '.join(missing)}")

        self.generation_backend = "ollama"
        self.adapter_loaded = True
        print(f"Ollama ready: base={base_model}, fine_tuned={fine_tuned_model}")

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
        use_adapter: bool = True,
        max_new_tokens: int = 384,
        do_sample: bool = False,
        temperature: float = 0.2,
        repetition_penalty: float = 1.1,
    ) -> str:
        if self.generation_backend == "ollama":
            return self._generate_ollama(
                question=question,
                contexts=contexts,
                use_adapter=use_adapter,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )

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

        adapter_context = nullcontext()
        if self.adapter_loaded and not use_adapter:
            adapter_context = self.model.disable_adapter()

        with adapter_context:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)

        answer_ids = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    def _generate_ollama(
        self,
        question: str,
        contexts: list[dict],
        use_adapter: bool,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float,
    ) -> str:
        model_name = (
            self.ollama_fine_tuned_model if use_adapter else self.ollama_base_model
        )
        payload = {
            "model": model_name,
            "messages": self.build_prompt(question, contexts),
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": max_new_tokens,
                "num_ctx": 4096,
                "temperature": temperature if do_sample else 0,
                "repeat_penalty": repetition_penalty,
            },
        }
        request = Request(
            f"{self.ollama_host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=900) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (OSError, URLError) as exc:
            raise RuntimeError(f"Ollama generation failed for model: {model_name}") from exc
        return str(result.get("message", {}).get("content", "")).strip()

    def answer(
        self,
        question: str,
        top_k: int = 5,
        mode: str | None = None,
        **generation_kwargs,
    ) -> dict:
        mode = mode or ("fine_tuned" if self.adapter_loaded else "base")
        if mode not in {"base", "fine_tuned"}:
            raise ValueError("mode must be 'base' or 'fine_tuned'.")
        if mode == "fine_tuned" and not self.adapter_loaded:
            raise RuntimeError("Fine-tuned mode requires a loaded LoRA adapter.")

        contexts = self.retrieve(question, top_k=top_k)
        answer = self.generate(
            question,
            contexts,
            use_adapter=mode == "fine_tuned",
            **generation_kwargs,
        )
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "mode": mode,
        }

    def compare(self, question: str, top_k: int = 5, **generation_kwargs) -> dict:
        """Generate base and fine-tuned answers from the exact same retrieval result."""
        if not self.adapter_loaded:
            raise RuntimeError("Comparison mode requires a loaded LoRA adapter.")

        contexts = self.retrieve(question, top_k=top_k)
        base_answer = self.generate(
            question,
            contexts,
            use_adapter=False,
            **generation_kwargs,
        )
        fine_tuned_answer = self.generate(
            question,
            contexts,
            use_adapter=True,
            **generation_kwargs,
        )
        return {
            "question": question,
            "base_answer": base_answer,
            "fine_tuned_answer": fine_tuned_answer,
            "contexts": contexts,
            "mode": "compare",
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
