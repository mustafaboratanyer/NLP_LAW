import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from custom_benchmark import (  # noqa: E402
    average_available,
    evaluate_answer,
    load_benchmark_rows,
    retrieval_metrics,
)
from rag_answer import LegalRAG  # noqa: E402


DEFAULT_PATHS = {
    "index": ROOT_DIR / "data" / "index" / "faiss_bge_m3.index",
    "metadata": ROOT_DIR / "data" / "index" / "metadata_bge_m3.json",
    "config": ROOT_DIR / "data" / "index" / "index_config_bge_m3.json",
    "corpus": ROOT_DIR / "data" / "processed" / "retrieval_corpus.json",
    "adapter": ROOT_DIR / "models" / "qwen_7b_lora_v2" / "final_600",
}


def local_storage_root() -> Path:
    configured = os.environ.get("RAG_LOCAL_STORAGE")
    if configured:
        return Path(configured).expanduser()
    if os.environ.get("PUBLIC"):
        return Path(os.environ["PUBLIC"]) / "NLP_LAW"
    return Path.home() / ".nlp_law"


LOCAL_STORAGE_ROOT = local_storage_root()
LOCAL_BASE_MODEL = Path(
    os.environ.get(
        "RAG_LOCAL_MODEL_DIR",
        str(LOCAL_STORAGE_ROOT / "models" / "Qwen2.5-7B-Instruct"),
    )
)
BASE_MODEL = os.environ.get(
    "RAG_BASE_MODEL",
    str(LOCAL_BASE_MODEL) if LOCAL_BASE_MODEL.exists() else "Qwen/Qwen2.5-7B-Instruct",
)
CUSTOM_ROOT = Path(
    os.environ.get("RAG_CUSTOM_ROOT", str(LOCAL_STORAGE_ROOT / "custom"))
)
SUPPORTED_DOCUMENT_TYPES = ["json", "jsonl", "csv", "txt", "md", "pdf"]
SUPPORTED_BENCHMARK_TYPES = ["csv", "json", "jsonl"]


def env_path(name: str, default: Path) -> str:
    return os.environ.get(name, str(default))


def safe_filename(name: str, fallback: str) -> str:
    filename = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return cleaned or fallback


def uploaded_files_fingerprint(uploaded_files) -> str:
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8", errors="ignore"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()[:16]


def run_command(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(detail or f"Command failed: {' '.join(command)}")
    return completed.stdout.strip()


def build_custom_artifacts(
    uploaded_files,
    collection_name: str,
    embedding_device: str,
) -> dict[str, str | int]:
    fingerprint = uploaded_files_fingerprint(uploaded_files)
    session_dir = CUSTOM_ROOT / fingerprint
    upload_dir = session_dir / "uploads"
    processed_dir = session_dir / "processed"
    index_dir = session_dir / "index"
    upload_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    for index, uploaded_file in enumerate(uploaded_files, start=1):
        filename = safe_filename(uploaded_file.name, f"document_{index:03d}.txt")
        (upload_dir / filename).write_bytes(uploaded_file.getvalue())

    corpus_path = processed_dir / "retrieval_corpus_custom.json"
    chunks_path = processed_dir / "retrieval_chunks_custom.json"
    index_path = index_dir / "faiss_custom_bge_m3.index"
    metadata_path = index_dir / "metadata_custom_bge_m3.json"
    config_path = index_dir / "index_config_custom_bge_m3.json"

    prepare_output = run_command(
        [
            sys.executable,
            str(SCRIPTS_DIR / "prepare_custom_corpus.py"),
            "--input",
            str(upload_dir),
            "--corpus-out",
            str(corpus_path),
            "--chunks-out",
            str(chunks_path),
            "--collection-name",
            collection_name.strip() or "Custom Documents",
        ]
    )
    prepare_summary = json.loads(prepare_output)
    if int(prepare_summary.get("chunks", 0)) == 0:
        raise ValueError("Yüklenen dosyalardan aranabilir metin çıkarılamadı.")

    run_command(
        [
            sys.executable,
            str(SCRIPTS_DIR / "build_faiss_index.py"),
            "--chunks",
            str(chunks_path),
            "--index-out",
            str(index_path),
            "--metadata-out",
            str(metadata_path),
            "--config-out",
            str(config_path),
            "--model",
            "BAAI/bge-m3",
            "--query-prefix",
            "query: ",
            "--passage-prefix",
            "passage: ",
            "--device",
            embedding_device,
            "--batch-size",
            "8" if embedding_device == "cuda" else "4",
        ]
    )

    return {
        "fingerprint": fingerprint,
        "corpus": str(corpus_path),
        "chunks": str(chunks_path),
        "index": str(index_path),
        "metadata": str(metadata_path),
        "config": str(config_path),
        "documents": int(prepare_summary["documents"]),
        "chunk_count": int(prepare_summary["chunks"]),
    }


def save_benchmark_file(uploaded_file) -> Path:
    digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:16]
    benchmark_dir = CUSTOM_ROOT / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower()
    path = benchmark_dir / f"benchmark_{digest}{suffix}"
    path.write_bytes(uploaded_file.getvalue())
    return path


@st.cache_resource(show_spinner=False)
def load_rag(
    index_path: str,
    metadata_path: str,
    config_path: str,
    corpus_path: str,
    adapter_path: str,
    embedding_device: str,
    load_in_4bit: bool,
    llm_backend: str,
) -> LegalRAG:
    rag = LegalRAG(
        index_path=index_path,
        metadata_path=metadata_path,
        config_path=config_path,
        corpus_path=corpus_path,
        alpha=0.70,
        dense_candidates=300,
        bm25_candidates=100,
        preliminary_top_k=50,
        use_query_expansion=False,
    )
    rag.load_retriever(embedding_device=embedding_device)
    if llm_backend == "ollama":
        rag.load_ollama()
    else:
        rag.load_llm(
            base_model=BASE_MODEL,
            adapter_path=adapter_path or None,
            load_in_4bit=load_in_4bit,
        )
    return rag


def render_sources(contexts: list[dict]) -> None:
    st.subheader("Kaynaklar")
    for context in contexts:
        label = f"{context['rank']}. {context['citation']}"
        with st.expander(label):
            st.caption(
                f"Hybrid: {context['score']:.4f} | "
                f"Dense: {context['dense_score']:.4f} | "
                f"BM25: {context['bm25_score']:.4f}"
            )
            st.write(context["text"])
            if context.get("source_url"):
                st.caption(f"Kaynak: {context['source_url']}")


def display_metric(label: str, value: float | None) -> None:
    st.metric(label, "-" if value is None else f"{value:.4f}")


def results_to_csv(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    output = io.StringIO()
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def run_benchmark(
    rag: LegalRAG,
    benchmark_rows: list[dict],
    compare_models: bool,
    max_new_tokens: int,
    progress_bar,
    status_text,
) -> tuple[list[dict], dict]:
    results = []
    total = len(benchmark_rows)

    for index, example in enumerate(benchmark_rows, start=1):
        status_text.write(f"{index}/{total}: {example['question']}")
        if compare_models:
            generated = rag.compare(
                example["question"],
                top_k=5,
                max_new_tokens=max_new_tokens,
            )
            model_answers = {
                "base": generated["base_answer"],
                "fine_tuned": generated["fine_tuned_answer"],
            }
        else:
            generated = rag.answer(
                example["question"],
                top_k=5,
                mode="fine_tuned",
                max_new_tokens=max_new_tokens,
            )
            model_answers = {"fine_tuned": generated["answer"]}

        retrieval = retrieval_metrics(
            generated["contexts"],
            example["gold_document_ids"],
        )
        retrieved_ids = "|".join(
            str(context.get("parent_id", "")) for context in generated["contexts"]
        )

        for model_name, answer in model_answers.items():
            answer_metrics = evaluate_answer(answer, example["expected_answer"])
            results.append(
                {
                    "question_id": example["question_id"],
                    "model": model_name,
                    "question": example["question"],
                    "expected_answer": example["expected_answer"],
                    "generated_answer": answer,
                    "gold_document_ids": "|".join(example["gold_document_ids"]),
                    "retrieved_document_ids": retrieved_ids,
                    **retrieval,
                    **answer_metrics,
                }
            )
        progress_bar.progress(index / total)

    summaries = {}
    for model_name in sorted({row["model"] for row in results}):
        model_rows = [row for row in results if row["model"] == model_name]
        summaries[model_name] = {
            "questions": len(model_rows),
            "em": average_available(model_rows, "em"),
            "f1": average_available(model_rows, "f1"),
            "rouge_l": average_available(model_rows, "rouge_l"),
            "retrieval_hit": average_available(model_rows, "retrieval_hit"),
            "mrr": average_available(model_rows, "reciprocal_rank"),
        }
    return results, summaries


st.set_page_config(page_title="Türk Hukuku RAG", layout="wide")
st.title("Türk Hukuku RAG Asistanı")
st.caption(
    "BGE-M3 + FAISS + BM25 hybrid retrieval. "
    "Yanıtlar yalnızca getirilen kaynaklara dayandırılır."
)

if "custom_artifacts" not in st.session_state:
    st.session_state.custom_artifacts = None
if "data_source" not in st.session_state:
    st.session_state.data_source = "Hazır hukuk corpusu"

with st.sidebar:
    mode_label = st.radio(
        "Model",
        ["Base RAG", "Fine-tuned RAG", "Karşılaştır"],
        help="Retrieval her modda aynıdır; yalnızca LoRA adapter durumu değişir.",
    )
    llm_backend_label = st.selectbox(
        "LLM backend",
        ["Ollama (laptop için önerilen)", "Transformers"],
    )
    embedding_device = st.selectbox("Embedding cihazı", ["cpu", "cuda"])
    load_in_4bit = st.checkbox(
        "LLM'i 4-bit yükle",
        value=True,
        disabled=llm_backend_label.startswith("Ollama"),
    )
    max_new_tokens = st.slider("Maksimum yeni token", 64, 768, 384, 32)
    st.selectbox(
        "Aktif doküman koleksiyonu",
        ["Hazır hukuk corpusu", "Yüklenen custom corpus"],
        key="data_source",
    )

question_tab, custom_tab, benchmark_tab = st.tabs(
    ["Soru-Cevap", "Custom Dokümanlar", "Benchmark"]
)

with custom_tab:
    st.subheader("Doküman koleksiyonu")
    st.write(
        "Sistemin üzerinde çalışacağı doküman koleksiyonunu yükleyin."
    )
    collection_name = st.text_input("Koleksiyon adı", value="Özel Doküman Koleksiyonu")
    uploaded_documents = st.file_uploader(
        "Dokümanları yükle",
        type=SUPPORTED_DOCUMENT_TYPES,
        accept_multiple_files=True,
        help="JSON, JSONL, CSV, TXT, MD ve metin içeren PDF desteklenir.",
    )
    if st.button(
        "Koleksiyonu hazırla",
        type="primary",
        disabled=not uploaded_documents,
    ):
        with st.spinner(
            "Dokümanlar işleniyor ve BGE-M3 FAISS indexi oluşturuluyor..."
        ):
            try:
                artifacts = build_custom_artifacts(
                    uploaded_documents,
                    collection_name=collection_name,
                    embedding_device=embedding_device,
                )
            except Exception as exc:
                st.exception(exc)
            else:
                st.session_state.custom_artifacts = artifacts
                st.session_state.data_source = "Yüklenen custom corpus"
                st.session_state.pop("benchmark_results", None)
                st.success(
                    f"{artifacts['documents']} doküman ve "
                    f"{artifacts['chunk_count']} chunk hazırlandı."
                )
                st.rerun()

    if st.session_state.custom_artifacts:
        artifacts = st.session_state.custom_artifacts
        st.success(
            f"Aktif custom koleksiyon hazır: {artifacts['documents']} doküman, "
            f"{artifacts['chunk_count']} chunk."
        )
        st.code(
            "\n".join(
                [
                    f"Corpus: {artifacts['corpus']}",
                    f"FAISS: {artifacts['index']}",
                    f"Metadata: {artifacts['metadata']}",
                ]
            )
        )

if st.session_state.data_source == "Yüklenen custom corpus":
    selected = st.session_state.custom_artifacts
    if selected:
        index_path = str(selected["index"])
        metadata_path = str(selected["metadata"])
        config_path = str(selected["config"])
        corpus_path = str(selected["corpus"])
    else:
        index_path = metadata_path = config_path = corpus_path = ""
else:
    index_path = env_path("RAG_INDEX_PATH", DEFAULT_PATHS["index"])
    metadata_path = env_path("RAG_METADATA_PATH", DEFAULT_PATHS["metadata"])
    config_path = env_path("RAG_CONFIG_PATH", DEFAULT_PATHS["config"])
    corpus_path = env_path("RAG_CORPUS_PATH", DEFAULT_PATHS["corpus"])

adapter_path = env_path("RAG_ADAPTER_PATH", DEFAULT_PATHS["adapter"])
rag = None
load_error = None

if st.session_state.data_source == "Yüklenen custom corpus" and not st.session_state.custom_artifacts:
    load_error = "Önce Custom Dokümanlar sekmesinden bir koleksiyon oluştur."
else:
    required_paths = {
        "FAISS index": Path(index_path),
        "Metadata": Path(metadata_path),
        "Index config": Path(config_path),
        "Retrieval corpus": Path(corpus_path),
        "LoRA adapter": Path(adapter_path),
    }
    missing_paths = [
        f"{name}: {path}" for name, path in required_paths.items() if not path.exists()
    ]
    if missing_paths:
        load_error = "Eksik dosyalar:\n" + "\n".join(missing_paths)

if not load_error:
    with st.spinner("Retriever ve dil modeli ilk kez yükleniyor..."):
        try:
            rag = load_rag(
                index_path=index_path,
                metadata_path=metadata_path,
                config_path=config_path,
                corpus_path=corpus_path,
                adapter_path=adapter_path,
                embedding_device=embedding_device,
                load_in_4bit=load_in_4bit,
                llm_backend=(
                    "ollama"
                    if llm_backend_label.startswith("Ollama")
                    else "transformers"
                ),
            )
        except Exception as exc:
            load_error = str(exc)

with question_tab:
    if load_error:
        st.warning(load_error)
    else:
        source_label = (
            "custom koleksiyon"
            if st.session_state.data_source == "Yüklenen custom corpus"
            else "hazır hukuk corpusu"
        )
        st.success(
            f"RAG hazır ({source_label}). Model ve index sonraki sorularda yeniden yüklenmez."
        )
        with st.form("question_form"):
            question = st.text_area(
                "Hukuki sorunuzu yazın",
                placeholder="Örnek: İşçi hangi durumlarda kıdem tazminatına hak kazanır?",
                height=120,
            )
            submitted = st.form_submit_button("Soruyu yanıtla", type="primary")

        if submitted:
            question = question.strip()
            if not question:
                st.warning("Lütfen bir soru yazın.")
            else:
                with st.spinner("Kaynaklar aranıyor ve yanıt hazırlanıyor..."):
                    try:
                        if mode_label == "Karşılaştır":
                            result = rag.compare(
                                question,
                                top_k=5,
                                max_new_tokens=max_new_tokens,
                            )
                        else:
                            result = rag.answer(
                                question,
                                top_k=5,
                                mode=(
                                    "fine_tuned"
                                    if mode_label == "Base RAG"
                                    else "base"
                                ),
                                max_new_tokens=max_new_tokens,
                            )
                    except Exception as exc:
                        st.exception(exc)
                        result = None

                if result and result["mode"] == "compare":
                    base_column, fine_tuned_column = st.columns(2)
                    with base_column:
                        st.subheader("Fine-tuned RAG")
                        st.write(result["base_answer"])
                    with fine_tuned_column:
                        st.subheader("Base RAG")
                        st.write(result["fine_tuned_answer"])
                elif result:
                    st.subheader("Yanıt")
                    st.write(result["answer"])

                if result:
                    render_sources(result["contexts"])

with benchmark_tab:
    st.subheader("Custom benchmark değerlendirmesi")
    st.write(
        "CSV, JSON veya JSONL benchmark dosyası yükleyin. Minimum soru alanı "
        "`question`/`soru`/`query` olmalıdır. Cevap metrikleri için "
        "`expected_answer` veya `answer`; retrieval metriği için "
        "`expected_document_id` veya `corpus_row_id` eklenebilir."
    )
    benchmark_file = st.file_uploader(
        "Benchmark dosyasını yükle",
        type=SUPPORTED_BENCHMARK_TYPES,
        key="benchmark_upload",
    )
    compare_models = st.checkbox(
        "Base ve Fine-tuned modelleri aynı retrieval sonuçlarıyla karşılaştır",
        value=False,
    )
    if benchmark_file:
        try:
            benchmark_path = save_benchmark_file(benchmark_file)
            benchmark_rows = load_benchmark_rows(benchmark_path)
        except Exception as exc:
            st.exception(exc)
            benchmark_rows = []
        else:
            st.info(f"{len(benchmark_rows)} benchmark sorusu bulundu.")
            run_limit = st.number_input(
                "Çalıştırılacak maksimum soru",
                min_value=1,
                max_value=len(benchmark_rows),
                value=len(benchmark_rows),
                step=1,
            )
            if st.button(
                "Benchmark'ı çalıştır",
                type="primary",
                disabled=rag is None,
            ):
                selected_rows = benchmark_rows[: int(run_limit)]
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                try:
                    results, summaries = run_benchmark(
                        rag,
                        selected_rows,
                        compare_models=compare_models,
                        max_new_tokens=max_new_tokens,
                        progress_bar=progress_bar,
                        status_text=status_text,
                    )
                except Exception as exc:
                    st.exception(exc)
                else:
                    status_text.success("Benchmark tamamlandı.")
                    st.session_state.benchmark_results = {
                        "rows": results,
                        "summaries": summaries,
                    }

    benchmark_result = st.session_state.get("benchmark_results")
    if benchmark_result:
        summaries = benchmark_result["summaries"]
        for model_name, summary in summaries.items():
            st.markdown(
                f"### {'Fine-tuned RAG' if model_name == 'fine_tuned' else 'Base RAG'}"
            )
            columns = st.columns(5)
            with columns[0]:
                display_metric("EM", summary["em"])
            with columns[1]:
                display_metric("F1", summary["f1"])
            with columns[2]:
                display_metric("ROUGE-L", summary["rouge_l"])
            with columns[3]:
                display_metric("Retrieval Hit@5", summary["retrieval_hit"])
            with columns[4]:
                display_metric("MRR@5", summary["mrr"])

        table_rows = benchmark_result["rows"]
        st.dataframe(table_rows, use_container_width=True, hide_index=True)
        st.download_button(
            "Sonuçları CSV olarak indir",
            data=results_to_csv(table_rows),
            file_name="custom_benchmark_results.csv",
            mime="text/csv",
        )
