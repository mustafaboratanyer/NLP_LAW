import os
import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from rag_answer import LegalRAG


DEFAULT_PATHS = {
    "index": ROOT_DIR / "data" / "index" / "faiss_bge_m3.index",
    "metadata": ROOT_DIR / "data" / "index" / "metadata_bge_m3.json",
    "config": ROOT_DIR / "data" / "index" / "index_config_bge_m3.json",
    "corpus": ROOT_DIR / "data" / "processed" / "retrieval_corpus.json",
    "adapter": ROOT_DIR / "models" / "qwen_7b_lora_v2" / "final_600",
}
LOCAL_BASE_MODEL = Path(r"C:\Users\Public\NLP_LAW_MODELS\Qwen2.5-7B-Instruct")
BASE_MODEL = os.environ.get(
    "RAG_BASE_MODEL",
    str(LOCAL_BASE_MODEL) if LOCAL_BASE_MODEL.exists() else "Qwen/Qwen2.5-7B-Instruct",
)


def env_path(name: str, default: Path) -> str:
    return os.environ.get(name, str(default))


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
                st.link_button("Kaynağı aç", context["source_url"])


st.set_page_config(
    page_title="Türk Hukuku RAG",
    layout="wide",
)

st.title("Türk Hukuku RAG Asistanı")
st.caption(
    "BGE-M3 + FAISS + BM25 hybrid retrieval. "
    "Yanıtlar yalnızca getirilen kaynaklara dayandırılır."
)

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

    with st.expander("Dosya yolları"):
        index_path = st.text_input(
            "FAISS index",
            env_path("RAG_INDEX_PATH", DEFAULT_PATHS["index"]),
        )
        metadata_path = st.text_input(
            "Metadata",
            env_path("RAG_METADATA_PATH", DEFAULT_PATHS["metadata"]),
        )
        config_path = st.text_input(
            "Index config",
            env_path("RAG_CONFIG_PATH", DEFAULT_PATHS["config"]),
        )
        corpus_path = st.text_input(
            "Retrieval corpus",
            env_path("RAG_CORPUS_PATH", DEFAULT_PATHS["corpus"]),
        )
        adapter_path = st.text_input(
            "LoRA adapter",
            env_path("RAG_ADAPTER_PATH", DEFAULT_PATHS["adapter"]),
        )

required_paths = {
    "FAISS index": Path(index_path),
    "Metadata": Path(metadata_path),
    "Index config": Path(config_path),
    "Retrieval corpus": Path(corpus_path),
}
if mode_label != "Base RAG" or adapter_path:
    required_paths["LoRA adapter"] = Path(adapter_path)

missing_paths = [f"{name}: `{path}`" for name, path in required_paths.items() if not path.exists()]
if missing_paths:
    st.error("RAG başlatılamadı. Eksik dosyalar:\n\n" + "\n\n".join(missing_paths))
    st.info(
        "Kaggle datasetindeki retrieval dosyalarını varsayılan `data/index` ve "
        "`data/processed` klasörlerine koyabilir veya kenar çubuğundan yolları değiştirebilirsin."
    )
    st.stop()

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
            llm_backend="ollama" if llm_backend_label.startswith("Ollama") else "transformers",
        )
    except Exception as exc:
        st.exception(exc)
        st.stop()

st.success("RAG hazır. Model ve index sonraki sorularda yeniden yüklenmez.")

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
        st.stop()

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
                    mode="fine_tuned" if mode_label == "Base RAG" else "base",
                    max_new_tokens=max_new_tokens,
                )
        except Exception as exc:
            st.exception(exc)
            st.stop()

    if result["mode"] == "compare":
        base_column, fine_tuned_column = st.columns(2)
        with base_column:
            st.subheader("Fine-tuned RAG")
            st.write(result["base_answer"])
        with fine_tuned_column:
            st.subheader("Base RAG")
            st.write(result["fine_tuned_answer"])
    else:
        st.subheader("Yanıt")
        st.write(result["answer"])

    render_sources(result["contexts"])
