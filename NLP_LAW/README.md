# Turkish Legal RAG

This repository contains the code and notebooks for the CENG493 term project:
**Improving Turkish Legal Question Answering with an Optimized RAG Pipeline**.

The system answers Turkish legal questions using a Retrieval-Augmented Generation
pipeline:

```text
Question -> Dense Retrieval -> BM25 Hybrid Retrieval -> LLM -> Grounded Answer
```

The final default retrieval setup is **BGE-M3 dense retrieval + BM25 hybrid
retrieval**. Query expansion is disabled by default and is only available as an
optional debug/ablation flag.

## Repository Structure

```text
scripts/      Core preprocessing, indexing, retrieval, evaluation, and RAG scripts
notebooks/    Kaggle notebooks for demo and experiments
docs/         Technical notes and report draft
configs/      Small reproducibility configs
examples/     Tiny custom-data examples for instructor testing
results/      Small metric summaries
```

Large data/model artifacts are intentionally not committed to GitHub. They should
be supplied through the Kaggle dataset or Hugging Face:

- `retrieval_corpus.json`
- `retrieval_chunks.json`
- `faiss_bge_m3.index`
- `metadata_bge_m3.json`
- `index_config_bge_m3.json`
- LoRA adapter files such as `adapter_model.safetensors`

## Main Notebooks

- `notebooks/rag_demo_kaggle.ipynb`: loads the final RAG pipeline on Kaggle and
  runs Turkish legal QA.
- `notebooks/custom_data_rag_kaggle.ipynb`: builds a new corpus/index from custom
  documents provided by the evaluator.
- `notebooks/retrieval_grid_search_kaggle.ipynb`: evaluates dense/BM25 hybrid
  retrieval configurations.
- `notebooks/reranker_finetune_kaggle_law.ipynb`: reranker fine-tuning experiment.
- `notebooks/embedding_finetune_kaggle.ipynb`: embedding fine-tuning experiment.

## Final Retrieval Configuration

The best benchmark retrieval configuration is:

```text
embedding_model = BAAI/bge-m3
retrieval = dense FAISS + BM25 hybrid
alpha = 0.70
dense_candidates = 300
bm25_candidates = 100
preliminary_top_k = 50
query_expansion = false
```

## Evaluation

Retrieval was evaluated using `qa_benchmark_gold.csv`. From 290 rows, 244 examples
were resolved to the current corpus.

Best retrieval result:

```text
Recall@5   = 0.6311
Recall@10  = 0.6762
Top-1 Acc. = 0.4795
MRR        = 0.5456
nDCG@10    = 0.5772
```

End-to-end RAG evaluation should be run on the same gold benchmark for both:

- Base RAG: base Qwen model + same retrieval
- Fine-tuned RAG: LoRA Qwen model + same retrieval

This comparison is run from the Kaggle demo/evaluation notebook.

## Custom Document Evaluation

The instructor can provide a custom document collection using one of these forms:

- `custom_documents.json`
- `custom_documents.jsonl`
- `custom_documents.csv`
- a `custom_docs/` directory containing `.txt`, `.md`, or `.pdf` files

Use:

```text
notebooks/custom_data_rag_kaggle.ipynb
```

The notebook converts the custom collection into a retrieval corpus, builds a
new FAISS index, and runs the same RAG pipeline on the supplied documents.

## Typical Kaggle Workflow

1. Add the project Kaggle dataset containing the retrieval corpus and FAISS index files. The LoRA adapter is included in this repository under `models/`.
2. Open `notebooks/rag_demo_kaggle.ipynb`.
3. Run setup/copy cells.
4. Load retriever and LLM.
5. Run the gold benchmark evaluation cell.
6. Repeat with `adapter_path=None` for Base RAG.

## Notes

This project is for academic research and is not legal advice. The system should
only answer from retrieved sources and should cite the relevant law/article when
possible.

## Included Fine-Tuned Model

The repository includes the LoRA adapter used for the fine-tuned RAG system:

```text
models/qwen_7b_lora_v2/final_600/
```

The adapter was trained on top of:

```text
Qwen/Qwen2.5-7B-Instruct
```

During inference, the base model is downloaded from Hugging Face and the local
LoRA adapter is loaded with PEFT. For Base RAG evaluation, use the same retriever
and set `adapter_path=None`. For Fine-tuned RAG evaluation, set:

```python
adapter_path = "models/qwen_7b_lora_v2/final_600"
```


