# Retrieval Evaluation on Gold Benchmark

Use `qa_benchmark_gold.csv` as the main benchmark because it resolves to the current retrieval corpus better than the external `gold_benchmark.json`.

Coverage check:

```bash
python /kaggle/working/legal-rag/scripts/evaluate_retrieval.py \
  --benchmark /kaggle/working/legal-rag/data/eval/qa_benchmark_gold.csv \
  --corpus /kaggle/working/legal-rag/data/processed/retrieval_corpus.json \
  --resolve-only \
  --output /kaggle/working/legal-rag/data/eval/gold_resolve.json
```

Dense FAISS baseline:

```bash
python /kaggle/working/legal-rag/scripts/evaluate_retrieval.py \
  --benchmark /kaggle/working/legal-rag/data/eval/qa_benchmark_gold.csv \
  --corpus /kaggle/working/legal-rag/data/processed/retrieval_corpus.json \
  --chunks /kaggle/working/legal-rag/data/processed/retrieval_chunks.json \
  --index /kaggle/working/legal-rag/data/index/faiss_bge_m3.index \
  --metadata /kaggle/working/legal-rag/data/index/metadata_bge_m3.json \
  --config /kaggle/working/legal-rag/data/index/index_config_bge_m3.json \
  --mode dense \
  --embedding-device cuda \
  --top-k 10 \
  --output /kaggle/working/legal-rag/data/eval/eval_dense.json
```

Hybrid retrieval:

```bash
python /kaggle/working/legal-rag/scripts/evaluate_retrieval.py \
  --benchmark /kaggle/working/legal-rag/data/eval/qa_benchmark_gold.csv \
  --corpus /kaggle/working/legal-rag/data/processed/retrieval_corpus.json \
  --chunks /kaggle/working/legal-rag/data/processed/retrieval_chunks.json \
  --index /kaggle/working/legal-rag/data/index/faiss_bge_m3.index \
  --metadata /kaggle/working/legal-rag/data/index/metadata_bge_m3.json \
  --config /kaggle/working/legal-rag/data/index/index_config_bge_m3.json \
  --mode hybrid \
  --embedding-device cuda \
  --top-k 10 \
  --output /kaggle/working/legal-rag/data/eval/eval_hybrid.json
```

Hybrid + fine-tuned reranker:

```bash
python /kaggle/working/legal-rag/scripts/evaluate_retrieval.py \
  --benchmark /kaggle/working/legal-rag/data/eval/qa_benchmark_gold.csv \
  --corpus /kaggle/working/legal-rag/data/processed/retrieval_corpus.json \
  --chunks /kaggle/working/legal-rag/data/processed/retrieval_chunks.json \
  --index /kaggle/working/legal-rag/data/index/faiss_bge_m3.index \
  --metadata /kaggle/working/legal-rag/data/index/metadata_bge_m3.json \
  --config /kaggle/working/legal-rag/data/index/index_config_bge_m3.json \
  --mode rerank \
  --reranker-model /kaggle/working/legal-rag/models/legal-berturk-reranker \
  --embedding-device cuda \
  --reranker-device cuda \
  --top-k 10 \
  --output /kaggle/working/legal-rag/data/eval/eval_rerank_v2.json
```

Hybrid + fine-tuned reranker fusion:

```bash
python /kaggle/working/legal-rag/scripts/evaluate_retrieval.py \
  --benchmark /kaggle/working/legal-rag/data/eval/qa_benchmark_gold.csv \
  --corpus /kaggle/working/legal-rag/data/processed/retrieval_corpus.json \
  --chunks /kaggle/working/legal-rag/data/processed/retrieval_chunks.json \
  --index /kaggle/working/legal-rag/data/index/faiss_bge_m3.index \
  --metadata /kaggle/working/legal-rag/data/index/metadata_bge_m3.json \
  --config /kaggle/working/legal-rag/data/index/index_config_bge_m3.json \
  --mode rerank_fusion \
  --reranker-model /kaggle/working/legal-rag/models/legal-berturk-reranker \
  --embedding-device cuda \
  --reranker-device cuda \
  --reranker-weight 0.35 \
  --hybrid-weight 0.65 \
  --top-k 10 \
  --output /kaggle/working/legal-rag/data/eval/eval_rerank_fusion_v2.json
```

The summary section of each output JSON contains:

```text
recall@5
recall@10
top1_accuracy
mrr
ndcg@10
```
