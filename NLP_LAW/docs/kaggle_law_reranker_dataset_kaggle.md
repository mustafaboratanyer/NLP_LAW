# Kaggle Law Dataset -> Reranker Training

Bu akış Kaggle `turkish_law_dataset.csv` dosyasını yüksek güvenli reranker eğitim çiftlerine çevirir.

## 1. Kaggle Dataset'e Eklenecek Dosyalar

`kaggle_upload` klasöründen şu dosyaların Kaggle dataset içinde olduğundan emin olun:

```text
build_kaggle_law_reranker.py
split_reranker_jsonl.py
train_reranker.py
rerank_search.py
evaluate_retrieval.py
turkish_law_dataset.csv
retrieval_chunks.json
retrieval_corpus.json
faiss_bge_m3.index
metadata_bge_m3.json
index_config_bge_m3.json
qa_benchmark_gold.csv
```

## 2. Notebook Setup Cell

```python
from pathlib import Path
import shutil

INPUT_DIR = Path("/kaggle/input/datasets/efealvs/turkish-legal-rag-data")
WORK_DIR = Path("/kaggle/working/legal-rag")

for path in [
    WORK_DIR / "scripts",
    WORK_DIR / "data/processed",
    WORK_DIR / "data/index",
    WORK_DIR / "data/reranker",
    WORK_DIR / "data/eval",
    WORK_DIR / "models",
]:
    path.mkdir(parents=True, exist_ok=True)

def copy_required(name, dest_dir):
    src = INPUT_DIR / name
    if not src.exists():
        raise FileNotFoundError(f"Missing in Kaggle dataset: {src}")
    dst = dest_dir / name
    shutil.copy2(src, dst)
    print(f"Copied {name} -> {dst}")

for name in [
    "build_kaggle_law_reranker.py",
    "split_reranker_jsonl.py",
    "train_reranker.py",
    "rerank_search.py",
    "evaluate_retrieval.py",
]:
    copy_required(name, WORK_DIR / "scripts")

for name in ["retrieval_chunks.json", "retrieval_corpus.json"]:
    copy_required(name, WORK_DIR / "data/processed")

for name in ["faiss_bge_m3.index", "metadata_bge_m3.json", "index_config_bge_m3.json"]:
    copy_required(name, WORK_DIR / "data/index")

copy_required("turkish_law_dataset.csv", WORK_DIR / "data/reranker")
copy_required("qa_benchmark_gold.csv", WORK_DIR / "data/eval")
```

## 3. Generate Kaggle-Based Reranker Data

```python
!python /kaggle/working/legal-rag/scripts/build_kaggle_law_reranker.py \
  --csv /kaggle/working/legal-rag/data/reranker/turkish_law_dataset.csv \
  --chunks /kaggle/working/legal-rag/data/processed/retrieval_chunks.json \
  --output /kaggle/working/legal-rag/data/reranker/kaggle_law_reranker.jsonl \
  --stats-out /kaggle/working/legal-rag/data/reranker/kaggle_law_reranker.stats.json \
  --audit-out /kaggle/working/legal-rag/data/reranker/kaggle_law_reranker.audit.csv \
  --min-score 9 \
  --min-context-overlap 0.55 \
  --min-answer-overlap 0.05 \
  --negatives-per-query 3 \
  --hard-negative-pool 80
```

Kontrol:

```python
!cat /kaggle/working/legal-rag/data/reranker/kaggle_law_reranker.stats.json
!head -n 3 /kaggle/working/legal-rag/data/reranker/kaggle_law_reranker.jsonl
```

## 4. Train/Dev Split

```python
!python /kaggle/working/legal-rag/scripts/split_reranker_jsonl.py \
  --input /kaggle/working/legal-rag/data/reranker/kaggle_law_reranker.jsonl \
  --train-out /kaggle/working/legal-rag/data/reranker/train_kaggle_law.jsonl \
  --dev-out /kaggle/working/legal-rag/data/reranker/dev_kaggle_law.jsonl \
  --dev-ratio 0.1 \
  --seed 493
```

## 5. Fine-Tune Reranker

```python
!rm -rf /kaggle/working/legal-rag/models/legal-berturk-reranker-kaggle-law

!CUDA_VISIBLE_DEVICES=0 python /kaggle/working/legal-rag/scripts/train_reranker.py \
  --train /kaggle/working/legal-rag/data/reranker/train_kaggle_law.jsonl \
  --dev /kaggle/working/legal-rag/data/reranker/dev_kaggle_law.jsonl \
  --base-model dbmdz/bert-base-turkish-cased \
  --output-dir /kaggle/working/legal-rag/models/legal-berturk-reranker-kaggle-law \
  --epochs 2 \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --max-length 512 \
  --device cuda \
  --evaluation-steps 200 \
  --save-best-model
```

## 6. Gold Benchmark Eval

```python
!python /kaggle/working/legal-rag/scripts/evaluate_retrieval.py \
  --benchmark /kaggle/working/legal-rag/data/eval/qa_benchmark_gold.csv \
  --corpus /kaggle/working/legal-rag/data/processed/retrieval_corpus.json \
  --index /kaggle/working/legal-rag/data/index/faiss_bge_m3.index \
  --metadata /kaggle/working/legal-rag/data/index/metadata_bge_m3.json \
  --config /kaggle/working/legal-rag/data/index/index_config_bge_m3.json \
  --mode rerank_fusion \
  --reranker-model /kaggle/working/legal-rag/models/legal-berturk-reranker-kaggle-law \
  --embedding-device cuda \
  --reranker-device cuda \
  --reranker-weight 0.35 \
  --top-k 10 \
  --output /kaggle/working/legal-rag/data/eval/eval_rerank_fusion_kaggle_law.json
```
