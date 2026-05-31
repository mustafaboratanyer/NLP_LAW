# Reranker Fine-Tuning on Kaggle

Bu adımlar BERTurk tabanlı hukuk reranker modeli eğitmek içindir.

## 1. Gerekli dosyalar

Kaggle dataset içinde şu dosyalar olmalı:

```text
retrieval_corpus.json
build_reranker_dataset.py
train_reranker.py
rerank_search.py
faiss_bge_m3.index
metadata_bge_m3.json
index_config_bge_m3.json
```

## 2. Paket kurulumu

```python
!pip install -q -U sentence-transformers datasets faiss-cpu
```

## 3. Dosyaları working klasörüne kopyalama

```python
from pathlib import Path
import shutil

INPUT_DIR = Path("/kaggle/input/datasets/efealvs/turkish-legal-rag-data")
WORK_DIR = Path("/kaggle/working/legal-rag")

for directory in [
    WORK_DIR / "data/processed",
    WORK_DIR / "data/index",
    WORK_DIR / "scripts",
    WORK_DIR / "data/reranker",
    WORK_DIR / "models",
]:
    directory.mkdir(parents=True, exist_ok=True)

for name in ["retrieval_corpus.json"]:
    shutil.copy2(INPUT_DIR / name, WORK_DIR / "data/processed" / name)

for name in ["build_reranker_dataset.py", "train_reranker.py", "rerank_search.py"]:
    shutil.copy2(INPUT_DIR / name, WORK_DIR / "scripts" / name)

for name in ["faiss_bge_m3.index", "metadata_bge_m3.json", "index_config_bge_m3.json"]:
    if (INPUT_DIR / name).exists():
        shutil.copy2(INPUT_DIR / name, WORK_DIR / "data/index" / name)
```

## 4. Training dataset üretme

İlk deneme için küçük ve hızlı:

```python
!python /kaggle/working/legal-rag/scripts/build_reranker_dataset.py \
  --corpus /kaggle/working/legal-rag/data/processed/retrieval_corpus.json \
  --output-dir /kaggle/working/legal-rag/data/reranker \
  --hf-datasets Renicames/turkish-law-chatbot \
  --max-hf-rows 3000 \
  --max-qa-examples 3000 \
  --synthetic-limit 3000 \
  --synthetic-per-article 3 \
  --negatives-per-query 2 \
  --positive-repeat 2 \
  --hard-negative-pool 30 \
  --min-positive-score 5.0
```

Eğer Hugging Face indirmesi sorun çıkarırsa sadece corpus başlıklarından dataset üret:

```python
!python /kaggle/working/legal-rag/scripts/build_reranker_dataset.py \
  --corpus /kaggle/working/legal-rag/data/processed/retrieval_corpus.json \
  --output-dir /kaggle/working/legal-rag/data/reranker \
  --synthetic-limit 5000 \
  --synthetic-per-article 3 \
  --negatives-per-query 2 \
  --positive-repeat 2 \
  --hard-negative-pool 30
```

## 5. BERTurk cross-encoder fine-tuning

İlk hızlı deneme:

```python
!python /kaggle/working/legal-rag/scripts/train_reranker.py \
  --train /kaggle/working/legal-rag/data/reranker/train_pairs.jsonl \
  --dev /kaggle/working/legal-rag/data/reranker/dev_pairs.jsonl \
  --base-model dbmdz/bert-base-turkish-cased \
  --output-dir /kaggle/working/legal-rag/models/legal-berturk-reranker \
  --epochs 1 \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --max-length 512 \
  --device cuda
```

GPU memory hatası olursa:

```text
--batch-size 4
```

## 6. Fine-tuned reranker ile arama

```python
!python /kaggle/working/legal-rag/scripts/rerank_search.py \
  "birini öldürmek suç mudur?" \
  --top-k 5 \
  --index /kaggle/working/legal-rag/data/index/faiss_bge_m3.index \
  --metadata /kaggle/working/legal-rag/data/index/metadata_bge_m3.json \
  --config /kaggle/working/legal-rag/data/index/index_config_bge_m3.json \
  --articles /kaggle/working/legal-rag/data/processed/retrieval_corpus.json \
  --reranker-model /kaggle/working/legal-rag/models/legal-berturk-reranker \
  --device cuda \
  --rerank-batch-size 8 \
  --show-text
```

Bu deney raporda şu şekilde anlatılabilir:

```text
The reranker was fine-tuned as a binary relevance classifier using positive question-article pairs and BM25-mined hard negatives. The base model was dbmdz/bert-base-turkish-cased. Each query was paired with one relevant article and four hard negative articles.
```
