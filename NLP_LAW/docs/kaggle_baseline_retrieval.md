# Kaggle Baseline Retrieval

Bu adımlar FAISS index'i Kaggle üzerinde üretmek ve hızlı retrieval testi yapmak içindir.

## 1. Kaggle Dataset Olarak Yüklenecek Dosyalar

Aşağıdaki dosyaları/klsörleri Kaggle'a bir dataset olarak yükleyin:

```text
data/processed/retrieval_chunks.json
data/processed/retrieval_corpus.json
scripts/build_faiss_index.py
scripts/search_faiss.py
scripts/rerank_search.py
```

Dataset adını örnek olarak `turkish-legal-rag-data` varsayıyorum. Kaggle path'i genelde şöyle olur:

```text
/kaggle/input/turkish-legal-rag-data/
```

Eğer `turkish_legal_rag_baseline.zip` dosyasını yüklediyseniz önce şunu çalıştırın:

```python
!unzip -q /kaggle/input/turkish-legal-rag-data/turkish_legal_rag_baseline.zip -d /kaggle/working/uploaded_data
```

Bu durumda aşağıdaki kopyalama komutlarında `/kaggle/input/turkish-legal-rag-data/` yerine
`/kaggle/working/uploaded_data/` kullanın.

## 2. Notebook Ayarları

Kaggle notebook'ta:

```text
Accelerator: GPU T4
Internet: On
```

## 3. Paket Kurulumu

```python
!pip install -q sentence-transformers faiss-cpu
```

## 4. Dosyaları Working Directory'ye Kopyalama

```python
!mkdir -p /kaggle/working/data/processed
!mkdir -p /kaggle/working/scripts

!cp /kaggle/input/turkish-legal-rag-data/data/processed/retrieval_chunks.json /kaggle/working/data/processed/
!cp /kaggle/input/turkish-legal-rag-data/data/processed/retrieval_corpus.json /kaggle/working/data/processed/
!cp /kaggle/input/turkish-legal-rag-data/scripts/build_faiss_index.py /kaggle/working/scripts/
!cp /kaggle/input/turkish-legal-rag-data/scripts/search_faiss.py /kaggle/working/scripts/
!cp /kaggle/input/turkish-legal-rag-data/scripts/rerank_search.py /kaggle/working/scripts/
```

Dataset path farklıysa `turkish-legal-rag-data` kısmını Kaggle'da görünen dataset klasör adıyla değiştirin.

## 5. FAISS Index Üretme

Baseline için hafif model:

```python
!python /kaggle/working/scripts/build_faiss_index.py \
  --chunks /kaggle/working/data/processed/retrieval_chunks.json \
  --index-out /kaggle/working/data/index/faiss.index \
  --metadata-out /kaggle/working/data/index/metadata.json \
  --config-out /kaggle/working/data/index/index_config.json \
  --model intfloat/multilingual-e5-small \
  --device cuda \
  --batch-size 64
```

Daha güçlü ama daha yavaş model:

```python
!python /kaggle/working/scripts/build_faiss_index.py \
  --chunks /kaggle/working/data/processed/retrieval_chunks.json \
  --index-out /kaggle/working/data/index/faiss_e5_base.index \
  --metadata-out /kaggle/working/data/index/metadata_e5_base.json \
  --config-out /kaggle/working/data/index/index_config_e5_base.json \
  --model intfloat/multilingual-e5-base \
  --device cuda \
  --batch-size 32
```

## 6. Retrieval Testi

```python
!python /kaggle/working/scripts/search_faiss.py \
  "İşçinin yıllık ücretli izin hakkı nasıl düzenlenir?" \
  --top-k 5 \
  --index /kaggle/working/data/index/faiss.index \
  --metadata /kaggle/working/data/index/metadata.json \
  --config /kaggle/working/data/index/index_config.json \
  --articles /kaggle/working/data/processed/retrieval_corpus.json \
  --device cuda \
  --show-text
```

Başka test soruları:

```python
!python /kaggle/working/scripts/search_faiss.py "Kişisel verilerin yurt dışına aktarılması hangi şartlara bağlıdır?" --top-k 5 --index /kaggle/working/data/index/faiss.index --metadata /kaggle/working/data/index/metadata.json --config /kaggle/working/data/index/index_config.json --articles /kaggle/working/data/processed/retrieval_corpus.json --device cuda

!python /kaggle/working/scripts/search_faiss.py "Kamulaştırma bedeli mahkeme tarafından nasıl tespit edilir?" --top-k 5 --index /kaggle/working/data/index/faiss.index --metadata /kaggle/working/data/index/metadata.json --config /kaggle/working/data/index/index_config.json --articles /kaggle/working/data/processed/retrieval_corpus.json --device cuda

!python /kaggle/working/scripts/search_faiss.py "Avukatlık mesleğine kabul şartları nelerdir?" --top-k 5 --index /kaggle/working/data/index/faiss.index --metadata /kaggle/working/data/index/metadata.json --config /kaggle/working/data/index/index_config.json --articles /kaggle/working/data/processed/retrieval_corpus.json --device cuda
```

## 7. Kaggle Output Olarak Saklanacak Dosyalar

Notebook çalışınca şu dosyaları output olarak indirebilirsiniz:

```text
/kaggle/working/data/index/faiss.index
/kaggle/working/data/index/metadata.json
/kaggle/working/data/index/index_config.json
```

Bu üç dosya baseline dense retrieval index'idir.

## 8. Hybrid + Reranker Testi

Dense retrieval bazen doğal sorularda doğru maddeyi ilk sıraya koyamayabilir. Bu yüzden final sistemde
daha güçlü deneme olarak önce FAISS + BM25 ile aday maddeler bulunur, sonra cross-encoder reranker
bu adayları tekrar sıralar.

Var olan FAISS index'i tekrar üretmeye gerek yoktur. Aşağıdaki komut mevcut index dosyalarıyla çalışır:

```python
!python /kaggle/working/scripts/rerank_search.py \
  "birini öldürmek suç mudur?" \
  --top-k 5 \
  --index /kaggle/working/data/index/faiss_bge_m3.index \
  --metadata /kaggle/working/data/index/metadata_bge_m3.json \
  --config /kaggle/working/data/index/index_config_bge_m3.json \
  --articles /kaggle/working/data/processed/retrieval_corpus.json \
  --device cuda \
  --rerank-batch-size 4 \
  --show-text
```

Eğer GPU memory hatası verirse batch size düşürün:

```python
!python /kaggle/working/scripts/rerank_search.py \
  "işçi 2 gün işe gelmezse ne olur?" \
  --top-k 5 \
  --index /kaggle/working/data/index/faiss_bge_m3.index \
  --metadata /kaggle/working/data/index/metadata_bge_m3.json \
  --config /kaggle/working/data/index/index_config_bge_m3.json \
  --articles /kaggle/working/data/processed/retrieval_corpus.json \
  --device cuda \
  --rerank-batch-size 1 \
  --preliminary-top-k 40 \
  --show-text
```

Eğer doğru madde adaylar arasına girmiyorsa aday sayısını artırın:

```python
--dense-candidates 300 --bm25-candidates 500 --preliminary-top-k 120
```
