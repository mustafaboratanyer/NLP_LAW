# 🚀 1 KİŞİ PROJE YAPILANDI - CENG493 Turkish Legal QA RAG

> **Başlama Tarihi:** May 7, 2026  
> **Önemli:** Teslim tarihini 📍 belirle - zamanlamayı buna göre ayarla

---

## 📊 GENEL ZAMAN ÇIZELGESI

| Faza | Hafta | Saat/Hafta | Açıklama |
|------|-------|-----------|---------|
| **FAZE 1** | 1-2 | 20-25 | Data Hazırlığı + Altyapı |
| **FAZE 2** | 2-3 | 25-30 | Baseline Retrieval |
| **FAZE 3** | 3-4 | 15-20 | Test Benchmark |
| **FAZE 4a** | 4-5 | 20-25 | Embedding Fine-tuning |
| **FAZE 4b** | ⏩ SKIP | - | Reranker (Zaman yok) |
| **FAZE 4c** | 5-6 | 30-40 | LLM Fine-tuning |
| **FAZE 5** | 6-7 | 20-25 | Evaluation + Report |
| | | | |
| **TOPLAM** | **6-7 hafta** | **150-165** | Baştan sona |

---

## 📋 FAZE 1: DATA HAZIRLIĞI + ALTYAPISI (Hafta 1-2)

**Hedef:** Temiz Turkish legal dataset + geliştirme ortamı

### Adımlar:
- [ ] **1.1** GitHub repo oluştur (public - teslim için)
- [ ] **1.2** Python venv kur: `python -m venv venv`
- [ ] **1.3** requirements.txt + setup.py yazı (BERT, LLM lib'ler, etc)
- [ ] **1.4** Proje folder structure:
  ```
  project/
  ├── data/
  │   ├── raw/          (indirilen datasets)
  │   ├── processed/    (temizlenmiş dokümanlar)
  │   └── benchmarks/   (test soruları)
  ├── src/
  │   ├── retrieval.py
  │   ├── embedding.py
  │   ├── reranker.py
  │   └── llm.py
  ├── notebooks/
  │   ├── 01_data_exploration.ipynb
  │   ├── 02_retrieval_baseline.ipynb
  │   └── ...
  ├── models/           (fine-tuned weights)
  ├── configs/          (hyperparameters)
  └── results/          (experiments output)
  ```

### Veri Kaynakları:
- **Kaggle:** https://www.kaggle.com/datasets/batuhankalem/turkishlaw-dataset-for-llm-finetuning
- **HuggingFace:** https://huggingface.co/datasets/Renicames/turkish-lawchatbot

### Veri İşleme:
- [ ] **1.5** Datasetleri indir + explore (boyut, format, örnek dokümanlar)
- [ ] **1.6** Dokümanları chunk et (madde/paragraf bazında, max 512 token)
- [ ] **1.7** Tekil dokümanları `documents.json`'a yaz:
  ```json
  [
    {
      "id": "doc_001",
      "text": "Madde 1: Lorem ipsum...",
      "source": "Turkish_Civil_Law_Article_123"
    }
  ]
  ```
- [ ] **1.8** EDA notebook: dokuman sayısı, token distribution, dil özelliği analiz

### Çıktı:
- ✅ `data/processed/documents.json` (clean Turkish legal docs)
- ✅ `notebooks/01_data_exploration.ipynb` (analiz)
- ✅ Development ortamı hazır

**⏱️ Tahmini Süre:** 10-15 saat

---

## 🔍 FAZE 2: BASELINE RETRIEVAL (Hafta 2-3)

**Hedef:** Dense + BM25 hybrid retrieval sistemi çalışan durumda

### 2a. Dense Retrieval Setup:
- [ ] **2.1** Embedding model seç: `sentence-transformers/LaBSE` (multilingual Turkish)
- [ ] **2.2** FAISS vector database kur:
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer('LaBSE')
  embeddings = model.encode(documents)
  # FAISS'a yükle
  ```
- [ ] **2.3** Query embedding + retrieval test et
- [ ] **2.4** Baseline metrics hesapla: Recall@5, Recall@10, MRR
  - Test queries: ilk 20-30 soruyu manual yazı
  - Relevant documents: manuel annotation

### 2b. BM25 Retrieval:
- [ ] **2.5** BM25 index kur (Elasticsearch veya `rank-bm25` Python lib)
- [ ] **2.6** Hybrid retrieval: dense results + BM25 results → ensemble
  ```python
  # Pseudo-code
  dense_results = dense_search(query, top_20)
  bm25_results = bm25_search(query, top_20)
  hybrid = combine(dense_results, bm25_results, alpha=0.5)
  ```

### 2c. Baseline Evaluation:
- [ ] **2.7** 30-50 test soruda metrics hesapla
- [ ] **2.8** Sonuçlar: `results/baseline_retrieval.json`
  ```json
  {
    "Recall@5": 0.45,
    "Recall@10": 0.62,
    "MRR": 0.38,
    "config": "dense+BM25"
  }
  ```

### Çıktı:
- ✅ `src/retrieval.py` (dense + BM25 functions)
- ✅ `notebooks/02_retrieval_baseline.ipynb` (experiments)
- ✅ FAISS index + BM25 index
- ✅ `results/baseline_retrieval.json` (metrics)

**⏱️ Tahmini Süre:** 15-20 saat

---

## ✅ FAZE 3: TEST BENCHMARK + GOLD GROUNDING (Hafta 3-4)

**Hedef:** 150-300 Turkish legal soru + **golden answers** + **gold documents** (Rubrik: Gold Q+A+Doc)

⚠️ **ÖNEMLİ:** Her soru için golden answer ve supporting documents ZORUNLU!

### Adımlar:

#### 3.1 Questions Topla:
- [ ] **3.1a** Mevcut `turkish-lawchatbot` dataset'ten 150+ question çek
- [ ] **3.1b** Duplicate'leri çıkar
- [ ] **3.1c** Turkish hukuk konularını cover ettiğini kontrol et

#### 3.2 Her Soru için Golden Answer Yazı:
- [ ] **3.2a** Soru örneği:
  ```
  Q: "Türkiye'de evlilik yaşı kaç?"
  ```

- [ ] **3.2b** Golden answer yazı (30-50 kelime, kaynak referansı ile):
  ```
  A: "Türk Medeni Kanunu'nun 124. maddesine göre, 
  evlilik yaşı 18'dir. İstisnaen, mahkeme izniyle 
  16 yaşından itibaren evlilik yapılabilir.
  (Ref: TMK madde 124-125)"
  ```

- [ ] **3.2c** Manual verify: Cevap doğru mu? Citation doğru mu?

#### 3.3 Her Answer için Gold Documents Seç:
- [ ] **3.3a** Golden answer'da bahsedilen madde/bölümleri `documents.json`'dan bul
- [ ] **3.3b** İlgili dokuman ID'lerini kaydet (2-4 dokuman typical)
  ```json
  "gold_documents": ["doc_124", "doc_125"]  
  // Madde 124 (evlilik yaşı) ve Madde 125 (istisna)
  ```

- [ ] **3.3c** Kontrol: Her gold doc, gerçekten answer'da bahsedilen bilgi içeriyor mu?

#### 3.4 Test Set JSON Format (ÖNEMLİ!):
```json
{
  "question_id": "q_001",
  "question": "Türkiye'de evlilik yaşı kaç?",
  
  "gold_answer": "Türk Medeni Kanunu'nun 124. maddesine göre, 
                   evlilik yaşı 18'dir. İstisnaen, mahkeme izniyle 
                   16 yaşından itibaren evlilik yapılabilir.",
  
  "gold_documents": [
    "doc_124",
    "doc_125"
  ],
  
  "category": "Civil Law",
  "difficulty": "easy"
}
```

- [ ] **3.5** Tüm 150-300 soruda bunu yap
- [ ] **3.6** `data/benchmarks/test_set.json` save et (single file)

### Kalite Kontrol:
- [ ] **3.7** Spot check: ilk 20 Q-A-Doc çifti manuel review
  - Golden answer, soruya cevap veriyor mu?
  - Gold documents, answer'ı support ediyor mu?
  - Citation accuracy %95+ mi?

### Çıktı:
- ✅ `data/benchmarks/test_set.json` (150-300 Q-A-Doc triplets)
- ✅ Her sorunun: question + golden_answer + gold_documents listesi
- ✅ Quality report: spot check sonuçları

**⏱️ Tahmini Süre:** 15-20 saat (çok önemli, acele etme!)

---

## ⚙️ FAZE 4a: EMBEDDING FINE-TUNING (Hafta 4-5)

**Hedef:** Domain-adapted Turkish legal embedding model

### 4a.1 Training Data Hazırlığı:
- [ ] **4a.1** `documents.json`'dan positive pairs oluştur:
  ```json
  {
    "anchor": "Madde 1: Lorem ipsum",
    "positive": "Madde 1 ile ilişkili başka bir paragraf"
  }
  ```
- [ ] **4a.2** Hard negatives mining:
  ```python
  # Basit yöntem: random sampling + BM25 distance
  anchor = doc[i]
  positive = doc_related_to_anchor
  hard_negative = random_doc_with_low_similarity  # BM25'e göre
  ```
- [ ] **4a.3** Training pairs dataset: `data/training_pairs.json` (1000-5000 pairs)

### 4a.2 Fine-Tuning:
- [ ] **4a.4** Sentence-transformers contrastive learning:
  ```python
  from sentence_transformers import SentenceTransformer, InputExample, losses
  from torch.utils.data import DataLoader
  
  model = SentenceTransformer('sentence-transformers/LaBSE')
  train_examples = [InputExample(texts=[anchor, positive, hard_neg])]
  train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
  train_loss = losses.TripletLoss(model=model)
  model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2)
  ```
- [ ] **4a.5** Model save: `models/finetuned_embedding_v1.pt`

### 4a.3 Evaluation:
- [ ] **4a.6** Fine-tuned embedding baseline ile karşılaştır:
  ```
  Baseline:  Recall@5=0.45, Recall@10=0.62, MRR=0.38
  Fine-tuned: Recall@5=0.52, Recall@10=0.68, MRR=0.44  ← +7% improvement
  ```
- [ ] **4a.7** Sonuçlar: `results/embedding_finetuning.json`

### Çıktı:
- ✅ `data/training_pairs.json` (contrastive pairs)
- ✅ `models/finetuned_embedding_v1.pt`
- ✅ `results/embedding_finetuning.json` (ablation results)
- ✅ `notebooks/03_embedding_tuning.ipynb`

**⏱️ Tahmini Süre:** 15-20 saat

---

## ⏩ FAZE 4b: RERANKER FINE-TUNING (SKIP - Zaman Yok!)

⚠️ **Bu fazı ATLA başta.** Zamanın varsa sonra ekle.

Eğer eklemek istersen:
- Cross-encoder model train et (30-40 saat)
- Top-20 retrieval → top-5 yeniden sırala
- Örnek: `cross-encoders/mmarcotr-multilingual-v1` pre-trained + fine-tune

**Skip Alternatifi:** Pre-trained cross-encoder kullan (fine-tuning yapma)

---

## 🤖 FAZE 4c: LLM FINE-TUNING (Hafta 5-6)

**Hedef:** Turkish legal question answering için optimize LLM

### 4c.1 LLM Seçim:
- [ ] **4c.1** Model seç: **Llama2-Turkish 7B** veya **Mistral7B**
  - GPU memory: ~15GB (7B model + training overhead)
  - Inference: ~5-10 saniye/question
- [ ] **4c.2** Model download:
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained('metu-nlp/turkish-llama-7b')
  tokenizer = AutoTokenizer.from_pretrained('metu-nlp/turkish-llama-7b')
  ```

### 4c.2 Training Data Hazırlığı:
- [ ] **4c.3** Q-A pairs → fine-tuning format:
  ```json
  {
    "instruction": "Şu soruyu Türkçe hukuk bağlamında cevapla:",
    "input": "Evlilik yaşı kaç?",
    "context": "Madde 124: Evlilik on sekiz yaşını tamamlamış kişilerin..."
    "output": "Türk Medeni Kanunu'na göre evlilik yaşı 18'dir..."
  }
  ```
- [ ] **4c.4** Training set: `data/llm_training.json` (500-1000 examples)

### 4c.3 LoRA Fine-Tuning:
- [ ] **4c.5** LoRA config (low-rank adaptation - çok az memory):
  ```python
  from peft import LoraConfig, get_peft_model
  
  lora_config = LoraConfig(
      r=16,
      lora_alpha=32,
      target_modules=["q_proj", "v_proj"],
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM"
  )
  model = get_peft_model(model, lora_config)
  ```
- [ ] **4c.6** Training loop (HuggingFace Trainer):
  ```python
  from transformers import Trainer, TrainingArguments
  
  args = TrainingArguments(
      output_dir='models/llm_finetuned',
      num_train_epochs=2,
      per_device_train_batch_size=4,
      learning_rate=3e-4,
  )
  trainer = Trainer(model=model, args=args, train_dataset=train_data)
  trainer.train()
  ```
- [ ] **4c.7** Model save: `models/llm_finetuned_lora_v1.pt`

### 4c.4 RAG Pipeline Integration:
- [ ] **4c.8** Full pipeline:
  ```
  Question
    ↓
  [Dense + BM25 Retrieval] → Top-10 docs
    ↓
  [Reranker - SKIP bu seferlik]
    ↓
  [Top-5 docs] → format into context
    ↓
  [LLM Prompt]
    Prompt: "Konteks:\n{docs}\n\nSoru: {question}\n\nCevap:"
    ↓
  [Fine-tuned LLM generates answer]
    ↓
  Answer + Citations
  ```
- [ ] **4c.9** Citation tracking: cevap hangi dokümanlardan geldiğini kaydet

### 4c.5 Evaluation:
- [ ] **4c.10** 50-100 test soruda QA metrics:
  - **F1 Score** (token-level overlap)
  - **BLEU** (n-gram similarity)
  - **ROUGE** (recall-oriented understudy for gisting evaluation)
  - **Faithfulness:** Citation doğruluk (manuel check)

- [ ] **4c.11** Sonuçlar: `results/llm_finetuning.json`

### Çıktı:
- ✅ `data/llm_training.json` (Q-A-Context triplets)
- ✅ `models/llm_finetuned_lora_v1.pt` (LoRA weights)
- ✅ `src/llm.py` (LLM + RAG pipeline)
- ✅ `notebooks/04_llm_finetuning.ipynb`
- ✅ `results/llm_finetuning.json` (metrics)

**⏱️ Tahmini Süre:** 25-35 saat

---

## 📊 FAZE 5: EVALUATION + REPORT (Hafta 6-7)

**Hedef:** Tüm ablations test + technical report + presentation

### 🎯 Scoring Rubric (Hocanın istediği):

**Final Score = (0.35 × R) + (0.4 × A) + (0.25 × G)**

Burada:
- **R (Retrieval Score):** Doğru dokumentları bulabildin mi?
  - Metric: Recall@5, Recall@10, MRR
  - Gold documents içinde kaçını buldun?
  
- **A (Answer Quality):** Cevap doğru mu?
  - Metric: Exact Match (EM), F1 Score, BLEU/ROUGE
  - Generated answer ≈ golden_answer?
  
- **G (Grounding/Faithfulness):** Cevap gold docs'a dayalı mı?
  - Metric: Citation Accuracy
  - Generated answer'daki claims, gold docs'ta var mı?
  - Hallucination yokmu?

⚠️ **KEY:** Eğer retrieval yanlışsa, A ve G penalize edilir!

### 5.1 Ablation Experiments:
Test 5 configuration (her biri ~150 test query):

| Config | Embedding | Retrieval | Reranker | LLM | R | A | G | Final |
|--------|-----------|-----------|----------|-----|---|---|---|-------|
| **V1** | Baseline | Dense | ❌ | Vanilla | ? | ? | ? | 0.35R+0.4A+0.25G |
| **V2** | Baseline | Dense+BM25 | ❌ | Vanilla | ? | ? | ? | 0.35R+0.4A+0.25G |
| **V3** | Fine-tuned | Dense+BM25 | ❌ | Vanilla | ? | ? | ? | 0.35R+0.4A+0.25G |
| **V4** | Fine-tuned | Dense+BM25 | ❌ | Fine-tuned | ? | ? | ? | 0.35R+0.4A+0.25G |
| **V5** | Fine-tuned | Dense+BM25 | Pre-trained | Fine-tuned | ? | ? | ? | 0.35R+0.4A+0.25G |

- [ ] **5.1** Tüm configlerde run et
- [ ] **5.2** Metrics table'ını organize et

### 5.2 Metric Calculation:

#### 5.2a Retrieval Metrics (R):
```python
# Her test query için:
for each question in test_set:
    retrieved_docs = system_retrieves(question)  # Top-10
    gold_docs = test_set[question].gold_documents
    
    # Recall@5, @10
    recall_5 = len(set(retrieved_docs[:5]) & set(gold_docs)) / len(gold_docs)
    recall_10 = len(set(retrieved_docs[:10]) & set(gold_docs)) / len(gold_docs)
    
    # MRR (Mean Reciprocal Rank)
    mrr = 1 / rank_of_first_gold_doc  # eğer bulunmadıysa 0

# Average all queries
R_score = avg(recall_5, recall_10, mrr)
```

#### 5.2b Answer Quality Metrics (A):
```python
# Her test query için:
for each question in test_set:
    generated_answer = system_generates(question)
    gold_answer = test_set[question].gold_answer
    
    # EM (Exact Match)
    em = 1 if normalize(generated) == normalize(gold) else 0
    
    # F1 Score (token overlap)
    f1 = compute_f1(generated_tokens, gold_tokens)
    
    # BLEU / ROUGE
    bleu = compute_bleu(generated, gold)
    rouge = compute_rouge(generated, gold)

# Average all queries
A_score = avg(em, f1, bleu, rouge)
```

#### 5.2c Grounding/Faithfulness Metrics (G):
```python
# Her test query için MANUEL REVIEW:
for each question in test_set:
    generated_answer = system_generates(question)
    gold_docs = test_set[question].gold_documents
    
    # Check each claim in generated_answer
    claims = extract_claims(generated_answer)
    
    for claim in claims:
        found_in_docs = search_claim_in_documents(claim, gold_docs)
        if not found_in_docs:
            hallucination_count += 1
    
    # Citation accuracy
    cited_docs = extract_citations(generated_answer)
    correct_citations = len(set(cited_docs) & set(gold_docs)) / len(cited_docs)

# Grounding score
G_score = 1 - (hallucination_count / total_claims) 
G_score *= correct_citation_rate
```

- [ ] **5.2** Python script yaz (evaluation.py)
- [ ] **5.3** Tüm 5 config'de run et
- [ ] **5.4** Results table oluştur

### 5.3 Hallucination Analysis:
- [ ] **5.5** LLM çıktılarından 50-100 answer manuel review
- [ ] **5.6** Hallucination examples identify et:
  - "Sistem doğru olmayan şey söyledi mi?"
  - "Söylediği bilgi, gold docs'ta var mı?"
  - "Citation doğru mu?"
- [ ] **5.7** Hallucination sayısı + oranı kaydet
  ```
  Hallucination Rate: X% (Y hallucinations out of Z answers)
  ```

### 5.4 Error Analysis:
- [ ] **5.8** Başarısız queryler analiz et (R < 0.3 veya A < 0.4 olanlar)
- [ ] **5.9** Failure patterns: categorize et
  ```
  Pattern 1 - NOT RETRIEVED (Retrieval fail):
    - Soruda bahsedilen madde/konu vector DB'de bulunmadı
    - Example: [soru], [expected docs], [retrieved docs]
    - Fix: Better chunking? Better embedding?
  
  Pattern 2 - WRONG RANKING (Retrieval rank yanlış):
    - Gold doc var ama top-10'da değil
    - Example: [soru], [gold doc rank=15]
    - Fix: Reranker yardım eder
  
  Pattern 3 - HALLUCINATION (Grounding fail):
    - LLM doğru olmayan şey söyledi
    - Example: [generated]: "...", [gold docs]: "..."
    - Fix: Better prompting? Better grounding?
  
  Pattern 4 - CITATION ERROR (Citation accuracy fail):
    - Cevap yanlış doc'u reference etti
    - Example: [claim]: "...", [cited doc]: "..."
    - Fix: Citation tracking system gerekli
  
  Pattern 5 - ANSWER MISMATCH (Answer quality fail):
    - Cevap doğru ama farklı şekilde yazıldı
    - Example: [generated]: "...", [gold]: "..."
    - Not critical - F1 Score yardım eder
  ```
- [ ] **5.10** Pattern distribution: pie chart yap
  ```
  Total errors: 250
  - Not retrieved: 50 (20%)
  - Wrong ranking: 40 (16%)
  - Hallucination: 80 (32%) ← En büyük sorun!
  - Citation error: 50 (20%)
  - Answer mismatch: 30 (12%)
  ```

### 5.5 Technical Report (10-15 sayfa):
- [ ] **5.11** Yazı:
  1. **Abstract** (1 s): Sistem özeti + final scores (R, A, G, Final)
  2. **Introduction** (2 s): 
     - Problem tanımı
     - RAG'ın Turkish legal context'te neden önemli
  3. **Related Work** (1.5 s): RAG, Turkish NLP, LLM literature
  4. **Methodology** (3 s):
     - Data preparation (documents + gold test set)
     - Retrieval system (dense + BM25)
     - Embedding fine-tuning details
     - LLM fine-tuning details
  5. **Experiments & Results** (3 s):
     - Ablation results table (V1-V5 scores)
     - R vs A vs G analysis
     - Which component helped most?
  6. **Hallucination & Error Analysis** (2 s):
     - Hallucination examples
     - Error pattern breakdown (pie chart)
     - Top 5 failure cases
  7. **Conclusion & Future Work** (1 s):
     - Best configuration: V4 vs V5?
     - Lessons learned
     - Next steps

- [ ] **5.12** Diagrams/Charts:
  - Pipeline architecture diagram
  - Ablation results bar chart (V1-V5 final scores)
  - R vs A vs G breakdown (stacked bar)
  - Error distribution pie chart
  - Hallucination examples table

### 5.6 Presentation (15 dakika):
- [ ] **5.13** Slides oluştur (~15-20 slide):
  - Slide 1: Title + team
  - Slide 2-3: Problem definition + rubric (Gold Q+A+Doc)
  - Slide 4-5: System architecture + pipeline
  - Slide 6-8: **Results** (Ablation table V1-V5 with R, A, G, Final)
  - Slide 9: **Which component helped most?** (R improvement, A improvement, G improvement)
  - Slide 10-11: **Hallucination Analysis** (examples + rate)
  - Slide 12: **Error Patterns** (pie chart)
  - Slide 13: **Best Configuration** (V4 vs V5, why?)
  - Slide 14: **Limitations & Future Work**
  - Slide 15: Q&A

### 5.7 Live Demo:
- [ ] **5.14** Interactive demo (Gradio):
  ```python
  import gradio as gr
  
  def answer_question_with_grounding(question):
      # Retrieval
      retrieved_docs = retrieve(question)  # Show these!
      
      # LLM answer
      answer, citations = generate_answer_with_citations(question, retrieved_docs)
      
      # Format output
      output = f"""
      SORU: {question}
      
      BULUNAN DOKÜMANLAR:
      {format_docs(retrieved_docs)}
      
      CEVAP:
      {answer}
      
      KAYNAKLARı:
      {citations}
      """
      return output
  
  gr.Interface(
      fn=answer_question_with_grounding,
      inputs="text",
      outputs="text",
      title="Turkish Legal QA System"
  ).launch()
  ```

### 5.8 Code Cleanup + GitHub:
- [ ] **5.15** GitHub'a final push
  - README.md:
    ```
    # Turkish Legal QA with RAG
    
    ## How to run
    1. pip install -r requirements.txt
    2. Download data from ../data/
    3. python src/main.py --config config_v4.json
    4. gradio demo.py
    
    ## Results
    Best Config (V4): R=0.62, A=0.58, G=0.71, Final=0.63
    ```
  - requirements.txt
  - All notebooks cleaned + documented
  - src/ folder with final code
  - configs/ folder with all hyperparameters
  - models/ folder or download links

### Çıktı:
- ✅ `results/ablations.json` (tüm 5 configuration: R, A, G, Final scores)
  ```json
  {
    "v1": {"R": 0.45, "A": 0.35, "G": 0.42, "Final": 0.41},
    "v2": {"R": 0.52, "A": 0.38, "G": 0.45, "Final": 0.46},
    "v3": {"R": 0.58, "A": 0.41, "G": 0.48, "Final": 0.51},
    "v4": {"R": 0.62, "A": 0.58, "G": 0.71, "Final": 0.63},
    "v5": {"R": 0.68, "A": 0.61, "G": 0.74, "Final": 0.67}
  }
  ```
- ✅ `results/hallucination_analysis.json` (examples + rate)
- ✅ `results/error_analysis.json` (pattern breakdown)
- ✅ `evaluation.py` (metrics calculation script)
- ✅ `REPORT.pdf` (10-15 sayfa with all analysis)
- ✅ `presentation.pptx` (15 slide)
- ✅ `demo.py` (Gradio interface with citations)
- ✅ GitHub repository (clean, documented, reproducible)

**⏱️ Tahmini Süre:** 25-30 saat

---

## 📌 TOPLAM ÖZET

| Faza | Görev | Süre | Çıktı | Rubrik Bağlantısı |
|------|-------|------|-------|-------------------|
| 1 | Data + Setup | 10-15h | documents.json + dev env | - |
| 2 | Baseline Retrieval | 15-20h | Dense + BM25 pipeline | R (Retrieval) |
| 3 | **Test Benchmark** | **15-20h** | **150-300 Q+A+Doc triplets** | **Gold Q+A+Doc** |
| 4a | Embedding Tuning | 15-20h | Fine-tuned embedding | R improvement |
| 4b | Reranker | ⏩ SKIP | - | - |
| 4c | LLM Fine-tuning | 25-35h | Fine-tuned LLM + RAG | A + G scores |
| 5 | **Eval + Report** | **25-30h** | **R, A, G metrics + analysis** | **Rubric scoring** |
| | | | | |
| **TOPLAM** | **1 Kişi** | **130-160h** | **Full working system** | **Rubric-aligned** |

---

## 🎯 MINIMAL VIABLE PROJE (acele varsa)

Eğer süre kısıysa (ama hocanın rubrik'ini karşılamalısın):
1. ✅ FAZE 1-2: Data + Baseline (2 hafta)
2. ✅ **FAZE 3: Gold Q+A+Doc Benchmark** (1.5 hafta - ÖNEMLİ!)
3. ❌ FAZE 4a: Embedding tuning SKIP
4. ✅ FAZE 4c: LLM - fine-tuning yerine few-shot prompt (1 hafta)
5. ✅ **FAZE 5: R+A+G Scoring + Report** (1.5 hafta)

**Minimum Zaman:** 7 hafta, ~100-120 saat  
**Beklenen Puan:** 5-6/10 (baseline olur ama minimal)

⚠️ **NOT:** FAZE 3 (Gold Benchmark) ve FAZE 5 (Rubric Scoring) ASLA skip etme!

---

## 💡 HIZLI İPUÇLARİ

1. **Veri Collection:** Mevcut datasets'i kullan, crawl etme
2. **GPU:** Google Colab Pro veya Lambda Labs kullan (15GB+ memory gerekli)
3. **Fine-tuning:** LoRA + QLoRA ile bellek tasarrufu yap
4. **Gold Benchmark (ÖZEL):**
   - Her soru için golden answer YAZMAN GEREK (copy-paste yetmez)
   - Her answer için gold docs'ları manuel seç
   - Citation'ları doğru yap (madde numaraları, kaynak linkler)
5. **Metrics Calculation (ÖZEL):**
   - R (Retrieval): Retrieved docs ∩ gold docs / |gold docs|
   - A (Answer): EM, F1, BLEU with golden_answer
   - G (Grounding): Hallucination rate + citation accuracy
   - Final = 0.35R + 0.4A + 0.25G
6. **Error Analysis:** Başarısız cases'leri kategorize et (Not retrieved, Wrong ranking, Hallucination, Citation error, Answer mismatch)
7. **Demo:** Citation'ları göster (output: Question + Retrieved Docs + Answer + Sources)

---

## ⏰ ACELE TESLİME GÖRE TAKVIM

**Eğer 2 haftan var:**
- Skip FAZE 4a (embedding tuning)
- Skip FAZE 4c (LLM tuning)
- Sadece baseline RAG + prompt-tuned LLM
- Puan: 5-6/10

**Eğer 4 haftan var:**
- FAZE 1-3 (baseline sistem)
- FAZE 4c (LLM lite fine-tuning)
- Skip FAZE 4a + 4b
- Puan: 6-7/10

**Eğer 6+ haftan var:**
- Tüm fazalar (tam sistem)
- Puan: 8-9/10

---

---

## 📍 CHECKLIST - ÖNEMLİ NOKTALAR

✅ **FAZE 3 (Gold Benchmark) - Başarısı için:**
- [ ] Her soru için manuel golden answer yazıldı
- [ ] Her answer için ilgili gold documents seçildi (2-4 doc)
- [ ] Citation'lar doğru (madde numaraları, kaynaklar)
- [ ] Quality control: Spot check 20+ Q-A-Doc çifti manuel verify

✅ **FAZE 5 (Evaluation) - Başarısı için:**
- [ ] R (Retrieval) metric'leri hesaplanıyor
- [ ] A (Answer Quality) metric'leri hesaplanıyor
- [ ] G (Grounding) metric'leri hesaplanıyor
- [ ] Final Score = 0.35R + 0.4A + 0.25G
- [ ] Hallucination analysis yapılıyor (manuel)
- [ ] Error patterns kategorize ediliyor
- [ ] Report'ta tüm sonuçlar gösteriliyor

---

**Teslim Tarihi:** `_____________`  
**Şu an:** May 7, 2026  
**Süre Kalan:** `_____ hafta`

➡️ **Teslim tarihini ver, zamanlamayı ayarlayalım!**
