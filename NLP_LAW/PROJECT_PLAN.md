# CENG493 - Türkçe Hukuk QA RAG Sistemi 
## 3 Kişilik Takım - Proje Planı

---

## 🎯 TAKIM YAPILANDI VE ROL ATAMALARI

### **👤 Person 1: Data + Baseline Retrieval**
Sorumlu: **[İsim]**

**Ana Görevler:**
- Türkçe hukuk metinleri topla ve temizle
- Madde/bölüm bazında chunk
- Dataset/index oluştur ve vektör DB'ne yükle
- Baseline dense retrieval implement
- BM25 retrieval implement
- Hybrid retrieval (dense + BM25) kur ve test
- Baseline retrieval metrics hesapla (Recall@5, Recall@10, MRR)
- Vektör arama performansı optimize et

**Çıktı:** Clean legal dataset + fully working baseline retrieval pipeline

---

### **👤 Person 2: Embedding + Reranker**
Sorumlu: **[İsim]**

**Ana Görevler:**
- Training pairs ve hard negatives dataset hazırla
- Türkçe embedding model (SBERT) seç
- Embedding model contrastive fine-tuning (Üye 1 ile synch)
- Cross-encoder reranker model seç ve train
- Top-k retrieval quality iyileştir ve test et
- Baseline vs. tuned embedding retrieval karşılaştır
- Reranker etkisini ablation with/without test et
- Fine-tuned models save et (safeguard)

**Çıktı:** Domain-adapted embedding model + trained reranker

---

### **👤 Person 3: LLM + Final System Integration**
Sorumlu: **[İsim]**

**Ana Görevler:**
- Answer generation module tasarla
- Citation-based prompt templates design (Üye 1 ile input)
- Türkçe LLM (Llama2-Turkish/Mistral) seç
- LoRA/QLoRA adapter config
- LLM supervised fine-tuning (training data Üye 1'den)
- Full pipeline integrate: retrieval → reranker → LLM → answer
- Citation accuracy control implement
- Demo app/gradio interface hazırla
- End-to-end sistem test et

**Çıktı:** Complete working RAG system + demo interface


---

## 👥 SHARED TASKS (Tüm 3 Kişi Birlikte)

### **Hafta 1-2: Gold QA Benchmark Build**
- 150-300 Türkçe hukuk sorusu topla/hazırla
- Referanslı cevaplar doğrula
- Test split oluştur (train/val/test)
- Question-answer pairs JSON format'ında save et
- **Checkpoint:** 150+ verified Q-A pairs

### **Hafta 10: Evaluation Run**
- Tüm 5 ablation sistema test et
- Tüm metrics hesapla (retrieval + QA metrics)
- Sonuçları tablo halinde organize et
- Baseline vs optimized karşılaştırma

### **Hafta 11: Hallucination & Error Analysis**
- Generated answers'ları manual review
- Hallucination örnekleri identify et
- Citation accuracy deep dive
- Error patterns analiz et
- Failure cases document et

### **Hafta 12: Report & Presentation**
- **Report:** 10-15 sayfa technical report
  - Abstract, Introduction, Related Work
  - Methodology (her component)
  - Experiments & Ablation Results
  - Error Analysis & Hallucination Study
  - Conclusion & Future Work
- **Slides:** 15-minute presentation
- **Demo:** Live system demonstration
- Code cleanup + documentation

---

## 2. PROJE FAZLARİ VE ZAMAN ÇIZELGESI

### FAZE 1: Data Preparation (Hafta 1-2)
**Person 1 Sorumlu**

```
Tasks:
□ Kaggle Turkish Law Dataset indir
□ HuggingFace turkish-lawchatbot dataset indir
□ Veri yapısını explore et (format, boyut, örnek docs)
□ Data cleaning script yaz (text normalization, duplicate removal)
□ Madde/bölüm bazında chunking implement
□ Chunk size optimize et (256-512 tokens)
□ Cleaned dataset JSON/CSV save et
□ Shared: Gold QA benchmark build (150-300 sorular)
```

**Checkpoint:** Data + benchmark ready

---

### FAZE 2: Vector Database + Baseline Dense Retrieval (Hafta 2-3)
**Person 1 Sorumlu**

```
Tasks:
□ Embedding model seç (mBERT, xlm-roberta, or SBERT)
□ Chroma/Pinecone setup
□ All chunks embed et (baseline model ile)
□ Vector store populate
□ Basic search test et
□ Dense retriever implement (vector similarity search)
□ Test queries ile test et
□ Retrieval metrics implement (Recall@5, Recall@10, MRR)
□ Baseline retrieval performance measure
```

**Checkpoint:** Baseline dense retrieval working

---

### FAZE 3: BM25 + Hybrid Retrieval (Hafta 3-4)
**Person 1 Sorumlu**

```
Tasks:
□ BM25 indexing implement
□ BM25 retriever implement
□ Dense + BM25 hybrid combine (weighted)
□ Hybrid performance compare vs. dense alone
□ Aggregation strategy optimize
□ Baseline metrics calculate: Recall@5, Recall@10, MRR
□ Person 3 with: Evaluation metrics implement (EM, F1, BLEU, ROUGE)
□ Person 3 with: Citation accuracy checker
□ Final baseline retrieval pipeline document
```

**Checkpoint:** Baseline RAG running, all metrics ready

---

### FAZE 4: Embedding Training Data Prep (Hafta 5)
**Person 2 Sorumlu (with Person 1)**

```
Tasks:
□ Training pairs dataset create (query-positive-negative)
□ Hard negative mining: difficult non-relevant docs
□ Control set for validation
□ Training data validation
□ Data split: train/val for embedding tuning
```

---

### FAZE 5: Embedding Model Fine-tuning (Hafta 5-6)
**Person 2 Sorumlu**

```
Tasks:
□ Türkçe embedding model seç (SBERT)
□ Contrastive learning setup (triplet/simclaim)
□ Training loop implement
□ Hyperparameters: epochs, batch size, learning rate
□ Training monitor (loss curves)
□ Validation on held-out pairs
□ Fine-tuned embedding model save
□ Compare: baseline embedding vs. fine-tuned
```

**Checkpoint:** +Embedding ablation results ready

---

### FAZE 6: Reranker Training (Hafta 6-7)
**Person 2 Sorumlu**

```
Tasks:
□ Cross-encoder model seç (mBERT-based)
□ Ranking training data prepare (query-doc pairs with relevance score)
□ Cross-encoder fine-tuning setup
□ Training: epochs, batch size, learning rate
□ Loss curves monitor
□ Validation performance check
□ Fine-tuned reranker save
□ With Person 1: Reranker to retrieval output apply
□ Top-k quality measure
□ Ablation: baseline vs +embedding vs +reranker
```

**Checkpoint:** +Reranker ablation results ready

---

### FAZE 7: LLM Fine-tuning (Hafta 8-9)
**Person 3 Sorumlu (with Person 1 for training data)**

```
Tasks:
□ Türkçe LLM seç (Llama2-Turkish or Mistral-7B)
□ Model loading yapı kur
□ GPU memory requirement calculate
□ LoRA/QLoRA adapter config
□ 8-bit quantization setup (memory optimization)
□ Training data: (context + question, answer) pairs
□ Training loop implement
□ Hyperparameters: epochs, learning_rate, batch_size
□ Training monitor ve checkpoints save
□ Validation loss track
□ Best model select
□ Fine-tuned weights save (LoRA adapters)
□ Design citation-based prompt templates
□ Instruction format finalize
```

**Checkpoint:** +LLM ablation results ready

---

### FAZE 8: System Integration (Hafta 10)
**Person 3 Sorumlu (all support)**

```
Tasks:
□ Retrieval → Reranker → LLM pipeline orchestrate
□ Input processing (question formatting)
□ Output processing (answer parsing + citation extraction)
□ Error handling implement (timeout, OOM, etc)
□ End-to-end test et
□ Performance measure (latency, throughput)
□ Evaluate all ablations (shared task all 3)
□ Fully optimized system results ready
```

**Checkpoint:** Fully optimized system ready

---

### FAZE 9: Demo Interface (Hafta 10-11)
**Person 3 Sorumlu**

```
Tasks:
□ Gradio interface design
□ Input: Türkçe hukuk sorusu
□ Output: Answer + retrieved documents + sources
□ Citation highlighting
□ Real-time processing
□ Test et loop
□ User experience optimize
```

**Checkpoint:** Working demo interface

## 📅 ZAMAN ÇIZELGESI & MİLESTONELAR

| Hafta | Person 1 | Person 2 | Person 3 | Shared | Milestone |
|-------|----------|----------|----------|--------|-----------|
| 1 | Data download & explore | - | - | Build benchmark (150+ Q-A) | Data ready ✅ |
| 2 | Data clean & chunk | - | Setup LLM | Verify benchmark | All data ready ✅ |
| 3 | Baseline dense retrieval | - | LLM setup test | - | Baseline pipeline live |
| 4 | BM25 + Hybrid retrieval | - | Basic inference test | Evaluate baseline metrics | Baseline metrics ✅ |
| 5 | Dataset finalize (Üye 2 için) | Embedding tuning start | - | - | - |
| 6 | Support Üye 2 | Embedding tuning finish | - | - | +Embedding ablation ✅ |
| 7 | Support Üye 2 | Reranker training | - | - | +Reranker ablation ✅ |
| 8 | - | Reranker finish | LLM fine-tuning | - | - |
| 9 | - | - | LLM fine-tuning finish | - | +LLM ablation ✅ |
| 10 | Integration test | Integration test | Integration test | Evaluate fully optimized | Fully optimized ✅ |
| 11 | Support analysis | Support analysis | Support analysis | Hallucination & error analysis | Analysis complete ✅ |
| 12 | Report write | Report write | Report write | Finalize deliverables | Final report ✅ |

---

## 3. GITHUB REPOSITORY YAPISI

```
nlp_law_rag/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                      # Person 1 sorumlu
│   │   ├── turkish_law_raw.json
│   │   └── raw_dataset_info.md
│   ├── processed/                # Person 1 sorumlu
│   │   ├── chunked_documents.json
│   │   └── chunk_statistics.md
│   ├── benchmarks/               # Shared (all 3)
│   │   ├── test_questions.json
│   │   └── golden_answers.json
│   └── training/
│       ├── embedding_pairs.json           # Person 2
│       └── lora_training_data.json        # Person 3
│
├── models/
│   ├── baseline/
│   │   └── checkpoint (vector embeddings)
│   ├── embedding_tuned/                  # Person 2 output
│   │   └── fine_tuned_model
│   ├── reranker/                         # Person 2 output
│   │   └── fine_tuned_reranker
│   └── llm_lora/                         # Person 3 output
│       └── adapter_weights
│
├── src/
│   ├── __init__.py
│   ├── person1_retrieval/
│   │   ├── data_loader.py
│   │   ├── chunking.py
│   │   ├── vector_store.py
│   │   ├── dense_retriever.py
│   │   ├── bm25_retriever.py
│   │   └── hybrid_retriever.py
│   ├── person2_embedding_reranker/
│   │   ├── embedding_trainer.py
│   │   ├── reranker_trainer.py
│   │   └── reranker_inference.py
│   ├── person3_llm/
│   │   ├── llm_loader.py
│   │   ├── prompt_templates.py
│   │   ├── llm_finetuner.py
│   │   ├── llm_inference.py
│   │   └── demo_app.py
│   ├── evaluation/
│   │   ├── retrieval_metrics.py
│   │   ├── qa_metrics.py
│   │   ├── hallucination_detector.py
│   │   └── citation_checker.py
│   └── pipeline/
│       └── rag_pipeline.py
│
├── configs/
│   ├── baseline_config.yaml
│   ├── embedding_config.yaml
│   ├── reranker_config.yaml
│   └── llm_lora_config.yaml
│
├── experiments/
│   ├── ablation_1_baseline/
│   ├── ablation_2_embedding/
│   ├── ablation_3_reranker/
│   ├── ablation_4_llm/
│   └── ablation_5_optimized/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_retrieval.ipynb
│   ├── 03_embedding_training.ipynb
│   ├── 04_reranker_training.ipynb
│   ├── 05_llm_finetuning.ipynb
│   └── 06_evaluation.ipynb
│
├── scripts/
│   ├── download_data.sh
│   ├── run_baseline.py
│   ├── run_ablations.py
│   └── evaluate_all.py
│
└── reports/
    ├── technical_report.md
    ├── ablation_results.md
    ├── error_analysis.md
    └── figures/
```

## 4. KRITIK MİLESTONELAR

| Hafta | Milestone | Sorumlu | Checkpoint |
|-------|-----------|---------|-----------|
| 2 | Veri + Benchmark Hazır | P1 + Shared | 150+ Q-A pair |
| 4 | Baseline RAG Çalışıyor | P1 | Baseline metrics ready |
| 6 | +Embedding Tuning Done | P2 | Ablation 2 results |
| 7 | +Reranker Done | P2 | Ablation 3 results |
| 9 | +LLM Fine-tune Complete | P3 | Ablation 4 results |
| 10 | Fully Optimized System | All | Ablation 5 results |
| 11 | Analysis Complete | Shared | Error analysis ready |
| 12 | Report + Presentation | All | Final deliverables ✅ |

---

## 5. ABLATION EXPERIMENTS TRACKING

| # | Ablation | Retrieval Recall@10 | EM | F1 | BLEU | Notes |
|---|----------|---------------------|----|----|------|-------|
| 1 | Baseline RAG | ___ | ___ | ___ | ___ | P1 baseline |
| 2 | +Embedding tuning | ___ | ___ | ___ | ___ | P2 tuned embedding |
| 3 | +Reranker | ___ | ___ | ___ | ___ | P2 cross-encoder |
| 4 | +LLM fine-tuning | ___ | ___ | ___ | ___ | P3 LoRA |
| 5 | FULLY OPTIMIZED | ___ | ___ | ___ | ___ | All components |

---

## 6. TEKNIK STACK ÖNERİLERİ

```yaml
Embedding: sentence-transformers (Turkish SBERT)
Reranker: cross-encoder (Turkish)
LLM: Llama2-Turkish-7B or Mistral-7B
Vector DB: Chroma (local) veya Pinecone
Framework: LangChain + HuggingFace Transformers
Fine-tuning: bitsandbytes (QLoRA support)
Evaluation: NLTK, rouge-metric, transformers (bertscore)
Interface: Gradio
```

---

## 7. GPU GEREKSINIMLERI

```
Embedding tuning: 1x GPU (8GB+)
Reranker training: 1x GPU (8GB+)
LLM fine-tuning: 2-4x GPUs (distributed training recommended)
Toplam: ~16GB-24GB GPU memory optimal
```

---

## 8. HAFTALIK SYNC (Pazartesi 10:00 - 30 min)

1. **Person 1:** Retrieval pipeline status (data, dense, BM25)
2. **Person 2:** Model tuning progress (embedding/reranker training)
3. **Person 3:** LLM + integration blockers
4. **All:** Next week's priorities
5. **All:** Data/resource sharing needs
6. **All:** Any blockers?

---

## 9. BAŞARILI TAMAMLAMA İÇİN İPUÇLARİ

✅ **Early Integration:** Hafta 4'ten itibaren sistem integration test et  
✅ **Experiment Tracking:** Tüm runs için logs tutarak GPU usage kaydet  
✅ **Regular Checkpoints:** Eğitim sırasında model checkpoints save et  
✅ **Error Prevention:** Regular data validation çek (duplicate, corruption)  
✅ **Communication:** Blockers'ı hemen kaldırmaya yardımcı ol  
✅ **Backup Plans:** Model timeout/OOM için alternatifler hazırla  
✅ **Version Control:** Git'te meaningful commits + branches per person

---

## 10. RISKLER VE MİTİGASYON

| Risk | Etki | Mitigation |
|------|------|-----------|
| GPU OOM | High | Model quantization (8-bit), batch size reduce |
| Veri kalitesi | Medium | Early exploration + validation scripts |
| Integration issues | High | Weekly integration tests, API contracts |
| Time crunch | High | Agile + parallelization, prioritize ablations |
| Model convergence | Medium | Multiple seeds, early stopping, learning rate decay |
| Hallucination | High | Citation-guided prompts, constrained generation |

---

## 11. EMERGENCY CONTINGENCIES

**Zaman Geride Kaldık:**
- Skip embedding tuning → use baseline embedding only
- Use pre-trained reranker → skip fine-tuning
- Use instruction prompts → skip LLM fine-tuning (focus on prompt engineering)
- Smaller test set (100 q's instead of 150-300)

**Model OOM:**
- Model quantization (8-bit)
- Smaller batch sizes
- LoRA adapter size reduce
- Use smaller LLM (7B instead of 13B)

**Data Issues:**
- Alternative datasets ready
- Backup benchmark prepared
- Data quality checker automat

---

## 12. İLETİŞİM & COLLABORATION

- **GitHub:** Hergün commit + meaningful messages
- **Discord/Telegram:** Günlük ~5 min sync (blockers, quick wins)
- **Weekly Sync:** Pazartesi 10:00 (30 min detailed meeting)
- **Shared Document:** Progress tracking sheet (ablation results live update)
- **Code Reviews:** Pull request → another person reviews before merge

---

## 📊 SIMPLE SUMMARY

```
Person 1: Doğru document'leri bul 📄
   ↓
Person 2: Ranking kalitesini artır 📊
   ↓
Person 3: Final answer ver + sistem birleştir 💬
   ↓
All Together: Test, analyze, report! 🚀
```

---

**LET'S BUILD THIS! 🚀**
**Başarılar dileğiniz!**
