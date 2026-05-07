# HIZLI BAŞLANGÇ KONTROL LİSTESİ
## 3 Kişi Takım - Week by Week Checklist

---

## ✅ İLK GÜN YAPILACAKLAR (Tüm Üyeler)

- [ ] GitHub repo oluştur
- [ ] Python venv kur: `python -m venv venv`
- [ ] Requirements.txt + setup.py hazırla
- [ ] Discord/Telegram grubu aç (günlük sync)
- [ ] PROJECT_PLAN.md oku ve rolleri onayla
- [ ] Repository struktur oluştur (data/, src/, configs/, etc)
- [ ] İlk commit: initial setup

---

## 👤 HAFTA 1-2: DATA + BENCHMARK PREP

### Person 1: Data & Baseline Retrieval (HAFTA 1 START)
```
Data Preparation:
□ Kaggle Turkish Law Dataset indir
□ HuggingFace turkish-lawchatbot dataset indir
□ data/raw/ klasörüne koy
□ Dataset yapısını explore et (format, boyut, örnek docs)
□ Exploratory data analysis notebook başlat (01_data_exploration.ipynb)
□ Veri boyutu, format, örnek docs kaydet

Shared: Benchmark Building (tüm 3 kişi)
□ 150-300 Türkçe hukuk sorusu hazırla (manual veya crawl sites)
□ Referanslı cevaplar doğrula (golden answers)
□ Question-answer pairs JSON format'ında save et
□ data/benchmarks/test_questions.json + golden_answers.json
□ Verification: Q-A çiftleri checksum/validate
```

### Person 2: LLM Araştırma (HAFTA 1)
```
LLM Setup & Research:
□ Türkçe LLM model araştır (Llama2-Turkish vs Mistral7B)
□ HuggingFace model cards oku
□ Model size + memory requirements hesapla
□ GPU compatibility check
□ Local/cloud access plan yapı kur
□ Prompt template template oluştur (placeholder)
□ Hugging Face token setup et
```

### Person 3: Support (HAFTA 1-2)
```
Support & Setup:
□ GPU setup verify et (torch + cuda)
□ Evaluation metrics framework başlat
□ Test utils kodu hazırla
□ Monitoring/logging infrastructure setup
```

**Hafta 1 End Checkpoint:**
- ✅ All data ready + benchmark 150+ Q-A
- ✅ Repository structured
- ✅ All GPUs working
- ✅ First daily sync completed

---

## 👤 HAFTA 2-3: DATA CLEANING + BASELINE DENSE

### Person 1: Data Cleaning & Vector Store (HAFTA 2)
```
Data Cleaning:
□ Data cleaning script yaz (text normalization, duplicate removal)
□ Türkçe-specific preprocessing apply et
□ Outliers ve corrupted documents remove et
□ Cleaned data: data/processed/cleaned_documents.json
□ Data statistics: chunk count, avg length, etc

Baseline Embedding Setup:
□ Embedding model seç (mBERT, xlm-roberta, Turkish SBERT)
□ Model download et + test load
□ Chunking strategy decide (256-512 tokens)
□ Chunk script implement et
□ Chunked documents: data/processed/chunked_documents.json
□ Chunk statistics document et
```

### Person 1: Dense Retrieval (HAFTA 3)
```
Vector Store & Dense Retrieval:
□ Vector DB seç (Chroma local, Pinecone, vb)
□ Embedding inference script implement et
□ Embed all chunks (baseline embedding ile)
□ Vector store populate: vektör DB'ne tüm chunks
□ Basic dense search test et (manual tests)
□ Dense retriever implement (similarity search)
□ Test ranking üzerinde manual checks
□ Baseline dense retrieval metrics calculate:
   - Recall@5, Recall@10, MRR
□ Baseline results: experiments/ablation_1_baseline/
□ Document: embed times, search latency, memory usage
```

**Hafta 3 End Checkpoint:**
- ✅ Clean dataset ready
- ✅ Dense retrieval working
- ✅ Baseline metrics calculated (Recall@5, @10, MRR)
- ✅ 02_baseline_retrieval.ipynb notebook

---

## 👤 HAFTA 4: BM25 + HYBRID + EVAL SETUP

### Person 1: BM25 + Hybrid Retrieval (HAFTA 4)
```
BM25 Implementation:
□ BM25 indexing implement et (Elasticsearch veya pyserini)
□ BM25 retriever implement et
□ Hybrid combine logic (dense + BM25 weighted)
□ Aggregation strategy test et (αD + (1-α)B tuning)
□ Hybrid retrieval test et
□ Hybrid performance measure:
   - Recall@5, Recall@10, MRR (dense vs hybrid compare)
□ Final baseline retrieval pipeline (dense + BM25 + hybrid)
□ Best config save: configs/baseline_config.yaml
```

### Person 3: Evaluation Framework (HAFTA 4)
```
Evaluation Setup (with Person 1):
□ Retrieval metrics implement: Recall@5, @10, MRR, nDCG
□ QA metrics implement: EM (Exact Match)
□ QA metrics: F1 Score, BLEU, ROUGE scores
□ Citation accuracy checker script
□ Hallucination detection baseline version
□ Evaluation results template + tracking sheet
□ Metrics calculate on all test questions
□ Create: utils/evaluation_metrics.py
```

### Person 2: Basic LLM Inference (HAFTA 4)
```
LLM Setup (quick):
□ Seçili LLM model indir
□ Basic inference test et (2-3 dummy inputs)
□ Prompt formatting template finalize et
□ LLM output example save et
□ Memory profiling + latency measure
□ Initial thoughts on prompt design
```

**Hafta 4 End Checkpoint:**
- ✅ Baseline RAG fully working (dense + BM25 + hybrid)
- ✅ All baseline metrics calculated
- ✅ Evaluation framework ready
- ✅ Ablation 1 results: experiments/ablation_1_baseline/results.json
- ✅ Tech stack confirmed + all tests passing

---

## 👤 HAFTA 5-6: EMBEDDING TUNING (Person 2)

### Person 2: Embedding Fine-tuning (HAFTA 5-6)
```
Training Data Prep (Hafta 5 start - with Person 1):
□ Query-positive-negative triplet dataset create
□ Hard negative mining logic (difficult non-relevant docs)
□ Training data validation (semantic check)
□ Data split: train/val/test (e.g., 80/10/10)
□ Save: data/training/embedding_pairs.json

Embedding Model Fine-tuning (Hafta 5-6):
□ Turkish SBERT model load et
□ Contrastive learning setup (triplet/simclaim loss)
□ Training hyperparameters set:
   - epochs, batch_size (e.g., 16-32), learning_rate (1e-5 to 5e-5)
   - warmup steps, scheduler
□ Training loop implement et
□ Loss curves monitor + save
□ Validation on held-out pairs
□ Model convergence check
□ Best model checkpoint save: models/embedding_tuned/
□ Compare baseline vs. fine-tuned embedding:
   - Retrieval Recall@5, @10, MRR on test set
□ Results save: experiments/ablation_2_embedding/results.json
□ GPU hours log: experiments/ablation_2_embedding/gpu_log.txt
```

**Hafta 6 End Checkpoint:**
- ✅ Embedding model fine-tuned
- ✅ Ablation 2 results ready (+Embedding)
- ✅ 03_embedding_training.ipynb notebook
- ✅ Compare: baseline embedding vs fine-tuned

---

## 👤 HAFTA 6-7: RERANKER TRAINING (Person 2)

### Person 2: Reranker Fine-tuning (HAFTA 6-7)
```
Ranking Training Data (Hafta 6):
□ Query-document pairs with relevance scores create
□ Negative sampling: hard negatives include
□ Positive/negative balance check
□ Training data validation
□ Data split: train/val/test
□ Save: data/training/ranking_pairs.json

Cross-Encoder Fine-tuning (Hafta 6-7):
□ Turkish cross-encoder model seç (mBERT-based)
□ Ranking loss setup (classification or regression)
□ Hyperparameters:
   - epochs, batch_size, learning_rate
   - warmup, scheduler, dropout
□ Training loop implement
□ Validation loss monitor
□ Early stopping implement
□ Best model checkpoint: models/reranker/
□ Reranker to pipeline integrate:
   - Take top-20 from dense+BM25
   - Rerank with cross-encoder
   - Top-k select (e.g., k=5)
□ Reranker performance measure:
   - Recall@5, @10 (with reranking)
□ Ablation: baseline vs +embedding vs +reranker
□ Results save: experiments/ablation_3_reranker/results.json
□ GPU hours log: experiments/ablation_3_reranker/gpu_log.txt
```

**Hafta 7 End Checkpoint:**
- ✅ Reranker model trained
- ✅ Reranker integrated to pipeline
- ✅ Ablation 3 results ready (+Reranker)
- ✅ 04_reranker_training.ipynb notebook
- ✅ Performance gain measured

---

## 👤 HAFTA 8-9: LLM FINE-TUNING (Person 3)

### Person 3: LLM Fine-tuning (HAFTA 8-9)
```
LLM Setup & Prompting (Hafta 8 start):
□ Seçili LLM model (Llama2-Turkish or Mistral) indir
□ Model config examine et
□ GPU memory requirement final check
□ LoRA/QLoRA adapter configuration:
   - rank, lora_alpha, target_modules set
   - configs/llm_lora_config.yaml save
□ 8-bit quantization setup (bitsandbytes)
□ Token limit verify et

Citation-Based Prompting (Hafta 8):
□ Prompt template design (with retrieval context):
   "Şu kaynaklara dayanarak soruyu cevapla:
    [retrieved + reranked documents]
    Soru: [question]
    Cevap (kaynakları belirterek):"
□ Few-shot examples prepare
□ Instruction format finalize
□ Hallucination prevention prompts add:
   - "Sadece verilen kaynaklarda bulunan bilgileri kullan"
   - Citation format enforce
□ Test prompts with base model (dummy tests)
□ 03_llm_prompt_templates.py save

LLM Supervised Fine-tuning (Hafta 9):
□ Training data prepare (with Person 1):
   - Context (retrieved docs) + Question → Answer pairs
   - Format: instruction + input + output
   - 500-1000 examples minimum
   - Save: data/training/lora_training_data.json
□ Training hyperparameters:
   - epochs (2-3), batch_size (8-16)
   - learning_rate (1e-4 to 5e-4)
   - gradient_accumulation_steps
   - max_steps vs epochs
□ Training loop implement with LoRA
□ Training monitor:
   - Loss curves, validation loss
   - Checkpoints save
□ Best model select (lowest validation loss)
□ LoRA weights save: models/llm_lora/
□ Test inference with fine-tuned LLM
□ Citation accuracy spot-check (manual)
□ Results save: experiments/ablation_4_llm/results.json
□ GPU hours log: experiments/ablation_4_llm/gpu_log.txt
```

**Hafta 9 End Checkpoint:**
- ✅ LLM fine-tuned with LoRA
- ✅ Ablation 4 results ready (+LLM)
- ✅ 05_llm_finetuning.ipynb notebook
- ✅ Citation-guided prompts working
- ✅ Manual hallucination checks passed

---

## 👥 HAFTA 10: FULL SYSTEM INTEGRATION (All 3)

### All Together: Full Integration
```
Person 1: Data + Retrieval support
□ Final data validation
□ Retrieval pipeline optimization
□ Hybrid retrieval parameters tune

Person 2: Embedding + Reranker support
□ Embedding inference optimization
□ Reranker latency check

Person 3: LLM + Full Pipeline lead
□ LLM inference optimization
□ Pipeline orchestration:
  Question → Embedding → Retrieval → Reranker → LLM → Answer
□ End-to-end latency measure
□ Error handling:
  - Timeout handling
  - OOM fallback
  - Empty retrieval results handling

Integration Testing (all):
□ Full pipeline test on 10 sample questions
□ Output quality check (answers + citations)
□ Performance measure:
  - Latency per component
  - Total end-to-end latency
  - Memory usage peak
□ Debug any issues

Ablation 5: Fully Optimized System
□ Run on full test set (150-300 questions)
□ All metrics calculate:
  - Retrieval: Recall@5, @10, MRR, nDCG
  - QA: EM, F1, BLEU, ROUGE
  - Citations: accuracy, F1
  - Hallucination: count of hallucinations
□ Results save: experiments/ablation_5_optimized/results.json
□ GPU hours log total
□ Create final results table: results_table.json
```

**Hafta 10 End Checkpoint:**
- ✅ Fully integrated RAG system working
- ✅ Ablation 5 results ready
- ✅ 06_evaluation.ipynb all ablations
- ✅ Performance bottlenecks identified
- ✅ All 5 ablation experiments completed

---

## 👥 HAFTA 11: ERROR ANALYSIS + HALLUCINATION (All 3)

### All: Error Analysis & Hallucination Study
```
Error Analysis:
□ Failed test cases identify (errors, wrong answers)
□ Error categories: retrieval failure, reranking failure, LLM hallucination
□ Analysis per component:
  - Retrieval errors: gold document not in top-k
  - Reranker errors: relevant doc pushed down
  - LLM errors: fabricated facts, wrong reasoning
□ Error examples document with detailed analysis

Hallucination Analysis:
□ Generated answers manually review (5-10% sample)
□ Hallucination instances identify
□ Hallucination types:
  - Extrinsic hallucination (outside training data)
  - Intrinsic hallucination (contradicts context)
□ Hallucination percentage estimate
□ Citation accuracy measurement
□ Citations vs. facts correspondence check

Detailed Metrics:
□ Per-component ablation analysis
□ Embedding vs. reranker impact measure
□ LLM fine-tuning contribution estimate
□ Best performing configuration identify

Report Preparation (parallel write):
□ Person 1: Data + Retrieval sections
□ Person 2: Embedding + Reranker sections
□ Person 3: LLM + Architecture sections + Conclusion

Save outputs:
□ error_analysis.md
□ hallucination_report.md
□ results_summary.md with all ablations
```

**Hafta 11 End Checkpoint:**
- ✅ Error analysis completed
- ✅ Hallucination study done
- ✅ Detailed ablation analysis
- ✅ Report draft sections ready

---

## 👥 HAFTA 12: FINAL REPORT + PRESENTATION (All 3)

### All: Report Writing
```
Final Report (10-15 pages):
□ Abstract (motivation + key results)
□ Introduction (problem definition)
□ Related Work (RAG systems, Turkish NLP)
□ Methodology section:
  ✓ Data preparation (Person 1)
  ✓ Baseline retrieval (Person 1)
  ✓ Embedding + Reranker (Person 2)
  ✓ LLM fine-tuning (Person 3)
  ✓ System architecture (Person 3)
□ Experiments:
  ✓ Ablation study setup
  ✓ Hyperparameters
  ✓ Training details (GPU hours, convergence)
□ Results (table + graphs)
□ Error Analysis
□ Hallucination Study
□ Conclusion + Future Work
□ References + Appendix

Presentation Slides (15 min):
□ Slide 1: Title slide (problem, team)
□ Slide 2: Problem definition
□ Slides 3-4: System architecture diagram
□ Slides 5-7: Methodology (retrieval, embedding, LLM)
□ Slides 8-10: Ablation results (with graphs)
□ Slide 11: Error analysis + hallucination findings
□ Slide 12: Performance bottlenecks
□ Slides 13-14: Key insights + limitations
□ Slide 15: Conclusion + future work

Live Demo Script:
□ Demo input questions prepare (3-5 good examples)
□ Expected outputs document
□ Live system walkthrough prepare
□ Q&A prep (common questions answer)
□ Timing: 5 min demo

Code Cleanup:
□ All code commented + docstrings
□ README complete with setup instructions
□ Requirements.txt finalize
□ configs/ all documented
□ experiments/ folder structure clean
□ Git history clean + meaningful commits

Final Checks:
□ Report proofread
□ Slides reviewed by all
□ Demo tested 3x
□ GitHub final state
□ Code reproducibility verify
□ Hyperparameters documented
□ GPU usage reported
```

**Hafta 12 END - FINAL CHECKPOINT:**
- ✅ 10-15 page technical report
- ✅ Presentation slides finished
- ✅ Live demo working + script prepared
- ✅ GitHub repo polished
- ✅ All deliverables submitted

---

## 📊 ABLATION RESULTS TRACKING TABLE

| # | Ablation | Timeline | Recall@10 | EM | F1 | BLEU | Halluc. | GPU Hours | Who |
|---|----------|----------|-----------|----|----|------|---------|-----------|-----|
| 1 | Baseline | Hafta 4 | ___ | ___ | ___ | ___ | ___ | ___ | P1 |
| 2 | +Embedding | Hafta 6 | ___ | ___ | ___ | ___ | ___ | ___ | P2 |
| 3 | +Reranker | Hafta 7 | ___ | ___ | ___ | ___ | ___ | ___ | P2 |
| 4 | +LLM | Hafta 9 | ___ | ___ | ___ | ___ | ___ | ___ | P3 |
| 5 | FULLY OPT | Hafta 10 | ___ | ___ | ___ | ___ | ___ | ___ | All |

---

## 📞 HAFTALIK SYNC (Pazartesi 10:00 - 30 min)

**Agenda:**
1. Block update (2 min)
2. Person 1: Retrieval status (5 min)
3. Person 2: Model training progress (5 min)
4. Person 3: LLM + integration bugs (5 min)
5. Next week priorities + data sharing (5 min)
6. Q&A + troubleshooting (3 min)

---

## ⚠️ EMERGENCY CONTINGENCIES

**Time Running Out:**
- Skip embedding tuning → use pre-trained SBERT
- Use standard cross-encoder → minimize reranker tuning
- Focus on LLM prompting → skip fine-tuning
- Smaller test set (100 questions)

**GPU OOM:**
- 8-bit quantization enable
- Batch sizes reduce (4-8)
- LoRA rank reduce (r=4)
- Use 7B instead of 13B LLM

**Data Issues:**
- Alternative dataset ready (backup)
- Hybrid generation + retrieval (synthetic QA pairs)
- Smaller benchmark (50 high-quality questions)

**Model Not Converging:**
- Learning rate reduce
- More training data verify
- Architecture simpler (smaller model)
- Gradient clipping + warmup add

---

**Last Updated:** Day 1 ✅  
**Status:** Ready to launch! 🚀
