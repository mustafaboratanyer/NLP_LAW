# Improving Turkish Legal Question Answering with an Optimized RAG Pipeline

**Course:** CENG493 Term Project  
**Students:** Mustafa Bora Tanyer, Arda Gönül, Nurettin Efe Alver  
**Project Type:** Turkish legal question answering with retrieval-augmented generation

## Abstract

This project develops a Turkish legal question answering system based on a Retrieval-Augmented Generation (RAG) pipeline. The goal is to answer Turkish legal questions using retrieved legal sources and to reduce hallucination by grounding answers in cited legal articles. The current implementation focuses on building a clean Turkish legal retrieval corpus, constructing a FAISS-based dense index, adding hybrid retrieval with BM25, and fine-tuning a BERTurk-based cross-encoder reranker. Early qualitative results show that the system can retrieve the correct article for several natural legal questions, while colloquial labor-law paraphrases remain challenging. To address this, we added a second reranker-training stage with hard negative mining and curated colloquial legal queries.

## 1. Introduction

Legal question answering requires higher grounding reliability than general-domain question answering. A model must not only generate a fluent answer, but also support the answer with the correct statute, article, or legal source. This is especially important in Turkish law, where the same concept may appear across multiple laws, and small wording differences can change the relevant article.

In this project, we build a RAG system for Turkish legal QA. The baseline architecture follows:

```text
Question -> Embedding -> Vector Search -> Optional Reranker -> LLM -> Answer
```

Our work so far focuses mainly on the retrieval and reranking stages. These stages are critical because the LLM can only produce a grounded answer if the correct legal context is retrieved first.

## 2. Data Preparation

### 2.1 Retrieval Corpus

We constructed an article-level Turkish legal retrieval corpus from official law PDFs. The current internal corpus contains:

| Item | Count |
|---|---:|
| Laws | 34 |
| Article-level documents | 8,455 |
| Retrieval chunks | 8,776 |

Each article-level document includes:

```text
id
law_name
law_no
article_no
article_title
text
source_url
```

For retrieval, long articles are also converted into chunk-level records with parent article IDs. This allows the retriever to search smaller passages while preserving citation consistency at the article level.

### 2.1.1 Chunking Strategy

The corpus is stored at two levels:

| Level | Purpose |
|---|---|
| Article-level corpus | Keeps each law article as the canonical citation unit |
| Chunk-level corpus | Splits long articles into retrievable passages for dense/BM25 search |

The article-level corpus is used for citation consistency. Each chunk keeps a `parent_id` pointing back to the full article. This is important because the retrieval model may retrieve a smaller chunk, but the final answer should cite the legal article, not only an arbitrary text window.

For short articles, one chunk is created for the full article. For long articles, the text is split into overlapping word windows. The current chunking configuration keeps the article title, law name, law number, article number, chunk index, chunk count, parent word count, and source URL. In practice:

```text
retrieval_corpus.json  -> article-level legal documents
retrieval_chunks.json  -> chunk-level searchable passages
chunk.parent_id        -> article id used for evaluation and citation
```

The LLM stage should preferably receive the full parent article or neighboring chunks when the retrieved chunk is only part of a long article. This avoids losing legal conditions that may appear in another part of the same article.

### 2.2 External Dataset Inspection

We also inspected an external dataset prepared by another team. It contains:

| File | Purpose | Size |
|---|---|---:|
| `corpus.jsonl` | External legal corpus | 7,579 records |
| `reranker.jsonl` | Query-passage relevance pairs | 6,752 pairs |
| `gold_benchmark.json` | Gold benchmark | 240 questions |
| `rag_eval.json` | Retrieval evaluation queries | 1,000 queries |
| `llm.jsonl` | SFT-style LLM data | 35 MB |
| `embedding.jsonl` | Embedding training pairs | 4.7 MB |

The most useful file for the reranker stage is `reranker.jsonl`, because it already follows a cross-encoder training structure:

```text
query
candidate_passage
label
candidate_id
citation_label
source
```

We verified the following properties:

| Property | Value |
|---|---:|
| Total reranker pairs | 6,752 |
| Query groups | 1,689 |
| Positive pairs | 2,453 |
| Negative pairs | 4,299 |
| Missing/invalid records | 0 |
| Duplicate query-candidate-label pairs | 0 |
| Candidate IDs found in external corpus | 100% |

This dataset is suitable for reranker fine-tuning, although it contains some template-like questions such as "kaynağa göre ne söylenebilir?". Therefore, we use it as a practical reranker training source but still evaluate carefully on separate legal QA examples.

### 2.3 Gold QA Benchmark

We inspected `qa_benchmark_gold.csv`, which contains 290 manually verified QA examples. Its main properties are:

| Property | Value |
|---|---:|
| Total questions | 290 |
| Active-law questions | 266 |
| Repealed/inactive-law questions | 24 |
| Duplicate questions | 0 |
| Open-ended questions | 0 |
| Context starts mid-sentence | 0 |
| Average answer-support overlap | 0.90 |

This file is highly useful as a final RAG evaluation benchmark. To avoid data leakage, we should not train on the same questions that we use for final evaluation.

## 3. System Architecture

### 3.1 Dense Retrieval

The dense retriever uses `BAAI/bge-m3` embeddings. We build a FAISS index over the chunk-level corpus:

```text
retrieval_chunks.json -> BGE-M3 embeddings -> FAISS index
```

The current index files are:

```text
faiss_bge_m3.index
metadata_bge_m3.json
index_config_bge_m3.json
```

### 3.2 Hybrid Retrieval

Dense retrieval alone sometimes fails on colloquial Turkish questions. For example, the query:

```text
işçi 2 gün işe gelmezse ne olur?
```

did not reliably retrieve the relevant article using dense retrieval alone. We therefore added hybrid retrieval:

```text
Hybrid score = dense retrieval score + BM25 lexical score
```

BM25 helps when the query contains legally important lexical signals such as "iki gün", "işe gelmezse", "devamsızlık", or "haklı nedenle fesih".

### 3.3 Reranker

We implemented reranking with a cross-encoder model. A cross-encoder receives:

```text
(question, candidate passage)
```

and outputs a relevance score. Unlike the embedding model, it does not search the whole corpus by itself. Therefore, the pipeline must first retrieve candidate passages and then rerank them:

```text
Question -> Hybrid candidate retrieval -> Cross-encoder reranker -> Ranked top-k contexts
```

We tested two reranker modes:

| Mode | Description |
|---|---|
| Pretrained reranker | `BAAI/bge-reranker-v2-m3` without project-specific fine-tuning |
| Fine-tuned reranker | `dbmdz/bert-base-turkish-cased` fine-tuned as binary relevance classifier |

For the main experimental result, we use pure cross-encoder scoring:

```text
--ranking-mode rerank
```

We also implemented an optional score fusion mode, but this is treated as an optional demo/ablation variant, not as the main fine-tuned reranker result.

## 4. Reranker Fine-Tuning

### 4.1 Training Data Format

The fine-tuning data follows this binary relevance format:

```json
{
  "query": "birini öldürmek suç mudur?",
  "candidate_passage": "Kasten öldürme Madde 81- Bir insanı kasten öldüren kişi...",
  "label": 1
}
```

Negative examples use passages that are similar but not the correct answer:

```json
{
  "query": "birini öldürmek suç mudur?",
  "candidate_passage": "Taksirle öldürme Madde 85- ...",
  "label": 0
}
```

This teaches the model to distinguish the directly relevant article from related but incorrect articles.

### 4.2 Model

We fine-tune:

```text
dbmdz/bert-base-turkish-cased
```

as a cross-encoder sequence classification model with one output score. The model is trained as a binary relevance classifier.

Current training configuration:

| Hyperparameter | Value |
|---|---|
| Base model | `dbmdz/bert-base-turkish-cased` |
| Epochs | 1 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Max sequence length | 512 |
| Device | Kaggle GPU |
| Train/dev split | Query-ID based, 90/10 |

The query-ID based split prevents the same query from appearing in both training and development data.

### 4.3 Training Data Statistics

From `reranker.jsonl`:

| Split | Rows |
|---|---:|
| Total | 6,752 |
| Train | 6,165 |
| Dev | 587 |

The split is performed by query group:

| Item | Count |
|---|---:|
| Total query groups | 1,689 |
| Dev query groups | 168 |

### 4.4 Hard Negative Mining Update

After the first reranker experiment, we observed that some wrong but lexically similar articles were ranked above the correct article for colloquial questions. For example, the query "işçi 2 gün işe gelmezse ne olur?" should retrieve İş Kanunu Madde 25, but the reranker sometimes ranked leave-related articles higher.

To improve this, we added a hard-negative mining step. The mining script takes the existing reranker queries and searches our current retrieval corpus using the same dense + BM25 candidate generation stage used in the RAG pipeline. Retrieved candidates that are similar to the query but not protected as positives are added as negative examples. We also added a small curated seed file with everyday Turkish legal questions and their correct article IDs, such as:

```text
işçi 2 gün işe gelmezse ne olur? -> İş Kanunu Madde 25
birini öldürmek suç mudur? -> Türk Ceza Kanunu Madde 81
kişisel veriler yurt dışına hangi şartlarda aktarılır? -> KVKK Madde 9
kiracı depozitoyu hangi şartlarda geri alabilir? -> Türk Borçlar Kanunu Madde 342
```

The updated reranker training flow is:

```text
reranker.jsonl + curated queries
-> current corpus hard negative mining with dense + BM25
-> query-level train/dev split
-> BERTurk cross-encoder fine-tuning
```

The Kaggle run produced the following augmented reranker dataset:

| Item | Count |
|---|---:|
| Original reranker rows | 6,752 |
| Original query groups | 1,689 |
| Curated positive rows added | 23 |
| Aligned current-corpus positives added | 642 |
| Hard negatives added | 5,130 |
| Final augmented rows | 12,547 |
| Final positive rows | 3,118 |
| Final negative rows | 9,429 |
| Final query groups | 1,710 |

The augmented dataset was split by query group:

| Split | Rows | Query groups |
|---|---:|---:|
| Train | 11,327 | 1,539 |
| Dev | 1,220 | 171 |

### 4.5 Reranker V2 Training Results

The second reranker was trained for 2 epochs on the augmented dataset. The training completed in about 34.8 minutes on Kaggle GPU. The final training loss was 0.2405.

Best observed dev metrics during training:

| Metric | Value |
|---|---:|
| Accuracy | 0.9418 |
| F1 | 0.8778 |
| Precision | 0.9251 |
| Recall | 0.8725 |
| Average precision | 0.9381 |

The reranker scores are used for ranking rather than as calibrated probabilities. Therefore, absolute score values can differ across questions; the most important signal is whether the correct article is ranked above competing candidates.

### 4.6 Kaggle Law Dataset Reranker Experiment

We also tested the mandatory Kaggle Turkish law fine-tuning dataset as a reranker source. The raw dataset contains question, answer, source, context, and quality score fields. Since it is not directly a reranker dataset, we converted it into binary relevance pairs:

```text
question + matched legal context -> positive pair
question + similar wrong chunk   -> hard negative pair
```

To reduce noisy supervision, we used strict filtering:

| Filter | Purpose |
|---|---|
| `Score >= 9` | Keep only higher-quality generated QA rows |
| Source-law match required | Avoid mapping a question to a law not present in our corpus |
| Context overlap threshold | Accept only rows whose context strongly matches a current corpus chunk |
| Answer overlap threshold | Ensure the answer is supported by the matched passage |
| Same-parent protection | Avoid using another chunk from the same article as a negative |

The generated reranker model was evaluated on the gold benchmark, but it performed worse than the hybrid baseline. This suggests that automatically generated QA-context data does not necessarily teach the reranker the same relevance behavior required by the manually verified benchmark.

### 4.7 Clean 1000-Question Reranker Dataset

After the previous reranker attempts reduced benchmark performance, we created a smaller but cleaner reranker training set. The goal was to prioritize reliable `question -> positive_parent_id` supervision over large noisy pair counts.

The clean dataset was built from three sources:

| Source | Selection rule | Selected questions |
|---|---|---:|
| HuggingFace law QA | Only questions with explicit law/article reference and answer overlap | 123 |
| Kaggle law QA | High-score rows whose context matched our corpus | 490 |
| Corpus title fallback | Questions generated from clear article titles with known parent IDs | 387 |
| Total |  | 1,000 |

We explicitly filtered out artificial or low-quality query styles such as:

```text
ORICON kaynagina gore ...
kaynaga gore ne soylenebilir?
metne gore ...
KAYIT 0845 ...
olayin ozu ve karar sonucu ...
```

The final selected set covers 666 unique parent articles. It was then converted into reranker pairs by adding BM25 hard negatives from our own retrieval chunks:

| Item | Count |
|---|---:|
| Clean questions | 1,000 |
| Positive rows | 1,021 |
| BM25 hard negative rows | 5,000 |
| Total pair rows | 6,021 |
| Train rows | 5,419 |
| Dev rows | 602 |

This is the cleanest reranker training set we created, but the gold benchmark result still showed poor generalization in pure reranking mode. This indicates that even clean data can hurt if its question distribution differs from the final benchmark distribution.

## 5. Qualitative Results So Far

The following results are from the V2 fine-tuned BERTurk reranker using:

```text
--ranking-mode rerank
```

### 5.1 Criminal Law Query

Query:

```text
birini öldürmek suç mudur?
```

Top result:

```text
Türk Ceza Kanunu (5237), Madde 81 - Kasten öldürme
```

This is correct. The retrieved article states that a person who intentionally kills another person is punished with life imprisonment. The correct article was ranked first, above related criminal-law articles.

### 5.2 KVKK Query

Query:

```text
kişisel veriler yurt dışına hangi şartlarda aktarılır?
```

Top result:

```text
Kişisel Verilerin Korunması Kanunu (6698), MADDE 9 - Kişisel verilerin yurt dışına aktarılması
```

This is correct. The system retrieves the article specifically regulating international transfer of personal data. The reranker assigns this article a much higher score than the following KVKK-related articles.

### 5.3 Labor Law Query

Query:

```text
işçi 2 gün işe gelmezse ne olur?
```

Expected relevant article:

```text
İş Kanunu (4857), Madde 25 - İşverenin haklı nedenle derhal fesih hakkı
```

Observed top result after hard-negative mining:

```text
İş Kanunu (4857), Madde 25 - İşverenin haklı nedenle derhal fesih hakkı
```

This is correct. This query was previously a failure case because the reranker ranked leave-related articles above the correct termination article. After adding curated colloquial examples and hard negatives, the correct article is ranked first.

## 6. Quantitative Evaluation

We evaluated retrieval performance using `qa_benchmark_gold.csv`. Out of 290 rows, 24 inactive examples were excluded and 22 examples could not be mapped to our current corpus. The final retrieval evaluation used 244 resolved gold questions.

### 6.1 Retrieval Metrics

| System | Recall@5 | Recall@10 | Top-1 Acc. | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| Dense FAISS baseline | 0.5533 | 0.6230 | 0.4016 | 0.4685 | 0.5052 |
| Hybrid retrieval | 0.6311 | 0.6680 | 0.4631 | 0.5348 | 0.5674 |
| Hybrid retrieval, grid-optimized | 0.6311 | 0.6762 | 0.4795 | 0.5456 | 0.5772 |
| Fine-tuned embedding dense | 0.1066 | 0.1270 | 0.0574 | 0.0801 | 0.0914 |
| Fine-tuned embedding hybrid | 0.2459 | 0.2459 | 0.2131 | 0.2275 | 0.2322 |
| Qwen3-Embedding-0.6B dense | 0.5287 | 0.5820 | 0.3689 | 0.4388 | 0.4732 |
| Qwen3-Embedding-0.6B hybrid | 0.6148 | 0.6434 | 0.4754 | 0.5369 | 0.5630 |
| Hybrid + fine-tuned reranker V2 | 0.4877 | 0.5697 | 0.3402 | 0.4046 | 0.4437 |
| Hybrid + reranker score fusion | 0.6148 | 0.6516 | 0.4467 | 0.5175 | 0.5502 |
| Kaggle-law reranker | 0.2828 | 0.3443 | 0.1844 | 0.2300 | 0.2569 |
| Kaggle-law reranker fusion | 0.4713 | 0.5451 | 0.3361 | 0.3921 | 0.4280 |
| Clean 1000-question reranker | 0.0820 | 0.1107 | 0.0410 | 0.0573 | 0.0696 |

The hybrid system improves over the dense FAISS baseline on all retrieval metrics. This shows that BM25 contributes useful lexical matching, especially for legal questions where exact law terms and article-specific expressions matter.

We then performed a grid search over the hybrid retrieval parameters. The best configuration was:

```text
alpha = 0.70
dense_candidates = 300
bm25_candidates = 100
preliminary_top_k = 50
metadata_profile = none
```

This improved Recall@10, Top-1 accuracy, MRR, and nDCG@10 compared with the initial hybrid setting. Recall@5 stayed the same. Since the best result did not require metadata boosting, the improvement comes from retrieval score weighting and candidate selection rather than benchmark-specific law/article hints.

We also tested embedding fine-tuning using the clean question-positive-hard-negative training data. This experiment performed much worse than the original BGE-M3 embedding model. The likely reason is that the fine-tuning data was too small and not sufficiently representative of the final gold benchmark. Since BGE-M3 is already a strong multilingual retrieval model, narrow fine-tuning on noisy or mismatched pairs caused overfitting and reduced general retrieval quality.

We also tested a stronger off-the-shelf embedding model, `Qwen/Qwen3-Embedding-0.6B`, without fine-tuning. Qwen3 hybrid retrieval slightly improved Top-1 accuracy and MRR compared with BGE-M3 hybrid, but BGE-M3 hybrid still achieved better Recall@5, Recall@10, and nDCG@10. Since legal RAG needs the correct source to appear in the retrieved context, we prioritize recall and citation coverage over a small Top-1 gain. Therefore, BGE-M3 hybrid remains the safer default retrieval configuration, while Qwen3-Embedding-0.6B is a strong alternative candidate.

The grid-optimized BGE-M3 hybrid system is currently the best-performing retrieval configuration by Recall@10, Top-1 accuracy, MRR, and nDCG@10. Embedding fine-tuning and all fine-tuned reranker variants reduced gold benchmark performance. Score fusion reduced the damage for some reranker variants because it preserved part of the hybrid retrieval score, but it still did not outperform the optimized BGE-M3 hybrid retrieval. Therefore, the current final retrieval choice is grid-optimized BGE-M3 hybrid retrieval, while embedding tuning, Qwen3 embedding replacement, and reranking are reported as ablation experiments.

### 6.2 QA Metrics

| System | EM | F1 | ROUGE-L | Faithfulness | Citation Accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline RAG | TBD | TBD | TBD | TBD | TBD |
| RAG + reranker | TBD | TBD | TBD | TBD | TBD |
| Fully optimized RAG | TBD | TBD | TBD | TBD | TBD |

## 7. Hallucination and Citation Analysis Plan

Hallucination analysis will focus on cases where:

1. The retrieved context does not contain the answer.
2. The LLM produces legal claims not supported by the retrieved article.
3. The answer cites the wrong law or article.
4. The model answers when the context is insufficient.

For citation consistency, we will check whether the final answer cites the same law and article as the retrieved supporting context.

## 8. Error Analysis So Far

### 8.1 Colloquial Paraphrase Failure and Fix

The labor-law query "işçi 2 gün işe gelmezse ne olur?" is a difficult paraphrase. The legally relevant concept is:

```text
izinsiz ve haklı sebep olmaksızın ardı ardına iki işgünü işe devam etmeme
```

The user query does not explicitly mention:

```text
haklı nedenle fesih
devamsızlık
işverenin derhal fesih hakkı
```

In the first reranker version, the model sometimes ranked leave-related or wage-related articles above the correct termination article. This motivated the hard-negative mining update described in Section 4.4. After retraining with curated examples and hard negatives, the same query ranks İş Kanunu Madde 25 first. This suggests that targeted hard negatives are useful for improving colloquial legal paraphrases.

### 8.2 Related-Article Confusion

For criminal-law questions, the reranker correctly ranks TCK Article 81 first, but related articles such as Article 82, Article 85, and insurance-related killing provisions also appear in the top results. This is expected because they contain semantically related terms. The top-1 ranking is correct in the tested example.

### 8.3 Reranker Generalization Issue

The benchmark results show that the V2 reranker performs worse than hybrid retrieval overall, even though it fixes some selected qualitative examples. This likely comes from a distribution mismatch: the reranker was trained on external query-passage pairs, mined hard negatives, and a small set of curated colloquial questions, while the gold benchmark contains many direct article-specific questions. In these cases, hybrid retrieval already ranks exact law/article matches well, and the reranker can sometimes demote the correct article because it learned broader semantic relevance rather than exact citation matching.

The Kaggle-law and clean-1000 reranker experiments confirmed the same issue. The training data was either generated from context-specific QA rows or from explicit article-title questions, while the gold benchmark includes a different mixture of natural legal questions and citation-oriented questions. As a result, the reranker learned relevance signals that did not transfer well to the final benchmark.

For this reason, the current best retrieval configuration is hybrid retrieval. The reranker should not be used as the default final ranking component unless a new training set is created from benchmark-like questions with manually verified positive article IDs and carefully mined hard negatives.

### 8.4 Embedding Fine-Tuning Overfitting

The fine-tuned embedding model performed worse than the original BGE-M3 model. Dense retrieval Recall@5 dropped from 0.5533 to 0.1066, and hybrid retrieval Recall@5 dropped from 0.6311 to 0.2459. This suggests that the embedding model overfit to the limited training pairs and lost some of its broader multilingual/legal semantic retrieval ability.

For the final system, we therefore keep the original BGE-M3 embeddings and use BM25 hybrid retrieval. The embedding fine-tuning result is still useful as an ablation experiment because it shows that optimization does not automatically improve the system unless the training data is large, clean, and aligned with the final benchmark.

### 8.5 Stronger Off-the-Shelf Embedding Model

We evaluated `Qwen/Qwen3-Embedding-0.6B` as a stronger ready-made embedding model without fine-tuning. The dense Qwen3 model performed below the BGE-M3 dense baseline. With BM25 hybrid retrieval, Qwen3 became competitive and slightly improved Top-1 accuracy and MRR:

```text
BGE-M3 hybrid Top-1: 0.4631, MRR: 0.5348
Qwen3 hybrid Top-1: 0.4754, MRR: 0.5369
```

However, Qwen3 hybrid was lower on recall-oriented metrics:

```text
BGE-M3 hybrid Recall@5: 0.6311, Recall@10: 0.6680, nDCG@10: 0.5674
Qwen3 hybrid Recall@5: 0.6148, Recall@10: 0.6434, nDCG@10: 0.5630
```

For legal QA, missing the correct source is more harmful than ranking a correct source one position higher in some cases. Therefore, we keep BGE-M3 hybrid as the default, but Qwen3 remains a useful alternative for future tuning or ensemble experiments.

### 8.6 Hybrid Parameter Grid Search

The initial hybrid retrieval setup used `alpha=0.55`, meaning dense retrieval received 55% of the combined score and BM25 received 45%. Grid search showed that a stronger dense weight worked better on the gold benchmark:

```text
alpha = 0.70
dense_candidates = 300
bm25_candidates = 100
preliminary_top_k = 50
metadata_profile = none
```

The optimized setting improved the ranking of correct articles without relying on metadata boost. This is useful because it is less tied to benchmark questions that explicitly mention law names or article numbers. The result suggests that BM25 is still useful, but the current corpus and BGE-M3 embeddings benefit from giving dense retrieval more weight while keeping BM25 as a lexical correction signal.

## 9. Next Steps

1. Use grid-optimized BGE-M3 hybrid retrieval as the current default retrieval configuration.
2. Do not use the current fine-tuned embedding model in the final pipeline; keep it as an ablation result.
3. Keep Qwen3-Embedding-0.6B as an alternative embedding ablation, but use optimized BGE-M3 hybrid as the default because it has better recall/ranking balance.
4. Improve reranker training with more current-corpus positives and benchmark-like questions before using it as the default ranker.
5. Integrate the top hybrid-retrieved contexts into the LLM answer generation stage.
6. Evaluate answer faithfulness and citation accuracy.
7. Prepare ablation table for the required systems:

```text
Baseline RAG
+ Embedding tuning
+ Reranker
+ LLM fine-tuning
Fully optimized system
```

## 10. Current Conclusion

So far, the project has a working Turkish legal retrieval corpus, a BGE-M3 FAISS index, optimized hybrid retrieval, embedding fine-tuning experiments, a Qwen3 embedding replacement experiment, and a fine-tuned BERTurk cross-encoder reranker. The benchmark evaluation shows that grid-optimized BGE-M3 hybrid retrieval is currently the strongest retrieval configuration for recall-oriented legal retrieval. Qwen3 hybrid is competitive and slightly better than the initial BGE-M3 hybrid in Top-1/MRR, but it is still below the optimized BGE-M3 hybrid configuration. The fine-tuned embedding model and reranker variants performed worse on the gold benchmark, so they should be reported as ablation experiments rather than used as final defaults.

## Appendix A. Reranker Dataset Experiment Summary

| Experiment | Dataset | Hard negatives | Gold benchmark result |
|---|---|---|---|
| External reranker.jsonl | External positive/negative query-passage pairs | Already contained negative examples | Used as initial reranker source |
| External augmented reranker | External reranker.jsonl + curated queries + current-corpus positives | Added by us using dense + BM25 mining | Pure rerank Recall@5 0.4877, fusion Recall@5 0.6148 |
| Kaggle law reranker | Kaggle QA/context rows converted to reranker pairs | Added by us using BM25 | Pure rerank Recall@5 0.2828, fusion Recall@5 0.4713 |
| Clean 1000-question reranker | 1,000 selected question-parent pairs from HF, Kaggle, and corpus titles | Added by us using BM25 | Pure rerank Recall@5 0.0820 |
| Fine-tuned embedding | Clean question-positive-hard-negative pairs | Used hard negatives from clean reranker data | Dense Recall@5 0.1066, hybrid Recall@5 0.2459 |
| Qwen3 embedding replacement | `Qwen/Qwen3-Embedding-0.6B` without fine-tuning | No training; corpus was re-embedded | Dense Recall@5 0.5287, hybrid Recall@5 0.6148 |
| Hybrid parameter grid search | BGE-M3 dense + BM25, no training | No new training; retrieval parameters optimized | Recall@5 0.6311, Recall@10 0.6762, Top-1 0.4795, nDCG@10 0.5772 |

The main lesson is that retrieval optimization depends more on benchmark-aligned supervision than on pair count. A smaller clean dataset is preferable to a noisy dataset, but it must still match the final evaluation distribution. In our current experiments, hybrid retrieval with the original BGE-M3 embeddings is the most reliable final retrieval configuration.
