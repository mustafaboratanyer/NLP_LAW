"""
FAZE 2: BASELINE RETRIEVAL SYSTEM
- Dense Retrieval (FAISS + LaBSE embedding)
- BM25 (Keyword-based)
- Hybrid (Dense + BM25)
- Metrics: Recall@5, @10, MRR
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

print("=" * 70)
print("🔍 FAZE 2: BASELINE RETRIEVAL SYSTEM")
print("=" * 70)

# ============================================================================
# PART 1: Load Documents
# ============================================================================
print("\n📖 PART 1: Loading documents.json...")

doc_path = Path("data/processed/documents.json")
with open(doc_path, 'r', encoding='utf-8') as f:
    documents = json.load(f)

print(f"✅ Loaded {len(documents)} documents")
print(f"   Sample doc: {documents[0]}")

# ============================================================================
# PART 2: Setup Embedding Model (LaBSE - Turkish multilingual)
# ============================================================================
print("\n" + "=" * 70)
print("🧠 PART 2: Setting up embedding model (LaBSE)...")
print("=" * 70)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    
    print("⏳ Loading LaBSE model (first time = slow)...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    
    print(f"✅ Model loaded!")
    print(f"   Device: {model.device}")
    print(f"   Embedding dim: {model.get_sentence_embedding_dimension()}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# ============================================================================
# PART 3: Create Embeddings + FAISS Index
# ============================================================================
print("\n" + "=" * 70)
print("🔢 PART 3: Creating embeddings (FAISS index)...")
print("=" * 70)

try:
    import faiss
    
    print("⏳ Embedding documents (this may take 1-2 minutes)...")
    
    # Extract text from documents
    doc_texts = [doc['text'] for doc in documents]
    
    # Batch embedding for speed
    batch_size = 32
    embeddings_list = []
    
    for i in range(0, len(doc_texts), batch_size):
        batch = doc_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings_list.append(batch_embeddings)
        
        if (i + batch_size) % 1000 == 0:
            print(f"   Embedded: {i + batch_size}/{len(doc_texts)}")
    
    embeddings = np.vstack(embeddings_list).astype('float32')
    
    print(f"✅ Created embeddings: {embeddings.shape}")
    
    # Create FAISS index
    print("⏳ Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(embeddings)
    
    print(f"✅ FAISS index created!")
    print(f"   Index size: {index.ntotal}")
    
    # Save for later use
    faiss.write_index(index, "models/faiss_dense_index.bin")
    print(f"   Saved: models/faiss_dense_index.bin")
    
except ImportError:
    print("❌ Missing: pip install faiss-cpu")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# ============================================================================
# PART 4: BM25 Index
# ============================================================================
print("\n" + "=" * 70)
print("📚 PART 4: Creating BM25 index...")
print("=" * 70)

try:
    from rank_bm25 import BM25Okapi
    
    print("⏳ Tokenizing documents for BM25...")
    
    # Simple Turkish tokenization (split by space and punctuation)
    tokenized_docs = []
    for doc in documents:
        # Simple tokenization
        tokens = doc['text'].lower().split()
        tokenized_docs.append(tokens)
    
    print(f"⏳ Building BM25 model...")
    bm25_model = BM25Okapi(tokenized_docs)
    
    print(f"✅ BM25 index created!")
    print(f"   Indexed docs: {len(tokenized_docs)}")
    
except ImportError:
    print("❌ Missing: pip install rank-bm25")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# ============================================================================
# PART 5: Test Retrieval on Sample Queries
# ============================================================================
print("\n" + "=" * 70)
print("🧪 PART 5: Testing retrieval...")
print("=" * 70)

# Create some test queries (from HuggingFace dataset)
print("⏳ Loading test queries from HuggingFace dataset...")

hf_path = Path("data/processed/hf_lawchatbot.json")
with open(hf_path, 'r', encoding='utf-8') as f:
    hf_data = json.load(f)

# Get 20 test queries
test_queries = []
if isinstance(hf_data, dict) and 'samples' in hf_data:
    for sample in hf_data['samples'][:20]:
        if 'Soru' in sample or 'Question' in sample:
            q = sample.get('Soru', sample.get('Question', ''))
            test_queries.append(q)

print(f"✅ Loaded {len(test_queries)} test queries")

# Helper function: Retrieve with Dense
def dense_retrieve(query_text, top_k=10):
    query_embedding = model.encode([query_text]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return [(idx, documents[idx]) for idx in indices[0]]

# Helper function: Retrieve with BM25
def bm25_retrieve(query_text, top_k=10):
    query_tokens = query_text.lower().split()
    scores = bm25_model.get_scores(query_tokens)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(idx, documents[idx]) for idx in top_indices]

# Helper function: Hybrid Retrieve
def hybrid_retrieve(query_text, top_k=10, alpha=0.5):
    # Dense results
    query_embedding = model.encode([query_text]).astype('float32')
    _, dense_indices = index.search(query_embedding, top_k * 2)
    dense_indices = dense_indices[0]
    
    # BM25 results
    query_tokens = query_text.lower().split()
    scores = bm25_model.get_scores(query_tokens)
    bm25_indices = np.argsort(scores)[-(top_k * 2):][::-1]
    
    # Combine scores
    combined_scores = {}
    for rank, idx in enumerate(dense_indices):
        score = 1.0 / (rank + 1)
        combined_scores[idx] = combined_scores.get(idx, 0) + alpha * score
    
    for rank, idx in enumerate(bm25_indices):
        score = 1.0 / (rank + 1)
        combined_scores[idx] = combined_scores.get(idx, 0) + (1 - alpha) * score
    
    # Sort and return
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [(idx, documents[idx]) for idx, score in sorted_results]

# Test each method
print("\n📊 RETRIEVAL RESULTS (First 5 queries):\n")

for i, query in enumerate(test_queries[:5]):
    print(f"Query {i+1}: {query[:70]}...")
    print("-" * 70)
    
    # Dense
    dense_results = dense_retrieve(query, top_k=5)
    print(f"  Dense (Top 5):")
    for rank, (idx, doc) in enumerate(dense_results):
        print(f"    {rank+1}. [doc_{idx:05d}] {doc['text'][:60]}...")
    
    # BM25
    bm25_results = bm25_retrieve(query, top_k=5)
    print(f"  BM25 (Top 5):")
    for rank, (idx, doc) in enumerate(bm25_results):
        print(f"    {rank+1}. [doc_{idx:05d}] {doc['text'][:60]}...")
    
    # Hybrid
    hybrid_results = hybrid_retrieve(query, top_k=5)
    print(f"  Hybrid (Top 5):")
    for rank, (idx, doc) in enumerate(hybrid_results):
        print(f"    {rank+1}. [doc_{idx:05d}] {doc['text'][:60]}...")
    
    print()

# ============================================================================
# PART 6: Save Configuration
# ============================================================================
print("=" * 70)
print("💾 PART 6: Saving configuration...")
print("=" * 70)

config = {
    "system": "Baseline RAG",
    "embedding_model": "sentence-transformers/LaBSE",
    "retrieval_methods": ["dense", "bm25", "hybrid"],
    "vector_db": "FAISS",
    "document_count": len(documents),
    "embedding_dimension": dimension,
    "models_saved": {
        "dense": "models/faiss_dense_index.bin",
        "bm25": "In memory (BM25Okapi)"
    }
}

Path("models").mkdir(exist_ok=True)
config_path = Path("configs/retrieval_config.json")
config_path.parent.mkdir(exist_ok=True)

with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(f"✅ Config saved: {config_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ FAZE 2 BASELINE RETRIEVAL - COMPLETE!")
print("=" * 70)

print("""
📊 Systems Ready:
   ✅ Dense Retrieval (FAISS + LaBSE)
   ✅ BM25 Retrieval
   ✅ Hybrid Retrieval (Dense + BM25)

📁 Generated Files:
   - models/faiss_dense_index.bin (FAISS index)
   - configs/retrieval_config.json (config)

🎯 Next Steps:
   1. FAZE 3: Create Gold Test Benchmark (150-300 Q-A-Doc triplets)
   2. FAZE 4: Fine-tuning (Embedding + LLM)
   3. FAZE 5: Evaluation (R+A+G metrics)

⏱️  Estimated timeline:
   - FAZE 3: 2-3 hafta (manual gold benchmark)
   - FAZE 4: 2-3 hafta (fine-tuning)
   - FAZE 5: 1 hafta (evaluation + report)
""")

print("=" * 70)
