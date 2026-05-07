"""
MERGE DATASETS: Kaggle + HuggingFace
Create comprehensive document database
Re-embed with FAISS
"""

import json
import numpy as np
from pathlib import Path

print("=" * 70)
print("🔄 MERGING DATASETS: Kaggle + HuggingFace")
print("=" * 70)

# ============================================================================
# PART 1: Load Existing Kaggle Documents
# ============================================================================
print("\n📖 PART 1: Loading Kaggle documents...")

kaggle_path = Path("data/processed/documents.json")
with open(kaggle_path, 'r', encoding='utf-8') as f:
    kaggle_docs = json.load(f)

print(f"✅ Loaded {len(kaggle_docs)} Kaggle documents")

# ============================================================================
# PART 2: Load HuggingFace Q-A Pairs (Convert to Documents)
# ============================================================================
print("\n" + "=" * 70)
print("📥 PART 2: Loading HuggingFace Q-A pairs...")
print("=" * 70)

hf_path = Path("data/processed/hf_lawchatbot.json")
with open(hf_path, 'r', encoding='utf-8') as f:
    hf_data = json.load(f)

# Extract samples
if isinstance(hf_data, dict) and 'samples' in hf_data:
    hf_samples = hf_data['samples']
else:
    hf_samples = hf_data if isinstance(hf_data, list) else []

print(f"✅ Loaded {len(hf_samples)} HuggingFace samples")

# Convert HF Q-A to documents (use answers as documents)
hf_docs = []
for idx, sample in enumerate(hf_samples):
    # Get text from various possible column names
    text = sample.get('Cevap', sample.get('Answer', sample.get('cevap', '')))
    question = sample.get('Soru', sample.get('Question', sample.get('soru', '')))
    
    if text:  # Only add if has text
        doc = {
            "id": f"hf_doc_{idx:05d}",
            "text": str(text),
            "source": "HuggingFace",
            "type": "qa_answer",
            "question": str(question),
        }
        hf_docs.append(doc)

print(f"✅ Converted {len(hf_docs)} HuggingFace answers to documents")

# ============================================================================
# PART 3: Merge Datasets
# ============================================================================
print("\n" + "=" * 70)
print("🔀 PART 3: Merging datasets...")
print("=" * 70)

merged_docs = kaggle_docs + hf_docs
print(f"✅ Merged: {len(kaggle_docs)} + {len(hf_docs)} = {len(merged_docs)} documents")

# Save merged
merged_path = Path("data/processed/documents_merged.json")
with open(merged_path, 'w', encoding='utf-8') as f:
    json.dump(merged_docs, f, ensure_ascii=False, indent=2)

size_mb = merged_path.stat().st_size / (1024 * 1024)
print(f"✅ Saved: {merged_path}")
print(f"   Size: {size_mb:.2f} MB")

# ============================================================================
# PART 4: Re-embed All Documents with Merged Dataset
# ============================================================================
print("\n" + "=" * 70)
print("🧠 PART 4: Embedding merged documents (FAISS)...")
print("=" * 70)

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    
    # Load model (should be cached now)
    print("⏳ Loading LaBSE model...")
    model = SentenceTransformer('sentence-transformers/LaBSE')
    print(f"✅ Model loaded (Device: {model.device})")
    
    # Extract texts
    doc_texts = [doc['text'] for doc in merged_docs]
    
    # Batch embedding
    print(f"⏳ Embedding {len(doc_texts)} documents...")
    batch_size = 32
    embeddings_list = []
    
    for i in range(0, len(doc_texts), batch_size):
        batch = doc_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings_list.append(batch_embeddings)
        
        if (i + batch_size) % 2000 == 0 or (i + batch_size) >= len(doc_texts):
            print(f"   Embedded: {min(i + batch_size, len(doc_texts))}/{len(doc_texts)}")
    
    embeddings = np.vstack(embeddings_list).astype('float32')
    print(f"✅ Created embeddings: {embeddings.shape}")
    
    # Create FAISS index
    print("⏳ Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"✅ FAISS index created!")
    print(f"   Index size: {index.ntotal}")
    
    # Save index
    index_path = Path("models/faiss_merged_index.bin")
    faiss.write_index(index, str(index_path))
    print(f"✅ Saved: {index_path}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# ============================================================================
# PART 5: Update BM25 Index
# ============================================================================
print("\n" + "=" * 70)
print("📚 PART 5: Creating BM25 index (merged)...")
print("=" * 70)

try:
    from rank_bm25 import BM25Okapi
    
    print("⏳ Tokenizing merged documents...")
    tokenized_docs = []
    for doc in merged_docs:
        tokens = doc['text'].lower().split()
        tokenized_docs.append(tokens)
    
    print("⏳ Building BM25 model...")
    bm25_model = BM25Okapi(tokenized_docs)
    
    print(f"✅ BM25 index created!")
    print(f"   Indexed docs: {len(tokenized_docs)}")
    
except Exception as e:
    print(f"⚠️  BM25 warning: {e}")

# ============================================================================
# PART 6: Test Merged System
# ============================================================================
print("\n" + "=" * 70)
print("🧪 PART 6: Testing merged retrieval system...")
print("=" * 70)

print("\n📝 Test Queries (from HuggingFace):\n")

test_queries = []
for sample in hf_samples[:3]:  # First 3
    q = sample.get('Soru', sample.get('Question', ''))
    if q:
        test_queries.append(q)

for i, query in enumerate(test_queries):
    print(f"Query {i+1}: {query[:70]}...")
    
    # Dense search
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, 5)
    
    print(f"  Top 5 Results:")
    for rank, idx in enumerate(indices[0]):
        doc = merged_docs[idx]
        print(f"    {rank+1}. [{doc['id']}] {doc['text'][:60]}...")
    print()

# ============================================================================
# PART 7: Summary
# ============================================================================
print("=" * 70)
print("✅ MERGED SYSTEM READY!")
print("=" * 70)

print(f"""
📊 Merged Database:
   - Kaggle articles: {len(kaggle_docs):,}
   - HuggingFace Q-A: {len(hf_docs):,}
   - Total documents: {len(merged_docs):,}

📁 New Files:
   - data/processed/documents_merged.json
   - models/faiss_merged_index.bin

🎯 Next:
   - FAZE 3: Create Gold Test Benchmark
   - Use merged retrieval system for testing

⏱️  Performance Impact:
   + Better retrieval (larger DB)
   + More comprehensive coverage
   - Slightly slower search (marginal)
""")

print("=" * 70)
