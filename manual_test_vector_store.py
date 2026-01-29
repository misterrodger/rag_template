import json
import numpy as np
from src.vector_store import create_index, save_index, load_index, search

chunks = []
embeddings_list = []

for filename in ["climate_change", "ai_overview"]:
    with open(f"storage/{filename}_chunks.json", "r") as f:
        chunks.extend(json.load(f))

    with open(f"storage/{filename}_embeddings.json", "r") as f:
        embeddings_list.extend(json.load(f))

embeddings = np.array(embeddings_list)

print(f"Loaded {len(chunks)} chunks")
print(f"Embeddings shape: {embeddings.shape}\n")

index = create_index(embeddings)
print(f"Created FAISS index")

save_index(index, chunks)
print(f"Saved index and chunks to storage\n")

loaded_index, loaded_chunks = load_index()
print(f"Loaded {len(loaded_chunks)} chunks from storage")
print(f"Index dimension: {loaded_index.ntotal}\n")

query_embedding = embeddings[0]
distances, indices = search(loaded_index, query_embedding, k=3)

print(f"Top 3 results for first embedding:\n")
for i, (idx, dist) in enumerate(zip(indices, distances)):
    print(f"Result {i+1} (index={idx}, distance={dist:.4f}):")
    print(f"{loaded_chunks[idx][:100]}...\n")
