import json
import numpy as np
from src.vector_store import create_index, save_index, load_index
from src.retriever import retrieve

chunks = []
embeddings_list = []

for filename in ["climate_change", "ai_overview"]:
    with open(f"storage/{filename}_chunks.json", "r") as f:
        chunks.extend(json.load(f))

    with open(f"storage/{filename}_embeddings.json", "r") as f:
        embeddings_list.extend(json.load(f))

embeddings = np.array(embeddings_list)

print(f"Loaded {len(chunks)} chunks\n")

index = create_index(embeddings)
save_index(index, chunks)

loaded_index, loaded_chunks = load_index()

queries = [
    "What is climate change?",
    "What is artificial intelligence?",
    "Tell me about greenhouse gases"
]

for query in queries:
    print(f"Query: {query}")
    results = retrieve(query, loaded_index, loaded_chunks, k=3)

    for i, result in enumerate(results):
        print(f"\n  Result {i+1} (distance={result['distance']:.4f}):")
        print(f"  {result['chunk'][:80]}...")

    print()
