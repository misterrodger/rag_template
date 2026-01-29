import json
import numpy as np
from src.vector_store import create_index, save_index, load_index
from src.rag import query_rag

chunks = []
embeddings_list = []

for filename in ["climate_change", "ai_overview"]:
    with open(f"storage/{filename}_chunks.json", "r") as f:
        chunks.extend(json.load(f))

    with open(f"storage/{filename}_embeddings.json", "r") as f:
        embeddings_list.extend(json.load(f))

embeddings = np.array(embeddings_list)

index = create_index(embeddings)
save_index(index, chunks)

loaded_index, loaded_chunks = load_index()

queries = [
    "What is climate change?",
    "What is artificial intelligence?",
    "Tell me about greenhouse gases"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    result = query_rag(query, loaded_index, loaded_chunks, k=3)

    print(f"\nAnswer:\n{result['answer']}")

    print(f"\nSources:")
    for source in result['sources']:
        print(f"  - Index {source['index']} (distance={source['distance']:.4f})")
        print(f"    {source['chunk']}\n")
