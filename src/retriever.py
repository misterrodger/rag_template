from src.embedder import embed_query
from src.vector_store import search


def retrieve(query: str, index, chunks: list[str], k: int = 5):
    query_embedding = embed_query(query)
    distances, indices = search(index, query_embedding, k)

    return [
        {
            "chunk": chunks[idx],
            "distance": float(dist),
            "index": int(idx)
        }
        for idx, dist in zip(indices, distances)
    ]
