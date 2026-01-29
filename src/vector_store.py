import json
import faiss
import numpy as np
from pathlib import Path

from src.config import config


def create_index(embeddings: np.ndarray) -> faiss.Index:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    return index


def save_index(index: faiss.Index, chunks: list[str], index_path: Path = config.vector_index_path) -> None:
    faiss.write_index(index, str(index_path))

    chunks_path = config.chunks_path
    with open(chunks_path, 'w') as f:
        json.dump(chunks, f)


def load_index(index_path: Path = config.vector_index_path) -> tuple[faiss.Index, list[str]]:
    index = faiss.read_index(str(index_path))

    chunks_path = config.chunks_path
    with open(chunks_path, 'r') as f:
        chunks = json.load(f)

    return index, chunks


def search(index: faiss.Index, query_embedding: np.ndarray, k: int = 5) -> tuple[list[float], list[int]]:
    query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return distances[0].tolist(), indices[0].tolist()
