import pytest
import numpy as np
import faiss
import json
from pathlib import Path
from src.vector_store import create_index, save_index, load_index, search


def test_create_index_correct_dimensions():
    embeddings = np.random.randn(10, 128).astype(np.float32)

    index = create_index(embeddings)

    assert isinstance(index, faiss.Index)
    assert index.d == 128
    assert index.ntotal == 10


def test_create_index_various_dimensions():
    for dim in [384, 1536]:
        embeddings = np.random.randn(5, dim).astype(np.float32)

        index = create_index(embeddings)

        assert index.d == dim
        assert index.ntotal == 5


def test_save_index_writes_files(storage_dir, monkeypatch):
    embeddings = np.random.randn(5, 64).astype(np.float32)
    chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
    index = create_index(embeddings)

    from src import config as config_module
    monkeypatch.setattr(config_module.config, 'storage_dir', Path(storage_dir))

    index_path = storage_dir / "vectors.index"
    save_index(index, chunks, index_path=index_path)

    assert index_path.exists()
    assert (storage_dir / "chunks.json").exists()
    assert json.loads((storage_dir / "chunks.json").read_text()) == chunks


def test_load_index_roundtrip(storage_dir, monkeypatch):
    embeddings = np.random.randn(5, 64).astype(np.float32)
    chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
    index = create_index(embeddings)

    from src import config as config_module
    monkeypatch.setattr(config_module.config, 'storage_dir', Path(storage_dir))

    index_path = storage_dir / "vectors.index"
    save_index(index, chunks, index_path=index_path)
    loaded_index, loaded_chunks = load_index(index_path=index_path)

    assert loaded_index.ntotal == 5
    assert loaded_index.d == 64
    assert loaded_chunks == chunks


def test_search_returns_distances_and_indices(sample_embeddings):
    index = create_index(sample_embeddings)
    query = sample_embeddings[0]

    distances, indices = search(index, query, k=3)

    assert isinstance(distances, list)
    assert isinstance(indices, list)
    assert len(distances) == 3
    assert len(indices) == 3


def test_search_exact_match():
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    index = create_index(embeddings)

    distances, indices = search(index, embeddings[1], k=1)

    assert indices[0] == 1
    assert distances[0] < 0.01


def test_search_respects_k(sample_embeddings):
    index = create_index(sample_embeddings)

    for k in [1, 3, 5]:
        distances, indices = search(index, sample_embeddings[0], k=k)
        assert len(distances) == k
        assert len(indices) == k


def test_search_sorted_by_distance(sample_embeddings):
    index = create_index(sample_embeddings)

    distances, _ = search(index, sample_embeddings[0], k=5)

    assert distances == sorted(distances)


def test_create_index_single_vector():
    embeddings = np.random.randn(1, 128).astype(np.float32)

    index = create_index(embeddings)

    assert index.ntotal == 1
    assert index.d == 128


def test_search_returns_floats(sample_embeddings):
    index = create_index(sample_embeddings)

    distances, _ = search(index, sample_embeddings[0], k=3)

    assert all(isinstance(d, (float, np.floating)) for d in distances)


def test_search_returns_valid_indices(sample_embeddings):
    index = create_index(sample_embeddings)

    _, indices = search(index, sample_embeddings[0], k=3)

    for idx in indices:
        assert isinstance(idx, (int, np.integer))
        assert 0 <= idx < len(sample_embeddings)


def test_save_load_preserves_functionality(storage_dir, monkeypatch):
    embeddings = np.random.randn(10, 128).astype(np.float32)
    chunks = [f"chunk {i}" for i in range(10)]
    index = create_index(embeddings)

    from src import config as config_module
    monkeypatch.setattr(config_module.config, 'storage_dir', Path(storage_dir))

    index_path = storage_dir / "vectors.index"
    query = embeddings[3]

    distances_before, indices_before = search(index, query, 3)
    save_index(index, chunks, index_path=index_path)
    loaded_index, _ = load_index(index_path=index_path)
    distances_after, indices_after = search(loaded_index, query, 3)

    assert indices_before == indices_after
    np.testing.assert_allclose(distances_before, distances_after, rtol=1e-5)
