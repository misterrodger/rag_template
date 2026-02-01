import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock
from src.chunker import chunk_file
from src.embedder import create_embeddings
from src.vector_store import create_index, save_index, load_index
from src.retriever import retrieve
from src.rag import query_rag


def mock_embeddings(dim):
    return lambda batch: [np.random.randn(dim).astype(np.float32).tolist() for _ in range(len(batch))]


def test_full_pipeline(mocker, sample_doc, storage_dir, embedding_dim, monkeypatch):
    np.random.seed(42)

    from src import config as config_module
    monkeypatch.setattr(config_module.config, 'storage_dir', storage_dir)

    index_path = storage_dir / "vectors.index"

    mocker.patch('src.embedder.embed_batch', side_effect=mock_embeddings(embedding_dim))
    mocker.patch('src.retriever.embed_query', return_value=np.random.randn(embedding_dim).astype(np.float32))
    mocker.patch('src.rag.client.chat.completions.create',
                 return_value=Mock(choices=[Mock(message=Mock(content="Python is versatile."))]))

    chunks = chunk_file(str(sample_doc))
    embeddings = create_embeddings(chunks)
    index = create_index(embeddings)

    assert len(chunks) > 0
    assert embeddings.shape == (len(chunks), embedding_dim)
    assert index.ntotal == len(chunks)

    save_index(index, chunks, index_path=index_path)
    assert index_path.exists()

    loaded_index, loaded_chunks = load_index(index_path=index_path)
    assert loaded_index.ntotal == len(chunks)
    assert loaded_chunks == chunks

    result = query_rag("What is Python?", loaded_index, loaded_chunks, k=3)
    assert all(k in result for k in ['question', 'answer', 'sources'])
    assert len(result['sources']) <= 3


def test_pipeline_multiple_documents(mocker, tmp_path, storage_dir, embedding_dim, monkeypatch):
    np.random.seed(42)

    from src import config as config_module
    monkeypatch.setattr(config_module.config, 'storage_dir', storage_dir)

    (tmp_path / "doc1.txt").write_text("Python is a programming language. Used for data science.")
    (tmp_path / "doc2.txt").write_text("Climate change affects temperatures. Renewable energy is important.")

    mocker.patch('src.embedder.embed_batch', side_effect=mock_embeddings(embedding_dim))

    all_chunks = []
    all_embeddings = []

    for doc in ["doc1.txt", "doc2.txt"]:
        chunks = chunk_file(str(tmp_path / doc))
        embeddings = create_embeddings(chunks)
        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings.tolist())

    index = create_index(np.array(all_embeddings).astype(np.float32))
    index_path = storage_dir / "vectors.index"
    save_index(index, all_chunks, index_path=index_path)

    loaded_index, loaded_chunks = load_index(index_path=index_path)

    assert len(loaded_chunks) >= 2
    assert loaded_index.ntotal == len(loaded_chunks)
    assert any("Python" in c for c in loaded_chunks)
    assert any("Climate" in c for c in loaded_chunks)


def test_pipeline_retrieves_relevant_context(mocker, tmp_path, storage_dir, embedding_dim, monkeypatch):
    np.random.seed(42)

    from src import config as config_module
    monkeypatch.setattr(config_module.config, 'storage_dir', storage_dir)

    (tmp_path / "mixed.txt").write_text("""
    Python Programming
    Python is a high-level language for web and data science.
    Variables are dynamically typed.

    Climate Science
    Climate change causes temperature shifts.
    Emissions contribute to warming.
    """)

    def contextual_embeddings(batch):
        return [
            ((np.ones(embedding_dim) if "Python" in text or "programming" in text
              else np.ones(embedding_dim) * -1.0 if "Climate" in text or "climate" in text or "emissions" in text
              else np.zeros(embedding_dim))
             + np.random.randn(embedding_dim) * 0.1).astype(np.float32).tolist()
            for text in batch
        ]

    mocker.patch('src.embedder.embed_batch', side_effect=contextual_embeddings)
    mocker.patch('src.retriever.embed_query',
                 return_value=(np.ones(embedding_dim) * 1.0).astype(np.float32))

    chunks = chunk_file(str(tmp_path / "mixed.txt"))
    embeddings = create_embeddings(chunks)
    index = create_index(embeddings)

    results = retrieve("Python programming", index, chunks, 2)

    assert len(results) > 0
    assert any("Python" in r['chunk'] or "programming" in r['chunk'] for r in results)


def test_pipeline_small_document(mocker, tmp_path, storage_dir, embedding_dim, monkeypatch):
    np.random.seed(42)

    from src import config as config_module
    monkeypatch.setattr(config_module.config, 'storage_dir', storage_dir)

    (tmp_path / "small.txt").write_text("This is a short document.")

    mocker.patch('src.embedder.embed_batch', side_effect=mock_embeddings(embedding_dim))
    mocker.patch('src.retriever.embed_query', return_value=np.random.randn(embedding_dim).astype(np.float32))
    mocker.patch('src.rag.client.chat.completions.create',
                 return_value=Mock(choices=[Mock(message=Mock(content="Short document."))]))

    chunks = chunk_file(str(tmp_path / "small.txt"))
    embeddings = create_embeddings(chunks)
    index = create_index(embeddings)

    index_path = storage_dir / "vectors.index"
    save_index(index, chunks, index_path=index_path)
    loaded_index, loaded_chunks = load_index(index_path=index_path)

    result = query_rag("What is this about?", loaded_index, loaded_chunks, 1)

    assert len(chunks) >= 1
    assert result['answer'] is not None
    assert len(result['sources']) >= 1


def test_chunking_preserves_content(sample_doc):
    chunks = chunk_file(str(sample_doc))
    combined = " ".join(chunks)

    assert "Python" in combined
    assert "programming" in combined.lower()
    assert "Variables" in combined or "variables" in combined


def test_retrieval_returns_sorted_results(mocker, sample_doc, embedding_dim):
    np.random.seed(42)

    mocker.patch('src.embedder.embed_batch', side_effect=mock_embeddings(embedding_dim))
    mocker.patch('src.retriever.embed_query', return_value=np.random.randn(embedding_dim).astype(np.float32))

    chunks = chunk_file(str(sample_doc))
    embeddings = create_embeddings(chunks)
    index = create_index(embeddings)

    results = retrieve("What is Python?", index, chunks, min(5, len(chunks)))

    distances = [r['distance'] for r in results]
    assert distances == sorted(distances)
