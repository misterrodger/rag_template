import pytest
import numpy as np
from src.retriever import retrieve


def test_retrieve_calls_embed_query(mocker, faiss_index, sample_chunks, embedding_dim):
    query = "What is climate change?"
    query_embedding = np.random.randn(embedding_dim).astype(np.float32)

    mock_embed = mocker.patch('src.retriever.embed_query', return_value=query_embedding)
    mocker.patch('src.retriever.search', return_value=([0.5, 0.7], [0, 1]))

    retrieve(query, faiss_index, sample_chunks, k=3)

    mock_embed.assert_called_once_with(query)


def test_retrieve_calls_search_correctly(mocker, faiss_index, sample_chunks, embedding_dim):
    query_embedding = np.random.randn(embedding_dim).astype(np.float32)

    mocker.patch('src.retriever.embed_query', return_value=query_embedding)
    mock_search = mocker.patch('src.retriever.search', return_value=([0.3, 0.5, 0.8], [1, 3, 5]))

    retrieve("test query", faiss_index, sample_chunks, k=3)

    call_args = mock_search.call_args
    assert call_args[0][0] == faiss_index
    np.testing.assert_array_equal(call_args[0][1], query_embedding)
    assert call_args[0][2] == 3


def test_retrieve_returns_structured_results(mocker, faiss_index, sample_chunks, embedding_dim):
    mocker.patch('src.retriever.embed_query', return_value=np.random.randn(embedding_dim).astype(np.float32))
    mocker.patch('src.retriever.search', return_value=([0.5, 0.7], [2, 4]))

    results = retrieve("test", faiss_index, sample_chunks, k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all('chunk' in r and 'distance' in r and 'index' in r for r in results)


def test_retrieve_respects_k_parameter(mocker, faiss_index, sample_chunks, embedding_dim):
    mocker.patch('src.retriever.embed_query', return_value=np.random.randn(embedding_dim).astype(np.float32))
    mocker.patch('src.retriever.search', return_value=([0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 2, 3, 4]))

    results = retrieve("test", faiss_index, sample_chunks, k=5)

    assert len(results) == 5


def test_retrieve_maps_chunks_correctly(mocker, faiss_index, sample_chunks, embedding_dim):
    mocker.patch('src.retriever.embed_query', return_value=np.random.randn(embedding_dim).astype(np.float32))
    mocker.patch('src.retriever.search', return_value=([0.5, 0.7, 0.9], [2, 5, 1]))

    results = retrieve("test", faiss_index, sample_chunks, k=3)

    assert results[0]['chunk'] == sample_chunks[2]
    assert results[0]['index'] == 2
    assert results[1]['chunk'] == sample_chunks[5]
    assert results[1]['index'] == 5
    assert results[2]['chunk'] == sample_chunks[1]
    assert results[2]['index'] == 1


def test_retrieve_preserves_distances(mocker, faiss_index, sample_chunks, embedding_dim):
    expected_distances = [0.123, 0.456, 0.789]

    mocker.patch('src.retriever.embed_query', return_value=np.random.randn(embedding_dim).astype(np.float32))
    mocker.patch('src.retriever.search', return_value=(expected_distances, [0, 1, 2]))

    results = retrieve("test", faiss_index, sample_chunks, k=3)

    assert all(isinstance(r['distance'], float) for r in results)
    assert [r['distance'] for r in results] == expected_distances


def test_retrieve_handles_k_larger_than_available(mocker, faiss_index, embedding_dim):
    small_chunks = ["chunk1", "chunk2"]

    mocker.patch('src.retriever.embed_query', return_value=np.random.randn(embedding_dim).astype(np.float32))
    mocker.patch('src.retriever.search', return_value=([0.5, 0.7], [0, 1]))

    results = retrieve("test", faiss_index, small_chunks, k=5)

    assert len(results) == 2


def test_retrieve_maintains_search_order(mocker, faiss_index, sample_chunks, embedding_dim):
    mocker.patch('src.retriever.embed_query', return_value=np.random.randn(embedding_dim).astype(np.float32))
    mocker.patch('src.retriever.search', return_value=([0.1, 0.5, 1.2], [3, 0, 6]))

    results = retrieve("test", faiss_index, sample_chunks, k=3)

    distances = [r['distance'] for r in results]
    assert distances[0] < distances[1] < distances[2]


def test_retrieve_single_result(mocker, faiss_index, sample_chunks, embedding_dim):
    mocker.patch('src.retriever.embed_query', return_value=np.random.randn(embedding_dim).astype(np.float32))
    mocker.patch('src.retriever.search', return_value=([0.5], [2]))

    results = retrieve("test", faiss_index, sample_chunks, k=1)

    assert len(results) == 1
    assert results[0]['chunk'] == sample_chunks[2]
    assert results[0]['distance'] == 0.5
    assert results[0]['index'] == 2
