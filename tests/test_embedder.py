import numpy as np
from unittest.mock import Mock
from src.embedder import embed_batch, create_embeddings, embed_query


def test_embed_batch_calls_openai_correctly(mocker, mock_embedding_response, embedding_dim):
    texts = ["text one", "text two", "text three"]
    response = mock_embedding_response(3)
    mock_create = mocker.patch('src.embedder.client.embeddings.create', return_value=response)

    result = embed_batch(texts)

    call_args = mock_create.call_args
    assert call_args.kwargs['input'] == texts
    assert call_args.kwargs['model'] == 'text-embedding-3-small'
    assert len(result) == 3


def test_embed_batch_returns_embeddings_list(mocker, mock_embedding_response):
    texts = ["one", "two", "three"]
    mocker.patch('src.embedder.client.embeddings.create', return_value=mock_embedding_response(3))

    result = embed_batch(texts)

    assert isinstance(result, list)
    assert len(result) == 3


def test_create_embeddings_single_batch(mocker, embedding_dim):
    texts = [f"text {i}" for i in range(50)]

    mock_batch = mocker.patch(
        'src.embedder.embed_batch',
        return_value=[np.random.randn(embedding_dim).tolist() for _ in range(50)]
    )

    result = create_embeddings(texts)

    assert mock_batch.call_count == 1
    assert isinstance(result, np.ndarray)
    assert result.shape == (50, embedding_dim)


def test_create_embeddings_multiple_batches(mocker, embedding_dim):
    texts = [f"text {i}" for i in range(250)]

    mock_batch = mocker.patch(
        'src.embedder.embed_batch',
        side_effect=lambda batch: [np.random.randn(embedding_dim).tolist() for _ in range(len(batch))]
    )

    result = create_embeddings(texts)

    assert mock_batch.call_count == 3
    assert result.shape == (250, embedding_dim)


def test_embed_query_returns_array(mocker, mock_embedding_response, embedding_dim):
    query = "What is machine learning?"
    mock_create = mocker.patch(
        'src.embedder.client.embeddings.create',
        return_value=mock_embedding_response(1)
    )

    result = embed_query(query)

    assert mock_create.call_args.kwargs['input'] == [query]
    assert isinstance(result, np.ndarray)
    assert result.shape == (embedding_dim,)


def test_create_embeddings_correct_dtype(mocker, embedding_dim):
    mocker.patch('src.embedder.embed_batch', return_value=[[0.1] * embedding_dim])

    result = create_embeddings(["sample"])

    assert result.dtype in [np.float32, np.float64]


def test_create_embeddings_empty_input(mocker):
    mocker.patch('src.embedder.embed_batch', return_value=[])

    result = create_embeddings([])

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 0


def test_embed_batch_extracts_embeddings(mocker, embedding_dim):
    expected = [0.1, 0.2, 0.3] + [0.0] * (embedding_dim - 3)
    response = Mock()
    response.data = [Mock(embedding=expected)]
    mocker.patch('src.embedder.client.embeddings.create', return_value=response)

    result = embed_batch(["test"])

    assert result[0] == expected
