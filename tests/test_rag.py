import pytest
from unittest.mock import Mock
from src.rag import query_rag


def test_query_rag_calls_retrieve(mocker, faiss_index, sample_chunks):
    query = "What is climate change?"
    results = [{"chunk": "Climate change is a global challenge.", "distance": 0.5, "index": 0}]

    mock_retrieve = mocker.patch('src.rag.retrieve', return_value=results)
    mocker.patch('src.rag.client.chat.completions.create',
                 return_value=Mock(choices=[Mock(message=Mock(content="Answer"))]))

    query_rag(query, faiss_index, sample_chunks, k=3)

    call_args = mock_retrieve.call_args
    assert call_args[0] == (query, faiss_index, sample_chunks, 3)


def test_query_rag_formats_context(mocker, faiss_index, sample_chunks):
    results = [
        {"chunk": "First chunk.", "distance": 0.3, "index": 0},
        {"chunk": "Second chunk.", "distance": 0.5, "index": 1},
        {"chunk": "Third chunk.", "distance": 0.7, "index": 2},
    ]

    mocker.patch('src.rag.retrieve', return_value=results)
    mock_create = mocker.patch('src.rag.client.chat.completions.create',
                               return_value=Mock(choices=[Mock(message=Mock(content="Answer"))]))

    query_rag("test", faiss_index, sample_chunks, k=3)

    messages = mock_create.call_args.kwargs['messages']
    user_content = messages[1]['content']

    assert "[1] First chunk." in user_content
    assert "[2] Second chunk." in user_content
    assert "[3] Third chunk." in user_content


def test_query_rag_calls_openai(mocker, faiss_index, sample_chunks):
    mocker.patch('src.rag.retrieve', return_value=[{"chunk": "test", "distance": 0.5, "index": 0}])
    mock_create = mocker.patch('src.rag.client.chat.completions.create',
                               return_value=Mock(choices=[Mock(message=Mock(content="Answer"))]))

    query_rag("test", faiss_index, sample_chunks, k=3)

    assert mock_create.call_args.kwargs['model'] == 'gpt-4o-mini'
    assert mock_create.call_args.kwargs['temperature'] == 0.7


def test_query_rag_constructs_messages(mocker, faiss_index, sample_chunks):
    query = "What is renewable energy?"
    mocker.patch('src.rag.retrieve', return_value=[{"chunk": "Solar power.", "distance": 0.4, "index": 0}])
    mock_create = mocker.patch('src.rag.client.chat.completions.create',
                               return_value=Mock(choices=[Mock(message=Mock(content="Answer"))]))

    query_rag(query, faiss_index, sample_chunks, k=3)

    messages = mock_create.call_args.kwargs['messages']
    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert messages[1]['role'] == 'user'
    assert 'Context:' in messages[1]['content']
    assert f'Question: {query}' in messages[1]['content']


def test_query_rag_returns_structured_response(mocker, faiss_index, sample_chunks):
    results = [
        {"chunk": "chunk 1", "distance": 0.3, "index": 0},
        {"chunk": "chunk 2", "distance": 0.5, "index": 1},
    ]

    mocker.patch('src.rag.retrieve', return_value=results)
    mocker.patch('src.rag.client.chat.completions.create',
                 return_value=Mock(choices=[Mock(message=Mock(content="Answer"))]))

    result = query_rag("test", faiss_index, sample_chunks, k=2)

    assert 'question' in result
    assert 'answer' in result
    assert 'sources' in result
    assert result['question'] == "test"
    assert result['answer'] == "Answer"
    assert len(result['sources']) == 2


def test_query_rag_truncates_sources(mocker, faiss_index, sample_chunks):
    long_chunk = "A" * 150
    mocker.patch('src.rag.retrieve', return_value=[{"chunk": long_chunk, "distance": 0.5, "index": 0}])
    mocker.patch('src.rag.client.chat.completions.create',
                 return_value=Mock(choices=[Mock(message=Mock(content="Answer"))]))

    result = query_rag("test", faiss_index, sample_chunks, k=1)

    source_chunk = result['sources'][0]['chunk']
    assert len(source_chunk) == 103
    assert source_chunk.endswith("...")


def test_query_rag_respects_k(mocker, faiss_index, sample_chunks):
    results = [{"chunk": f"chunk {i}", "distance": 0.1 * i, "index": i} for i in range(7)]

    mock_retrieve = mocker.patch('src.rag.retrieve', return_value=results)
    mocker.patch('src.rag.client.chat.completions.create',
                 return_value=Mock(choices=[Mock(message=Mock(content="Answer"))]))

    result = query_rag("test", faiss_index, sample_chunks, k=7)

    assert mock_retrieve.call_args[0][3] == 7
    assert len(result['sources']) == 7


def test_query_rag_sources_have_metadata(mocker, faiss_index, sample_chunks):
    mocker.patch('src.rag.retrieve', return_value=[{"chunk": "Climate change.", "distance": 0.35, "index": 2}])
    mocker.patch('src.rag.client.chat.completions.create',
                 return_value=Mock(choices=[Mock(message=Mock(content="Answer"))]))

    result = query_rag("test", faiss_index, sample_chunks, k=1)

    source = result['sources'][0]
    assert source['index'] == 2
    assert source['distance'] == 0.35


def test_query_rag_extracts_llm_response(mocker, faiss_index, sample_chunks):
    expected = "This is the expected answer."

    mocker.patch('src.rag.retrieve', return_value=[{"chunk": "test", "distance": 0.5, "index": 0}])
    mocker.patch('src.rag.client.chat.completions.create',
                 return_value=Mock(choices=[Mock(message=Mock(content=expected))]))

    result = query_rag("test", faiss_index, sample_chunks, k=1)

    assert result['answer'] == expected


def test_query_rag_handles_no_context(mocker, faiss_index, sample_chunks):
    mocker.patch('src.rag.retrieve', return_value=[])
    mocker.patch('src.rag.client.chat.completions.create',
                 return_value=Mock(choices=[Mock(message=Mock(content="No context."))]))

    result = query_rag("test", faiss_index, sample_chunks, k=3)

    assert result['sources'] == []
    assert result['answer'] == "No context."
