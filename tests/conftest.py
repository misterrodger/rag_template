import pytest
import numpy as np
import faiss
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture
def embedding_dim():
    return 128


@pytest.fixture(autouse=True)
def prevent_api_calls(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")


@pytest.fixture
def sample_texts():
    return [
        "Python is a high-level programming language known for its readability.",
        "Machine learning is a subset of artificial intelligence focused on data-driven learning.",
        "Data structures like lists, dictionaries, and sets are fundamental to computer science.",
        "Climate change refers to long-term shifts in global temperatures and weather patterns.",
        "Renewable energy sources include solar, wind, and hydroelectric power.",
    ]


@pytest.fixture
def sample_embeddings(embedding_dim):
    np.random.seed(42)
    return np.random.randn(5, embedding_dim).astype(np.float32)


@pytest.fixture
def sample_chunks():
    return [
        "Climate change is a global challenge requiring immediate action.",
        "Renewable energy includes solar and wind power technologies.",
        "Carbon emissions from fossil fuels contribute to global warming.",
        "Electric vehicles help reduce transportation sector emissions.",
        "Sustainable practices protect the environment for future generations.",
        "Python is widely used for data science and machine learning applications.",
        "Artificial intelligence is transforming industries worldwide.",
    ]


@pytest.fixture
def faiss_index(sample_embeddings):
    dim = sample_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(sample_embeddings)
    return index


def create_mock_embedding_response(dim, count=1):
    np.random.seed(42)
    response = Mock()
    response.data = [
        Mock(embedding=np.random.randn(dim).tolist())
        for _ in range(count)
    ]
    return response


def create_mock_chat_response(content="Test response"):
    response = Mock()
    response.choices = [Mock(message=Mock(content=content))]
    return response


@pytest.fixture
def mock_embedding_response(embedding_dim):
    return lambda count=1: create_mock_embedding_response(embedding_dim, count)


@pytest.fixture
def mock_chat_response():
    return create_mock_chat_response


@pytest.fixture
def sample_doc():
    return Path(__file__).parent / "fixtures" / "sample_doc.txt"


@pytest.fixture
def storage_dir(tmp_path):
    storage = tmp_path / "storage"
    storage.mkdir()
    return storage
