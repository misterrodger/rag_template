import pytest
from src.chunker import chunk_file

@pytest.fixture(scope="module")
def ai_chunks():
    return chunk_file("data/ai_overview.txt")


def test_chunk_file_happy_path(ai_chunks: list[str]):
    assert len(ai_chunks) > 0, "Should produce at least one chunk"


def test_chunk_file_not_found():
    with pytest.raises(FileNotFoundError):
        chunk_file("data/nonexistent_file.txt")


def test_content_preservation(ai_chunks: list[str]):
    chunks = ai_chunks
    
    assert "Artificial Intelligence" in chunks[0], "First chunk should contain document title"
    assert len(chunks[0].strip()) > 0, "First chunk should have content"
