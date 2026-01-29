from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from src.config import config

logger = logging.getLogger(__name__)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
  )

def chunk_file(filepath: str):
  with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

  logger.info(f"Chunking document of length {len(text)}")
  chunks = splitter.split_text(text)
  logger.info(f"Created {len(chunks)} chunks")
  return chunks
