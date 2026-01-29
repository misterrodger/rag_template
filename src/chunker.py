from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import config

splitter = RecursiveCharacterTextSplitter(
    chunk_size=config.chunk_size,
    chunk_overlap=config.chunk_overlap,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
  )

def chunk_file(filepath: str):
  with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

  chunks = splitter.split_text(text)
  return chunks
