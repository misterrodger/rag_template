import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Any

load_dotenv()

class Config(BaseModel):
  openai_api_key: str = os.environ.get('OPENAI_API_KEY', '')
  embedding_model: str = "text-embedding-3-small"
  chat_model: str = "gpt-4o-mini"

  chunk_size: int = 1000
  chunk_overlap: int = 200

  top_k: int = 5

  data_dir: Path = Path("data")
  storage_dir: Path = Path("storage")

  model_config = {"arbitrary_types_allowed": True}

  def __init__(self, **data: Any) -> None:
    super().__init__(**data)
    self.storage_dir.mkdir(exist_ok=True)

  @property
  def vector_index_path(self):
    return self.storage_dir / "vectors.index"
    
  @property
  def chunks_path(self):
    return self.storage_dir / "chunks.json"
    
  @property
  def metadata_path(self):
    return self.storage_dir / "metadata.json"
    
config = Config()
