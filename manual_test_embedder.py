import json
from src.embedder import create_embeddings


with open("storage/ai_overview_chunks.json", "r") as file:
  chunks = json.load(file)

  batches = create_embeddings(chunks)

  print(batches)