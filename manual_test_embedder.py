import json
from src.embedder import create_embeddings


with open("storage/climate_change_chunks.json", "r") as file:
  chunks = json.load(file)

  embeddings = create_embeddings(chunks)
  embeddings_list = embeddings.tolist()

  with open("storage/climate_change_embeddings.json", "w", encoding='utf-8') as f:
    json.dump(embeddings_list, f)
