from src.chunker import chunk_file
import logging
import json

logging.basicConfig(level=logging.INFO)

chunks = chunk_file("data/ai_overview.txt")

print(f"\nTotal chunks: {len(chunks)}")

with open("storage/ai_overview_chunks.json", "w", encoding="utf-8") as f:
  json.dump(chunks, f, indent=2)

for i, chunk in enumerate(chunks):
  print(f"\n_____Chunk {i}_____")
  print(chunk)
