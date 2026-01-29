import numpy as np
from openai import OpenAI

from src.config import config

client = OpenAI(api_key=config.openai_api_key)

BATCH_SIZE = 100

def embed_batch(batch: list[str]):
    response = client.embeddings.create(input=batch, model=config.embedding_model)
    return [item.embedding for item in response.data]

def create_embeddings(texts: list[str]):
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    embeddings = [emb for batch in batches for emb in embed_batch(batch)]
    return np.array(embeddings)

def embed_query(query: str, model: str = config.embedding_model):
    response = client.embeddings.create(input=[query], model=model)
    return np.array(response.data[0].embedding)
