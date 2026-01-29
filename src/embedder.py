import numpy as np
from voyageai.client import Client

from src.config import config

client = Client(config.voyageai_api_key, 0, 5)

BATCH_SIZE = 16

def embed_batch(batch: list[str]):
        return client.embed(batch, config.embedding_model, "document").embeddings

def create_embeddings(texts: list[str]):
    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    print(batches)
    # embeddings = [emb for batch in batches for emb in embed_batch(batch)]

    # return np.array(embeddings)


def embed_query(query: str, model: str = config.embedding_model):
    result = client.embed(texts=[query], model=model, input_type="query")
    return np.array(result.embeddings[0])
