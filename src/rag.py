from openai import OpenAI
from src.config import config
from src.retriever import retrieve


client = OpenAI(api_key=config.openai_api_key)


def query_rag(query: str, index, chunks: list[str], k: int = 5) -> dict:
    results = retrieve(query, index, chunks, k)

    context = "\n\n".join([f"[{i+1}] {result['chunk']}" for i, result in enumerate(results)])

    system_prompt = "You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain relevant information, say so."

    user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=config.chat_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7
    )

    answer = response.choices[0].message.content

    return {
        "question": query,
        "answer": answer,
        "sources": [{"index": r["index"], "distance": r["distance"], "chunk": r["chunk"][:100] + "..."} for r in results]
    }
