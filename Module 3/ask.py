"""
Ask the Nimbus docs: the chatbot that lessons 9-13 built.

Every piece below was built and measured in an earlier lesson. With the
retrieval code in retriever.py, the whole RAG chatbot fits on one screen:
hybrid retrieval and reranking (lesson 13), top 3 chunks (lesson 12), and
a grounded prompt (lesson 9).

Setup:
    pip install -r requirements.txt     # from the repo root
    # .env in the repo root (see README):  OPENAI_API_KEY=sk-...

Run:
    python ask.py          (type a question; press Enter on an empty line to quit)
"""

from retriever import (client, embed_all, heading, keyword_ranking, load_chunks, rerank, rrf,
                       vector_ranking)

MODEL = "gpt-4o-mini"

SYSTEM = """You are a support engineer for Nimbus, a payments platform.
Answer ONLY using the documentation provided in the user's message.
If the answer is not in the documentation, say so."""

chunks = load_chunks()
chunk_vectors = embed_all([chunk["text"] for chunk in chunks])  # once, at startup

while question := input("\nAsk about Nimbus: ").strip():
    question_vector = embed_all([question])[0]
    fused = rrf([keyword_ranking(chunks, question),
                 vector_ranking(chunks, chunk_vectors, question_vector)])
    reranked = rerank(question, fused[:20])  # retrieve 20 cheaply, rerank, keep 3
    top = [chunk for score, chunk in reranked[:3]]

    context = "\n\n---\n\n".join(chunk["text"] for chunk in top)
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Documentation:\n\n{context}\n\nQuestion: {question}"},
        ],
    )
    print(f"\n{response.choices[0].message.content}")
    print("\nSources: " + "; ".join(f"{c['source']} {heading(c)}" for c in top))
