"""
Lesson 11 — Vector search: retrieval by meaning.

Lessons 9 and 10 ended on the same failure: "three declines in a row" means
"fails three consecutive times", but a keyword retriever matches strings,
and those strings don't match.

In lesson 3 an embedding was something happening inside the model: a vector
where similar meanings point in similar directions. Today embeddings become
something you call, store and search. Retrieval stops being string matching
and becomes geometry.

Setup:
    pip install -r requirements.txt     # from the repo root
    # .env in the repo root (see README):  OPENAI_API_KEY=sk-...

Run:
    python vector_search.py
"""

import math
import re
import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # searches upward, so the repo-root .env is found
client = OpenAI()
encoder = tiktoken.get_encoding("o200k_base")  # the tokenizer gpt-4o-mini uses

MODEL = "gpt-4o-mini"
PRICE_PER_1M_INPUT_TOKENS = 0.15  # gpt-4o-mini input price in USD; check current pricing

# Lesson 3 used text-embedding-3-small. On our docs, small ranks the right
# chunk 3rd for REPHRASED; large ranks it 1st. Embedding models differ, and
# which one fits your data is something you measure (lesson 12).
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_PRICE_PER_1M_TOKENS = 0.13  # USD; check current pricing

DOCS_FOLDER = Path(__file__).parent / "docs"

SYSTEM = """You are a support engineer for Nimbus, a payments platform.
Answer ONLY using the documentation provided in the user's message.
If the answer is not in the documentation, say so."""

REPHRASED = "What happens after three declines in a row?"

STOP_WORDS = {"how", "does", "do", "the", "a", "an", "of", "to", "in", "on", "for",
              "what", "when", "after", "is", "are", "it", "that", "with", "and", "or"}


def ask(system, user):
    """Send one question to the model and return its reply as text."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def count_tokens(text):
    return len(encoder.encode(text))


def ask_with_chunks(chunks, question):
    """Put one or more chunks in the prompt and ask the question."""
    context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks)
    prompt = f"Documentation:\n\n{context}\n\nQuestion: {question}"
    print(f"Input tokens: {count_tokens(SYSTEM + prompt):,}\n")
    print(ask(SYSTEM, prompt))


def section_chunks_with_title(source, text):
    """Lesson 10's best chunker: one chunk per section, prefixed with the doc title."""
    title = text.splitlines()[0].lstrip("# ")
    sections = re.split(r"\n(?=##)", text)
    chunks = []
    for section in sections:
        chunks.append({"source": source, "text": f"[{source} · {title}]\n\n{section.strip()}"})
    return chunks


def heading(chunk):
    return chunk["text"].splitlines()[2]  # line 0 is the title prefix, line 2 the heading


chunks = []
for path in sorted(DOCS_FOLDER.glob("*.md")):  # top level only, so docs/deprecated/ is left out
    chunks += section_chunks_with_title(path.name, path.read_text())


# ─────────────────────────────────────────────────────────────────────────
# v0 · Where lesson 10 left us
#
# Keyword scoring, rephrased question. Nine chunks tie, the first one in
# file order wins, and it's the wrong one.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v0 — keyword retrieval, rephrased question (lesson 10) =====")

keywords = [w for w in re.findall(r"[a-z0-9]+", REPHRASED.lower()) if w not in STOP_WORDS]
scores = [sum(keyword in chunk["text"].lower() for keyword in keywords) for chunk in chunks]
best = chunks[scores.index(max(scores))]  # the FIRST chunk with the top score
print(f"Keywords: {keywords}")
print(f"Best: {best['source']}  {heading(best)}  "
      f"({max(scores)}/{len(keywords)} keywords, {scores.count(max(scores))} chunks tied)\n")
ask_with_chunks([best], REPHRASED)


# ─────────────────────────────────────────────────────────────────────────
# v1 · Embed every chunk, embed the question, find the closest
#
# One chunk → one vector. Not one vector per token: the embedding model
# reads the whole chunk and returns a single list of numbers for it.
#
# Store the vectors next to the chunks, and that list IS our vector
# database. Searching it is a for-loop: compare the question's vector with
# every chunk's vector and keep the closest.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v1 — vector search, top 1 =====")


def cosine(a, b):
    """Cosine similarity: from -1 (opposite) through 0 (unrelated) to 1 (same direction)."""
    dot = sum(x * y for x, y in zip(a, b))
    length_a = math.sqrt(sum(x * x for x in a))
    length_b = math.sqrt(sum(y * y for y in b))
    return dot / (length_a * length_b)


def embed(text):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def search(question, k):
    """Return the k chunks closest to the question, as (score, chunk) pairs."""
    question_vector = embed(question)
    results = [(cosine(question_vector, chunk["vector"]), chunk) for chunk in chunks]
    results.sort(key=lambda pair: pair[0], reverse=True)  # highest score first
    return results[:k]


# All chunks in ONE request. This is the expensive step, and you do it once.
started = time.perf_counter()
response = client.embeddings.create(model=EMBEDDING_MODEL, input=[c["text"] for c in chunks])
embed_seconds = time.perf_counter() - started
for chunk, item in zip(chunks, response.data):
    chunk["vector"] = item.embedding

print(f"Embedded {len(chunks)} chunks → {len(chunks)} vectors "
      f"of {len(chunks[0]['vector']):,} numbers each\n")

score, top = search(REPHRASED, k=1)[0]
print(f"Q: {REPHRASED}")
print(f"Closest: {top['source']}  {heading(top)}  (cosine {score:.3f})\n")
ask_with_chunks([top], REPHRASED)
print("\n→ No shared words needed. \"Three declines in a row\" and \"fails three")
print("  consecutive times\" point in nearly the same direction.")


# ─────────────────────────────────────────────────────────────────────────
# v2 · Why it worked: look at the scores
#
# Keyword scores were small integers, so nine chunks tied at 1 and file
# order decided. Cosine scores are a ranking, and the gap between them is
# information keyword matching never gave us.
#
# Cosine can range from -1 to 1, but text embeddings rarely go below 0, and
# real scores bunch in a narrow band. Read them relative to each other,
# not as absolute "percent similar".
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v2 — the top 5, with scores =====")

for rank, (score, chunk) in enumerate(search(REPHRASED, k=5), start=1):
    print(f"  #{rank}  {score:.3f}  {chunk['source']:<20} {heading(chunk)}")
print("\n→ #1 stands clear of the rest, which bunch together. No ties.")


# ─────────────────────────────────────────────────────────────────────────
# v3 · Top 3 instead of top 1
#
# Keyword ties forced us to take one chunk. A ranking makes "take the best
# few" natural. That fixes lesson 10's leftover problem: "Promotion is not
# a reset" is its own subsection, separate from the section it explains.
# A question that needs both can't be answered from one chunk.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v3 — an answer that spans two sections =====")

MULTI = "After three declines in a row, how many attempts does a charge get in total?"
print(f"Q: {MULTI}\n")
results = search(MULTI, k=3)
for rank, (score, chunk) in enumerate(results, start=1):
    print(f"  #{rank}  {score:.3f}  {chunk['source']:<20} {heading(chunk)}")

print("\n--- top 1 ---")
ask_with_chunks([results[0][1]], MULTI)
print("\n--- top 3 ---")
ask_with_chunks([chunk for score, chunk in results], MULTI)
# Narration matches today's run: top 1 answered "four". Re-check before re-recording.
print("\n→ Top 1 worked out a total from half the facts, and got it wrong. The")
print("  real total (six) is in \"Promotion is not a reset\". Top 3 brings it in.")


# ─────────────────────────────────────────────────────────────────────────
# v4 · What it costs, and when you need a real vector database
#
# Embedding the docs is a one-time cost. Each question costs one tiny
# embedding call plus a loop over every vector.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v4 — cost, time, and when brute force stops being free =====")

index_cost = response.usage.total_tokens / 1_000_000 * EMBEDDING_PRICE_PER_1M_TOKENS
print(f"Embed all chunks (once):  {response.usage.total_tokens:,} tokens, "
      f"${index_cost:.5f}, {embed_seconds:.1f}s")

question_vector = embed(REPHRASED)
started = time.perf_counter()
for chunk in chunks:
    cosine(question_vector, chunk["vector"])
loop_seconds = time.perf_counter() - started
print(f"Search (every question):  {loop_seconds * 1000:.1f} ms to compare {len(chunks)} vectors")

answer_prompt_tokens = 300  # roughly what v1's generation call sent, as lesson 9 measured
answer_cost = answer_prompt_tokens / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
print(f"Generation (every question): ~{answer_prompt_tokens} input tokens, ${answer_cost:.6f}")

print("\nThe same for-loop over more chunks:")
dimensions = len(chunks[0]["vector"])
for chunk_count in [10_000, 1_000_000, 100_000_000]:
    seconds = loop_seconds / len(chunks) * chunk_count
    gigabytes = chunk_count * dimensions * 4 / 1e9  # 4 bytes per number, stored compactly
    print(f"  {chunk_count:>11,} chunks: ~{seconds:>9,.1f}s per question, {gigabytes:>8,.2f} GB of vectors")

print("\n→ You do not need a vector database for 46 chunks. You need one when")
print("  this loop stops being free.")
# (numpy would make the loop ~100x faster, which moves the line but doesn't remove it.)
#
# What a vector database buys you:
# - Approximate nearest neighbour (HNSW, IVF): skip most comparisons. Much
#   faster, but it can miss the true best match. A real trade, not a free win.
# - Persistence: this script re-embeds everything on every run. Fine for 46
#   chunks, not for 10,000 on every deploy.
# - Metadata filtering: "closest chunks, but only current docs for this
#   merchant", in one query.
# Options: pgvector, Chroma, Qdrant, Pinecone.


# ─────────────────────────────────────────────────────────────────────────
# v5 · What keyword search did better
#
# Semantic search fixed rephrasing. Now ask about one specific error code.
# To an embedding model, NBS-409 and NBS-429 are both just "an error code".
# Keyword matching would find the exact string instantly.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v5 — an exact identifier =====")

CODE_QUESTION = "What does error NBS-409 mean?"
print(f"Q: {CODE_QUESTION}\n")
for rank, (score, chunk) in enumerate(search(CODE_QUESTION, k=5), start=1):
    marker = "   ← mentions NBS-409" if "NBS-409" in chunk["text"] else ""
    print(f"  #{rank}  {score:.3f}  {chunk['source']:<20} {heading(chunk)}{marker}")

exact = [chunk for chunk in chunks if "NBS-409" in chunk["text"]]
print(f"\nPlain string match for \"NBS-409\": {len(exact)} chunks, both correct:")
for chunk in exact:
    print(f"  {chunk['source']:<20} {heading(chunk)}")

print("\n→ #1 is about NBS-429, a different error. The chunks that actually")
print("  mention NBS-409 rank below it. With seven docs they scrape into the")
print("  top 5; with a page per error code, they'd be buried.")
print("→ For two lessons the dumb retriever broke and we made it smarter. Now")
print("  the smart one breaks where the dumb one never did. Embeddings aren't")
print("  an upgrade, they're a trade. Run both and combine them: lesson 14.")
