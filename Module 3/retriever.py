"""
Shared retrieval code for lesson 13 onwards.

Lessons 9-12 each built this up from scratch, in one file, so you could see
every piece. From lesson 13 on, lessons import it from here instead, so each
lesson file shows only what's new.

Nothing in this file is new. It's copied from:
    chunking.py        section_chunks_with_title       (lesson 10)
    vector_search.py   cosine, embed_all, vector_ranking (lesson 11)
    eval_retrieval.py  keyword_ranking, EVAL_SET, rank_of_answer, recall_at (lesson 12)
    hybrid_search.py   rrf, rerank                        (lesson 13)

This file isn't a lesson. Run the lesson files, not this one.
"""

import math
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()  # searches upward, so the repo-root .env is found
client = OpenAI()

DOCS_FOLDER = Path(__file__).parent / "docs"

# Lesson 12 found no winner between small and large on our eval set, and
# small is 6.5x cheaper, so from here on we use small.
EMBEDDING_MODEL = "text-embedding-3-small"

STOP_WORDS = {"how", "does", "do", "the", "a", "an", "of", "to", "in", "on", "for",
              "what", "when", "after", "is", "are", "it", "that", "with", "and", "or"}


# ----------------------------------------------------------------------
# Chunks (lesson 10)
# ----------------------------------------------------------------------

def section_chunks_with_title(source, text):
    """One chunk per markdown section, prefixed with the file name and doc title."""
    title = text.splitlines()[0].lstrip("# ")
    chunks = []
    for section in re.split(r"\n(?=##)", text):
        chunks.append({"source": source, "text": f"[{source} · {title}]\n\n{section.strip()}"})
    return chunks


def load_chunks():
    """Every doc in docs/ (not docs/deprecated/), cut into heading chunks."""
    chunks = []
    for path in sorted(DOCS_FOLDER.glob("*.md")):
        chunks += section_chunks_with_title(path.name, path.read_text())
    return chunks


def heading(chunk):
    """The chunk's section heading (line 0 is the title prefix, line 2 the heading)."""
    return chunk["text"].splitlines()[2]


# ----------------------------------------------------------------------
# Retrievers. Each returns ALL chunks as (score, chunk) pairs, best first.
# ----------------------------------------------------------------------

def keyword_ranking(chunks, question):
    """Lesson 10: score = how many question keywords the chunk contains.

    Ties keep file order. "NBS-409" is split into "nbs" and "409".
    """
    words = re.findall(r"[a-z0-9]+", question.lower())
    keywords = [word for word in words if word not in STOP_WORDS]
    scored = []
    for chunk in chunks:
        score = sum(keyword in chunk["text"].lower() for keyword in keywords)
        scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)  # stable: ties keep file order
    return scored


def cosine(a, b):
    """Lesson 11: from -1 (opposite) through 0 (unrelated) to 1 (same direction)."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def embed_all(texts):
    """Embed a list of texts in one request."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def vector_ranking(chunks, chunk_vectors, question_vector):
    scored = []
    for chunk, vector in zip(chunks, chunk_vectors):
        scored.append((cosine(question_vector, vector), chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def rrf(rankings, k=60):
    """Lesson 13: Reciprocal Rank Fusion. Each retriever gives a chunk 1 / (k + rank)."""
    scores = {}
    chunk_for = {}
    for ranking in rankings:  # one ranking per retriever
        for rank, (score, chunk) in enumerate(ranking, start=1):
            key = chunk["text"]
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            chunk_for[key] = chunk
    fused = [(score, chunk_for[key]) for key, score in scores.items()]
    fused.sort(key=lambda pair: pair[0], reverse=True)
    return fused


class ChunkScore(BaseModel):
    chunk: int       # the number we gave the chunk in the prompt
    relevance: int   # 0 = unrelated, 10 = answers the question


class Relevance(BaseModel):
    scores: list[ChunkScore]


RERANK_MODEL = "gpt-4o-mini"
RERANK_SYSTEM = """You judge search results for Nimbus documentation.
For EVERY numbered chunk, score how well it answers the question:
0 = unrelated, 5 = related but doesn't answer, 10 = directly answers it."""


def rerank(question, candidates):
    """Lesson 13: a model scores each candidate for the question. Best first."""
    numbered = "\n\n".join(f"[{i}]\n{chunk['text']}" for i, (score, chunk) in enumerate(candidates))
    response = client.chat.completions.parse(
        model=RERANK_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": RERANK_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nChunks:\n\n{numbered}"},
        ],
        response_format=Relevance,
    )
    parsed = response.choices[0].message.parsed
    relevance = {item.chunk: item.relevance for item in parsed.scores} if parsed else {}
    reranked = [(relevance.get(i, 0), chunk) for i, (score, chunk) in enumerate(candidates)]
    reranked.sort(key=lambda pair: pair[0], reverse=True)  # stable: ties keep the input order
    return reranked


# ----------------------------------------------------------------------
# The eval set and scoring (lesson 12)
#
# "evidence" is text the right chunk must contain. None = no answer in docs.
# ----------------------------------------------------------------------

EVAL_SET = [
    # direct: worded like the docs
    {"category": "direct", "question": "What does the Nimbus retry ladder do after a charge fails three consecutive times?", "evidence": "the ladder resumes at **rung 2**"},
    {"category": "direct", "question": "How long is the settlement fence window?", "evidence": "90-second window"},
    {"category": "direct", "question": "How many times is a failed webhook delivery redelivered?", "evidence": "redelivered up to 8 times"},
    {"category": "direct", "question": "How many partial refunds can a charge have?", "evidence": "at most 20 partial refunds"},
    {"category": "direct", "question": "When does a carry token expire?", "evidence": "It expires 72 hours after"},
    {"category": "direct", "question": "What does ladder.fence_override do?", "evidence": "fence checks"},
    # rephrased: same meaning, different words
    {"category": "rephrased", "question": "What happens after three declines in a row?", "evidence": "the ladder resumes at **rung 2**"},
    {"category": "rephrased", "question": "If a customer's card has expired, will Nimbus keep trying to charge it?", "evidence": "Nimbus never retries a hard decline"},
    {"category": "rephrased", "question": "How long until my customer sees their money back?", "evidence": "take 5 to 10 business"},
    {"category": "rephrased", "question": "Can a timeout end up billing my customer twice?", "evidence": "the customer is billed twice"},
    {"category": "rephrased", "question": "How do I know a webhook really came from Nimbus?", "evidence": "HMAC-SHA256"},
    # identifier: an exact code or incident number
    {"category": "identifier", "question": "What does error NBS-409 mean?", "evidence": "`NBS-409`"},
    {"category": "identifier", "question": "What triggers NBS-512?", "evidence": "`NBS-512`"},
    {"category": "identifier", "question": "When is NBS-221 emitted?", "evidence": "`NBS-221`"},
    {"category": "identifier", "question": "What does NBS-429 mean?", "evidence": "`NBS-429`"},
    {"category": "identifier", "question": "What went wrong in INC-2207?", "evidence": "INC-2207"},
    # no answer: nothing in the docs answers these
    {"category": "no answer", "question": "Does Nimbus support Apple Pay?", "evidence": None},
    {"category": "no answer", "question": "What does error NBS-404 mean?", "evidence": None},
    {"category": "no answer", "question": "What is Nimbus's uptime SLA?", "evidence": None},
    {"category": "no answer", "question": "Can I refund a charge in a different currency?", "evidence": None},
]

CATEGORIES = ["direct", "rephrased", "identifier"]


def rank_of_answer(ranking, evidence):
    """1 if the first chunk is correct, 2 if the second is, ... None if never found."""
    for position, (score, chunk) in enumerate(ranking, start=1):
        if evidence in chunk["text"]:
            return position
    return None


def recall_at(rankings, k, category=None):
    """Share of answerable questions whose right chunk is in the top k.

    `rankings` has one ranking per question, in EVAL_SET order.
    """
    hits = 0
    total = 0
    for ranking, case in zip(rankings, EVAL_SET):
        if case["evidence"] is None:
            continue  # no right chunk to find
        if category is not None and case["category"] != category:
            continue
        total += 1
        rank = rank_of_answer(ranking, case["evidence"])
        if rank is not None and rank <= k:
            hits += 1
    return hits / total
