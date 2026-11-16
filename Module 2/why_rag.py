"""
Lesson 9 — Why RAG exists.

Lesson 8 ended with the model inventing a confident answer about a Nimbus
system it had never heard of. Asking more firmly didn't fix that (lesson 7),
and a schema didn't either (lesson 8). The model's knowledge is frozen at
training time. The only fix is to put the facts in front of it when it
generates the answer.

That's RAG: Retrieve the right document, add it to the prompt, Generate.
This file builds it with the simplest retriever possible: keyword search.

Setup:
    pip install -r requirements.txt     # from the repo root
    # .env in the repo root (see README):  OPENAI_API_KEY=sk-...

Run:
    python why_rag.py
"""

import re
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # searches upward, so the repo-root .env is found
client = OpenAI()
encoder = tiktoken.get_encoding("o200k_base")  # the tokenizer gpt-4o-mini uses

MODEL = "gpt-4o-mini"
PRICE_PER_1M_INPUT_TOKENS = 0.15  # gpt-4o-mini input price in USD; check current pricing

DOCS_FOLDER = Path(__file__).parent / "docs"

SYSTEM = "You are a support engineer for Nimbus, a payments platform. Answer questions about Nimbus."

QUESTION = "What does the Nimbus retry ladder do after a charge fails three consecutive times?"


def ask(system, user):
    """Send one question to the model and return its reply as text."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,  # low on purpose: we want the same answer on every run
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


def count_tokens(text):
    return len(encoder.encode(text))


def load_docs():
    """Read every .md file in docs/ into a dict: {file name: file text}.

    Only the top level of docs/ is read, so docs/deprecated/ is left out.
    """
    docs = {}
    for path in sorted(DOCS_FOLDER.glob("*.md")):
        docs[path.name] = path.read_text()
    return docs


def build_prompt(docs, question):
    """Put the documents and the question together into one user prompt."""
    context = ""
    for name, text in docs.items():
        context += f"\n\n### {name}\n\n{text}"
    return f"Documentation:{context}\n\nQuestion: {question}"


# ─────────────────────────────────────────────────────────────────────────
# v0 · No documents
#
# Last lesson the Nimbus scheduler was imaginary. Today Nimbus is real: its
# docs are sitting in docs/. But the model has never seen them, so it does
# what it always does with a gap: fills it with something plausible.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v0 — no documents =====")
print(f"Q: {QUESTION}\n")
print(ask(SYSTEM, QUESTION))


# ─────────────────────────────────────────────────────────────────────────
# v1 · Paste ALL the docs into the prompt
#
# The correct answer is in retry-ladder.md: Nimbus skips rung 4, moves the
# charge to the merchant's secondary acquirer, and resumes at rung 2 (not
# rung 1). Give the model every document and it finds it.
#
# The model learned nothing. Clear the prompt and it knows nothing again.
# Retrieval is a filter, not a memory.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v1 — paste every document into the prompt =====")

all_docs = load_docs()
print(f"Loaded {len(all_docs)} documents: {list(all_docs.keys())}\n")

prompt_all = build_prompt(all_docs, QUESTION)
print(ask(SYSTEM, prompt_all))


# ─────────────────────────────────────────────────────────────────────────
# v2 · What v1 costs
#
# It works, so why not always do it? Because you pay for every input token
# on every question, and a bigger prompt also takes longer before the first
# token comes back. Let's do the maths instead of asserting it.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v2 — what pasting everything costs =====")

tokens_all = count_tokens(SYSTEM + prompt_all)
tokens_per_doc = tokens_all / len(all_docs)
QUESTIONS_PER_DAY = 10_000

print(f"v1 prompt: {tokens_all:,} input tokens (about {tokens_per_doc:,.0f} per document)\n")

for doc_count in [len(all_docs), 100, 1_000]:
    tokens_per_question = round(tokens_per_doc * doc_count)
    cost_per_day = tokens_per_question * QUESTIONS_PER_DAY / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
    print(f"{doc_count:>5} docs: {tokens_per_question:>9,} tokens per question, "
          f"${cost_per_day:>9,.2f} per day at {QUESTIONS_PER_DAY:,} questions/day")

print("\ngpt-4o-mini's context window is 128,000 tokens. At 1,000 docs the prompt")
print("doesn't even fit. And the more irrelevant text surrounds the answer,")
print("the harder it is for the model to find it.")


# ─────────────────────────────────────────────────────────────────────────
# v3 · Keyword retrieval: send only the document that matches
#
# The dumbest retriever that works: take the important words from the
# question and keep only the files that contain ALL of them, like grep.
# Same answer, a fraction of the tokens. That's RAG.
# ─────────────────────────────────────────────────────────────────────────

STOP_WORDS = {"how", "does", "do", "the", "a", "an", "of", "to", "in", "on", "for",
              "what", "when", "after", "is", "are", "it", "that", "with", "and", "or"}


def keyword_search(docs, question):
    """Return only the documents that contain every important word in the question."""
    words = re.findall(r"[a-z0-9]+", question.lower())  # split into lowercase words
    keywords = [word for word in words if word not in STOP_WORDS]
    print(f"Searching for: {keywords}")

    matches = {}
    for name, text in docs.items():
        if all(keyword in text.lower() for keyword in keywords):
            matches[name] = text
    print(f"Found: {list(matches.keys())}\n")
    return matches


print("\n===== v3 — keyword retrieval =====")

found = keyword_search(all_docs, QUESTION)
prompt_found = build_prompt(found, QUESTION)
print(ask(SYSTEM, prompt_found))

tokens_found = count_tokens(SYSTEM + prompt_found)
print(f"\nInput tokens: {tokens_found:,} instead of {tokens_all:,} "
      f"({tokens_found / tokens_all:.0%} of v1).")


# ─────────────────────────────────────────────────────────────────────────
# v4 · Same question, different words
#
# "Three declines in a row" means the same as "fails three consecutive
# times". But "happens" and "row" don't appear in retry-ladder.md, so the
# search finds nothing. The right document is sitting right there.
#
# The model is fine. The prompt is fine. The retriever is the problem: it
# matches strings, and people ask questions in meanings. (Lesson 11.)
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v4 — same question, different words =====")

REPHRASED = "What happens after three declines in a row?"
print(f"Q: {REPHRASED}\n")

found = keyword_search(all_docs, REPHRASED)
prompt_empty = build_prompt(found, REPHRASED)
print(ask(SYSTEM, prompt_empty))
print("\n→ No documents were found, so the model is back to guessing, like v0.")


# ─────────────────────────────────────────────────────────────────────────
# v5 · Grounding: answer only from the documents
#
# We can't fix the retriever yet, but we can stop the model from covering
# for it. Tell it to answer only from the documentation, and to say so when
# the answer isn't there. "I don't know" is a correct answer here.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v5 — grounded: answer only from the documents =====")

GROUNDED_SYSTEM = SYSTEM + """

Answer ONLY using the documentation provided in the user's message.
If the answer is not in the documentation, reply exactly:
"I can't find that in the Nimbus documentation."
Do not use any other knowledge."""

print(f"Q: {REPHRASED}  (same empty search result as v4)\n")
print(ask(GROUNDED_SYSTEM, prompt_empty))


# ─────────────────────────────────────────────────────────────────────────
# Not in this lesson (on purpose)
#
# - Chunking: we load whole files because ours are small.       Lesson 10.
# - Embeddings and vector search: search by meaning.            Lesson 11.
# - Hybrid search and reranking.                                Lesson 14.
# - Why retrieval can still return the wrong document
#   (docs/deprecated/ is waiting for that).                     Lesson 15.
# ─────────────────────────────────────────────────────────────────────────
