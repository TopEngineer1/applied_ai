"""
Lesson 12 — Evaluating retrieval: how do you know it works?

So far every choice was made by eye: 300-character chunks, the embedding
model, top 3. Each looked right on one question. This lesson replaces
"looks right" with numbers.

Evaluate retrieval before you evaluate answers. If the right chunk never
arrives, the answer can't be right, and you'd be debugging the wrong layer.
Retrieval is also the easy case: for each test question we know which text
should come back, so checking it is a for-loop. No model judges anything.

No new API in this lesson. Every line is code you've already run. What's
new is the method.

Setup:
    pip install -r requirements.txt     # from the repo root
    # .env in the repo root (see README):  OPENAI_API_KEY=sk-...

Run:
    python eval_retrieval.py
"""

import math
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # searches upward, so the repo-root .env is found
client = OpenAI()

DOCS_FOLDER = Path(__file__).parent / "docs"
EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-3-large"]

STOP_WORDS = {"how", "does", "do", "the", "a", "an", "of", "to", "in", "on", "for",
              "what", "when", "after", "is", "are", "it", "that", "with", "and", "or"}

# ----------------------------------------------------------------------
# The eval set. Written BEFORE running anything, so the numbers decide
# the conclusion, not the other way round.
#
# "evidence" is a short piece of text the right chunk must contain. A
# retrieved chunk counts as correct if it contains it. Labelling by text
# (not by chunk number) means the same eval set works for any chunking.
# None means the docs don't answer the question at all.
# ----------------------------------------------------------------------

EVAL_SET = [
    # direct: worded like the docs
    {"category": "direct", "question": "What does the Nimbus retry ladder do after a charge fails three consecutive times?", "evidence": "the ladder resumes at **rung 2**"},
    {"category": "direct", "question": "How long is the settlement fence window?", "evidence": "90-second window"},
    {"category": "direct", "question": "How many times is a failed webhook delivery redelivered?", "evidence": "redelivered up to 8 times"},
    {"category": "direct", "question": "How many partial refunds can a charge have?", "evidence": "at most 20 partial refunds"},
    {"category": "direct", "question": "When does a carry token expire?", "evidence": "It expires 72 hours after"},
    {"category": "direct", "question": "What does ladder.fence_override do?", "evidence": "fence checks"},  # splits into 3 ordinary words
    # rephrased: same meaning, different words
    {"category": "rephrased", "question": "What happens after three declines in a row?", "evidence": "the ladder resumes at **rung 2**"},
    {"category": "rephrased", "question": "If a customer's card has expired, will Nimbus keep trying to charge it?", "evidence": "Nimbus never retries a hard decline"},
    {"category": "rephrased", "question": "How long until my customer sees their money back?", "evidence": "take 5 to 10 business"},
    {"category": "rephrased", "question": "Can a timeout end up billing my customer twice?", "evidence": "the customer is billed twice"},
    {"category": "rephrased", "question": "How do I know a webhook really came from Nimbus?", "evidence": "HMAC-SHA256"},
    # identifier: an exact code, flag or incident number
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

ANSWERABLE = [case for case in EVAL_SET if case["evidence"]]
CATEGORIES = ["direct", "rephrased", "identifier"]


# ----------------------------------------------------------------------
# Chunkers and retrievers, copied from lessons 10 and 11.
# Each retriever returns ALL chunks, best first, so we can see the rank.
# ----------------------------------------------------------------------

def fixed_size_chunks(source, text, size):
    chunks = []
    for start in range(0, len(text), size):
        chunks.append({"source": source, "text": text[start : start + size]})
    return chunks


def overlapping_chunks(source, text, size, overlap):
    chunks = []
    for start in range(0, len(text), size - overlap):
        chunks.append({"source": source, "text": text[start : start + size]})
    return chunks


def section_chunks_with_title(source, text):
    title = text.splitlines()[0].lstrip("# ")
    chunks = []
    for section in re.split(r"\n(?=##)", text):
        chunks.append({"source": source, "text": f"[{source} · {title}]\n\n{section.strip()}"})
    return chunks


def keyword_ranking(chunks, question):
    """Lesson 10's keyword scoring. Ties keep file order, as before.

    Note: "NBS-409" is split into "nbs" and "409". Keyword search never
    matches the whole code; it wins when the bare number is rare.
    """
    keywords = [w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in STOP_WORDS]
    scores = [sum(keyword in chunk["text"].lower() for keyword in keywords) for chunk in chunks]
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [(scores[i], chunks[i]) for i in order]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def embed_all(texts, model):
    """Embed a list of texts in one request."""
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def vector_ranking(chunks, chunk_vectors, question_vector):
    scored = [(cosine(question_vector, vector), chunk) for vector, chunk in zip(chunk_vectors, chunks)]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def rank_of_answer(ranking, evidence):
    """1 if the first chunk is correct, 2 if the second is, ... None if never found."""
    for position, (score, chunk) in enumerate(ranking, start=1):
        if evidence in chunk["text"]:
            return position
    return None


docs = {path.name: path.read_text() for path in sorted(DOCS_FOLDER.glob("*.md"))}

chunkings = {"headings": [], "fixed 300": [], "fixed 300+100": []}  # +100 = lesson 10's overlap
for name, text in docs.items():
    chunkings["headings"] += section_chunks_with_title(name, text)
    chunkings["fixed 300"] += fixed_size_chunks(name, text, size=300)
    chunkings["fixed 300+100"] += overlapping_chunks(name, text, size=300, overlap=100)


# ----------------------------------------------------------------------
# v0 · Check the eval set itself
#
# A wrong label makes every number after it wrong. Before measuring the
# retriever, confirm each piece of evidence really is in the docs.
# ----------------------------------------------------------------------

print("\n===== v0 — the eval set =====")

for category in CATEGORIES + ["no answer"]:
    count = sum(case["category"] == category for case in EVAL_SET)
    print(f"  {category:<11} {count} questions")

all_text = "\n".join(docs.values())
for case in ANSWERABLE:
    assert case["evidence"] in all_text, f"Evidence not in docs: {case['evidence']}"
print(f"\nAll {len(ANSWERABLE)} pieces of evidence found in the docs.")

# Being in the docs isn't enough. If a chunk boundary cuts the evidence in
# half, no chunk contains it, and that question scores zero at every k for
# reasons that have nothing to do with the retriever.
print("\nCan any chunk contain the evidence at all?")
for chunking, chunks in chunkings.items():
    unreachable = [case["evidence"] for case in ANSWERABLE
                   if not any(case["evidence"] in chunk["text"] for chunk in chunks)]
    print(f"  {chunking:<13} {len(ANSWERABLE) - len(unreachable)}/{len(ANSWERABLE)} reachable"
          + (f"   cut in half: {unreachable}" if unreachable else ""))
print("→ Bad chunking doesn't just rank worse. It makes some answers impossible")
print("  to retrieve at ANY k during evals.")


# ----------------------------------------------------------------------
# Run every retriever on every question, once. This is the only part
# that calls the API: embeddings for chunks and questions. No chat model.
# ----------------------------------------------------------------------

questions = [case["question"] for case in EVAL_SET]
rankings = {}  # (chunking, retriever) → one ranking per question

for chunking, chunks in chunkings.items():
    rankings[(chunking, "keyword")] = [keyword_ranking(chunks, q) for q in questions]
    for model in EMBEDDING_MODELS:
        chunk_vectors = embed_all([chunk["text"] for chunk in chunks], model)
        question_vectors = embed_all(questions, model)
        retriever = model.replace("text-embedding-3-", "")  # "small" / "large"
        rankings[(chunking, retriever)] = [
            vector_ranking(chunks, chunk_vectors, qv) for qv in question_vectors
        ]

# For each setup, turn every ranking into one number: where the right chunk came.
ranks = {}  # (chunking, retriever) → [rank for question 1, rank for question 2, ...]
for config in rankings:
    ranks[config] = []
    for i, case in enumerate(EVAL_SET):
        ranking = rankings[config][i]  # this setup's ranking for question i
        if case["evidence"]:
            rank = rank_of_answer(ranking, case["evidence"])
        else:
            rank = None  # a no-answer question has no right chunk to find
        ranks[config].append(rank)


def recall_at(config, k, category=None):
    """Share of answerable questions whose right chunk is in the top k."""
    hits = total = 0
    for rank, case in zip(ranks[config], EVAL_SET):
        if case["evidence"] and (category is None or case["category"] == category):
            total += 1
            hits += rank is not None and rank <= k
    return hits / total


# ----------------------------------------------------------------------
# v1 · Where did the right chunk rank?
#
# Recall@3 = "is the right chunk in the top 3?" But it hides whether it
# came 1st or scraped in 3rd, so look at the rank too.
# ----------------------------------------------------------------------

print("\n===== v1 — rank of the right chunk, headings chunking =====\n")

print(f"  {'question':<62} {'keyword':>7} {'small':>6} {'large':>6}")
for index, case in enumerate(EVAL_SET):
    if not case["evidence"]:
        continue
    cells = []
    for retriever in ["keyword", "small", "large"]:
        rank = ranks[("headings", retriever)][index]
        cells.append(str(rank) if rank else "-")
    print(f"  {case['question'][:62]:<62} {cells[0]:>7} {cells[1]:>6} {cells[2]:>6}")

# The narration below describes today's run: these docs, this eval set,
# today's embedding models. Re-run and re-check before re-recording.

print("\n→ Lesson 11's question is row 7: keyword 7th, small 3rd, large 1st.")
print("  That one row is why lesson 11 switched to large.")
print("→ Now read the other rows. Large ranks NBS-512 19th and INC-2207 15th.")
print("  We picked a model from one question.")


# ----------------------------------------------------------------------
# v2 · The matrix: every chunking × retriever × k
# ----------------------------------------------------------------------

print("\n===== v2 — recall@k over all answerable questions =====\n")

print(f"  {'chunking':<13} {'retriever':<9} {'@1':>5} {'@3':>5} {'@5':>5}")
for (chunking, retriever) in ranks:
    cells = [f"{recall_at((chunking, retriever), k):>5.0%}" for k in (1, 3, 5)]
    print(f"  {chunking:<13} {retriever:<9} {' '.join(cells)}")

print(f"\n→ {len(ANSWERABLE)} questions, so ONE question moves a score by "
      f"{1 / len(ANSWERABLE):.0%}. Small vs large is one or two")
print("  questions apart: no winner. Large costs 6.5x as much and isn't better here.")
print("→ Top 1 → top 3 is the big jump for embeddings. Top 3 → top 5 adds little.")
print("  That's lesson 11's k=3, now with a reason.")
print("→ Plain fixed 300 has the lowest scores in the table, and it can't even reach one answer (v0).")
print("→ Add lesson 10's overlap and it catches up. With small, fixed 300+100")
print("  ties or beats headings. Headings didn't earn \"best\" on this eval set;")
print(f"  they tie with overlap, using {len(chunkings['headings'])} chunks instead of "
      f"{len(chunkings['fixed 300+100'])}.")


# ----------------------------------------------------------------------
# v3 · Split by question type
# ----------------------------------------------------------------------

print("\n===== v3 — recall@3 by question type =====\n")

print(f"  {'chunking':<13} {'retriever':<9} " + " ".join(f"{c:>10}" for c in CATEGORIES))
for (chunking, retriever) in ranks:
    cells = [f"{recall_at((chunking, retriever), 3, c):>10.0%}" for c in CATEGORIES]
    print(f"  {chunking:<13} {retriever:<9} {' '.join(cells)}")

print("\n→ The overall score hid this. With headings: keyword gets 100% of")
print("  identifiers and 20% of rephrased questions. Embeddings get 100% of")
print("  rephrased and 60% of identifiers. Each misses what the other catches.")
print("→ Be precise about why keyword wins on codes: it splits NBS-409 into")
print("  \"nbs\" and \"409\" and matches the bare number, which is rare in these")
print("  docs. Embeddings blur rare tokens into \"some error code\".")
print("→ That's what combining them should recover, with numbers: lesson 13.")


# ----------------------------------------------------------------------
# v4 · Questions with no answer
#
# Recall can't score these: there's no right chunk. What we can ask is
# whether the retriever KNOWS. Does the top score for a question with no
# answer look any different from the top score for one that has one?
# ----------------------------------------------------------------------

print("\n===== v4 — top-1 score: answerable vs no answer (headings) =====\n")

for retriever in ["small", "large"]:
    per_question = rankings[("headings", retriever)]
    answerable = [per_question[i][0][0] for i, c in enumerate(EVAL_SET) if c["evidence"]]
    unanswerable = [per_question[i][0][0] for i, c in enumerate(EVAL_SET) if not c["evidence"]]
    print(f"  {retriever}: answerable {min(answerable):.3f}–{max(answerable):.3f}, "
          f"no answer {min(unanswerable):.3f}–{max(unanswerable):.3f}")

print("\n→ The ranges overlap. Some no-answer questions score higher than real")
print("  ones, so no score cutoff can separate them. The retriever always")
print("  returns SOMETHING; catching \"not in the docs\" is the grounding")
print("  prompt's job (lesson 9), and checking it means judging answers.")


# ----------------------------------------------------------------------
# Not in this lesson (on purpose)
#
# - Judging generated answers with a model.                Later lesson.
# - Hybrid search: keyword + embeddings combined.           Lesson 13.
# - A bigger eval set. 15 questions is enough to see big gaps, not small
#   ones. Real teams grow it from questions users actually ask.
# ----------------------------------------------------------------------
