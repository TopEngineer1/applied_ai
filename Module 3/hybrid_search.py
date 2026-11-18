"""
Lesson 13 — Hybrid search and reranking.

Lesson 12 measured our two retrievers. Keyword search found every
identifier question and almost none of the rephrased ones. Embeddings did
the reverse. Each misses what the other catches.

Today we combine them, and for the first time in the course we change
something and PROVE whether it helped: lesson 12's eval set runs again on
every version.

From this lesson on, the code from lessons 10-12 lives in retriever.py,
so this file shows only what's new.

Setup:
    pip install -r requirements.txt     # from the repo root
    # .env in the repo root (see README):  OPENAI_API_KEY=sk-...
    #                                       COHERE_API_KEY=...   (free trial key works)

Run:
    python hybrid_search.py
"""

import os
import time

import cohere
from pydantic import BaseModel

from retriever import (CATEGORIES, EVAL_SET, client, embed_all, heading, keyword_ranking,
                       load_chunks, rank_of_answer, recall_at, vector_ranking)

MODEL = "gpt-4o-mini"
PRICE_PER_1M_INPUT_TOKENS = 0.15   # gpt-4o-mini, USD; check current pricing
PRICE_PER_1M_OUTPUT_TOKENS = 0.60

REPHRASED = "What happens after three declines in a row?"

chunks = load_chunks()
questions = [case["question"] for case in EVAL_SET]

# Embed every chunk and every question once, up front.
chunk_vectors = embed_all([chunk["text"] for chunk in chunks])
question_vectors = embed_all(questions)

keyword_rankings = []
vector_rankings = []
for question, question_vector in zip(questions, question_vectors):
    keyword_rankings.append(keyword_ranking(chunks, question))
    vector_rankings.append(vector_ranking(chunks, chunk_vectors, question_vector))


def print_row(name, rankings):
    """One line of lesson 12's table: recall@1/3/5, then recall@3 per question type."""
    overall = [f"{recall_at(rankings, k):>5.0%}" for k in (1, 3, 5)]
    by_type = [f"{recall_at(rankings, 3, category):>13.0%}" for category in CATEGORIES]
    print(f"  {name:<14} {' '.join(overall)}  {' '.join(by_type)}")


def print_header():
    print(f"  {'':<14} {'@1':>5} {'@3':>5} {'@5':>5}  "
          + " ".join(f"{category + ' @3':>13}" for category in CATEGORIES))


# ----------------------------------------------------------------------
# v0 · Where lesson 12 left us
# ----------------------------------------------------------------------

print("\n===== v0 — the two retrievers, from lesson 12 =====\n")

print_header()
print_row("keyword", keyword_rankings)
print_row("vector", vector_rankings)


# ----------------------------------------------------------------------
# v1 · Naive union: top 3 from each, send them all
# ----------------------------------------------------------------------

print("\n===== v1 — union: top 3 keyword + top 3 vector =====\n")


def union(keyword_ranked, vector_ranked, k=3):
    """Top k from each retriever, duplicates removed. Up to 2k chunks."""
    picked = []
    for score, chunk in keyword_ranked[:k] + vector_ranked[:k]:
        if chunk not in picked:
            picked.append(chunk)
    return [(None, chunk) for chunk in picked]


union_rankings = [union(kw, vec) for kw, vec in zip(keyword_rankings, vector_rankings)]
chunks_sent = sum(len(ranking) for ranking in union_rankings) / len(union_rankings)

# Union isn't ranked (it's keyword's 3, then vector's 3), so @1 and @3 would
# just repeat keyword's numbers. Compare what matters here instead: is the
# right chunk ANYWHERE in what we send, and how much do we send?
print("Is the right chunk anywhere in what we send?\n")
print(f"  {'':<14} {'sent':>5} {'recall':>7}  "
      + " ".join(f"{category:>13}" for category in CATEGORIES))
for name, rankings, k, sent in [("keyword top 3", keyword_rankings, 3, 3),
                                ("vector top 3", vector_rankings, 3, 3),
                                ("union", union_rankings, 6, chunks_sent)]:
    by_type = [f"{recall_at(rankings, k, category):>13.0%}" for category in CATEGORIES]
    print(f"  {name:<14} {sent:>5.1f} {recall_at(rankings, k):>7.0%}  {' '.join(by_type)}")

print("\n→ Every right chunk now reaches the prompt. But most of the extra chunks")
print("  are noise: lesson 9's problem again, more tokens and more to read past.")
print("  It works, and it's wasteful. We want one ranked list.")


# ----------------------------------------------------------------------
# v2 · The obvious fix: average the two scores
#
# Everyone's first idea. Watch what the two kinds of score look like
# side by side.
# ----------------------------------------------------------------------

print("\n===== v2 — 0.5 × keyword score + 0.5 × cosine =====\n")


def weighted(keyword_ranked, vector_ranked, keyword_weight=0.5):
    keyword_score = {chunk["text"]: score for score, chunk in keyword_ranked}
    combined = []
    for cosine_score, chunk in vector_ranked:
        score = keyword_weight * keyword_score[chunk["text"]] + (1 - keyword_weight) * cosine_score
        combined.append((score, chunk))
    combined.sort(key=lambda pair: pair[0], reverse=True)
    return combined


index = questions.index(REPHRASED)
keyword_score = {chunk["text"]: score for score, chunk in keyword_rankings[index]}
cosine_score = {chunk["text"]: score for score, chunk in vector_rankings[index]}

print(f"Q: {REPHRASED}\n")
print(f"  {'keyword':>7} {'cosine':>7} {'average':>8}  chunk")
for score, chunk in weighted(keyword_rankings[index], vector_rankings[index])[:5]:
    print(f"  {keyword_score[chunk['text']]:>7} {cosine_score[chunk['text']]:>7.3f} {score:>8.3f}"
          f"  {chunk['source']:<20} {heading(chunk)}")

shown = [cosine_score[chunk["text"]] for score, chunk in weighted(keyword_rankings[index], vector_rankings[index])[:5]]
print(f"\n→ Keyword scores are whole numbers. These cosine scores are all within "
      f"{max(shown) - min(shown):.2f} of each other.")
print("  Half of a 1-point keyword gap is 0.5; half of the cosine gap is tiny.")
print("  So keyword decides, and cosine only breaks ties.")

weighted_rankings = [weighted(kw, vec) for kw, vec in zip(keyword_rankings, vector_rankings)]
print()
print_header()
print_row("keyword", keyword_rankings)
print_row("weighted 0.5", weighted_rankings)
print("\n→ Better than keyword, but only because keyword ties are so common, so")
print("  cosine gets to break them. When a wrong chunk has ONE more keyword hit,")
print("  no cosine score can overtake it: rephrased questions stay low.")
print("→ The scores are on different scales. Adding them doesn't combine two")
print("  retrievers; it builds a keyword retriever with a tie-breaker.")


# ----------------------------------------------------------------------
# v3 · Reciprocal Rank Fusion: throw the scores away, keep the positions
#
# If the scores can't be compared, don't use them. Every retriever has a
# 1st, 2nd, 3rd... and a position means the same thing in any retriever.
# A chunk near the top of BOTH lists beats one at the top of just one.
# ----------------------------------------------------------------------

print("\n===== v3 — Reciprocal Rank Fusion =====\n")


def rrf(rankings, k=60):
    """Each retriever gives a chunk 1 / (k + its rank). Add them up."""
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


rrf_rankings = [rrf([kw, vec]) for kw, vec in zip(keyword_rankings, vector_rankings)]

print_header()
print_row("keyword", keyword_rankings)
print_row("vector", vector_rankings)
print_row("weighted 0.5", weighted_rankings)
print_row("RRF k=60", rrf_rankings)

missed = [case["question"] for ranking, case in zip(rrf_rankings, EVAL_SET)
          if case["evidence"] and (rank_of_answer(ranking, case["evidence"]) or 99) > 3]
print(f"\n→ Best top 3 so far ({recall_at(rrf_rankings, 3):.0%}), and it keeps keyword's "
      f"identifiers ({recall_at(rrf_rankings, 3, 'identifier'):.0%}).")
print(f"  Still missed at top 3: {missed}")
print(f"→ At top 5 it finds {recall_at(rrf_rankings, 5):.0%}: union's recall, but as one ranked"
      " list, so WE choose how many chunks to send.")

print("\nk=60 comes from the original RRF paper. It's a convention, not a law.")
print("We have an eval set now, so we can check it:\n")
for k in [1, 10, 60, 200]:
    fused = [rrf([kw, vec], k=k) for kw, vec in zip(keyword_rankings, vector_rankings)]
    print(f"  k={k:<4} recall@1 {recall_at(fused, 1):>4.0%}   recall@3 {recall_at(fused, 3):>4.0%}")
print("\n→ Small k trusts the very top of each list: better at #1, worse at top 3.")
print("  Above 10 it makes no difference on our set. 60 is fine, and now we've")
print("  checked instead of copying it.")


# ----------------------------------------------------------------------
# v4 · Reranking: score the question and the chunk TOGETHER
#
# Not a better fusion, a different mechanism. Every score so far compared
# two things computed separately: a chunk vector made with no idea what
# question would come. A reranker reads the question and the chunk
# together, so it can judge whether the chunk actually ANSWERS it.
#
# Nothing can be precomputed, so it's slow and costs money per question.
# The pattern: retrieve 20 cheaply, rerank those 20 expensively, keep 3.
#
# Dedicated rerankers (Cohere, cross-encoders) score each question-chunk
# pair. We use what we already have: gpt-4o-mini with lesson 8's
# structured output, scoring all 20 in one call.
# ----------------------------------------------------------------------

print("\n===== v4 — rerank the top 20 with a model =====\n")


class ChunkScore(BaseModel):
    chunk: int       # the number we gave the chunk in the prompt
    relevance: int   # 0 = unrelated, 10 = answers the question


class Relevance(BaseModel):
    scores: list[ChunkScore]


RERANK_SYSTEM = """You judge search results for Nimbus documentation.
For EVERY numbered chunk, score how well it answers the question:
0 = unrelated, 5 = related but doesn't answer, 10 = directly answers it."""


def rerank(question, candidates):
    """Ask the model to score each candidate chunk. Returns the reranked list and usage."""
    numbered = "\n\n".join(f"[{i}]\n{chunk['text']}" for i, (score, chunk) in enumerate(candidates))
    response = client.chat.completions.parse(
        model=MODEL,
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
    reranked.sort(key=lambda pair: pair[0], reverse=True)  # stable: ties keep RRF order
    return reranked, response.usage


CANDIDATES = 20

# A reranker can only reorder what retrieval hands it. If the right chunk
# isn't in the 20 candidates, no reranker can bring it back. So the number
# that matters now is recall@20: it's the reranker's ceiling.
print(f"RRF recall@{CANDIDATES} (the most reranking can reach): "
      f"{recall_at(rrf_rankings, CANDIDATES):.0%}")
print("→ Fetch too few candidates and you cap quality before the expensive step")
print("  runs. Fetch too many and you pay to score every one.\n")

reranked_rankings = []
input_tokens = output_tokens = 0
started = time.perf_counter()
for question, ranking in zip(questions, rrf_rankings):
    reranked, usage = rerank(question, ranking[:CANDIDATES])
    reranked_rankings.append(reranked)
    input_tokens += usage.prompt_tokens if usage else 0
    output_tokens += usage.completion_tokens if usage else 0
seconds = time.perf_counter() - started
seconds_llm = seconds / len(questions)

print(f"Q: {REPHRASED}\n")
print("  before (RRF)                                after (reranked)")
for (s1, before), (s2, after) in zip(rrf_rankings[index][:5], reranked_rankings[index][:5]):
    print(f"  {heading(before)[:40]:<42}  {s2:>2}  {heading(after)[:40]}")
evidence = EVAL_SET[index]["evidence"]
before_rank = rank_of_answer(rrf_rankings[index], evidence)
after_rank = rank_of_answer(reranked_rankings[index], evidence)
top_score = reranked_rankings[index][0][0]
wrong_at_top = [heading(chunk) for score, chunk in reranked_rankings[index]
                if score == top_score and evidence not in chunk["text"]]
print(f"\n→ The right section moved from #{before_rank} to #{after_rank}.")
print(f"→ But the model also gave {top_score}/10 to chunks that don't answer it: {wrong_at_top}.")
print("  A model judging relevance has blind spots too. And it's a model: run")
print("  this again and these scores can change, even at temperature 0.")

score_counts = {}
for score, chunk in reranked_rankings[index]:
    score_counts[score] = score_counts.get(score, 0) + 1
summary = ", ".join(f"{count} × {score}" for score, count in sorted(score_counts.items(), reverse=True))
print(f"\nScores the model gave all {CANDIDATES} candidates: {summary}")
print("→ Look how coarse that is. Chunks with the same score keep RRF's order,")
print("  so this is mostly \"RRF, sorted into a few buckets\". A dedicated")
print("  reranker (a cross-encoder) gives every pair a continuous score and")
print("  orders all 20. We get most of the benefit with no new dependency,")
print("  but know what we're trading.")

print()
print_header()
print_row("RRF k=60", rrf_rankings)
print_row("RRF + rerank", reranked_rankings)

cost = (input_tokens * PRICE_PER_1M_INPUT_TOKENS + output_tokens * PRICE_PER_1M_OUTPUT_TOKENS) / 1_000_000
print(f"\nReranking adds, per question: {input_tokens / len(questions):,.0f} input tokens, "
      f"${cost / len(questions):.5f}, {seconds / len(questions):.1f}s")

# Narration matches today's run. The reranker is a model, so re-check before re-recording.
print(f"\n→ The right chunk is in the top 3 for {recall_at(reranked_rankings, 3):.0%} of questions,"
      f" and first for {recall_at(reranked_rankings, 1):.0%}.")
print("→ The money is small. The time isn't: retrieval took milliseconds, and")
print("  reranking adds seconds before the answer can even start. Worth it?")
print("  That depends on the product, and now you have numbers to decide with.")
print("→ Careful: we've tuned against these same 16 questions for two lessons.")
print("  100% means \"no misses we know of\". Real teams keep adding questions")
print("  that real users ask.")



# ----------------------------------------------------------------------
# v5 · A dedicated reranker (Cohere)
#
# Same job as v4, but a model built only for this: it reads the question
# and each chunk together and returns a continuous relevance score from
# 0 to 1, so it orders all 20 candidates, not a few buckets.
#
# New vendor, new key (COHERE_API_KEY in .env). Trial keys are limited to
# a few calls per minute, so this section waits when it hits the limit.
# ----------------------------------------------------------------------

print("\n===== v5 — rerank the top 20 with Cohere =====\n")

COHERE_MODEL = "rerank-v5.0-fast"  # "rerank-v5.0-pro" scored the same on our eval set
cohere_client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])


def cohere_rerank(question, candidates):
    """Returns the candidates best first, with Cohere's score, and how long the call took."""
    documents = [chunk["text"] for score, chunk in candidates]
    while True:
        try:
            started = time.perf_counter()
            response = cohere_client.rerank(model=COHERE_MODEL, query=question,
                                            documents=documents, top_n=len(documents))
            call_seconds = time.perf_counter() - started
            break
        except cohere.errors.TooManyRequestsError:
            time.sleep(10)  # trial-key rate limit: wait and try again
    # Each result says which document it was (index) and how relevant it is.
    reranked = [(result.relevance_score, candidates[result.index][1]) for result in response.results]
    return reranked, call_seconds


cohere_rankings = []
cohere_seconds = 0
for question, ranking in zip(questions, rrf_rankings):
    reranked, call_seconds = cohere_rerank(question, ranking[:CANDIDATES])
    cohere_rankings.append(reranked)
    cohere_seconds += call_seconds

print(f"Q: {REPHRASED}\n")
for score, chunk in cohere_rankings[index][:5]:
    print(f"  {score:.3f}  {chunk['source']:<20} {heading(chunk)}")
print("\n→ Every chunk gets its own score. No buckets, no ties.")

print()
print_header()
print_row("RRF k=60", rrf_rankings)
print_row("RRF + gpt", reranked_rankings)
print_row("RRF + Cohere", cohere_rankings)

print(f"\nPer question:  gpt-4o-mini rerank {seconds_llm:.1f}s, ${cost / len(questions):.5f}"
      f"   |   Cohere {cohere_seconds / len(questions):.1f}s, 1 search unit")
# Cohere bills per "search unit" (one query, up to 100 documents). Check
# its pricing page for the current price per 1,000 searches.

# Lesson 12 showed cosine scores can't tell "no answer" from "answer".
# A reranker reads the question and chunk together. Can its score?
answerable = [cohere_rankings[i][0][0] for i, case in enumerate(EVAL_SET) if case["evidence"]]
unanswerable = [cohere_rankings[i][0][0] for i, case in enumerate(EVAL_SET) if not case["evidence"]]
print(f"\nCohere's top score:  answerable {min(answerable):.2f}–{max(answerable):.2f},"
      f"  no answer {min(unanswerable):.2f}–{max(unanswerable):.2f}")
if max(unanswerable) < min(answerable):
    print("→ No overlap. On our eval set, a cutoff between those ranges catches")
    print("  every no-answer question BEFORE generation: skip the model and say")
    print("  \"not in the docs\". Cosine couldn't do that (lesson 12, v4).")
    print("  With 4 no-answer questions that's a hint, not a proof.")
else:
    print("→ The ranges overlap, so no single cutoff separates them.")


# ----------------------------------------------------------------------
# Where this leaves us
# ----------------------------------------------------------------------

print("\n===== where this leaves us =====\n")
print("On our eval set, the right chunk now reaches the prompt every time.")
print("So every wrong answer left is a GENERATION problem, not a retrieval one.")
print("We've isolated the failure to one layer.")
print("\nMeasuring that layer is harder: there's no evidence string to check, so")
print("we need a model to judge the answers. Same technique as the reranker,")
print("different job. That's lesson 14.")
