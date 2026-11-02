"""
Embeddings — short, real examples with the actual OpenAI API.

An embedding = a list of numbers that captures the MEANING of text.
Similar meaning -> vectors point in a similar direction.

Setup:
    pip install openai python-dotenv
    # .env in the repo root (see README):  OPENAI_API_KEY=sk-...

Run:
    python embeddings.py
"""

import os
import math
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()  # searches upward, so the repo-root .env is found
except ImportError:
    pass

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "text-embedding-3-small"   # 1536 dimensions


def embed(texts):
    """Get embeddings for a list of texts in ONE API call."""
    resp = client.embeddings.create(model=MODEL, input=texts)
    return [item.embedding for item in resp.data]


def cosine(a, b):
    """Cosine similarity: 1.0 = same direction, 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def line(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ----------------------------------------------------------------------
# 1. GET AN EMBEDDING — text in, vector out
# ----------------------------------------------------------------------
line("1. Get an embedding")

resp = client.embeddings.create(model=MODEL, input="king")
vector = resp.data[0].embedding

print(f"Model      : {resp.model}")
print(f"Dimensions : {len(vector)}")
print(f"Tokens used: {resp.usage.total_tokens}")


# ----------------------------------------------------------------------
# 2. THE RAW VECTOR — it's just a list of floats
# ----------------------------------------------------------------------
line("2. The raw vector")

print(f"First 10 numbers: {[round(x, 4) for x in vector[:10]]}")
print(f"Last 5 numbers  : {[round(x, 4) for x in vector[-5:]]}")
print(f"Length (norm)   : {math.sqrt(sum(x * x for x in vector)):.4f}")
print("\n-> No single number means anything on its own. Meaning lives in the")
print("   whole pattern. OpenAI vectors are normalized to length 1.0.")


# ----------------------------------------------------------------------
# 3. WORD SIMILARITY — king / queen / banana
# ----------------------------------------------------------------------
line("3. Cosine similarity between words")

words = ["king", "queen", "banana"]
vecs = dict(zip(words, embed(words)))

pairs = [("king", "queen"), ("king", "banana"), ("queen", "banana")]
for a, b in pairs:
    print(f"  {a:>6} vs {b:<7}: {cosine(vecs[a], vecs[b]):.4f}")

print("\n-> king/queen score higher than king/banana: related meaning,")
print("   closer vectors. Scores are relative — compare them, don't read")
print("   them as absolute percentages.")


# ----------------------------------------------------------------------
# 4. SENTENCE SIMILARITY — same meaning, different words
# ----------------------------------------------------------------------
line("4. Sentence similarity: meaning beats wording")

base = "How do I reset my password?"
candidates = [
    "I forgot my login credentials, how can I change them?",  # same meaning, different words
    "How do I reset my sleep schedule?",                      # similar words, different meaning
    "What's the best pizza topping?",                         # unrelated
]

base_vec, *cand_vecs = embed([base] + candidates)

print(f"Base: {base!r}\n")
for text, v in zip(candidates, cand_vecs):
    print(f"  {cosine(base_vec, v):.4f}  {text}")

print("\n-> The paraphrase shares almost no words with the base, yet it")
print("   scores high. This is why embeddings power semantic search and RAG.")


# ----------------------------------------------------------------------
# 5. (OPTIONAL) ANALOGY — king - man + woman ≈ ?
# ----------------------------------------------------------------------
line("5. Optional: the analogy demo (king - man + woman)")

options = ["queen", "princess", "prince", "woman", "banana", "car"]
king, man, woman, *option_vecs = embed(["king", "man", "woman"] + options)

result = [k - m + w for k, m, w in zip(king, man, woman)]

ranked = sorted(
    ((cosine(result, v), word) for word, v in zip(options, option_vecs)),
    reverse=True,
)
for score, word in ranked:
    print(f"  {score:.4f}  {word}")

print("\n-> Vector math can capture relationships (royalty, gender).")
print("   Modern embedding models aren't trained for this, so the result")
print("   is often fuzzy — 'woman' may rank close to the top too.")


# ----------------------------------------------------------------------
# The takeaway
# ----------------------------------------------------------------------
line("Takeaway")
print("""
Text -> embedding -> a vector of numbers that encodes meaning.
  cosine similarity    -> how close two meanings are
  paraphrases          -> high score even with different words
  same words, new idea -> lower score

Next: store these vectors and search them (vector databases + RAG).
""")
