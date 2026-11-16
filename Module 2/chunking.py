"""
Lesson 10 — Chunking: how to cut documents into retrievable pieces.

In lesson 9 our retriever returned whole files. That worked because our
files are small. But the answer to our question lives in one section of
retry-ladder.md, and we sent the model all of it anyway. Real documents
are much longer, so we cut them into chunks and retrieve only the chunks
we need.

How you cut matters. Cut in the wrong place and the fact you need is
split across two chunks, and neither one answers the question.

Setup:
    pip install -r requirements.txt     # from the repo root
    # .env in the repo root (see README):  OPENAI_API_KEY=sk-...

Run:
    python chunking.py
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
DOCS_FOLDER = Path(__file__).parent / "docs"

SYSTEM = """You are a support engineer for Nimbus, a payments platform.
Answer ONLY using the documentation provided in the user's message.
If the answer is not in the documentation, say so."""

QUESTION = "What does the Nimbus retry ladder do after a charge fails three consecutive times?"

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


def load_docs():
    """Read every .md file in docs/ (not docs/deprecated/) into {file name: text}."""
    docs = {}
    for path in sorted(DOCS_FOLDER.glob("*.md")):
        docs[path.name] = path.read_text()
    return docs


def best_chunk(chunks, question):
    """Keyword retrieval, adapted for chunks.

    In lesson 9 a file had to contain ALL the keywords. Chunks are small,
    so that's too strict: almost nothing would match. Instead we score each
    chunk by how many keywords it contains, and keep the highest scorer.
    If several chunks tie, the first one in the list wins.
    """
    words = re.findall(r"[a-z0-9]+", question.lower())
    keywords = [word for word in words if word not in STOP_WORDS]

    scores = []
    for chunk in chunks:
        score = 0
        for keyword in keywords:
            if keyword in chunk["text"].lower():
                score += 1
        scores.append(score)

    best_score = max(scores)
    best = chunks[scores.index(best_score)]  # the FIRST chunk with the top score
    tied = scores.count(best_score)

    print(f"Searching for: {keywords}")
    print(f"Best chunk: from {best['source']}, matched {best_score}/{len(keywords)} keywords"
          f" ({tied} chunk(s) tied for the top score)")
    return best


def ask_with_chunk(chunk, question):
    prompt = f"Documentation:\n\n{chunk['text']}\n\nQuestion: {question}"
    print(f"Input tokens: {count_tokens(SYSTEM + prompt):,}\n")
    print(ask(SYSTEM, prompt))


docs = load_docs()
ladder = docs["retry-ladder.md"]


# ─────────────────────────────────────────────────────────────────────────
# v0 · Why chunk at all?
#
# Lesson 9's retriever found the right file and sent all of it. But the
# answer is one section. Everything else in the file is cost, and noise
# the model has to read past.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v0 — why chunk at all? =====")

start = ladder.index("## Consecutive failure handling")
end = ladder.index("### Promotion is not a reset")
answer_section = ladder[start:end].strip()
print(f"Whole retry-ladder.md:           {count_tokens(ladder):,} tokens")
print(f"The section that has the answer: {count_tokens(answer_section):,} tokens")


# ─────────────────────────────────────────────────────────────────────────
# v1 · Fixed-size chunks
#
# The simplest strategy: cut every N characters, no matter what's there.
# It's easy, and every chunk is the same size. But the cuts land wherever
# the count runs out: mid-paragraph, mid-sentence, mid-word.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v1 — fixed-size chunks (300 characters) =====")


def fixed_size_chunks(source, text, size):
    """Cut text into pieces of `size` characters."""
    chunks = []
    for start in range(0, len(text), size):  # start = 0, 300, 600, ...
        piece = text[start : start + size]   # characters 0-299, then 300-599, ...
        chunks.append({"source": source, "text": piece})
    return chunks


fixed_chunks = []
for name, text in docs.items():
    fixed_chunks += fixed_size_chunks(name, text, size=300)

print(f"{len(docs)} documents → {len(fixed_chunks)} chunks\n")

chunk = best_chunk(fixed_chunks, QUESTION)
print("-" * 40)
print(chunk["text"])
print("-" * 40)
# This narration describes where the 300-character cut lands in today's
# retry-ladder.md. Edit that file and it can silently stop matching what
# prints above, so re-run and re-check it before re-recording.
print("\n→ The chunk stops at \"...secondary acquirer, and\". The rest of the")
print("  sentence, \"the ladder resumes at rung 2\", is in the next chunk.\n")

ask_with_chunk(chunk, QUESTION)
print("\n→ Compare with lesson 9's answer. Does it say where the ladder resumes?")
print("  Nothing here is false, but it can't mention rung 2: that's in another")
print("  chunk. An incomplete answer looks just as confident as a complete one.")


# ─────────────────────────────────────────────────────────────────────────
# v2 · Fixed-size chunks with overlap
#
# Start each chunk a little before the previous one ended. Now a sentence
# cut at one boundary appears whole in the neighbouring chunk.
#
# Overlap is a safety net, not a fix. It costs extra (the same text is
# stored and sent more than once), and the retriever still has to pick
# the right chunk.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v2 — fixed-size chunks with 100 characters of overlap =====")


def overlapping_chunks(source, text, size, overlap):
    """Like fixed_size_chunks, but each chunk starts `overlap` characters early."""
    chunks = []
    step = size - overlap                    # 300 - 100 = move forward 200 each time
    for start in range(0, len(text), step):  # start = 0, 200, 400, ...
        piece = text[start : start + size]   # still 300 characters long
        chunks.append({"source": source, "text": piece})
    return chunks


def count_characters(chunks):
    total = 0
    for chunk in chunks:
        total += len(chunk["text"])
    return total


def count_chunks_with_whole_fact(chunks):
    """How many chunks contain BOTH halves of the answer?"""
    count = 0
    for chunk in chunks:
        if "*promoted*" in chunk["text"] and "**rung 2**" in chunk["text"]:
            count += 1
    return count


no_overlap = fixed_size_chunks("retry-ladder.md", ladder, size=300)
with_overlap = overlapping_chunks("retry-ladder.md", ladder, size=300, overlap=100)

characters_before = count_characters(no_overlap)
characters_after = count_characters(with_overlap)
whole_before = count_chunks_with_whole_fact(no_overlap)
whole_after = count_chunks_with_whole_fact(with_overlap)

print("retry-ladder.md, 300-character chunks:")
print(f"  no overlap:   {len(no_overlap)} chunks, {characters_before:,} characters, "
      f"{whole_before} chunk(s) with the whole fact")
print(f"  100 overlap:  {len(with_overlap)} chunks, {characters_after:,} characters, "
      f"{whole_after} chunk(s) with the whole fact")

extra = characters_after / characters_before - 1
print(f"\n→ Overlap kept the fact together, for {extra:.0%} more text stored and sent.")
print("  That's a lot to pay to patch one bad cut, and it's still luck: a")
print("  different chunk size could split the fact again.")


# ─────────────────────────────────────────────────────────────────────────
# v3 · Structure-aware chunks: split on headings
#
# The document's author already divided it into sections, each about one
# thing. Cut where they did. A section is a complete thought, so a fact
# is far less likely to be split in half.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v3 — split on markdown headings =====")


def section_chunks(source, text):
    """Cut text into one chunk per markdown section (## or ### heading)."""
    sections = re.split(r"\n(?=##)", text)  # split just before each line starting with ##
    chunks = []
    for section in sections:
        chunks.append({"source": source, "text": section.strip()})
    return chunks


heading_chunks = []
for name, text in docs.items():
    heading_chunks += section_chunks(name, text)

print(f"{len(docs)} documents → {len(heading_chunks)} chunks")
print("\nretry-ladder.md was cut into:")
for chunk in section_chunks("retry-ladder.md", ladder):
    heading = chunk["text"].split("\n")[0]
    print(f"  {count_tokens(chunk['text']):>4} tokens  {heading}")
print()

chunk = best_chunk(heading_chunks, QUESTION)
print("-" * 40)
print(chunk["text"])
print("-" * 40 + "\n")

ask_with_chunk(chunk, QUESTION)


# ─────────────────────────────────────────────────────────────────────────
# v4 · Carry the heading into every chunk
#
# A chunk that starts "### Promotion is not a reset" means little on its
# own: which system? which document? Prefix every chunk with where it came
# from. It's the cheapest win in RAG: the retriever has more to match, and
# the model knows what it's reading.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v4 — carry the document's title into every chunk =====")


def section_chunks_with_title(source, text):
    """Like section_chunks, but each chunk starts with its file and document title."""
    title = text.splitlines()[0].lstrip("# ")  # the first line is the "# Title" heading
    sections = re.split(r"\n(?=##)", text)
    chunks = []
    for section in sections:
        chunks.append({"source": source, "text": f"[{source} · {title}]\n\n{section.strip()}"})
    return chunks


titled_chunks = []
for name, text in docs.items():
    titled_chunks += section_chunks_with_title(name, text)

for chunk in section_chunks_with_title("retry-ladder.md", ladder):
    if "### Promotion is not a reset" in chunk["text"]:
        print("\n".join(chunk["text"].split("\n")[:7]))
        print("...")

print("\n→ The subsection now says which document and system it belongs to.\n")

chunk = best_chunk(titled_chunks, QUESTION)
print(f"  {chunk['text'].splitlines()[2]}")  # line 0 is the title prefix, line 2 the heading
print("→ Same winner as v3: carrying the title didn't break what already worked.")


# ─────────────────────────────────────────────────────────────────────────
# v5 · Same question, different words
#
# Our best chunks yet, and a question a real user might type. It means the
# same as QUESTION, but shares almost none of its words.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== v5 — same question, different words =====")

REPHRASED = "What happens after three declines in a row?"
print(f"Q: {REPHRASED}\n")

chunk = best_chunk(titled_chunks, REPHRASED)
print("-" * 40)
print(chunk["text"][:400])
print("-" * 40 + "\n")
ask_with_chunk(chunk, REPHRASED)

# Like v1's narration, this describes today's docs: the tie count and the
# winning file come from the current text. Re-run before re-recording.
print("\n→ Nine chunks, from three files, tied at 1 of 4 keywords, and the loop")
print("  kept the first one it saw: decline-classes.md's overview. That isn't")
print("  retrieval, it's file order.")
print("→ This time the grounding instruction caught it: the wrong chunk was")
print("  obviously unrelated. But we still paid for retrieval and got no answer,")
print("  and the right section was one of the nine tied chunks. File order")
print("  buried it.")
print("→ A wrong chunk that LOOKS relevant, like an outdated version of the")
print("  same doc, won't be caught so easily. That's lesson 15.")


# ─────────────────────────────────────────────────────────────────────────
# The trade-offs
#
# - Fixed-size: simple and predictable, but cuts through meaning.
# - Overlap: fewer broken facts, more tokens stored and sent.
# - Headings: chunks match meaning, but sizes vary a lot (look at the
#   token counts above), and it only works when documents have structure.
#   A huge section still needs splitting, and a subsection like
#   "Promotion is not a reset" is separate from the section it explains
#   (carrying the title, as in v4, softens that).
#
# Real systems mix these: split on structure first, then cut any chunk
# that's still too long, with a little overlap.
#
# And whatever the chunks, our retriever is still matching strings, so
# "three declines in a row" can't find "consecutive failure handling".
# Next: matching meaning, with embeddings and a vector database.
# ─────────────────────────────────────────────────────────────────────────
