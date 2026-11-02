"""
Tokenization 101 with tiktoken: English vs Hindi token counts and cost.

Setup:
    pip install tiktoken openai python-dotenv
    # .env next to this script:  OPENAI_API_KEY=sk-...

Run:
    python tokens.py
"""

import os
import tiktoken

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    print("Note: python-dotenv not installed, .env not loaded.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-dummy-key-1234567890abcdef")

# Dummy price for illustration
PRICE_PER_1M_INPUT_TOKENS = 2.50


def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# Part 1: Load a tokenizer
line("PART 1: Loading a tokenizer")

enc = tiktoken.get_encoding("o200k_base")  # used by GPT-4o
# or: enc = tiktoken.encoding_for_model("gpt-4o")

print(f"Encoding name : {enc.name}")
print(f"Vocab size    : {enc.n_vocab:,} possible tokens")


# Part 2: Encode and decode
line("PART 2: Text -> token IDs -> text")

text = "Hello, how are you?"
token_ids = enc.encode(text)

print(f"Text       : {text!r}")
print(f"Token IDs  : {token_ids}")
print(f"# tokens   : {len(token_ids)}")
print(f"Decoded    : {enc.decode(token_ids)!r}")


def show_pieces(text):
    pieces = []
    for tid in enc.encode(text):
        raw = enc.decode_single_token_bytes(tid)
        try:
            pieces.append(raw.decode("utf-8"))
        except UnicodeDecodeError:  # token is part of a character
            pieces.append(f"<{raw.hex()}>")
    print(" | ".join(repr(p) for p in pieces))


print("\nEach token as text:")
show_pieces(text)

print("\nCommon vs rare words:")
for word in ["cat", "unbelievable", "antidisestablishmentarianism"]:
    print(f"  {word:30} -> {len(enc.encode(word))} token(s): ", end="")
    show_pieces(word)


# Part 3: English vs Hindi
line("PART 3: English vs Hindi")

examples = [
    ("Hello, how are you?",
     "नमस्ते, आप कैसे हैं?"),
    ("What is the weather like today?",
     "आज मौसम कैसा है?"),
    ("I love learning about artificial intelligence.",
     "मुझे कृत्रिम बुद्धिमत्ता के बारे में सीखना पसंद है।"),
    ("Please summarize this document in three bullet points.",
     "कृपया इस दस्तावेज़ का तीन बिंदुओं में सारांश दें।"),
]

print(f"{'Language':10} {'Chars':>6} {'Tokens':>7}  Text")
print("-" * 70)
for en, hi in examples:
    for lang, s in [("English", en), ("Hindi", hi)]:
        print(f"{lang:10} {len(s):>6} {len(enc.encode(s)):>7}  {s}")
    print()

print("A Hindi word split into tokens:")
show_pieces("नमस्ते")


# Part 4: Cost
line("PART 4: Hindi costs more (same meaning, more tokens)")

english_doc = " ".join(en for en, _ in examples)
hindi_doc = " ".join(hi for _, hi in examples)

en_tokens = len(enc.encode(english_doc))
hi_tokens = len(enc.encode(hindi_doc))

print(f"English tokens : {en_tokens}")
print(f"Hindi tokens   : {hi_tokens}")
print(f"Hindi uses {hi_tokens / en_tokens:.2f}x the tokens of English")

messages_per_month = 1_000_000
en_cost = en_tokens * messages_per_month / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
hi_cost = hi_tokens * messages_per_month / 1_000_000 * PRICE_PER_1M_INPUT_TOKENS
print(f"\nMonthly input cost for {messages_per_month:,} messages "
      f"(@ ${PRICE_PER_1M_INPUT_TOKENS}/1M tokens, dummy price):")
print(f"  English : ${en_cost:,.2f}")
print(f"  Hindi   : ${hi_cost:,.2f}   (+${hi_cost - en_cost:,.2f})")

print("""
Why?
  * Tokenizers are trained mostly on English, so English words become
    single tokens while Hindi words get split into smaller pieces.
  * Devanagari characters are 3 bytes each in UTF-8 (English letters are 1).

Result for Hindi users: higher cost, context window fills faster,
slower responses.
""")


# Part 5: Real API call
line("PART 5: Token counts from a real API call")

if OPENAI_API_KEY.startswith("sk-dummy"):
    print("Dummy API key -> skipping. Add OPENAI_API_KEY to .env to run this.")
else:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    for prompt in ["Hello, how are you?", "नमस्ते, आप कैसे हैं?"]:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        u = resp.usage
        print(f"{prompt!r}: prompt={u.prompt_tokens}, "
              f"completion={u.completion_tokens}, total={u.total_tokens}")
    print("(prompt_tokens includes a few extra chat formatting tokens.)")
