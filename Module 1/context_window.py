"""
Context Windows — short, real examples with the actual OpenAI API.

The context window = the max tokens a model can handle in ONE request.
It's shared by everything: your input + the model's output.

Setup:
    pip install tiktoken openai python-dotenv
    # .env next to this file:  OPENAI_API_KEY=sk-...

Run:
    python context_window.py
"""

import os
import tiktoken
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
enc = tiktoken.get_encoding("o200k_base")

MODEL = "gpt-4o-mini"


def tokens(text):
    return len(enc.encode(text))


def line(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ----------------------------------------------------------------------
# 1. HAPPY PATH — a normal request that fits fine
# ----------------------------------------------------------------------
line("1. Happy path: a normal request")

prompt = "In one sentence, what is a context window?"
print(f"Input: {tokens(prompt)} tokens")

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
)
print(f"Reply: {resp.choices[0].message.content}")
print(f"finish_reason: {resp.choices[0].finish_reason}")   # 'stop' = finished cleanly
print(f"Usage: {resp.usage.prompt_tokens} in + "
      f"{resp.usage.completion_tokens} out = {resp.usage.total_tokens} total")


# ----------------------------------------------------------------------
# 2. OUTPUT CUT OFF — cap max_tokens so the reply can't finish
# ----------------------------------------------------------------------
line("2. Output overflow: the reply gets cut off")

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "List 20 uses of AI with a sentence each."}],
    max_tokens=30,   # not enough room to finish
)
print(f"Reply: {resp.choices[0].message.content!r}")
print(f"finish_reason: {resp.choices[0].finish_reason}")   # 'length' = ran out of room
print("\n-> 'length' means the model was cut off, not done. The answer is incomplete.")


# ----------------------------------------------------------------------
# 3. INPUT TOO BIG — send more tokens than the window holds
# ----------------------------------------------------------------------
line("3. Input overflow: too many tokens going in")

big_input = "The quick brown fox jumps over the lazy dog. " * 40000
print(f"Input size: {tokens(big_input):,} tokens (gpt-4o-mini holds 128,000)")

try:
    client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": big_input}],
    )
except Exception as e:
    print(f"\nAPI rejected it:\n  {e}")
    print("\n-> The API tells you the limit and your size. It simply won't fit.")


# ----------------------------------------------------------------------
# The takeaway
# ----------------------------------------------------------------------
line("Takeaway")
print("""
One window, shared by input + output.
  finish_reason='stop'   -> finished cleanly
  finish_reason='length' -> reply hit the cap, got cut off
  input too big          -> request rejected before it even runs

Next: how we handle a full window (trimming, summarizing, and RAG).
""")