"""
Prompt Caching with the OpenAI API.

In a chat, every new request re-sends the WHOLE chat history.
OpenAI remembers the start of prompts it saw recently (1,024+ tokens).
If the start matches, those tokens are "cached": cheaper and faster.

Setup:
    pip install openai python-dotenv
    # .env in the repo root (see README):  OPENAI_API_KEY=sk-...

Run:
    python promp_cache.py
"""

import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


# ----------------------------------------------------------------------
# A long chat history (pretend the user has been chatting for a while)
# ----------------------------------------------------------------------
# The first message has the current time in it, so every run starts
# with a brand new history -> call 1 is always a cache miss.
chat = [{"role": "user", "content": f"Hi! This chat started at {time.time()}"}]
chat.append({"role": "assistant", "content": "Hello! How can I help?"})

for i in range(1, 150):
    chat.append({"role": "user", "content": f"What is fun fact number {i} about the ocean?"})
    chat.append({"role": "assistant", "content": f"Fun fact {i}: the ocean covers about 71% of Earth and holds about 97% of its water."})


# ----------------------------------------------------------------------
# Call 1: send the history + a new question
# ----------------------------------------------------------------------
chat.append({"role": "user", "content": "In one sentence, what have we been talking about?"})

start = time.time()
resp1 = client.chat.completions.create(model="gpt-4o-mini", messages=chat, max_tokens=40)
time1 = time.time() - start

print("CALL 1")
print("Reply         :", resp1.choices[0].message.content)
print("prompt_tokens :", resp1.usage.prompt_tokens)
print("cached_tokens :", resp1.usage.prompt_tokens_details.cached_tokens)
print("cached %      :", round(resp1.usage.prompt_tokens_details.cached_tokens / resp1.usage.prompt_tokens * 100, 1), "%")
print("time          :", round(time1, 2), "s")


# ----------------------------------------------------------------------
# Call 2: the chat continues -> add the reply + a new question
# Everything before these two new messages is exactly the same as call 1.
# ----------------------------------------------------------------------
chat.append({"role": "assistant", "content": resp1.choices[0].message.content})
chat.append({"role": "user", "content": "Thanks! Now say goodbye in one sentence."})

start = time.time()
resp2 = client.chat.completions.create(model="gpt-4o-mini", messages=chat, max_tokens=40)
time2 = time.time() - start

print("\nCALL 2")
print("Reply         :", resp2.choices[0].message.content)
print("prompt_tokens :", resp2.usage.prompt_tokens)
print("cached_tokens :", resp2.usage.prompt_tokens_details.cached_tokens)
print("cached %      :", round(resp2.usage.prompt_tokens_details.cached_tokens / resp2.usage.prompt_tokens * 100, 1), "%")
print("time          :", round(time2, 2), "s")


# ----------------------------------------------------------------------
# Cost (gpt-4o-mini: $0.15 per 1M input tokens, cached tokens are half price)
# ----------------------------------------------------------------------
price = 0.15 / 1_000_000
cached_price = 0.075 / 1_000_000

cost1 = resp1.usage.prompt_tokens * price

cached = resp2.usage.prompt_tokens_details.cached_tokens
not_cached = resp2.usage.prompt_tokens - cached
cost2 = not_cached * price + cached * cached_price

print("\nINPUT COST")
print("call 1: $", round(cost1, 6))
print("call 2: $", round(cost2, 6))
