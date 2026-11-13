"""
Lesson 7 — Prompt engineering, one layer at a time.

Each version adds exactly one thing. Run them in order and watch the
output change. The last section is the one that matters: it shows why
asking for a format is not the same as getting one.

Setup:
    pip install -r requirements.txt     # from the repo root
    # .env in the repo root (see README):  OPENAI_API_KEY=sk-...

Run:
    python prompt_engineering.py
"""

import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # searches upward, so the repo-root .env is found
client = OpenAI()
MODEL = "gpt-4o-mini"
QUESTION = "Explain retries."


def ask(system, user, temperature=0.7):
    """Send one question to the model and return its reply as text."""
    messages = []
    if system:  # v1 has no system prompt, every later version does
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=messages,
    )
    return response.choices[0].message.content or ""   # content can be None (e.g. a refusal)


# ───────────────────────────────────────────────────────────────────────
# v1 · Nothing but the question
#
# Every gap you leave is filled with whatever is most probable in the
# training data. That average is a textbook, so you get a textbook.
# ───────────────────────────────────────────────────────────────────────

answer = ask(None, QUESTION)
print("\n===== v1 — bare question =====")
print(answer)


# ───────────────────────────────────────────────────────────────────────
# v2 · + ROLE and CONTEXT
#
# Who is answering, and about what. This alone moves the answer from
# "what is a retry" to "retries in the system you actually have".
# ───────────────────────────────────────────────────────────────────────

v2 = """You are a senior backend engineer at a payments company.
You are answering a colleague who is designing the retry path for card charges."""

answer = ask(v2, QUESTION)
print("\n===== v2 — + role and context =====")
print(answer)


# ───────────────────────────────────────────────────────────────────────
# v3 · + CONSTRAINTS
#
# Each constraint is a decision you make instead of leaving it to the
# model. Length, audience, and what to leave out.
# ───────────────────────────────────────────────────────────────────────

v3 = v2 + """

Constraints:
- Under 120 words.
- The reader knows HTTP and has shipped production services.
- Do not define basic terms.
- No introduction and no closing summary."""

answer = ask(v3, QUESTION)
print("\n===== v3 — + constraints =====")
print(answer)


# ───────────────────────────────────────────────────────────────────────
# v4 · + OUTPUT FORMAT
#
# Now the shape is predictable enough to put on a screen or in a ticket.
# ───────────────────────────────────────────────────────────────────────

v4 = v3 + """

Respond in exactly this shape:

RISK: <the failure this causes in production>
APPROACH: <what to do about it>
GOTCHA: <the part people get wrong>"""

answer = ask(v4, QUESTION)
print("\n===== v4 — + output format =====")
print(answer)


# ───────────────────────────────────────────────────────────────────────
# v5 · + ONE EXAMPLE
#
# The instruction above described the shape. It could not describe the
# level of specificity. One example does that in a way prose cannot.
# ───────────────────────────────────────────────────────────────────────

v5 = v4 + """

Match this level of specificity:

RISK: A retried charge writes a second ledger entry and double-bills the customer.
APPROACH: Derive an idempotency key from the order ID and send it on every attempt.
GOTCHA: The key has to survive a process restart, so generate it upstream of the retry loop."""

answer = ask(v5, QUESTION)
print("\n===== v5 — + one example =====")
print(answer)


# ───────────────────────────────────────────────────────────────────────
# v6 · Now ask for JSON — and try to actually use it
#
# This is the part that sets up the next lesson. Everything above was
# for a human to read. The moment code has to consume the output, a
# formatting *instruction* stops being good enough.
# ───────────────────────────────────────────────────────────────────────

json_prompt = """You are a senior backend engineer at a payments company.

Return ONLY valid JSON matching this schema.
No markdown, no code fences, no prose before or after:

{"risk": string, "approach": string, "gotcha": string, "confidence": number}"""

RUNS = 10
REQUIRED_KEYS = ["risk", "approach", "gotcha", "confidence"]
failures = 0

print(f"\n===== v6 — asking for JSON, {RUNS} runs =====")

for run in range(1, RUNS + 1):
    raw = ask(json_prompt, QUESTION, temperature=1.0)  # higher temperature = more variety

    # Step 1: can code even parse the reply as JSON?
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        failures += 1
        print(f"run {run}: NOT VALID JSON ({error})")
        print(f"       reply starts with: {raw[:70]}")
        continue

    # Step 2: it parsed, but does it have every field we asked for?
    missing = []
    for key in REQUIRED_KEYS:
        if key not in data:
            missing.append(key)

    if missing:
        failures += 1
        print(f"run {run}: parsed, but missing {missing}")
    else:
        print(f"run {run}: ok")

print(f"\n{failures}/{RUNS} runs were not usable by code.")


# ───────────────────────────────────────────────────────────────────────
# v7 · The failure you can reproduce on demand
#
# Even when v6 happens to pass, this breaks it reliably: asking the
# model to reason AND to emit strict JSON puts two instructions in
# direct conflict. It is failure mode #5 from the lecture.
# ───────────────────────────────────────────────────────────────────────

conflict_prompt = json_prompt + "\n\nThink through the problem step by step before you answer."

raw = ask(conflict_prompt, QUESTION, temperature=1.0)

print("\n===== v7 — reasoning + strict format, in conflict =====")
print(raw[:400])  # first 400 characters is enough to see what came back

try:
    json.loads(raw)
    print("\n→ parsed (try again, this one is unstable)")
except json.JSONDecodeError as error:
    print(f"\n→ NOT VALID JSON ({error})")


# ───────────────────────────────────────────────────────────────────────
# The point
#
# You asked for a format. You did not get a guarantee.
#
# And "it parses 95% of the time" is not a thing you can ship: 5% of
# production traffic throwing a JSONDecodeError is an incident, and
# there is no way to reach 100% by asking more firmly.
#
# ───────────────────────────────────────────────────────────────────────