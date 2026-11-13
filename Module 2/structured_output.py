"""
Lesson 8 — Structured outputs: from asking to enforcing.

Lesson 7 ended on a promise the model could not keep: we asked for JSON
and got no guarantee. This file walks the three levels of enforcement,
and then shows the three places where even the strongest one stops.

Setup:
    pip install -r requirements.txt     # from the repo root
    # .env in the repo root (see README):  OPENAI_API_KEY=sk-...

Run:
    python structured_output.py
"""

import json
from dotenv import load_dotenv
from openai import OpenAI, omit
from pydantic import BaseModel

load_dotenv()  # searches upward, so the repo-root .env is found
client = OpenAI()

MODEL = "gpt-4o-mini"
QUESTION = "Explain retries."

SYSTEM = (
    "You are a senior backend engineer at a payments company. "
    "You are answering a colleague designing the retry path for card charges."
)

REQUIRED_KEYS = ["risk", "approach", "gotcha", "confidence"]


def ask(system, user, response_format=None, max_tokens=None):
    """Send one question and return the model's choice.

    choice.message.content -> the reply text
    choice.finish_reason   -> why it stopped ('stop' = done, 'length' = cut off)
    """
    response = client.chat.completions.create(
        model=MODEL,
        temperature=1.0,  # high on purpose: lots of randomness to test against
        max_tokens=max_tokens,
        response_format=response_format or omit,  # omit = "don't send this setting"
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0]


def check(raw):
    """Return "ok" if raw is JSON with every required key, otherwise what's wrong."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        return f"not JSON ({error})"

    missing = []
    for key in REQUIRED_KEYS:
        if key not in data:
            missing.append(key)

    if missing:
        return f"missing keys {missing}"
    return "ok"


def ask_typed(system, user, output_class):
    """Like ask(), but returns an object of output_class instead of text."""
    completion = client.chat.completions.parse(
        model=MODEL,
        response_format=output_class,  # the class itself is the schema
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    message = completion.choices[0].message
    if message.parsed is None:  # only happens if the model refused (see 5b)
        raise RuntimeError(f"Model refused: {message.refusal}")
    return message.parsed


# ─────────────────────────────────────────────────────────────────────────
# LEVEL 0 · Ask, validate, retry
#
# This is what everyone writes first, and it is not wrong — it is just
# expensive. You pay for every failed attempt, the user waits through
# every retry, and on a prompt that genuinely cannot be satisfied the
# loop never succeeds. It runs out of attempts, then you ship an error anyway.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== LEVEL 0 — ask, validate, retry =====")

JSON_PROMPT = SYSTEM + """

Return ONLY valid JSON matching this shape.
No markdown, no code fences, no prose before or after:

{"risk": string, "approach": string, "gotcha": string, "confidence": number}"""

MAX_ATTEMPTS = 3
attempts = 0
succeeded = False

while attempts < MAX_ATTEMPTS and not succeeded:
    attempts += 1
    choice = ask(JSON_PROMPT, QUESTION)
    result = check(choice.message.content or "")
    print(f"attempt {attempts}: {result}")
    if result == "ok":
        succeeded = True

if not succeeded:
    print("gave up — the caller still needs an answer")

print(f"\n{attempts} API call(s) for one answer. You paid for all of them.")


# ─────────────────────────────────────────────────────────────────────────
# LEVEL 1 · JSON mode
#
#     response_format={"type": "json_object"}
#
# This guarantees the output parses. It guarantees NOTHING about your
# keys. The prompt below deliberately doesn't name them, so the model
# picks its own: valid JSON every time, and data["risk"] can still
# raise KeyError.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== LEVEL 1 — json_object: parses, but the shape is the model's choice =====")

VAGUE_PROMPT = SYSTEM + "\n\nReturn your advice as JSON."

usable_runs = 0

for run in range(1, 6):
    choice = ask(VAGUE_PROMPT, QUESTION, response_format={"type": "json_object"})
    raw = choice.message.content or ""
    data = json.loads(raw)  # safe: JSON mode guarantees it parses
    print(f"run {run}: keys = {list(data.keys())}")
    if check(raw) == "ok":
        usable_runs += 1

print(f"\n5 runs, all valid JSON. Only {usable_runs} had the keys our code needs: {REQUIRED_KEYS}")
print("Valid is not the same as usable.")


# ─────────────────────────────────────────────────────────────────────────
# LEVEL 2 · Structured Outputs — strict mode
#
#     response_format={"type": "json_schema", "json_schema": {... "strict": True}}
#
# Now the schema is enforced WHILE the model generates: at every step,
# tokens that would break the schema have their probability set to zero.
# They are not unlikely. They are impossible.
#
# strict: True has rules. Every property must be listed in "required",
# and "additionalProperties" must be False. You cannot have a casually
# optional field — make it a union with null instead.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== LEVEL 2 — json_schema strict: 20 runs at temperature 1.0 =====")

STRICT_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "retry_advice",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "risk": {"type": "string", "description": "the failure this causes in production"},
                "approach": {"type": "string", "description": "what to do about it"},
                "gotcha": {"type": "string", "description": "the part people get wrong"},
                "confidence": {"type": "number", "description": "0 to 1, how sure you are"},
            },
            "required": ["risk", "approach", "gotcha", "confidence"],
            "additionalProperties": False,
        },
    },
}

RUNS = 20
matched = 0

for run in range(1, RUNS + 1):
    choice = ask(SYSTEM, QUESTION, response_format=STRICT_FORMAT)
    result = check(choice.message.content or "")
    if result == "ok":
        matched += 1
    else:
        print(f"run {run}: {result}")

print(f"{matched}/{RUNS} runs matched the schema exactly.")
print("Temperature 1.0. The randomness is still there — it just has")
print("nowhere to put a malformed key.")


# ─────────────────────────────────────────────────────────────────────────
# LEVEL 3 · Pydantic — the same guarantee, without hand-writing schemas
#
# Describe the shape as a Python class. ask_typed() passes it to .parse(),
# which turns the class into the schema and hands back an object of that
# class instead of text.
# (On older SDK versions this lived at client.beta.chat.completions.parse.)
# ─────────────────────────────────────────────────────────────────────────

print("\n===== LEVEL 3 — Pydantic: a typed object, not a dict =====")


class RetryAdvice(BaseModel):
    risk: str
    approach: str
    gotcha: str
    confidence: float


advice = ask_typed(SYSTEM, QUESTION, RetryAdvice)  # a RetryAdvice object

print(f"type:       {type(advice).__name__}")
print(f"risk:       {advice.risk}")
print(f"confidence: {advice.confidence}")
print("\nNo json.loads. No KeyError. Your editor autocompletes advice.gotcha.")


# ─────────────────────────────────────────────────────────────────────────
# LEVEL 4 · The lesson 7 conflict, re-run under strict mode
#
# Last lesson this prompt was unstable: sometimes JSON, sometimes steps,
# never reliably both. The conflict hasn't gone away, but it's no longer
# a coin flip. The schema wins every time, so the reasoning instruction
# loses every time.
#
# If you actually want the reasoning, don't ask for it in prose.
# Give it a field.
# ─────────────────────────────────────────────────────────────────────────

print("\n===== LEVEL 4 — the conflicting prompt, now under a schema =====")

CONFLICT_PROMPT = SYSTEM + "\n\nThink through the problem step by step before you answer."

choice = ask(CONFLICT_PROMPT, QUESTION, response_format=STRICT_FORMAT)
print((choice.message.content or "")[:300])
print("\n→ Valid JSON. No steps. The schema left no room for 'Step 1:'.")


class ReasonedAdvice(BaseModel):
    reasoning: str  # declared first, so the model writes it first
    risk: str
    approach: str
    gotcha: str
    confidence: float


reasoned = ask_typed(SYSTEM, QUESTION, ReasonedAdvice)

print(f"\nWith a reasoning field instead:\n  {reasoned.reasoning[:200]}")
print("\nFields are generated in order. Reasoning first means the answer")
print("is written after, and based on, the reasoning.")


# ─────────────────────────────────────────────────────────────────────────
# LEVEL 5 · Three ways the guarantee still breaks
#
# "Guaranteed valid JSON" is a promise about the shape of the output,
# not about your application. Here is what it does not cover.
# ─────────────────────────────────────────────────────────────────────────

# 5a. Truncation: strict mode controls WHICH tokens come next, not how many.

print("\n===== LEVEL 5a — truncation: a cut-off reply is not valid JSON =====")

choice = ask(SYSTEM, QUESTION, response_format=STRICT_FORMAT, max_tokens=20)
print(f"finish_reason: {choice.finish_reason}")          # 'length' = cut off
print(f"check:         {check(choice.message.content or '')}")
print("\nAlways check finish_reason. 'length' means you got the start of a")
print("valid object, which is not a valid object.")


# 5b. Refusals: if the model declines, the reason arrives in .refusal
#     and .content is None. It's a separate field, not an exception.

print("\n===== LEVEL 5b — refusals: a separate field, not an exception =====")

choice = ask(SYSTEM, QUESTION, response_format=STRICT_FORMAT)
if choice.message.refusal:
    print(f"refused: {choice.message.refusal}")
else:
    print("No refusal on this call. But .refusal is where it would arrive,")
    print("and .content would be None. That's why you check for it.")


# 5c. The one that should worry you: perfect shape, invented content.

print("\n===== LEVEL 5c — the one that should worry you =====")

FICTIONAL = "How does our Imaginary retry scheduler handle a third consecutive card failure?"

invented = ask_typed(SYSTEM, FICTIONAL, RetryAdvice)

print(f"question:   {FICTIONAL}")
print(f"\nrisk:       {invented.risk}")
print(f"approach:   {invented.approach}")
print(f"confidence: {invented.confidence}")
print("\nThere is no Imaginary scheduler. I made the name up.")
print("Every field is correctly typed. The schema is perfect.")
print("The content is fiction, delivered with high confidence.")


# ─────────────────────────────────────────────────────────────────────────
# The point
#
# Structured outputs solved a parsing problem completely and a
# correctness problem not at all. You can now trust the SHAPE of the
# answer completely, and the CONTENT exactly as much as before: not at
# all, on anything the model was never told.
#
# Lesson 9: where the facts come from.
# ─────────────────────────────────────────────────────────────────────────
