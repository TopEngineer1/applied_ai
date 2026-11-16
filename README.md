# Applied AI

Hands-on Python demos for the Applied AI course. Each module is a folder of
short, runnable scripts that show one concept at a time using real API calls.

## Modules

### Module 1: LLM Fundamentals

| Script | What it shows |
| --- | --- |
| `tokens.py` | How text is split into tokens with `tiktoken`, why English and Hindi cost different amounts, and how token counts turn into price. |
| `embeddings.py` | Turning text into vectors with `text-embedding-3-small` and comparing meaning with cosine similarity. |
| `context_window.py` | What a context window is, how input and output share it, and what happens when you go over the limit. |
| `promp_cache.py` | How OpenAI prompt caching makes repeated long prompts cheaper and faster. |

### Module 2: Building with LLMs

| Script | What it shows |
| --- | --- |
| `prompt_engineering.py` | Building a prompt one layer at a time (context, constraints, output format, examples), and why asking for JSON is not the same as getting it. |
| `structured_output.py` | Going from asking for JSON to enforcing it: retries, JSON mode, strict JSON schemas, and Pydantic, plus the cases where even strict mode breaks. |
| `why_rag.py` | Why RAG exists: pasting every document vs. retrieving one with keyword search, what each costs, and grounding the model so it says "I don't know". |
| `chunking.py` | Cutting documents into chunks (fixed-size, overlapping, by heading, with the title carried in) and how a bad cut breaks the answer. |
| `vector_search.py` | Retrieval by meaning: embedding chunks, brute-force cosine search, top-k, when you actually need a vector database, and where keyword search still wins. |

More modules will be added as the course progresses.

## Setup

Requires Python 3.10+.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies (shared by all modules)
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env             # then edit .env and paste your key
```

`.env` is listed in `.gitignore`, so your key is never committed. Only
`.env.example`, which holds a placeholder value, is tracked.

## Running a demo

```bash
cd "Module 1"
python tokens.py
```

The demos call the OpenAI API, so running them uses a small amount of credit
on your account.

## Project structure

```
applied_ai/
├── Module 1/          # one folder per module
├── requirements.txt   # shared dependencies
├── .env               # your API key (not committed)
└── README.md
```

When adding a module, create a new `Module N/` folder. Add any new packages to
the root `requirements.txt`.
