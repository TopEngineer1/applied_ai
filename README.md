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
echo "OPENAI_API_KEY=sk-..." > .env
```

`.env` is listed in `.gitignore`, so your key is never committed.

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
