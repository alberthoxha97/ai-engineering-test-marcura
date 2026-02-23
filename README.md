# Charter Party Document Parser

A Python application that extracts legal clauses from voyage charter party PDFs and outputs them as structured JSON. It works entirely from layout analysis — no page-number constants — and optionally uses an LLM to improve accuracy.

## How it works

The pipeline runs in two stages:

### Stage 1 — Layout-aware PDF extraction (always runs)

Every page is classified automatically by its content, and the appropriate parser is applied:

| Parser | Detected when… | Example section |
|---|---|---|
| **Two-column** | Page has title-fragment text at `x < 100 pt` AND numbered clause bodies at `x ≥ 100 pt` | SHELLVOY 5 Part II (clauses 1–44) |
| **Single-column / standalone number** | Clause number alone at far-left margin (`x < 60 pt`), title on next line | Shell Additional Clauses (`SAC-1` … `SAC-N`) |
| **Single-column / inline number** | Clause number + title on the same far-left line | Essar Rider Clauses (`ERC-1` … `ERC-N`) |

No hardcoded page ranges are used — the same code works on any PDF with these layouts.

**Strikethrough detection** is applied at two levels:
- *Line level*: lines where ≥ 50 % of the width is covered by a thin filled rectangle (height < 1 pt) are excluded entirely.
- *Word level*: for partially struck lines (< 50 % coverage), individual words overlapping a strike rectangle are removed, preserving the valid remainder.

The two-column parser uses a **two-pass title-matching** algorithm with absolute y-coordinates to correctly assign heading fragments to clauses even when struck-out clauses interleave with their replacements.

### Stage 2 — LLM enhancement (optional)

When an OpenAI or Anthropic API key is available, the clean plain text is forwarded to the LLM (GPT-4o by default, Claude as fallback), which re-extracts the clause list with richer contextual understanding. The LLM step is skipped gracefully when no key is present; the layout result is used as-is.

## Requirements

- Python 3.9+
- An OpenAI **or** Anthropic API key (optional — layout extraction works without one)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-engineering-test-marcura.git
cd ai-engineering-test-marcura

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and fill in your key(s):

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and/or ANTHROPIC_API_KEY
```

Or export directly in your shell:

```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Layout extraction only (no API key needed)
python main.py --no-llm

# Full pipeline — uses LLM if a key is set in the environment
python main.py

# Custom PDF and output path
python main.py --pdf path/to/charter.pdf --output clauses.json

# Pass API keys directly (overrides env vars)
python main.py --openai-key sk-... --output clauses.json

# Verbose logging
python main.py -v
```

### All options

```
usage: main.py [-h] [--pdf PATH] [--output PATH] [--no-llm]
               [--openai-key KEY] [--anthropic-key KEY] [-v]

  --pdf PATH          Path to the charter party PDF (default: voyage-charter-example.pdf)
  --output PATH       Output JSON file path (default: output.json)
  --no-llm            Skip LLM step; use layout-based extraction only
  --openai-key KEY    OpenAI API key (overrides OPENAI_API_KEY env var)
  --anthropic-key KEY Anthropic API key (overrides ANTHROPIC_API_KEY env var)
  -v, --verbose       Enable verbose/debug logging
```

## Output format

Each clause has three fields:

- `id` — clause identifier as it appears in the document. SHELLVOY 5 clauses use plain numbers (`"1"`, `"2"`, …); Shell Additional Clauses are prefixed `SAC-`; Essar Rider Clauses are prefixed `ERC-`.
- `title` — clause heading.
- `text` — full clause body (strikethrough text excluded).

```json
{
  "total_clauses": 84,
  "clauses": [
    {
      "id": "1",
      "title": "Condition Of vessel",
      "text": "Owners shall exercise due diligence to ensure that..."
    },
    {
      "id": "SAC-1",
      "title": "Indemnity Clause",
      "text": "If Charterers by telex, facsimile or other form of written communication..."
    },
    {
      "id": "ERC-1",
      "title": "INTERNATIONAL REGULATIONS CLAUSE",
      "text": "Vessel to comply with all national and international regulations..."
    }
  ]
}
```

## Project structure

```
.
├── main.py                  # Entry point and CLI
├── src/
│   ├── __init__.py
│   ├── models.py            # Clause and ExtractionResult dataclasses
│   ├── pdf_extractor.py     # Layout detection, three parsers, strikethrough filtering
│   └── llm_extractor.py     # Optional LLM extraction (OpenAI GPT-4o / Anthropic Claude)
├── requirements.txt
├── .env.example             # API key template
└── voyage-charter-example.pdf  # Source PDF (not tracked in git)
```

## Notes

- **Strikethrough text** is excluded throughout. This document contains numerous amended clauses where original SHELLVOY 5 text is struck through and replaced by rider text; only the live (non-struck) text appears in the output.
- **Missing SHELLVOY 5 clause IDs** (e.g. 21, 27–29, 38, 39) correspond to clauses that were struck through in their entirety and replaced by custom rider clauses — their content is excluded per the task requirements.
- **Source PDF** is not included in the repository (it is listed in `.gitignore`). Download it from the [original source](https://shippingforum.wordpress.com/wp-content/uploads/2012/09/voyage-charter-example.pdf) and place it in the project root as `voyage-charter-example.pdf`.
