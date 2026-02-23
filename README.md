# Charter Party Document Parser

A Python application that parses the SHELLVOY 5 voyage charter party PDF and extracts legal clauses in structured JSON format using a combination of layout-aware PDF parsing and LLM-powered extraction.

## How it works

The pipeline runs in two stages:

### 1. Layout-aware PDF extraction (always runs)

The SHELLVOY 5 Part II pages use a characteristic two-column layout:

- **Left column** (`x < 100 pt`): clause title / heading
- **Right column** (`x ≥ 100 pt`): clause body, prefixed by the clause number (`N. text…`)

PyMuPDF is used to:
- Read every text line with its position and bounding box.
- Detect **strikethrough text** — rendered in this PDF as thin filled rectangles (height < 1 pt) overlaid on text. Lines where ≥ 50 % of the width is covered are excluded automatically.
- Apply a **two-pass title-matching** algorithm that assigns left-column heading fragments to the correct clause, even when struck-out clauses interleave with their replacements.

### 2. LLM enhancement (optional)

When an OpenAI or Anthropic API key is available, the clean plain text is forwarded to the LLM (GPT-4o by default, Claude as fallback) which re-extracts the full clause list with richer understanding of context, handles edge cases, and produces cleaner titles and body text.

The LLM step is skipped gracefully when no key is found; the layout-based result is used instead.

## Requirements

- Python 3.9+
- An OpenAI **or** Anthropic API key (optional but recommended for best accuracy)

## Installation

```bash
# Clone the repository and enter the project directory
git clone https://github.com/your-username/ai-engineering-test-marcura.git
cd ai-engineering-test-marcura

# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and add your API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY or ANTHROPIC_API_KEY
```

Alternatively, export the key in your shell:

```bash
export OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Full pipeline (layout extraction + LLM enhancement if key is available)
python main.py

# Skip LLM step — use layout-based extraction only
python main.py --no-llm

# Specify a different PDF path or output file
python main.py --pdf path/to/charter.pdf --output clauses.json

# Verbose logging
python main.py -v
```

### All options

```
usage: main.py [-h] [--pdf PATH] [--output PATH] [--no-llm]
               [--openai-key KEY] [--anthropic-key KEY] [-v]

  --pdf PATH          Path to the charter party PDF (default: voyage-charter-example.pdf)
  --output PATH       Output JSON file path (default: output.json)
  --no-llm            Skip the LLM step and use the layout-based extraction only
  --openai-key KEY    OpenAI API key (overrides OPENAI_API_KEY env var)
  --anthropic-key KEY Anthropic API key (overrides ANTHROPIC_API_KEY env var)
  -v, --verbose       Enable verbose/debug logging
```

## Output format

```json
{
  "total_clauses": 38,
  "clauses": [
    {
      "id": "1",
      "title": "Condition Of vessel",
      "text": "Owners shall exercise due diligence to ensure that..."
    },
    {
      "id": "2",
      "title": "Cleanliness Of tanks",
      "text": "Whilst loading, carrying and discharging the cargo..."
    }
  ]
}
```

## Project structure

```
.
├── main.py                  # Entry point
├── src/
│   ├── __init__.py
│   ├── models.py            # Clause and ExtractionResult dataclasses
│   ├── pdf_extractor.py     # Layout-aware PDF parsing (strikethrough detection, title matching)
│   └── llm_extractor.py     # LLM extraction (OpenAI GPT-4o / Anthropic Claude)
├── requirements.txt
├── .env.example
├── output.json              # Pre-generated output (38 SHELLVOY 5 Part II clauses)
└── voyage-charter-example.pdf
```

## Notes on the document

- **Strikethrough text** is excluded throughout. This document contains numerous amended clauses where the original SHELLVOY 5 text is struck through and replaced by rider text. Only the replacement (non-struck) text is included in the output.
- **Missing clause IDs** (e.g. 21, 27–29, 38, 39) correspond to clauses that were struck through *in their entirety* and replaced by custom rider clauses — their content is excluded as per the challenge requirements.
- Pages 18–39 contain additional "Shell" and "Essar" rider clauses in a single-column format. The layout-based extractor focuses on the standard two-column SHELLVOY 5 clauses (pages 6–17); the LLM path handles the full page range 6–39.
