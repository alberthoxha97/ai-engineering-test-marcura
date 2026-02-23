# Charter Party Document Parser

A Python application that extracts legal clauses from voyage charter party PDFs and outputs them as structured JSON. Extraction is driven entirely by layout analysis — no hardcoded page numbers or section names — with an optional LLM pass for enhanced accuracy.

## How it works

The pipeline runs in two stages.

### Stage 1 — Layout-aware PDF extraction (always runs)

Every page is classified automatically from its content and routed to the appropriate parser. Three parser strategies are implemented:

| Parser | Detected when… | Typical section |
|---|---|---|
| **Two-column** | Page has title-fragment text at `x < 100 pt` AND numbered clause starters at `x ≥ 100 pt` | SHELLVOY 5 Part II (clauses `1`–`44`) |
| **Single-column / standalone number** | Clause number alone at far-left margin (`x < 60 pt`), title on the next line | Shell Additional Clauses (`SAC-1` … `SAC-N`) |
| **Single-column / inline number** | Clause number + title together on the same far-left line | Essar Rider Clauses (`ERC-1` … `ERC-N`) |

**No hardcoded page ranges are used.** The layout detector classifies each page individually, so the same code handles any PDF that follows these layout conventions.

#### Generic section detection

Section boundaries within single-column pages are detected without relying on known section names:

- **Position-based header detection** — lines starting at `x ≥ 150 pt` (visually centred, above the body-text margin of ≈149 pt) that are not clause numbers are flagged as section-header candidates.
- **`body_since_header` guard** — a candidate is only activated when a numbered clause follows it with no intervening body text. This ensures a deeply-indented line inside clause body text is never mistaken for a section header.
- **Numbering-reset detection** — if the clause number sequence resets (e.g. jumps from 43 back to 1), a new section is opened automatically even if no visible header was found.
- **Auto-generated prefixes** — section prefixes are derived from the header text by taking the first letter of each significant word (`"SHELL ADDITIONAL CLAUSES"` → `SAC`, `"Essar Rider Clauses"` → `ERC`, `"Maritime Freight Addendum"` → `MFA`). No prefix strings are hardcoded.

#### Strikethrough detection

Struck text is identified and excluded at two levels:

- **Line level** — lines where ≥ 50 % of the width is covered by a thin filled rectangle (height < 1 pt) are dropped entirely.
- **Word level** — for partially struck lines (< 50 % coverage), `get_text("words")` is used to resolve individual word bounding boxes; only words that overlap a strike rectangle by ≥ 30 % are removed. This handles cases like clause 13 where only the tail of a line is struck.

The two-column parser uses a **two-pass title-matching** algorithm with absolute y-coordinates (accumulated across pages) to correctly associate heading fragments with their clause even when struck-out clauses interleave with their replacements.

### Stage 2 — LLM enhancement (optional)

When an OpenAI or Anthropic API key is available, the clean plain text is forwarded to the LLM (GPT-4o by default, Claude as fallback), which re-extracts the clause list with richer contextual understanding. The LLM step is skipped gracefully when no key is present and the layout result is used as-is.

---

## Requirements

- Python 3.9+
- An OpenAI **or** Anthropic API key (optional — layout extraction works fully without one)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ai-engineering-test-marcura.git
cd ai-engineering-test-marcura

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and add your key(s):

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
# Layout extraction only — no API key needed
python main.py --no-llm

# Full pipeline — uses LLM if a key is configured
python main.py

# Custom PDF and output path
python main.py --pdf path/to/charter.pdf --output clauses.json

# Pass an API key directly (overrides env var)
python main.py --openai-key sk-...

# Verbose / debug logging
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
  -v, --verbose       Enable verbose / debug logging
```

---

## Output format

Each clause has three fields:

- `id` — the clause identifier. SHELLVOY 5 clauses use plain numbers (`"1"`, `"2"`, …); rider/additional sections get a prefix auto-derived from their section header (e.g. `SAC-1`, `ERC-2`).
- `title` — clause heading (empty string when the clause has no distinct heading).
- `text` — full clause body with all strikethrough text removed.

```json
{
  "total_clauses": 86,
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

For the reference document (`voyage-charter-example.pdf`) the extractor produces **86 clauses**:

| Section | Clauses | IDs |
|---|---|---|
| SHELLVOY 5 Part II | 38 | `1` … `44` (some struck, hence gaps) |
| Shell Additional Clauses | 28 | `SAC-1` … `SAC-43` |
| Essar Rider Clauses | 20 | `ERC-1` … `ERC-22` |

---

## Project structure

```
.
├── main.py                  # Entry point and CLI
├── src/
│   ├── __init__.py
│   ├── models.py            # Clause and ExtractionResult dataclasses
│   ├── pdf_extractor.py     # Page classifier, three parsers, strikethrough filtering
│   └── llm_extractor.py     # Optional LLM extraction (OpenAI GPT-4o / Anthropic Claude)
├── requirements.txt
├── .env.example             # API key template
└── voyage-charter-example.pdf  # Source PDF (excluded from git — see below)
```

---

## Notes

- **Strikethrough text** is excluded throughout. The document contains numerous amended clauses where original SHELLVOY 5 text is struck through and replaced by rider text; only the live (non-struck) text is included in the output.
- **Missing SHELLVOY 5 clause IDs** (e.g. 21, 27–29, 38, 39) correspond to clauses struck through in their entirety and replaced by custom rider clauses — their content is excluded per the task requirements.
