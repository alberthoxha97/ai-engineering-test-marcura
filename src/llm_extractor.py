"""LLM-based clause extraction and clean-up for charter party documents.

Supports both OpenAI (GPT-4o) and Anthropic (Claude) backends.
The LLM receives the plain-text of the document (strikethrough already removed)
and returns a structured JSON list of clauses.
"""

import json
import logging
import os
from typing import List, Optional

from .models import Clause

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

# Conservative chunk size in characters (well within 128 K-token context).
CHUNK_SIZE = 40_000
CHUNK_OVERLAP = 2_000


def _chunk_text(text: str) -> List[str]:
    """Split *text* into overlapping chunks no larger than CHUNK_SIZE chars."""
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            for sep in ("\n[PAGE ", "\n\n"):
                boundary = text.rfind(sep, start + CHUNK_SIZE // 2, end)
                if boundary != -1:
                    end = boundary
                    break
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP if end < len(text) else end

    logger.info("Split document into %d chunks for LLM processing", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a legal document parser specialising in maritime charter party contracts.

Your task: extract every legal clause from the provided text of a voyage charter party (SHELLVOY 5, Part II).

For each clause output a JSON object with exactly three keys:
  "id"    – the clause number as a plain string, e.g. "1", "2", "15"
  "title" – the clause title / heading (e.g. "Condition of vessel")
  "text"  – the complete clause body (exclude the clause number and title)

Rules:
1. Only extract clauses with an explicit numeric ID (e.g. "1.", "2.", "15.").
2. Strikethrough text has already been removed from the input – do not worry about it.
3. Return clauses in document order.
4. Do NOT duplicate clause IDs that appear in the already-extracted list.
5. Concatenate body text that spans multiple lines or page breaks into a single string.
6. Output ONLY a valid JSON array. Start with "[" and end with "]". No markdown fences, no commentary.\
"""


def _user_message(chunk: str, already_extracted: List[str]) -> str:
    prefix = (
        f"Already-extracted clause IDs (skip these): {already_extracted}\n\n"
        if already_extracted
        else ""
    )
    return f"{prefix}Extract all clauses from the following charter party text:\n\n{chunk}"


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(raw: str) -> List[dict]:
    """Parse the JSON returned by the LLM, tolerating markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        # Model wrapped the array: {"clauses": [...]}
        return next((v for v in parsed.values() if isinstance(v, list)), [])
    return parsed


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _call_openai(client, chunk: str, already_extracted: List[str]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_message(chunk, already_extracted)},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or "[]"


def _call_anthropic(client, chunk: str, already_extracted: List[str]) -> str:
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=8096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _user_message(chunk, already_extracted)}],
    )
    return response.content[0].text if response.content else "[]"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_clauses_with_llm(
    full_text: str,
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
) -> List[Clause]:
    """Extract clauses from *full_text* using an LLM.

    Tries OpenAI GPT-4o first; falls back to Anthropic Claude if the OpenAI
    key is not provided or the request fails.

    Args:
        full_text: Plain text of Part II (strikethrough already removed).
        openai_api_key: OpenAI API key (falls back to OPENAI_API_KEY env var).
        anthropic_api_key: Anthropic API key (falls back to ANTHROPIC_API_KEY env var).

    Returns:
        Ordered list of :class:`~src.models.Clause` objects.
    """
    oai_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    ant_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")

    # Pick the backend
    use_openai = bool(oai_key)
    use_anthropic = bool(ant_key)

    if not use_openai and not use_anthropic:
        raise ValueError(
            "No LLM API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
        )

    if use_openai:
        from openai import OpenAI
        client = OpenAI(api_key=oai_key)
        call_fn = lambda chunk, ids: _call_openai(client, chunk, ids)
        backend_name = "OpenAI GPT-4o"
    else:
        import anthropic
        client = anthropic.Anthropic(api_key=ant_key)
        call_fn = lambda chunk, ids: _call_anthropic(client, chunk, ids)
        backend_name = "Anthropic Claude"

    logger.info("Using LLM backend: %s", backend_name)

    chunks = _chunk_text(full_text)
    all_clauses: List[Clause] = []
    seen_ids: set = set()

    for i, chunk in enumerate(chunks):
        logger.info(
            "Processing chunk %d/%d (%d chars) …", i + 1, len(chunks), len(chunk)
        )
        already_extracted = [c.id for c in all_clauses]

        try:
            raw = call_fn(chunk, already_extracted)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM call failed for chunk %d: %s", i + 1, exc)
            if use_openai and use_anthropic:
                # Fall back to Anthropic
                logger.info("Falling back to Anthropic Claude …")
                import anthropic as _ant
                fb_client = _ant.Anthropic(api_key=ant_key)
                try:
                    raw = _call_anthropic(fb_client, chunk, already_extracted)
                except Exception as fb_exc:
                    logger.error("Anthropic fallback also failed: %s", fb_exc)
                    continue
            else:
                continue

        logger.debug("Raw LLM response (chunk %d): %.200s …", i + 1, raw)

        try:
            clause_list = _parse_response(raw)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error for chunk %d: %s", i + 1, exc)
            continue

        new_count = 0
        for item in clause_list:
            clause_id = str(item.get("id", "")).strip()
            if not clause_id or clause_id in seen_ids:
                continue
            all_clauses.append(
                Clause(
                    id=clause_id,
                    title=str(item.get("title", "")).strip(),
                    text=str(item.get("text", "")).strip(),
                )
            )
            seen_ids.add(clause_id)
            new_count += 1

        logger.info(
            "Chunk %d: +%d new clauses (running total: %d)",
            i + 1,
            new_count,
            len(all_clauses),
        )

    logger.info("LLM extraction complete. Total clauses: %d", len(all_clauses))
    return all_clauses
