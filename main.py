#!/usr/bin/env python3
"""Charter Party Document Parser

Extracts legal clauses from the SHELLVOY 5 voyage charter party PDF and outputs
them as structured JSON.

Extraction pipeline
-------------------
1. **PDF extraction** (PyMuPDF) – layout-aware, column-split parsing of Part II
   (pages 6–39).  Strikethrough text is detected via thin filled rectangles and
   excluded automatically.
2. **LLM enhancement** (OpenAI GPT-4o or Anthropic Claude, optional) – the LLM
   receives the clean plain text and returns a structured clause list. Enabled
   when an API key is available; falls back gracefully to the layout-based result.

Usage
-----
    python main.py [--pdf PATH] [--output PATH] [--no-llm] [-v]

Environment variables
---------------------
    OPENAI_API_KEY      OpenAI API key (preferred LLM backend)
    ANTHROPIC_API_KEY   Anthropic API key (fallback LLM backend)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.models import Clause, ExtractionResult
from src.pdf_extractor import extract_clauses_from_pdf, get_part_ii_plain_text

load_dotenv()

DEFAULT_PDF = Path(__file__).parent / "voyage-charter-example.pdf"
DEFAULT_OUTPUT = Path(__file__).parent / "output.json"


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract legal clauses from a charter party PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Path to the charter party PDF (default: {DEFAULT_PDF.name})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the LLM step and use the layout-based extraction only",
    )
    parser.add_argument(
        "--openai-key",
        dest="openai_key",
        default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--anthropic-key",
        dest="anthropic_key",
        default=None,
        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    return parser.parse_args()


def raw_to_clause(raw) -> Clause:
    return Clause(id=raw.id, title=raw.title, text=raw.text)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------------
    # Step 1 – Layout-based PDF extraction (always runs)
    # -------------------------------------------------------------------------
    logger.info("Extracting clauses from: %s", args.pdf)
    raw_clauses, pages_processed = extract_clauses_from_pdf(args.pdf)
    layout_clauses = [raw_to_clause(r) for r in raw_clauses]
    logger.info(
        "Layout extraction: %d clauses from %d pages", len(layout_clauses), pages_processed
    )

    # -------------------------------------------------------------------------
    # Step 2 – Optional LLM enhancement
    # -------------------------------------------------------------------------
    final_clauses = layout_clauses

    openai_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    anthropic_key = args.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
    has_api_key = bool(openai_key or anthropic_key)

    if args.no_llm:
        logger.info("LLM step skipped (--no-llm flag)")
    elif not has_api_key:
        logger.warning(
            "No API key found (OPENAI_API_KEY / ANTHROPIC_API_KEY). "
            "Using layout-based extraction only."
        )
    else:
        logger.info("Running LLM extraction for enhanced accuracy …")
        try:
            from src.llm_extractor import extract_clauses_with_llm

            plain_text, _ = get_part_ii_plain_text(args.pdf)
            llm_clauses = extract_clauses_with_llm(
                plain_text,
                openai_api_key=openai_key,
                anthropic_api_key=anthropic_key,
            )
            if llm_clauses:
                logger.info("LLM extraction: %d clauses", len(llm_clauses))
                final_clauses = llm_clauses
            else:
                logger.warning("LLM returned no clauses; using layout extraction.")
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM extraction failed (%s); using layout extraction.", exc)

    # -------------------------------------------------------------------------
    # Step 3 – Serialise output
    # -------------------------------------------------------------------------
    result = ExtractionResult(
        clauses=final_clauses, total_pages_processed=pages_processed
    )
    output_data = result.to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, indent=2, ensure_ascii=False)

    logger.info("Output written to: %s", args.output)

    # Brief summary to stdout
    print(f"\n{'=' * 60}")
    print(f"Extracted {len(final_clauses)} clauses from {pages_processed} pages")
    print(f"Output: {args.output}")
    print(f"{'=' * 60}")
    if final_clauses:
        print("\nSample (first 3 clauses):")
        for clause in final_clauses[:3]:
            snippet = clause.text[:90].replace("\n", " ")
            print(f"  [{clause.id}] {clause.title}")
            print(f"       {snippet}…")


if __name__ == "__main__":
    main()
