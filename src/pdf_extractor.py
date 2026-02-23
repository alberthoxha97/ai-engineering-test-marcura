"""PDF text extraction with layout-aware column parsing and strikethrough detection.

The SHELLVOY 5 Part II document uses a two-column layout on each page:
  - Left column  (x < ~100 pts): clause title / heading text
  - Right column (x >= ~100 pts): clause body text, prefixed by clause number

Strikethrough text is represented as thin filled rectangles drawn on top of
text spans. We detect and skip these lines automatically.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Pages 6–39 in the PDF (0-indexed: 5–38). Part I covers pages 1–5.
PART_II_START_PAGE = 5  # 0-indexed

# The SHELLVOY 5 standard Part II clauses (1–44) use the two-column layout
# and appear on pages 6–17 of this document (0-indexed: 5–16).  Pages 18–39
# contain additional/rider clauses in a single-column format which are handled
# by the LLM extraction step.
SHELLVOY5_END_PAGE = 17  # 0-indexed, exclusive (pages 6–17 inclusive)

# x-coordinate threshold separating the left (title) column from the right
# (body text) column.  Anything with x0 < TITLE_COLUMN_MAX_X is a title fragment.
TITLE_COLUMN_MAX_X = 100.0

# Strikethrough lines are thin filled rects with height below this threshold.
STRIKETHROUGH_MAX_HEIGHT = 1.0

# A strikethrough must cover at least this fraction of the line's width to
# cause the entire line to be dropped.  We also apply span-level filtering so
# that only the struck characters are removed, not the whole line.
STRIKETHROUGH_COVERAGE_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _get_strikethrough_rects(page: fitz.Page) -> List[fitz.Rect]:
    """Return bounding boxes of all strikethrough decorations on *page*."""
    rects = []
    for drawing in page.get_drawings():
        r = fitz.Rect(drawing["rect"])
        if 0 < r.height < STRIKETHROUGH_MAX_HEIGHT:
            rects.append(r)
    return rects


def _is_struck(line_bbox: Tuple, strike_rects: List[fitz.Rect]) -> bool:
    """Return True if strikethrough rects cover most of this line.

    A line is considered struck-through only when the combined horizontal
    coverage of overlapping strikethrough rectangles exceeds
    STRIKETHROUGH_COVERAGE_THRESHOLD of the line's width.  This prevents
    small inline amendments (e.g. a single struck word) from suppressing the
    entire line.
    """
    lr = fitz.Rect(line_bbox)
    line_width = lr.width
    if line_width <= 0:
        return False

    covered = 0.0
    for sr in strike_rects:
        if not lr.intersects(sr):
            continue
        # Horizontal overlap only
        overlap = min(lr.x1, sr.x1) - max(lr.x0, sr.x0)
        if overlap > 0:
            covered += overlap

    return (covered / line_width) >= STRIKETHROUGH_COVERAGE_THRESHOLD


# ---------------------------------------------------------------------------
# Per-page structured extraction
# ---------------------------------------------------------------------------

@dataclass
class _TextLine:
    """A text line with its position, visibility, and text."""
    x0: float
    y0: float
    text: str
    struck: bool = False  # True if the line is struck-through


def _extract_lines(page: fitz.Page, y_offset: float = 0.0) -> List[_TextLine]:
    """Extract all text lines from *page*, sorted top-to-bottom.

    *y_offset* is added to each line's y-coordinate so that lines across
    different pages have unique, monotonically increasing absolute y-values.

    Lines are returned with their ``struck`` flag set.  Struck body lines are
    still included so the parser can consume their associated title fragments.
    """
    strike_rects = _get_strikethrough_rects(page)
    lines: List[_TextLine] = []

    text_dict = page.get_text("dict", sort=True)
    for block in text_dict["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text or text.isdigit():
                continue
            struck = bool(strike_rects and _is_struck(line["bbox"], strike_rects))
            if struck:
                logger.debug("Struck line on page %d: %s…", page.number + 1, text[:40])
            lines.append(
                _TextLine(
                    x0=line["bbox"][0],
                    y0=y_offset + line["bbox"][1],
                    text=text,
                    struck=struck,
                )
            )

    return lines


# ---------------------------------------------------------------------------
# Clause-level data structure
# ---------------------------------------------------------------------------

@dataclass
class RawClause:
    """A clause as extracted from the PDF before any LLM post-processing."""
    id: str
    title: str
    text: str


# ---------------------------------------------------------------------------
# Document-level extraction
# ---------------------------------------------------------------------------

# Pattern matching a clause-number prefix in the body column, e.g. "1. ", "15. "
_CLAUSE_NUM_RE = re.compile(r"^(\d+)\.\s+(.*)$", re.DOTALL)


def _parse_clauses_from_lines(all_lines: List[_TextLine]) -> List[RawClause]:
    """Convert a flat list of positioned text lines into structured clauses.

    Two-pass algorithm
    ------------------
    Pass 1 – Title assignment:
        Each left-column title line is assigned to the clause whose body starts
        at the closest y-position AT OR BEFORE the title line's y-position.
        This correctly handles multi-line titles (which appear below the clause
        body start) and prevents titles for struck-out clauses from leaking
        into the following surviving clause.

    Pass 2 – Body assembly:
        Process right-column body lines sequentially.  Non-struck clause
        starters begin a new clause; struck starters are skipped.  Body
        continuation lines are appended to the current open clause.

        When a non-struck clause has no title (because its title was consumed
        by a struck version of the same clause), the struck version's title
        is inherited.
    """
    # ------------------------------------------------------------------ #
    # Pass 1: collect all clause starters (both struck & not) and assign  #
    # title lines to them.                                                 #
    # ------------------------------------------------------------------ #

    # All clause starters: (abs_y, clause_id, is_struck)
    starters: List[Tuple[float, str, bool]] = []
    for line in all_lines:
        if line.x0 >= TITLE_COLUMN_MAX_X:
            m = _CLAUSE_NUM_RE.match(line.text)
            if m:
                starters.append((line.y0, m.group(1), line.struck))
    starters.sort(key=lambda t: t[0])

    # Small tolerance (in points) to handle sub-pixel floating-point differences
    # between title line y-coordinates and their corresponding body line y-coordinates.
    Y_EPSILON = 1.5

    # Map clause_id → list of title texts, using "most recent starter ≤ title_y"
    clause_titles: Dict[str, List[str]] = {}
    for line in all_lines:
        if line.x0 >= TITLE_COLUMN_MAX_X:
            continue
        # Find the last starter whose y ≤ this title line's y (with tolerance)
        assigned_id: Optional[str] = None
        for (sy, sid, _struck) in reversed(starters):
            if sy <= line.y0 + Y_EPSILON:
                assigned_id = sid
                break
        if assigned_id is not None:
            clause_titles.setdefault(assigned_id, []).append(line.text)

    def _title_for(clause_id: str) -> str:
        parts = clause_titles.get(clause_id, [])
        return " ".join(parts).strip()

    # ------------------------------------------------------------------ #
    # Pass 2: assemble clause bodies                                       #
    # ------------------------------------------------------------------ #

    clauses: List[RawClause] = []
    # Track struck-clause titles so they can be inherited by their replacements
    struck_titles: Dict[str, str] = {}

    current_id: Optional[str] = None
    current_body_parts: List[str] = []
    in_struck_clause = False

    def _flush() -> None:
        if current_id is not None and not in_struck_clause:
            body = " ".join(current_body_parts).strip()
            title = _title_for(current_id)
            # Inherit from the struck version if this clause has no title of its own
            if not title and current_id in struck_titles:
                title = struck_titles[current_id]
            clauses.append(RawClause(id=current_id, title=title, text=body))

    for line in all_lines:
        if line.x0 < TITLE_COLUMN_MAX_X:
            continue  # title lines handled in pass 1

        m = _CLAUSE_NUM_RE.match(line.text)
        if m:
            _flush()
            current_id = m.group(1)
            current_body_parts = [m.group(2)] if m.group(2).strip() else []
            in_struck_clause = line.struck
            if line.struck:
                # Save struck title for potential inheritance
                struck_titles[current_id] = _title_for(current_id)
        else:
            if current_id is None or in_struck_clause or line.struck:
                continue
            current_body_parts.append(line.text)

    _flush()
    return clauses


def extract_clauses_from_pdf(pdf_path: Path) -> Tuple[List[RawClause], int]:
    """Extract raw clauses from Part II of the charter party PDF.

    Returns a tuple (clauses, pages_processed).
    Strikethrough text is automatically excluded.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    logger.info("Opened '%s' (%d pages total)", pdf_path.name, total_pages)

    all_lines: List[_TextLine] = []
    pages_processed = 0
    y_offset = 0.0
    # Only process the two-column SHELLVOY 5 pages for layout-based extraction
    end_page = min(SHELLVOY5_END_PAGE, total_pages)
    for page_num in range(PART_II_START_PAGE, end_page):
        page = doc[page_num]
        page_height = page.rect.height
        page_lines = _extract_lines(page, y_offset=y_offset)
        all_lines.extend(page_lines)
        pages_processed += 1
        y_offset += page_height
        logger.debug("Page %d: %d lines", page_num + 1, len(page_lines))

    doc.close()
    logger.info("Extracted lines from %d pages", pages_processed)

    clauses = _parse_clauses_from_lines(all_lines)
    logger.info("Parsed %d raw clauses", len(clauses))
    return clauses, pages_processed


# ---------------------------------------------------------------------------
# Plain-text view (for LLM post-processing)
# ---------------------------------------------------------------------------

def get_part_ii_plain_text(pdf_path: Path) -> Tuple[str, int]:
    """Return the full plain-text of Part II with strikethrough text removed.

    Also returns the number of pages processed.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    parts: List[str] = []
    pages = 0
    y_offset = 0.0
    for page_num in range(PART_II_START_PAGE, total_pages):
        page = doc[page_num]
        lines = _extract_lines(page, y_offset=y_offset)
        y_offset += page.rect.height
        visible = [l.text for l in lines if not l.struck]
        if visible:
            parts.append(f"[PAGE {page_num + 1}]\n" + "\n".join(visible))
            pages += 1

    doc.close()
    return "\n\n".join(parts), pages
