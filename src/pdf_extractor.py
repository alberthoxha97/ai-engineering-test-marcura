"""PDF text extraction with layout-aware column parsing and strikethrough detection.

Three parser strategies are applied based on detected page layout:

1. **Two-column parser** (SHELLVOY 5 style)
   Detected when a page has title-fragment text in the left column
   (x < TITLE_COLUMN_MAX_X) AND numbered clause starters in the right column
   (x >= TITLE_COLUMN_MAX_X).  Titles and bodies are matched via absolute
   y-coordinates across the full batch of two-column pages.

2. **Single-column / standalone-number parser** (Shell Additional style)
   Clause number appears alone at the far-left margin (x < RIDER_NUM_X_MAX).
   The title follows on the very next non-empty line; body text fills the rest.

3. **Single-column / inline-number parser** (Rider style)
   Clause number and title appear together on the same far-left line
   (e.g. ``1.  ARBITRATION CLAUSE``).  Body text follows.

Parsers 2 and 3 are unified in ``_extract_single_column_clauses``; the two
sub-formats are distinguished automatically by whether the clause-number line
carries inline text.

Section prefixes (``SAC-``, ``ERC-``, …) are derived from section-header lines
found in the page content — no page-number constants are used.

Strikethrough text is represented as thin filled rectangles drawn on top of
text spans.  Word-level filtering ensures only the struck characters are
removed when a strikethrough covers only part of a line.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Layout-detection thresholds
# ---------------------------------------------------------------------------

# x-coordinate threshold separating the title column (left) from the body
# column (right) in the two-column SHELLVOY 5 layout.
TITLE_COLUMN_MAX_X = 100.0

# In single-column sections the top-level clause number sits at the very left
# margin (x ≈ 50 pts).  Sub-clause labels and body text sit further right.
RIDER_NUM_X_MAX = 60.0

# Strikethrough lines are thin filled rects with height below this threshold.
STRIKETHROUGH_MAX_HEIGHT = 1.0

# A line is considered entirely struck when this fraction of its width is covered.
STRIKETHROUGH_COVERAGE_THRESHOLD = 0.5

# A word (from get_text("words")) is considered struck when this fraction of
# its width overlaps a strikethrough rectangle.
WORD_STRIKE_COVERAGE_THRESHOLD = 0.3


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


def _word_is_struck(
    word_bbox: Tuple[float, float, float, float],
    strike_rects: List[fitz.Rect],
) -> bool:
    """Return True if the word's bbox is sufficiently covered by a strike rect."""
    wr = fitz.Rect(word_bbox)
    if wr.width <= 0:
        return False
    for sr in strike_rects:
        if not wr.intersects(sr):
            continue
        overlap = min(wr.x1, sr.x1) - max(wr.x0, sr.x0)
        if overlap / wr.width >= WORD_STRIKE_COVERAGE_THRESHOLD:
            return True
    return False


def _extract_lines(page: fitz.Page, y_offset: float = 0.0) -> List[_TextLine]:
    """Extract all text lines from *page*, sorted top-to-bottom.

    *y_offset* is added to each line's y-coordinate so that lines across
    different pages have unique, monotonically increasing absolute y-values.

    Lines are returned with their ``struck`` flag set.  Struck body lines are
    still included so the parser can consume their associated title fragments.

    For partially-struck lines (below STRIKETHROUGH_COVERAGE_THRESHOLD), struck
    words are removed at the word level so that only the non-struck portion is
    kept.
    """
    strike_rects = _get_strikethrough_rects(page)

    # Build a word-level index keyed by (block_no, line_no) so we can reassemble
    # cleaned text for partially-struck lines.
    # get_text("words") returns (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    word_map: Dict[Tuple[int, int], List[str]] = {}
    if strike_rects:
        for w in page.get_text("words", sort=True):
            x0, y0, x1, y1, word, bno, lno, _ = w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]
            if not _word_is_struck((x0, y0, x1, y1), strike_rects):
                word_map.setdefault((bno, lno), []).append(word)

    lines: List[_TextLine] = []

    text_dict = page.get_text("dict", sort=True)
    for block in text_dict["blocks"]:
        if block["type"] != 0:
            continue
        bno = block["number"]
        for lno, line in enumerate(block["lines"]):
            raw_text = "".join(s["text"] for s in line["spans"]).strip()
            if not raw_text or raw_text.isdigit():
                continue

            struck = bool(strike_rects and _is_struck(line["bbox"], strike_rects))

            if struck:
                logger.debug("Struck line on page %d: %s…", page.number + 1, raw_text[:40])
                text = raw_text
            elif strike_rects:
                # Reassemble from non-struck words; fall back to raw if nothing filtered
                kept_words = word_map.get((bno, lno))
                if kept_words is not None and len(kept_words) < len(raw_text.split()):
                    text = " ".join(kept_words)
                    if text != raw_text:
                        logger.debug(
                            "Partial strike on page %d — kept: %s…",
                            page.number + 1,
                            text[:60],
                        )
                else:
                    text = raw_text
            else:
                text = raw_text

            if not text.strip():
                continue

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

# Rider / additional sections: clause starters are lines where the number sits
# at the far-left margin (x < RIDER_NUM_X_MAX).  The number may be followed
# immediately by the title (Essar format) or the title may be on the next line
# (Shell Additional format).
_RIDER_CLAUSE_NUM_RE = re.compile(r"^(\d+)[.)]+\s*(.*)")

# Known section headers in the rider pages and their short prefix codes.
_RIDER_SECTIONS: List[Tuple[str, str]] = [
    ("SHELL ADDITIONAL", "SAC"),
    ("ESSAR RIDER",      "ERC"),
]


def _detect_rider_section(text: str) -> Optional[str]:
    """Return the section prefix if *text* is a known rider section header."""
    upper = text.upper()
    for keyword, prefix in _RIDER_SECTIONS:
        if keyword in upper:
            return prefix
    return None


def _has_two_column_layout(page: fitz.Page) -> bool:
    """Return True when the page uses the two-column SHELLVOY 5 layout.

    A page is classified as two-column when it satisfies **both** conditions:

    1. It has meaningful text in the left title column
       (x < TITLE_COLUMN_MAX_X that is neither a pure digit nor a clause-number
       line such as ``2.`` or ``3.``).
    2. It has at least one numbered clause starter in the right body column
       (x >= TITLE_COLUMN_MAX_X matching ``N. …``).

    This reliably distinguishes SHELLVOY 5 pages from:
    - Part I form pages  (no right-column numbered clauses)
    - Shell Additional pages (title text at x ≈ 86 but clause starters are at
      x ≈ 50, not in the right column)
    - Essar Rider pages (all text at x ≈ 50, no right-column clauses at all)
    """
    has_title_col = False
    has_right_col = False

    for block in page.get_text("dict", sort=True)["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text or text.isdigit():
                continue
            x0 = line["bbox"][0]
            if x0 < TITLE_COLUMN_MAX_X:
                # Exclude lines that are themselves clause-number starters
                # (e.g. "2." or "3.  Insurance Clause") so that single-column
                # pages whose number sits at x ≈ 50 aren't misclassified.
                if not _RIDER_CLAUSE_NUM_RE.match(text):
                    has_title_col = True
            elif _CLAUSE_NUM_RE.match(text):
                has_right_col = True

        if has_title_col and has_right_col:
            return True

    return has_title_col and has_right_col


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


def _extract_single_column_clauses(
    doc: fitz.Document,
    page_nums: List[int],
) -> List[RawClause]:
    """Extract clauses from single-column pages (Shell Additional and Rider styles).

    Both sub-formats are handled transparently:

    * **Standalone-number style** (Shell Additional): the clause number sits
      alone at the far-left margin (x < RIDER_NUM_X_MAX) and the title appears
      on the very next non-empty line (e.g. pages that begin with ``2.`` then
      ``Original Bill of Lading Clause``).

    * **Inline-number style** (Rider): the clause number and title appear
      together on the same far-left line (e.g. ``1.  ARBITRATION CLAUSE``).

    The distinction is made automatically: if the number line contains text
    after the number, that text becomes the title (inline style); otherwise
    the next line is taken as the title (standalone style).

    Section prefixes such as ``SAC`` or ``ERC`` are determined from section-
    header lines encountered in the page content (e.g.
    ``SHELL ADDITIONAL CLAUSES``, ``Essar Rider Clauses``).  No page-number
    constants are used; the same function works for any PDF with this layout.

    Sub-clause labels (``1)``, ``2)``, ``(A)`` …) sit at a deeper x-coordinate
    and are incorporated into the clause body text.
    """
    current_section = "SAC"  # sensible default; overridden by any section header
    current_id: Optional[str] = None
    current_id_section = "SAC"  # snapshot the section when the clause starts
    current_title: Optional[str] = None
    current_body_parts: List[str] = []
    awaiting_title = False  # True when we have a number but no title yet

    clauses: List[RawClause] = []

    def _flush() -> None:
        if current_id is not None and current_body_parts:
            full_id = f"{current_id_section}-{current_id}"
            clauses.append(
                RawClause(
                    id=full_id,
                    title=current_title or "",
                    text=" ".join(current_body_parts).strip(),
                )
            )

    for page_num in page_nums:
        page = doc[page_num]
        strike_rects = _get_strikethrough_rects(page)

        # Build word-level strike map for partial-line filtering
        word_map: Dict[Tuple[int, int], List[str]] = {}
        if strike_rects:
            for w in page.get_text("words", sort=True):
                x0, y0, x1, y1 = w[0], w[1], w[2], w[3]
                word, bno, lno = w[4], w[5], w[6]
                if not _word_is_struck((x0, y0, x1, y1), strike_rects):
                    word_map.setdefault((bno, lno), []).append(word)

        text_dict = page.get_text("dict", sort=True)
        for block in text_dict["blocks"]:
            if block["type"] != 0:
                continue
            bno = block["number"]
            for lno, line in enumerate(block["lines"]):
                raw_text = "".join(s["text"] for s in line["spans"]).strip()
                if not raw_text or raw_text.isdigit():
                    continue

                x0 = line["bbox"][0]
                struck = bool(strike_rects and _is_struck(line["bbox"], strike_rects))

                if struck:
                    continue  # skip wholly struck lines in rider section

                # Apply partial word-level strike filtering
                if strike_rects:
                    kept = word_map.get((bno, lno))
                    text = (
                        " ".join(kept)
                        if kept is not None and len(kept) < len(raw_text.split())
                        else raw_text
                    )
                else:
                    text = raw_text

                if not text.strip():
                    continue

                # Detect section header (updates current_section, not a clause line)
                section = _detect_rider_section(text)
                if section:
                    current_section = section
                    continue

                # Detect top-level clause starter: number at leftmost margin
                if x0 < RIDER_NUM_X_MAX:
                    m = _RIDER_CLAUSE_NUM_RE.match(text)
                    if m:
                        _flush()
                        current_id = m.group(1)
                        current_id_section = current_section  # snapshot section at start
                        inline_title = m.group(2).strip()
                        if inline_title:
                            current_title = inline_title
                            awaiting_title = False
                        else:
                            current_title = None
                            awaiting_title = True
                        current_body_parts = []
                        continue

                if current_id is None:
                    continue

                # Assign first line after a standalone number as the title
                if awaiting_title:
                    current_title = text
                    awaiting_title = False
                    continue

                current_body_parts.append(text)

    _flush()
    logger.info("Single-column parser: %d clauses from %d pages", len(clauses), len(page_nums))
    return clauses


def extract_clauses_from_pdf(pdf_path: Path) -> Tuple[List[RawClause], int]:
    """Extract all clauses from a charter party PDF.

    The function scans every page and automatically selects the appropriate
    parser based on detected page layout — no page-number constants are used.

    Three parser strategies are applied:

    1. **Two-column parser** (``_parse_clauses_from_lines``) — used for pages
       where title-fragment text appears in the left column and numbered clause
       bodies appear in the right column (SHELLVOY 5 style).

    2 & 3. **Single-column parser** (``_extract_single_column_clauses``) — used
       for all remaining pages.  Within this parser, the standalone-number style
       (Shell Additional) and the inline-number style (Rider) are handled
       automatically based on whether the clause-number line carries inline text.

    Returns a tuple ``(clauses, pages_processed)``.
    Strikethrough text is excluded at both the word and line levels.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    logger.info("Opened '%s' (%d pages total)", pdf_path.name, total_pages)

    # ------------------------------------------------------------------ #
    # Step 1 – Classify every page as two-column or single-column         #
    # ------------------------------------------------------------------ #
    two_col_pages: List[int] = []
    single_col_pages: List[int] = []

    for page_num in range(total_pages):
        page = doc[page_num]
        if _has_two_column_layout(page):
            two_col_pages.append(page_num)
        else:
            single_col_pages.append(page_num)

    logger.info(
        "Layout detection: %d two-column pages, %d single-column pages",
        len(two_col_pages),
        len(single_col_pages),
    )

    all_clauses: List[RawClause] = []

    # ------------------------------------------------------------------ #
    # Step 2 – Parser 1: two-column (SHELLVOY 5 style)                   #
    # ------------------------------------------------------------------ #
    if two_col_pages:
        all_lines: List[_TextLine] = []
        y_offset = 0.0
        for page_num in two_col_pages:
            page = doc[page_num]
            page_lines = _extract_lines(page, y_offset=y_offset)
            all_lines.extend(page_lines)
            y_offset += page.rect.height
            logger.debug("Two-col page %d: %d lines", page_num + 1, len(page_lines))

        shellvoy_clauses = _parse_clauses_from_lines(all_lines)
        logger.info("Two-column parser: %d clauses", len(shellvoy_clauses))
        all_clauses.extend(shellvoy_clauses)

    # ------------------------------------------------------------------ #
    # Step 3 – Parsers 2 & 3: single-column (Shell Additional + Rider)   #
    # ------------------------------------------------------------------ #
    if single_col_pages:
        single_clauses = _extract_single_column_clauses(doc, single_col_pages)
        all_clauses.extend(single_clauses)

    doc.close()

    pages_processed = len(two_col_pages) + len(single_col_pages)
    logger.info("Total: %d clauses from %d pages", len(all_clauses), pages_processed)
    return all_clauses, pages_processed


# ---------------------------------------------------------------------------
# Plain-text view (for LLM post-processing)
# ---------------------------------------------------------------------------

def get_part_ii_plain_text(pdf_path: Path) -> Tuple[str, int]:
    """Return the plain text of all clause pages with strikethrough removed.

    Only pages that contain clause content (two-column or single-column with
    numbered clauses) are included.  Determined by layout detection, not page
    numbers.  Also returns the number of pages included.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    parts: List[str] = []
    pages = 0
    y_offset = 0.0
    for page_num in range(total_pages):
        page = doc[page_num]
        lines = _extract_lines(page, y_offset=y_offset)
        y_offset += page.rect.height
        visible = [ln.text for ln in lines if not ln.struck]
        if visible:
            parts.append(f"[PAGE {page_num + 1}]\n" + "\n".join(visible))
            pages += 1

    doc.close()
    return "\n\n".join(parts), pages
