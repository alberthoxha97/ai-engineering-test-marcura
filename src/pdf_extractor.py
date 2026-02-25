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
    """Return bounding boxes of all strikethrough decorations on *page*.

    Strikethrough is stored in the PDF graphics layer as a thin filled
    rectangle drawn on top of the text — not as a font property.  This
    function reads that graphics layer and filters for hair-thin shapes
    (height < 1 pt) which are exclusively used for strikethrough in these
    documents.  All other shapes (borders, table lines, rules) are taller
    and are ignored.
    """
    rects = []
    # Iterate every vector drawing object on the page (borders, lines, fills).
    # Each drawing dict contains a "rect" key with the shape's bounding box.
    for drawing in page.get_drawings():
        r = fitz.Rect(drawing["rect"])
        # Keep only hair-thin filled rectangles — these are strikethrough lines.
        # Page borders and table rules are always taller than 1 pt.
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
    # Accumulate the total horizontal width covered by all overlapping strike rects.
    for sr in strike_rects:
        if not lr.intersects(sr):
            # Fast reject — no overlap between this strike rect and the line at all.
            continue
        # Calculate the width of the horizontal overlap between the line and this rect.
        # max(left edges) gives the overlap's left bound;
        # min(right edges) gives the overlap's right bound.
        overlap = min(lr.x1, sr.x1) - max(lr.x0, sr.x0)
        if overlap > 0:
            covered += overlap

    # If the combined coverage exceeds the threshold, the whole line is struck.
    return (covered / line_width) >= STRIKETHROUGH_COVERAGE_THRESHOLD


# ---------------------------------------------------------------------------
# Per-page structured extraction
# ---------------------------------------------------------------------------

@dataclass
class _TextLine:
    """A single visual line of text extracted from a PDF page.

    x0    — left edge in PDF points; used to classify which column the line
             belongs to (title column vs body column vs section header zone).
    y0    — absolute y-coordinate (page-relative y + cumulative page offsets);
             used by the two-column parser to match titles to clauses across pages.
    text  — the visible text after strikethrough words have been removed.
    struck — True when the line-level coverage check determined the whole line
             is struck through.  Struck lines are kept in the list so the
             two-column title matcher can still use their coordinates, but their
             text is excluded from clause output.
    """
    x0: float
    y0: float
    text: str
    struck: bool = False


def _word_is_struck(
    word_bbox: Tuple[float, float, float, float],
    strike_rects: List[fitz.Rect],
) -> bool:
    """Return True if this individual word is sufficiently covered by a strike rect.

    Used for partial-strike lines where the line-level check (_is_struck) returned
    False because less than 50% of the line is covered.  At word level, a 30%
    coverage threshold is used because individual word bounding boxes are small
    enough that a 30% overlap is a clear visual strikethrough of that word.
    """
    wr = fitz.Rect(word_bbox)
    if wr.width <= 0:
        # Degenerate bounding box (e.g. whitespace character) — treat as not struck
        # to avoid division by zero.
        return False
    # Check each strike rect against this word's bounding box.
    for sr in strike_rects:
        if not wr.intersects(sr):
            # No overlap between this strike rect and the word — skip immediately.
            continue
        # Measure horizontal overlap between the word and the strike rect.
        overlap = min(wr.x1, sr.x1) - max(wr.x0, sr.x0)
        if overlap / wr.width >= WORD_STRIKE_COVERAGE_THRESHOLD:
            # At least 30% of the word's width is covered — consider it struck.
            return True
    return False


def _extract_lines(page: fitz.Page, y_offset: float = 0.0) -> List[_TextLine]:
    """Extract all text lines from *page* as _TextLine objects, sorted top-to-bottom.

    *y_offset* is added to each line's y-coordinate so that lines across
    different pages have unique, monotonically increasing absolute y-values.
    This is essential for the two-column parser which must match titles to
    clause bodies across page boundaries using y-coordinate comparisons.

    Lines are returned with their ``struck`` flag set.  Struck lines are still
    included (not discarded) so the two-column parser can use their coordinates
    when assigning titles — even though their text will not appear in output.

    For partially-struck lines (line-level coverage below 50%), individual
    struck words are removed using word-level bounding box analysis, and only
    the surviving words are kept.
    """
    strike_rects = _get_strikethrough_rects(page)

    # ------------------------------------------------------------------ #
    # Build a word-level survival index for partial-strike filtering.     #
    # Only populated when the page actually has strike rects.             #
    # ------------------------------------------------------------------ #
    # Maps (block_number, line_number) → list of non-struck word strings.
    # Both numbers come from the "words" API and match the "dict" API's
    # block/line indices, giving us a shared key between the two APIs.
    word_map: Dict[Tuple[int, int], List[str]] = {}
    if strike_rects:
        # get_text("words") returns one tuple per word:
        # (x0, y0, x1, y1, "word_text", block_no, line_no, word_no)
        for w in page.get_text("words", sort=True):
            x0, y0, x1, y1, word, bno, lno, _ = w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]
            # Only keep words that are NOT struck — dropped words are simply
            # never added to word_map, so they vanish when we reassemble.
            if not _word_is_struck((x0, y0, x1, y1), strike_rects):
                word_map.setdefault((bno, lno), []).append(word)

    lines: List[_TextLine] = []

    # ------------------------------------------------------------------ #
    # Walk the page's text structure: blocks → lines → spans.            #
    # ------------------------------------------------------------------ #
    # get_text("dict") returns the full hierarchy with bounding boxes.
    # sort=True ensures reading order (top-to-bottom, left-to-right).
    text_dict = page.get_text("dict", sort=True)
    for block in text_dict["blocks"]:
        # type=0 is a text block; type=1 is an image block — skip images.
        if block["type"] != 0:
            continue
        bno = block["number"]  # block index on this page, used as part of word_map key

        for lno, line in enumerate(block["lines"]):
            # A line can have multiple spans (e.g. bold + regular on same row).
            # Join all span texts to get the complete visible text of this line.
            raw_text = "".join(s["text"] for s in line["spans"]).strip()

            # Skip blank lines and standalone page numbers (e.g. "7" at page footer).
            if not raw_text or raw_text.isdigit():
                continue

            # Check if this entire line is struck (≥ 50% horizontal coverage).
            struck = bool(strike_rects and _is_struck(line["bbox"], strike_rects))

            if struck:
                # Line is fully struck — keep raw_text unchanged.
                # The two-column parser needs this line's coordinates to correctly
                # assign titles to clauses, even though its text won't appear in output.
                logger.debug("Struck line on page %d: %s…", page.number + 1, raw_text[:40])
                text = raw_text

            elif strike_rects:
                # Page has strike rects but this line is not fully struck.
                # Some individual words within it might still be struck —
                # use word_map to reconstruct only the surviving words.
                kept_words = word_map.get((bno, lno))
                if kept_words is not None and len(kept_words) < len(raw_text.split()):
                    # At least one word was removed — use the filtered version.
                    text = " ".join(kept_words)
                    if text != raw_text:
                        logger.debug(
                            "Partial strike on page %d — kept: %s…",
                            page.number + 1,
                            text[:60],
                        )
                else:
                    # No words were filtered — use raw_text to preserve spacing.
                    text = raw_text

            else:
                # No strike rects on this page at all — fastest path, no filtering needed.
                text = raw_text

            # After word-level filtering a line might become empty (all words struck).
            if not text.strip():
                continue

            lines.append(
                _TextLine(
                    x0=line["bbox"][0],           # left edge — used for column classification
                    y0=y_offset + line["bbox"][1], # absolute y — page-relative y + page offset
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

# Matches a SHELLVOY 5 two-column clause starter in the right body column.
# Format: "N. rest of text" — e.g. "1. Owners shall exercise due diligence"
# Group 1 = clause number, Group 2 = inline text after the number.
_CLAUSE_NUM_RE = re.compile(r"^(\d+)\.\s+(.*)$", re.DOTALL)

# Matches a single-column clause starter at the far-left margin.
# Handles Shell Additional format ("2."), Rider format ("1.  TITLE"),
# and edge cases like "22 .TITLE" (optional whitespace before punctuation).
# Group 1 = clause number, Group 2 = inline text after the number (may be empty).
_RIDER_CLAUSE_NUM_RE = re.compile(r"^(\d+)\s*[.)]+\s*(.*)")

# Maximum character length for text following a clause number to be treated
# as a title rather than the opening sentence of the body.
# "ARBITRATION CLAUSE" (18 chars) → title.
# "The Owners shall not be responsible for..." (45+ chars) → body.
RIDER_INLINE_TITLE_MAX_LEN = 60

# In single-column pages, body text sits at x ≈ 50–149 pt.  Real section
# headers (visually centred on the page) start at x ≥ 150 pt.  This gap is
# measured from the actual document; for other PDFs with different margins the
# constant can be adjusted.
SECTION_HEADER_MIN_X = 150.0

# A section-header candidate must contain at least this many characters to
# eliminate single-word fragments (e.g. "INDEPENDENT", "DIFFER,") that appear
# at high x-offsets due to table or list formatting.
SECTION_HEADER_MIN_LEN = 10

# A clause-number sequence is considered to have "reset" when the new number
# is at most this value and the previous maximum was at least this value.
# Detecting a reset lets us open a new section even when the header is absent
# or uses an unrecognised string.
SECTION_RESET_NEW_MAX = 3
SECTION_RESET_PREV_MIN = 5

# Words excluded from prefix generation (common English stop-words and
# ordinal/month suffixes that add no distinguishing value).
_PREFIX_STOP_WORDS = {
    "THE", "AND", "OF", "FOR", "IN", "TO", "A", "AN", "AT", "BY",
    "IS", "ARE", "ST", "ND", "RD", "TH",
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
}


def _is_section_header(text: str, x0: float) -> bool:
    """Return True when *text* looks like a section-break heading.

    All of the following must hold:

    * ``x0 >= SECTION_HEADER_MIN_X`` — the line starts to the right of normal
      body text (body text sits at x ≈ 50–149 pt; headers are centred at
      x ≥ 150 pt).
    * ``len(text) >= SECTION_HEADER_MIN_LEN`` — long enough to be a title (not
      a stray word or fragment from table formatting).
    * Does not start with a lower-case letter — section headers are always
      proper-case or all-caps; body-text continuations often start lower-case.
    * Is not itself a numbered clause starter.

    No specific section names are hard-coded; the detection is purely
    structural, so any new section in any document will be picked up.
    """
    stripped = text.strip()
    # Too short to be a meaningful heading — likely a stray word or list fragment.
    if len(stripped) < SECTION_HEADER_MIN_LEN:
        return False
    # Body text continuations often start with a lowercase letter; real headings don't.
    if stripped[0].islower():
        return False
    # A clause number like "1." or "22 .TITLE" is not a section header.
    if _RIDER_CLAUSE_NUM_RE.match(stripped):
        return False
    # Final check: must be positioned in the centred heading zone (x ≥ 150 pt).
    return x0 >= SECTION_HEADER_MIN_X


def _prefix_from_header(text: str) -> str:
    """Derive a short section prefix from a header string.

    Takes the first letter of each significant word (length > 2, not a stop
    word) and returns up to three initials, upper-cased.

    Examples
    --------
    ``"SHELL ADDITIONAL CLAUSES - 1st February, 1999"`` → ``"SAC"``
    ``"Essar Rider Clauses (1st Dec 2006)"``             → ``"ERC"``
    ``"Special Provisions Section"``                     → ``"SPS"``
    ``"Maritime Freight Addendum"``                      → ``"MFA"``
    """
    # Extract only alphabetic words — strips punctuation, numbers, brackets.
    words = re.findall(r"[A-Za-z]+", text)
    # Take the first letter of each significant word, skipping stop-words
    # and short words that carry no distinguishing information.
    initials = [
        w[0].upper()
        for w in words
        if w.upper() not in _PREFIX_STOP_WORDS and len(w) > 2
    ]
    # Use up to 3 initials for a compact prefix; fall back to "SEC" if nothing qualifies.
    return "".join(initials[:3]) if initials else "SEC"


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

    # Scan every line on the page looking for evidence of both columns.
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text or text.isdigit():
                continue
            x0 = line["bbox"][0]
            if x0 < TITLE_COLUMN_MAX_X:
                # Left column candidate — but exclude clause-number lines
                # (e.g. "2." at x≈50) that appear on single-column pages.
                # Those would otherwise trigger a false two-column classification.
                if not _RIDER_CLAUSE_NUM_RE.match(text):
                    has_title_col = True
            elif _CLAUSE_NUM_RE.match(text):
                # Right column clause starter found — confirms body column exists.
                has_right_col = True

        # Early exit as soon as both columns are confirmed — no need to scan further.
        if has_title_col and has_right_col:
            return True

    return has_title_col and has_right_col


def _parse_two_column_clauses(all_lines: List[_TextLine]) -> List[RawClause]:
    """Extract clauses from two-column (SHELLVOY 5 style) pages.

    In the two-column layout, clause titles appear in the left column
    (x < TITLE_COLUMN_MAX_X) and clause bodies appear in the right column
    (x >= TITLE_COLUMN_MAX_X).  Titles and bodies are matched by their
    absolute y-coordinates across all two-column pages.

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

    # Build a sorted list of every clause starter seen in the right column.
    # Each entry is (absolute_y, clause_id_string, is_struck).
    # Sorting by y ensures top-to-bottom document order regardless of
    # the order lines were extracted from individual pages.
    starters: List[Tuple[float, str, bool]] = []
    for line in all_lines:
        if line.x0 >= TITLE_COLUMN_MAX_X:
            m = _CLAUSE_NUM_RE.match(line.text)
            if m:
                starters.append((line.y0, m.group(1), line.struck))
    starters.sort(key=lambda t: t[0])

    # Small tolerance (in points) to handle sub-pixel floating-point differences
    # between title line y-coordinates and their corresponding body line y-coordinates.
    # A title rendered at y=302.4 and its clause at y=301.1 should be considered
    # the same vertical position — 1.5 pt absorbs that without crossing into
    # the clause above.
    Y_EPSILON = 1.5

    # Map clause_id → list of title text fragments belonging to that clause.
    # A clause can have multiple title lines (multi-line heading); they are
    # collected in document order and joined with spaces at output time.
    clause_titles: Dict[str, List[str]] = {}
    for line in all_lines:
        # Only process left-column lines — these are the title fragments.
        if line.x0 >= TITLE_COLUMN_MAX_X:
            continue
        # Walk backwards through the sorted starters list to find the last
        # clause starter whose y is at or above this title line's y.
        # "At or above" = the clause that this title visually aligns with.
        assigned_id: Optional[str] = None
        for (sy, sid, _struck) in reversed(starters):
            if sy <= line.y0 + Y_EPSILON:
                assigned_id = sid
                break
        if assigned_id is not None:
            clause_titles.setdefault(assigned_id, []).append(line.text)

    def _title_for(clause_id: str) -> str:
        """Join all collected title fragments for this clause into one string."""
        parts = clause_titles.get(clause_id, [])
        return " ".join(parts).strip()

    # ------------------------------------------------------------------ #
    # Pass 2: assemble clause bodies                                       #
    # ------------------------------------------------------------------ #

    clauses: List[RawClause] = []
    # When a clause is struck and replaced by a new version of the same number,
    # the title was assigned to the struck version in pass 1.  We save it here
    # so the replacement clause can inherit it.
    struck_titles: Dict[str, str] = {}

    current_id: Optional[str] = None
    current_body_parts: List[str] = []
    in_struck_clause = False  # True while accumulating lines for a struck clause

    def _flush() -> None:
        """Save the currently open clause to the output list.

        Only saves non-struck clauses that have body text.
        Inherits the title from a struck predecessor if the clause has none.
        """
        if current_id is not None and not in_struck_clause:
            body = " ".join(current_body_parts).strip()
            title = _title_for(current_id)
            # If this clause has no title of its own, it may be a replacement
            # for a struck clause whose title was assigned to the struck version.
            if not title and current_id in struck_titles:
                title = struck_titles[current_id]
            clauses.append(RawClause(id=current_id, title=title, text=body))

    # Walk right-column lines sequentially to assemble clause bodies.
    # Left-column lines (titles) were fully handled in pass 1 above.
    for line in all_lines:
        if line.x0 < TITLE_COLUMN_MAX_X:
            continue  # left-column title lines — already processed in pass 1

        m = _CLAUSE_NUM_RE.match(line.text)
        if m:
            # New clause starter found — save the previous clause first, then open this one.
            _flush()
            current_id = m.group(1)
            # Include any text on the same line as the number (e.g. "1. Owners shall...")
            current_body_parts = [m.group(2)] if m.group(2).strip() else []
            in_struck_clause = line.struck
            if line.struck:
                # Save this struck clause's title so the replacement clause can inherit it.
                struck_titles[current_id] = _title_for(current_id)
        else:
            # Body continuation line — skip if no clause is open, if we are inside
            # a struck clause, or if this individual line is struck.
            if current_id is None or in_struck_clause or line.struck:
                continue
            current_body_parts.append(line.text)

    # Save the last open clause — there is no following clause starter to trigger _flush.
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
    # ------------------------------------------------------------------ #
    # State variables — the parser's memory between lines and pages.     #
    # ------------------------------------------------------------------ #
    section_counter = 0          # incremented each time a new section opens; used for fallback prefix "S1", "S2"
    current_section = ""         # active section prefix e.g. "SAC", "ERC"
    pending_header: Optional[str] = None  # section heading seen but not yet confirmed
    body_since_header = False    # becomes True when body text appears after a pending_header,
                                 # which means the "header" was actually inside a clause body

    current_id: Optional[str] = None       # clause number currently being built e.g. "1", "43"
    current_id_section = ""                # section prefix at the moment this clause started;
                                           # snapshotted so section changes mid-clause don't corrupt the id
    current_title: Optional[str] = None   # title of the clause being built
    current_body_parts: List[str] = []    # accumulated body lines for the current clause
    awaiting_title = False                 # True after a standalone number line, before the title line arrives

    # Used to detect when the clause numbering resets (e.g. 43 → 1),
    # which signals a new section even if no section header was found.
    section_max_num = 0          # highest clause number seen in the current section
    section_clause_count = 0     # total clauses opened in the current section

    clauses: List[RawClause] = []

    def _open_section(header_text: Optional[str]) -> str:
        """Open a new section, reset per-section counters, and return the new prefix.

        If *header_text* is provided, the prefix is derived from it (e.g.
        "SHELL ADDITIONAL CLAUSES" → "SAC").  Otherwise a sequential fallback
        label "S1", "S2", … is used when a section boundary was detected by
        numbering reset with no visible heading.
        """
        nonlocal section_counter, section_max_num, section_clause_count
        section_counter += 1
        # Reset counters so numbering-reset detection starts fresh in the new section.
        section_max_num = 0
        section_clause_count = 0
        return _prefix_from_header(header_text) if header_text else f"S{section_counter}"

    def _flush() -> None:
        """Save the currently open clause to the output list.

        Guards: does nothing if no clause is open (current_id is None) or if
        the clause has no body text — a bare clause number with no content is
        not a real clause.
        """
        if current_id is not None and current_body_parts:
            # Build the full prefixed id e.g. "SAC-1", "ERC-22".
            # Uses current_id_section (snapshotted at clause open time) not
            # current_section, so a section change between open and flush
            # does not corrupt this clause's id.
            full_id = f"{current_id_section}-{current_id}"
            clauses.append(
                RawClause(
                    id=full_id,
                    title=current_title or "",
                    text=" ".join(current_body_parts).strip(),
                )
            )

    # ------------------------------------------------------------------ #
    # Main loop — iterate pages, then blocks, then lines.                #
    # State persists across pages so a clause spanning a page boundary   #
    # is assembled correctly.                                             #
    # ------------------------------------------------------------------ #
    for page_num in page_nums:
        page = doc[page_num]
        strike_rects = _get_strikethrough_rects(page)

        # Build the word-level survival index for partial-strike filtering.
        # Only runs when the page has strike rects — most single-column pages don't.
        word_map: Dict[Tuple[int, int], List[str]] = {}
        if strike_rects:
            for w in page.get_text("words", sort=True):
                x0, y0, x1, y1 = w[0], w[1], w[2], w[3]
                word, bno, lno = w[4], w[5], w[6]
                # Only add non-struck words; struck words are simply omitted from the map.
                if not _word_is_struck((x0, y0, x1, y1), strike_rects):
                    word_map.setdefault((bno, lno), []).append(word)

        text_dict = page.get_text("dict", sort=True)
        for block in text_dict["blocks"]:
            if block["type"] != 0:
                continue  # skip image blocks
            bno = block["number"]

            for lno, line in enumerate(block["lines"]):
                # Reconstruct the full visible text of this line by joining all spans.
                raw_text = "".join(s["text"] for s in line["spans"]).strip()
                if not raw_text or raw_text.isdigit():
                    continue  # skip blank lines and page-number footers

                x0 = line["bbox"][0]
                struck = bool(strike_rects and _is_struck(line["bbox"], strike_rects))

                if struck:
                    # Entire line is struck — discard immediately.
                    # Unlike the two-column parser we don't need struck lines for
                    # coordinate reference, so we can drop them right here.
                    continue

                # Apply partial word-level strike filtering for lines that are not
                # fully struck but may still contain individually struck words.
                if strike_rects:
                    kept = word_map.get((bno, lno))
                    text = (
                        " ".join(kept)
                        if kept is not None and len(kept) < len(raw_text.split())
                        else raw_text
                    )
                else:
                    # No strike rects on this page — use raw text directly.
                    text = raw_text

                # After word filtering the line might be empty (all words were struck).
                if not text.strip():
                    continue

                # ---------------------------------------------------- #
                # Decision 1 — Section header candidate                 #
                # Centred headings sit at x ≥ 150 pt, are ≥ 10 chars,  #
                # start with a capital, and are not clause numbers.     #
                # ---------------------------------------------------- #
                if _is_section_header(text, x0):
                    # Store as a candidate — do not open the section yet.
                    # The section only opens when a numbered clause immediately
                    # follows with no body text in between (body_since_header guard).
                    pending_header = text
                    body_since_header = False
                    logger.debug("Section header candidate: %s", text[:60])
                    continue

                # ---------------------------------------------------- #
                # Decision 2 — Top-level clause starter                 #
                # Number must be at the far-left margin (x < 60 pt).   #
                # Sub-clauses like "1) Oil Pollution" are indented to   #
                # x ≈ 86 and fail this check — they become body text.  #
                # ---------------------------------------------------- #
                if x0 < RIDER_NUM_X_MAX:
                    m = _RIDER_CLAUSE_NUM_RE.match(text)
                    if m:
                        new_num = int(m.group(1))

                        # Detect a numbering reset (e.g. clause counter jumps from
                        # 43 back to 1), which signals a new section even when there
                        # is no visible section heading between the two sections.
                        numbering_reset = (
                            new_num <= SECTION_RESET_NEW_MAX
                            and section_max_num >= SECTION_RESET_PREV_MIN
                            and section_clause_count >= SECTION_RESET_PREV_MIN
                        )

                        # A pending_header is only valid if no body text appeared
                        # between the heading and this clause number.  If body text
                        # did appear, the "heading" was a deeply indented body line
                        # that falsely looked like a header — discard it.
                        valid_header = pending_header and not body_since_header

                        # Open a new section when: a valid header was found, or no
                        # section has been opened yet, or the numbering reset.
                        if valid_header or (not current_section) or numbering_reset:
                            current_section = _open_section(
                                pending_header if valid_header else None
                            )
                            logger.debug(
                                "Opened section %r (reset=%s, header=%r)",
                                current_section, numbering_reset,
                                pending_header if valid_header else None,
                            )

                        # Consume the header candidate regardless of whether it was used.
                        pending_header = None
                        body_since_header = False

                        # Save the previous clause before opening this one.
                        _flush()

                        current_id = m.group(1)
                        # Snapshot the section prefix NOW so that if the section changes
                        # before this clause is flushed, the id still reflects the
                        # section where the clause actually started.
                        current_id_section = current_section
                        section_max_num = max(section_max_num, new_num)
                        section_clause_count += 1

                        inline_text = m.group(2).strip()
                        if inline_text and len(inline_text) <= RIDER_INLINE_TITLE_MAX_LEN:
                            # Short text after the number (≤ 60 chars) → clause title.
                            # e.g. "1.  ARBITRATION CLAUSE" → title = "ARBITRATION CLAUSE"
                            current_title = inline_text
                            awaiting_title = False
                            current_body_parts = []
                        elif inline_text:
                            # Long text after the number (> 60 chars) → opening sentence
                            # of the body.  No separate title line exists for this clause.
                            current_title = None
                            awaiting_title = False
                            current_body_parts = [inline_text]
                        else:
                            # Nothing after the number (e.g. bare "2.") → title is on
                            # the very next non-empty line.  Set the flag to capture it.
                            current_title = None
                            awaiting_title = True
                            current_body_parts = []
                        continue

                # ---------------------------------------------------- #
                # Decision 3 — No clause open yet                       #
                # Lines before the first clause number are discarded    #
                # (title page, Part I form fields, page headers, etc.). #
                # ---------------------------------------------------- #
                if current_id is None:
                    continue

                # ---------------------------------------------------- #
                # Decision 4 — Awaiting title                           #
                # Only active after a standalone number line ("2.").    #
                # The very next line becomes the clause title.          #
                # ---------------------------------------------------- #
                if awaiting_title:
                    current_title = text
                    awaiting_title = False
                    continue

                # ---------------------------------------------------- #
                # Decision 5 — Body text                                #
                # Everything that reaches here belongs to the body of   #
                # the currently open clause.                            #
                # ---------------------------------------------------- #
                current_body_parts.append(text)
                # Mark that real body content has appeared since the last header
                # candidate.  This invalidates any pending_header that was set
                # while we were inside this clause body.
                body_since_header = True

    # Save the last open clause — no following clause starter will trigger _flush.
    _flush()
    logger.info("Single-column parser: %d clauses from %d pages", len(clauses), len(page_nums))
    return clauses


def extract_clauses_from_pdf(pdf_path: Path) -> Tuple[List[RawClause], int]:
    """Extract all clauses from a charter party PDF.

    The function scans every page and automatically selects the appropriate
    parser based on detected page layout — no page-number constants are used.

    Three parser strategies are applied:

    1. **Two-column parser** (``_parse_two_column_clauses``) — used for pages
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
    # Step 1 – Classify every page as two-column or single-column.       #
    # Pages are inspected by content, not by page number.                #
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
    # Step 2 – Two-column parser (SHELLVOY 5 style).                     #
    # All two-column pages are merged into a single line list first so   #
    # that the two-pass title matcher can work across page boundaries    #
    # using absolute y-coordinates.                                      #
    # ------------------------------------------------------------------ #
    if two_col_pages:
        all_lines: List[_TextLine] = []
        y_offset = 0.0  # accumulates page heights to produce absolute y-coordinates
        for page_num in two_col_pages:
            page = doc[page_num]
            page_lines = _extract_lines(page, y_offset=y_offset)
            all_lines.extend(page_lines)
            # Advance offset by this page's height so the next page's lines
            # have y-coordinates that are strictly greater than this page's.
            y_offset += page.rect.height
            logger.debug("Two-col page %d: %d lines", page_num + 1, len(page_lines))

        shellvoy_clauses = _parse_two_column_clauses(all_lines)
        logger.info("Two-column parser: %d clauses", len(shellvoy_clauses))
        all_clauses.extend(shellvoy_clauses)

    # ------------------------------------------------------------------ #
    # Step 3 – Single-column parser (Shell Additional + Rider styles).   #
    # Pages are passed as a list; the parser's state machine carries     #
    # over between pages so cross-page clauses are assembled correctly.  #
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

    Produces a flat, human-readable string with page markers ([PAGE N]) that
    the LLM receives as input.  The LLM reads this as a document and extracts
    clauses with its language understanding — independent of the layout parser.

    Unlike the layout parsers, this function does not classify lines by column
    or x-position.  It simply collects every non-struck line from every page,
    preserving the original reading order.

    Also returns the total number of pages that contained visible text.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    parts: List[str] = []
    pages = 0
    y_offset = 0.0  # passed to _extract_lines but not used by this function —
                    # absolute y-coordinates are irrelevant when only collecting text

    # Process every page and collect its non-struck lines into a labelled block.
    for page_num in range(total_pages):
        page = doc[page_num]
        # _extract_lines handles strikethrough detection and returns _TextLine objects
        # with a struck flag — we filter to only the visible (non-struck) lines.
        lines = _extract_lines(page, y_offset=y_offset)
        y_offset += page.rect.height
        visible = [ln.text for ln in lines if not ln.struck]
        if visible:
            # Label each page so the LLM knows where page breaks occur.
            parts.append(f"[PAGE {page_num + 1}]\n" + "\n".join(visible))
            pages += 1

    doc.close()
    # Join all page blocks with a blank line between them.
    return "\n\n".join(parts), pages
