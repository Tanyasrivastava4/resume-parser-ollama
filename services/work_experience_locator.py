"""
services/work_experience_locator.py

Work Experience Locator (Milestone 17.7).

Reusable markdown infrastructure for future work_experience validation
rules (WEV-004, WEV-007, WEV-008, WEV-009, WEV-014, WEV-015 in
docs/WORK_EXPERIENCE_VALIDATION.md all need "the corresponding span of
text in the source markdown" for a job — this module is that shared
building block, built standalone in this milestone). It does not itself
validate anything, does not read any extracted JSON, and is not yet
wired into WorkExperienceValidator's rule logic — this milestone only
establishes the capability and its API; consuming it is future work.

Given raw markdown text, this locates every logical "work experience
block" — a company/role's own section of the document — inside the
Experience-type section, and returns each as a JobSpan: the block's
raw markdown text, its 1-indexed start/end line numbers within the
original text, and a best-effort extraction of company/designation/
date_range from the block's own heading/bold lines.

This is a heuristic, structural parser, not a semantic one — it never
looks at what a job entry's text *means*, only at markdown structure
(headings, bold lines, date-range patterns). Real resumes in this
project's dataset are formatted very inconsistently (see
docs/PARSER_ACCURACY_AUDIT.md and docs/WORK_EXPERIENCE_AUDIT.md for
prior findings about this), including within a single document —
heading levels are sometimes inconsistent even across jobs in the same
section (e.g. a level-3 "Professional Experience" section heading
followed by level-2 per-job headings), and some resumes have no
per-job headings at all, describing jobs as plain paragraph sentences.
This module degrades gracefully through several fallback tiers rather
than crashing or guessing, and its real, measured accuracy against
this project's own dataset is reported in
docs/WORK_EXPERIENCE_LOCATOR_RESULTS.md — it is not perfect, and is not
intended to be for every format; future validation rules built on top
of it must tolerate imprecise spans.

Milestone 17.9 adds `WorkExperienceLocator.match_jobs_to_spans()`: given
an extracted work_experience JSON array and the JobSpans `locate()`
found for the same document, it matches each job (by index) to at most
one span, using a confidence score in [0, 1] that combines company
similarity, designation similarity, date overlap, and occurrence order
(§ docs/JOBSPAN_MATCHING_RESULTS.md has the full design and real
measured match-rate improvement). This directly addresses
docs/TECHNOLOGY_VALIDATION_RESULTS.md's own finding that company-name-
only matching (as WEV-015 used in Milestone 17.8) left 36 of 47 real
jobs unmatched, largely because `_looks_like_company` frequently can't
recognize a plain company name with no corporate suffix — company
similarity here is one signal among several, not the only one, so a
job can still be confidently matched via designation and date overlap
alone when its company can't be extracted or matched. A job is only
matched when its best candidate span is both above a minimum
confidence threshold AND clearly (not marginally) its best candidate —
otherwise it is left unmatched, per this milestone's explicit
instruction to never guess. This method is standalone, additive
infrastructure: it does not change `locate()`'s own behavior at all,
and (per this milestone's scope) is not wired into
WorkExperienceValidator — WEV-015 continues to use its own existing,
unchanged company-only matching logic; see
docs/JOBSPAN_MATCHING_RESULTS.md for why.

Milestone V2.4A adds a narrow, additive plain-text fallback to
`_find_section_start`/`_find_section_end` only. Milestone V2.4
(docs/MILESTONE_V2_4_STAGE1_AB_BENCHMARK.md) found that native/
`pdftotext -layout`-extracted resumes have no `#`/`**` Markdown syntax
at all, so `_find_section_start` — gated entirely on `_HEADING_PATTERN`
— always returned `(None, None)` for such documents, making `locate()`
return zero spans and silently disabling WEV-015/016/017/018 (which all
require a located span to check anything). A diagnostic experiment
(same milestone) confirmed this precisely: reactivating span-location
via a plain-text fallback resolved 70 of 109 confirmed unsupported-
`employment_type` false positives across 18 affected resumes, with
**zero** behavior change on any of the 22 OCR-derived production
markdown files (the fallback is only ever consulted when the existing
Markdown-based check finds nothing) and **zero** new false negatives
introduced. `AshishKumarSingh_RBM_Pune_Stonelam`'s repeated
`"WORK EXPERIENCE (A)/(B)/(C)"` marker pattern is a separate,
pre-existing structural case handled by
`_find_repeated_section_name_markers` (Milestone 23.20) — untouched by
this addition, and still gated on `_HEADING_PATTERN` alone; resolving
it is explicitly out of this milestone's scope. The remaining 39
unresolved cases fail at the job-to-span *matching* confidence step
(`match_jobs_to_spans()`), not at section/span location — also
untouched here. See docs/MILESTONE_V2_4A_NATIVE_INPUT_COMPATIBILITY_FIXES.md.
"""

import re
from dataclasses import dataclass, replace
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from dateutil import parser as date_parser

# Heading text (after stripping leading #'s and surrounding whitespace/
# punctuation) that marks the start of the work experience section.
# Matched as a whole, not a substring, to avoid false positives like a
# "Related Experience" bullet inside a Projects section.
_EXPERIENCE_SECTION_NAMES = frozenset(
    {
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "career history",
        "work history",
        "employment",
        # Milestone 23.20: confirmed real, current heading on
        # AMAN_CV_BCA.md ("## Current Employment") -- the section's
        # single job (Asha Tech) was previously invisible to the
        # locator entirely (no _EXPERIENCE_SECTION_NAMES entry
        # matched), so WEV-016/017/018 could never null its
        # unsupported employment_type/location.
        "current employment",
        # Milestone 23.28: confirmed real, current heading on
        # Nikhil Khanna_Data Visualization Analyst-Power BI Developer_Intileo.md
        # ("## Employment Outline") -- both of the section's jobs were
        # previously invisible to the locator entirely (locate()
        # returned zero spans for this document at all), the same
        # category of gap as "current employment" above, just a
        # different literal string. See
        # docs/MILESTONE_23_27_WORK_EXPERIENCE_LOCATOR_NO_SPAN_CENSUS.md
        # §6/§7 for the evidence.
        "employment outline",
    }
)

# Heading text that marks a DIFFERENT section starting — i.e. the work
# experience section has ended. Matched as a substring (not a whole-line
# match like _EXPERIENCE_SECTION_NAMES) since real headings vary more
# here (e.g. "Academic Qualifications", "Technical Skills"). Deliberately
# keyword-based rather than heading-level-based: this project's real
# generated markdown sometimes uses inconsistent heading levels even
# within one resume (see module docstring), so "next heading at level
# <= the section heading's level" is not a reliable end-of-section
# signal on its own.
_NON_EXPERIENCE_SECTION_KEYWORDS = (
    "education",
    "academic",
    "qualification",
    "skill",
    "competenc",
    "certification",
    "project",
    "award",
    "achievement",
    "language",
    "personal profile",
    "personal information",
    "declaration",
    "extra-curricular",
    "extra curricular",
    "hobbies",
    "interest",
    "reference",
    "publication",
    "summary",
    "objective",
    "profile snapshot",
    "contact",
    "strength",
)

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s*(.+?)\s*#*\s*$")
_STANDALONE_BOLD_PATTERN = re.compile(r"^\*\*(.+?)\*\*$")
# A leading list-item marker ("- ", "* ", "+ ", "1. ") in front of an
# otherwise-standalone-bold or heading line — some generated markdown
# wraps a job's bold company name in a bullet.
_LEADING_LIST_MARKER_PATTERN = re.compile(r"^(?:[-*+]|\d+[.)])\s+")

# Milestone 23.20: a trailing parenthetical letter/number group (e.g.
# " (A)", " (b)", " (1)") stripped from a heading's text before
# comparing it against _EXPERIENCE_SECTION_NAMES — used ONLY by
# _find_repeated_section_name_markers below, never by the normal
# single-section-heading path (_find_section_start), which is
# unrelated and unaffected. Confirmed real, current pattern:
# AshishKumarSingh_RBM_Pune_Stonelam.md has no single "Work Experience"
# section heading at all — instead it repeats the section's own name
# as five separate per-job headings, "WORK EXPERIENCE (A)" through
# "Work Experience (E)", at inconsistent heading levels (##, #, #, ###,
# ###) that the existing single-level _find_block_markers tier cannot
# reconcile.
_TRAILING_PARENTHETICAL_SUFFIX_PATTERN = re.compile(r"^(.*?)\s*\([a-z0-9]+\)$", re.IGNORECASE)

# Milestone 23.20: a markdown table row or separator line (e.g. "|
# Company Name | Designation | ... |" or "|---|---|---|") — detected
# structurally: the line's first non-whitespace character is "|" AND
# it contains a second "|" later on. Deliberately anchored to the
# START of the line, not just "contains 2+ pipes" — a heading can
# legitimately use " | " as an inline separator between role/company/
# location (confirmed real, current pattern: Anushka_ (1).md has
# "### Software Development Engineer | The HR Ally | Remote (April
# 2025 - Present)"), which must NOT be mistaken for a table row just
# because it contains pipes; only a line that structurally *is* a
# table row (starts with "|") is skipped. Used so
# _extract_company_and_designation can bail out to (None, None) rather
# than misparse a table's column-header row as a company/designation
# guess (see the module docstring's "if detectable" contract; the
# job's real value is still found via match_jobs_to_spans()'s
# substring-presence fallback, since it appears verbatim in the row).
# Confirmed real pattern: AshishKumarSingh_RBM_Pune_Stonelam.md
# presents every job as a one-row markdown table, not a heading/bold
# line.
_TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|")

# Maximum word count for a heading's text to be eligible as a
# "non-experience section" match (see _NON_EXPERIENCE_SECTION_KEYWORDS).
# Real section headings are almost always short ("Education", "Awards &
# Achievements"); job/role titles that happen to contain an overlapping
# word (e.g. "Manager – Digital Transformation & Projects", which
# contains "project") are typically longer — this guards against
# exactly that kind of false positive.
_MAX_SECTION_HEADING_WORDS = 5

_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_MONTH_NAMES = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
    r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
# Ordered most-specific first, since regex alternation is first-match
# (not longest-match): a day-qualified token (either "DD Month YYYY" or
# "Month DD YYYY" order, both seen in this project's real dataset) must
# be tried before the bare-year fallback, or a phrase like "September
# 21 2020" gets misread as the token "September 21" (treating "21" as a
# 2-digit year) followed by a separate, wrong match on "2020".
_DATE_TOKEN = (
    rf"(?:{_DAY}\s+(?:{_MONTH_NAMES})|(?:{_MONTH_NAMES})\s+{_DAY},?)\.?\s+\d{{4}}"
    rf"|(?:{_MONTH_NAMES})\.?\s*'?\d{{2,4}}"
    rf"|\d{{1,2}}[/\-]\d{{4}}"
    rf"|\d{{4}}"
)
_ONGOING_WORD = r"Present|Current|Till\s+Date|Ongoing|Now"
# "till"/"until" as a separator (e.g. "March 2021 till now") is distinct
# from the ongoing-status phrase "Till Date" above — both are real,
# separate patterns seen in this project's dataset.
_DATE_RANGE_PATTERN = re.compile(
    rf"(?:{_DATE_TOKEN})\s*(?:[-–—]|to|till|until)\s*(?:{_DATE_TOKEN}|{_ONGOING_WORD})",
    re.IGNORECASE,
)

# Substrings that mark a line as naming a company/employer, rather than
# a job title. Deliberately simple and English-language-oriented — this
# is a best-effort heuristic (see "designation (if detectable)" in the
# module's own contract), not a comprehensive company-name detector.
_COMPANY_SUFFIX_KEYWORDS = (
    "ltd",
    "pvt",
    "llp",
    "llc",
    "inc",
    "corp",
    "company",
    "technologies",
    "solutions",
    "enterprises",
    "group",
    "& co",
    "associates",
    "services",
    "systems",
    "industries",
    "international",
    "bank",
    "plc",
    "consultants",
    "consulting",
)


# Milestone 17.9: job-to-span confidence matching. A pair must score at
# least this high, combined, to be considered a match candidate at all.
_MIN_CONFIDENCE = 0.5

# A job's best-scoring candidate span must beat its second-best
# candidate by at least this much to be considered *uniquely* best —
# i.e. "unique position in markdown" from the milestone's own signal
# list. If two candidates are this close, the match is ambiguous and
# the job is left unmatched rather than guessed at.
_AMBIGUITY_MARGIN = 0.05

# How much each available *identity* signal (company, designation,
# date) contributes to the primary combined score. Renormalized over
# whichever of these three are actually available for a given pair
# (see _score_pair) — a job missing a company, for instance, is scored
# using only designation + date, with their weights rescaled to sum to
# 1.0.
_IDENTITY_SIGNAL_WEIGHTS = {"company": 0.5, "designation": 0.3, "date": 0.2}

# Occurrence order is applied separately, as a small nudge on top of
# the primary identity-signal score (see _score_pair) — NOT blended
# into the same renormalized weight pool as company/designation/date.
# It must only break near-ties between two candidates whose identity
# signals are themselves nearly equal; if it were renormalized
# alongside the others, it would end up representing a much *larger*
# share of the total whenever few identity signals are available (e.g.
# company alone), and could then override a clear, real difference in
# company similarity — exactly what "weak, tie-breaking signal only"
# is meant to rule out. Confirmed by a real regression this milestone
# found: two spans "Tech Solutions" (exact match) and "Tech Solutions
# Group" (a different company, ~0.82 fuzzy similarity) — with order
# blended in at full weight, occurrence order alone flipped the
# ranking in favor of the wrong, merely-position-aligned span. This is
# a blend fraction (see _score_pair: score = primary*(1-w) + order*w),
# not an additive nudge — a blend keeps the result naturally in
# [0, 1] with no clipping, which matters because clipping an additive
# nudge at 1.0 can silently erase order's differentiating power
# exactly when it's needed most (two candidates both scoring a
# perfect 1.0 on identity signals — e.g. genuine re-employment at the
# same company with no other differentiating signal — where order is
# the only thing left to break the tie).
_ORDER_TIEBREAK_WEIGHT = 0.15

# Reused for date-signal parsing — see the note on the identical
# constant in services/work_experience_validator.py: fixed, never
# today's real date, so a "Month Year" or bare "Year" string always
# parses to the same result. Duplicated here (not imported from
# work_experience_validator) since that module already imports *this*
# one — importing back would be circular.
_DATE_PARSE_DEFAULT = datetime(1900, 1, 1)

_DATE_RANGE_SPLIT_PATTERN = re.compile(
    rf"^\s*({_DATE_TOKEN})\s*(?:[-–—]|to|till|until)\s*({_DATE_TOKEN}|{_ONGOING_WORD})\s*$",
    re.IGNORECASE,
)
_ONGOING_WORD_PATTERN = re.compile(rf"^(?:{_ONGOING_WORD})$", re.IGNORECASE)


@dataclass
class JobSpan:
    """One logical work-experience block located in the source markdown.

    Attributes:
        company: Best-effort extracted employer name, or None if not
            confidently detected.
        designation: Best-effort extracted job title, or None if not
            confidently detected.
        date_range: The first date-range-shaped substring found within
            the block (e.g. "09/2021 - Present"), or None if none was
            found.
        markdown_text: The block's raw markdown text, exactly as it
            appears in the source (not cleaned or reformatted).
        start_line: 1-indexed line number (within the original markdown
            text passed to locate()) where this block begins.
        end_line: 1-indexed line number (inclusive) where this block ends.
        confidence: 0.0 for every JobSpan `locate()` itself returns —
            `locate()` only parses markdown, it never sees a job to
            compare against, so there's nothing to score yet. Set to a
            real value in [0, 1] only on the copy of a JobSpan returned
            by `match_jobs_to_spans()` (Milestone 17.9), where it's the
            winning combined confidence score for that job/span pairing.
    """

    company: Optional[str]
    designation: Optional[str]
    date_range: Optional[str]
    markdown_text: str
    start_line: int
    end_line: int
    confidence: float = 0.0


class WorkExperienceLocator:
    """Locates logical work-experience blocks within raw resume markdown.

    Stateless: a single instance can be reused freely across resumes.
    """

    def locate(self, markdown_text: str) -> List[JobSpan]:
        """Find every logical work-experience block in `markdown_text`.

        Args:
            markdown_text: The full raw markdown of one resume.

        Returns:
            A list of JobSpan objects, in the order they appear in the
            document. Empty if no work-experience-type section heading
            is found at all.
        """
        lines = markdown_text.splitlines()

        # Milestone 23.20: some resumes repeat the section's own name as
        # each individual job's heading instead of using one section
        # heading followed by per-job sub-headings — see
        # _find_repeated_section_name_markers's docstring. This is
        # structurally different enough (multiple top-level occurrences,
        # often at inconsistent heading levels relative to each other)
        # that it needs its own path; every other resume is completely
        # unaffected, since this only activates when 2+ such headings
        # are found.
        repeated_markers = self._find_repeated_section_name_markers(lines)
        if repeated_markers:
            return self._build_spans_from_repeated_section_name_markers(lines, repeated_markers)

        section_start, section_level = self._find_section_start(lines)
        if section_start is None:
            return []

        section_end = self._find_section_end(lines, section_start, section_level)

        # Body is everything strictly between the section heading and
        # the section end, in 0-indexed line positions.
        body_start = section_start + 1
        body_end = section_end  # exclusive

        markers = self._find_block_markers(lines, body_start, body_end)
        return self._build_spans(lines, body_start, body_end, markers)

    @staticmethod
    def _plain_text_heading_text(line: str) -> Optional[str]:
        """Milestone V2.4A: candidate heading text for a line with no
        Markdown syntax at all — the part before the first ':' if any
        (native/`pdftotext -layout` lines like "WORK EXPERIENCE: ----- 5
        Year & 2 Months" attach trailing decoration after the colon),
        else the whole stripped line. Returns None for a blank line.

        Used only as a fallback by `_find_section_start`/
        `_find_section_end` below, never in place of the existing
        `_HEADING_PATTERN` check — see module docstring.
        """
        stripped = line.strip()
        if not stripped:
            return None
        head = stripped.split(":", 1)[0]
        return head.strip().lower()

    @staticmethod
    def _find_section_start(lines: List[str]) -> Tuple[Optional[int], Optional[int]]:
        """Find the work-experience section's own heading line.

        Returns:
            A (0-indexed line index, heading level) tuple, or
            (None, None) if no such heading exists.
        """
        for index, line in enumerate(lines):
            match = _HEADING_PATTERN.match(line)
            if not match:
                continue
            level = len(match.group(1))
            text = match.group(2).strip().rstrip(":").strip().lower()
            if text in _EXPERIENCE_SECTION_NAMES:
                return index, level

        # Milestone V2.4A: plain-text fallback, consulted ONLY when no
        # Markdown heading matched anything above — every resume where
        # the check above already succeeds (i.e. every OCR-derived
        # document in this project's own corpus, confirmed via a
        # zero-regression check across all 22) is completely unaffected
        # by the code below. See module docstring.
        for index, line in enumerate(lines):
            text = WorkExperienceLocator._plain_text_heading_text(line)
            if text is None:
                continue
            if len(text.split()) > _MAX_SECTION_HEADING_WORDS:
                continue
            if text in _EXPERIENCE_SECTION_NAMES:
                return index, 0
        return None, None

    @staticmethod
    def _find_section_end(lines: List[str], section_start: int, section_level: int) -> int:
        """Find where the work-experience section ends: the next heading
        whose text matches a known non-experience section keyword, or
        the end of the document. Deliberately not based on heading
        level — see the module docstring for why."""
        for index in range(section_start + 1, len(lines)):
            match = _HEADING_PATTERN.match(lines[index])
            if not match:
                continue
            text = match.group(2).strip().lower()
            if len(text.split()) > _MAX_SECTION_HEADING_WORDS:
                continue
            if any(keyword in text for keyword in _NON_EXPERIENCE_SECTION_KEYWORDS):
                return index

        # Milestone V2.4A: plain-text fallback (see _find_section_start
        # above for why, and the module docstring for the evidence).
        # Only reached when the Markdown-based scan above found no
        # non-experience heading — i.e. every resume that already had a
        # real Markdown section end (the entire OCR-derived corpus) is
        # unaffected; this only ever refines an otherwise-defaulted
        # end-of-document boundary.
        #
        # Milestone V2.4E: the plain-text fallback's own keyword check
        # (`_plain_text_heading_text` + substring match) matched on the
        # part of a line *before its first colon* — the rest of the
        # line was never inspected, so a body sentence like "Projects:
        # Big Data Engineering & for an Electronics Giant..." or a bullet
        # like "Collaborated ... applying concepts practiced in academic
        # projects like EcoVision and MockMate." matched "project"/
        # "skill"/"award"/"objective"/"achievement" just as readily as a
        # real heading, cutting the section off mid-job. Confirmed via a
        # corpus-wide scan (docs/MILESTONE_V2_4E_WE_SECTION_BOUNDARY_PRECISION.md
        # §3): 6 of 22 resumes' native-extracted text hit this exact
        # failure, each losing at least one real, later job entirely from
        # the search body — directly explaining several of Milestone
        # V2.4B's own "job located a span but couldn't be confidently
        # matched" cases as spans that were never in the search body at
        # all. `_looks_like_heading` below adds four structural checks —
        # whole-line word count, no leading bullet marker, no trailing
        # sentence punctuation, no large (3+ space) internal gap (the
        # signature of two unrelated columns' text glued onto one native-
        # extraction line, per Milestone V2.4D) — to the *whole* line,
        # not just the pre-colon fragment, before treating a keyword match
        # as a real boundary. Verified zero effect on the Markdown-based
        # tier above (untouched) and zero change on any of the 22 OCR-
        # derived production documents (same milestone, §4/§5) — this
        # only ever makes the *fallback* tier more conservative about
        # what it accepts as a boundary.
        for index in range(section_start + 1, len(lines)):
            stripped_line = lines[index].strip()
            text = WorkExperienceLocator._plain_text_heading_text(lines[index])
            if text is None:
                continue
            if not any(keyword in text for keyword in _NON_EXPERIENCE_SECTION_KEYWORDS):
                continue
            if WorkExperienceLocator._looks_like_heading(stripped_line):
                return index
        return len(lines)

    _BULLET_CHARS = ("•", "-", "*", "▪", "◦", "–", "—")
    _LARGE_INTERNAL_GAP_PATTERN = re.compile(r" {3,}")

    @staticmethod
    def _looks_like_heading(stripped_line: str) -> bool:
        """Milestone V2.4E: structural heading-likeness check for
        `_find_section_end`'s plain-text fallback only (see that
        method's docstring for the evidence this is based on).

        Requires ALL of, checked against the WHOLE line (not just a
        pre-colon fragment):
        - word count within `_MAX_SECTION_HEADING_WORDS`
        - does not start with a bullet marker
        - does not end with terminal sentence punctuation (. , ;)
        - no run of 3+ consecutive spaces (the signature of two
          unrelated columns' text glued onto one native-extraction
          line — a real single-column heading never has this)
        """
        if not stripped_line:
            return False
        if len(stripped_line.split()) > _MAX_SECTION_HEADING_WORDS:
            return False
        if stripped_line[0] in WorkExperienceLocator._BULLET_CHARS:
            return False
        if stripped_line.rstrip()[-1:] in (".", ",", ";"):
            return False
        if WorkExperienceLocator._LARGE_INTERNAL_GAP_PATTERN.search(stripped_line):
            return False
        return True

    @staticmethod
    def _strip_trailing_parenthetical_suffix(text: str) -> str:
        """Milestone 23.20: '"Work Experience (A)"' -> '"Work
        Experience"'. Returns `text` unchanged if it has no trailing
        parenthetical group. See _find_repeated_section_name_markers."""
        match = _TRAILING_PARENTHETICAL_SUFFIX_PATTERN.match(text)
        return match.group(1).strip() if match else text

    @classmethod
    def _find_repeated_section_name_markers(cls, lines: List[str]) -> List[int]:
        """Milestone 23.20: find every heading, anywhere in the document,
        whose text — after stripping a trailing parenthetical letter/
        number suffix like "(A)" — exactly matches a known
        work-experience section name (the same _EXPERIENCE_SECTION_NAMES
        set _find_section_start uses).

        Confirmed real, current pattern (AshishKumarSingh_RBM_Pune_Stonelam.md):
        some resumes repeat the section's own name as each individual
        job's own heading — "WORK EXPERIENCE (A)", "WORK EXPERIENCE
        (B)", ... one per job — instead of one section heading followed
        by consistently-leveled per-job sub-headings. These repeats are
        often at inconsistent heading levels relative to each other
        (confirmed: ##, #, #, ###, ### for this resume's five jobs),
        which the existing single-level `_find_block_markers` tier
        cannot reconcile — the fix is to recognize the repeated marker
        itself as each job's boundary, not to loosen the level filter
        (that would risk merging or splitting on unrelated headings in
        every other resume).

        Milestone 23.28 narrowed this rule's trigger condition after
        Milestone 23.27's dataset-wide census found the original
        condition ("2+ headings anywhere in the document each belong to
        _EXPERIENCE_SECTION_NAMES") over-fires on two real resumes it
        was never intended for: `Rima_Karmakar_CV` (a `"Professional
        Experience"` section followed, after a page-break-derived
        restart, by a separate `"Work Experience"` heading) and
        `Ankush Mehta_Data Scientist_Intileo` (an `"Experience"` heading
        repeated the same way). In both cases the two headings are
        *different, both-legitimate* section-name synonyms, each
        genuinely introducing its own set of distinct jobs — not the
        same literal marker repeated once per job the way
        `AshishKumarSingh`'s `"WORK EXPERIENCE (A)"`..`"(E)"` is. Treating
        them as repeated per-job markers merged 3-4 real jobs under each
        heading into one oversized span apiece, so only the first job
        under each heading could ever be confidently matched — see
        docs/MILESTONE_23_27_WORK_EXPERIENCE_LOCATOR_NO_SPAN_CENSUS.md
        §6/§7/§11 and docs/MILESTONE_23_28_WORK_EXPERIENCE_LOCATOR_PRECISION_FIXES_RESULTS.md
        for the full evidence and verification.

        The fix: headings now only qualify as "repeated markers" when
        their *stripped* text (the same stripping this method already
        used) is identical to each other, not merely each independently
        a member of `_EXPERIENCE_SECTION_NAMES`. `AshishKumarSingh`'s
        pattern is unaffected — every one of its five headings strips
        to the identical `"work experience"`. `Rima_Karmakar_CV`'s
        `"professional experience"` and `"work experience"` are both
        real set members but are no longer treated as repeats of each
        other; same for any other resume where two different recognized
        synonyms happen to both appear.

        Returns the list of 0-indexed line positions for whichever
        single stripped-text group has 2+ occurrences, in document
        order — empty if no such group exists. A single occurrence of
        any one stripped text is not this pattern; it is the ordinary
        case `_find_section_start` already handles, and this method
        deliberately does not interfere with it. In the (currently
        unobserved, in this project's own 23-resume dataset) case where
        two *different* stripped texts each independently repeat 2+
        times in the same document, the larger group is used — this
        method returns one marker list, matching
        `_build_spans_from_repeated_section_name_markers`'s own
        single-pattern design, which this milestone does not change.
        """
        positions_by_stripped_text: Dict[str, List[int]] = {}
        for index, line in enumerate(lines):
            match = _HEADING_PATTERN.match(line)
            if not match:
                continue
            text = match.group(2).strip().rstrip(":").strip()
            stripped = cls._strip_trailing_parenthetical_suffix(text).lower()
            if stripped in _EXPERIENCE_SECTION_NAMES:
                positions_by_stripped_text.setdefault(stripped, []).append(index)

        repeated_groups = [
            positions for positions in positions_by_stripped_text.values() if len(positions) >= 2
        ]
        if not repeated_groups:
            return []
        return max(repeated_groups, key=len)

    def _build_spans_from_repeated_section_name_markers(
        self, lines: List[str], markers: List[int]
    ) -> List[JobSpan]:
        """Milestone 23.20: build one JobSpan per repeated-section-name
        heading found by _find_repeated_section_name_markers — each such
        heading IS a job's own boundary (unlike the normal path, where
        one section heading is followed by separate per-job markers).

        The section body runs from the first marker to the ordinary
        end-of-section scan (the next heading matching a known
        non-experience keyword, exactly as `_find_section_end` already
        does), scanning from after the LAST marker — content between
        the markers themselves is, by definition, inside one job block
        or another and is never inspected for section-end purposes.
        """
        body_start = markers[0]
        section_end = self._find_section_end(lines, markers[-1], section_level=0)
        return self._build_spans(lines, body_start, section_end, markers)

    @staticmethod
    def _find_block_markers(lines: List[str], body_start: int, body_end: int) -> List[int]:
        """Find the 0-indexed line positions that mark the start of each
        job block within [body_start, body_end).

        Tries, in order, the first strategy that finds anything:
        1. Headings, filtered to only the level of the first heading
           found (deeper headings, e.g. a "#### Responsibilities:"
           sub-label within one job, are treated as part of that job's
           own content, not a new block) -- UNLESS Milestone 23.30's
           narrow intermediate-heading fallback (below) applies, in
           which case the deeper, real per-job headings are used
           instead.
        2. Standalone bold lines (a line whose entire content is
           "**...**"), for resumes that mark each job with bold text
           instead of a heading.
        3. Lines containing a date-range pattern, for resumes that
           present jobs as plain paragraphs with no heading or bold
           markup at all.

        Returns an empty list if none of the three find anything — the
        caller then treats the whole section body as one block.
        """
        heading_positions: List[Tuple[int, int]] = []  # (line index, level)
        for index in range(body_start, body_end):
            match = _HEADING_PATTERN.match(lines[index])
            if match:
                heading_positions.append((index, len(match.group(1))))

        if heading_positions:
            marker_level = heading_positions[0][1]
            markers = [index for index, level in heading_positions if level == marker_level]
            if len(markers) == 1:
                fallback = _find_intermediate_heading_fallback_markers(
                    lines, heading_positions, markers[0], marker_level, body_end
                )
                if fallback is not None:
                    return fallback
            return markers

        bold_positions = [
            index
            for index in range(body_start, body_end)
            if _STANDALONE_BOLD_PATTERN.match(lines[index].strip())
        ]
        if bold_positions:
            return bold_positions

        date_positions = [
            index
            for index in range(body_start, body_end)
            if _DATE_RANGE_PATTERN.search(lines[index])
        ]
        return date_positions

    def _build_spans(
        self, lines: List[str], body_start: int, body_end: int, markers: List[int]
    ) -> List[JobSpan]:
        """Turn marker positions into JobSpan objects, covering the
        whole section body — including any leading content before the
        first marker (e.g. a first job with no heading of its own, only
        bold/plain text — see the Dattatray S. Phad case in
        docs/WORK_EXPERIENCE_LOCATOR_RESULTS.md)."""
        if not markers:
            # No heading, bold, or date-range marker was found anywhere
            # in the section at all — there's nothing to confidently
            # call a job title or company name, only unstructured
            # prose, so company/designation stay None rather than
            # guessing from an arbitrary first line (see
            # TestMarkdownFormattingVariations::
            # test_no_structural_markers_at_all_returns_one_whole_section_span).
            block_lines = lines[body_start:body_end]
            block_text = "\n".join(block_lines).strip()
            if not block_text:
                return []
            trimmed_end = body_end
            while trimmed_end > body_start and not lines[trimmed_end - 1].strip():
                trimmed_end -= 1
            return [
                JobSpan(
                    company=None,
                    designation=None,
                    date_range=self._extract_date_range(block_text),
                    markdown_text="\n".join(lines[body_start:trimmed_end]),
                    start_line=body_start + 1,
                    end_line=trimmed_end,
                )
            ]

        boundaries = sorted(markers)
        spans: List[JobSpan] = []

        # Leading content before the first marker, if any, is its own
        # block (only if it contains real content, not just blank lines).
        if boundaries[0] > body_start:
            leading_text = "\n".join(lines[body_start : boundaries[0]]).strip()
            if leading_text:
                spans.append(self._span_from_block(lines, body_start, boundaries[0]))

        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else body_end
            spans.append(self._span_from_block(lines, start, end))

        return spans

    def _span_from_block(self, lines: List[str], start: int, end: int) -> JobSpan:
        """Build one JobSpan from a 0-indexed [start, end) line range."""
        block_lines = lines[start:end]
        # Trim trailing blank lines from the block so end_line reflects
        # real content, not incidental whitespace before the next block.
        trimmed_end = end
        while trimmed_end > start and not lines[trimmed_end - 1].strip():
            trimmed_end -= 1
        block_lines = lines[start:trimmed_end] if trimmed_end > start else block_lines

        markdown_text = "\n".join(block_lines)
        company, designation = self._extract_company_and_designation(block_lines)
        date_range = self._extract_date_range(markdown_text)

        return JobSpan(
            company=company,
            designation=designation,
            date_range=date_range,
            markdown_text=markdown_text,
            start_line=start + 1,
            end_line=max(trimmed_end, start + 1),
        )

    @staticmethod
    def _clean_marker_text(line: str) -> str:
        """Strip heading '#'/bold '**' markup, a leading list-item
        marker (e.g. a bullet wrapping a bold company name), and
        surrounding whitespace from one line, to get its plain text
        content."""
        stripped = _LEADING_LIST_MARKER_PATTERN.sub("", line.strip())
        heading_match = _HEADING_PATTERN.match(stripped)
        if heading_match:
            return heading_match.group(2).strip()
        bold_match = _STANDALONE_BOLD_PATTERN.match(stripped)
        if bold_match:
            return bold_match.group(1).strip()
        return stripped

    @classmethod
    def _looks_like_company(cls, text: str) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in _COMPANY_SUFFIX_KEYWORDS)

    @staticmethod
    def _looks_like_table_row(line: str) -> bool:
        """Milestone 23.20: a markdown table row or separator (e.g. "|
        Company Name | Designation | ... |" or "|---|---|---|") —
        detected structurally (2+ pipe characters), never by content.
        See _TABLE_ROW_PATTERN's own comment for why."""
        return bool(_TABLE_ROW_PATTERN.match(line.strip()))

    @classmethod
    def _extract_company_and_designation(
        cls, block_lines: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Best-effort extraction from the first couple of non-blank
        lines of a block. See the module docstring: this is a
        heuristic, not a guarantee — designation in particular is
        "if detectable" by design."""
        non_blank = [line for line in block_lines if line.strip()]
        if not non_blank:
            return None, None

        # Milestone 23.20: a repeated-section-name marker line (e.g.
        # "## WORK EXPERIENCE (A)") carries no company/designation
        # information of its own — it's the same generic label
        # _find_repeated_section_name_markers matched, just repeated as
        # this job's own heading. Skip it and look at the block's real
        # first content line instead. This can only ever match a line
        # that literally IS the (parenthetical-stripped) section name,
        # so it never affects the normal path, where a block's first
        # line is always a per-job heading/bold line, not the section
        # name itself.
        first_line_text = cls._clean_marker_text(non_blank[0])
        if cls._strip_trailing_parenthetical_suffix(first_line_text).lower() in _EXPERIENCE_SECTION_NAMES:
            non_blank = non_blank[1:]
            if not non_blank:
                return None, None

        # Milestone 23.20: a markdown table row can't be reliably split
        # into company/designation without parsing table syntax — bail
        # out to (None, None) rather than guess (see this method's own
        # docstring). match_jobs_to_spans()'s substring-presence
        # fallback still finds the real value verbatim inside the row's
        # own text when it's there.
        if cls._looks_like_table_row(non_blank[0]):
            return None, None

        first_text = cls._clean_marker_text(non_blank[0])
        if not first_text:
            return None, None

        if cls._looks_like_company(first_text):
            return first_text, None

        # First line reads as a designation (or is ambiguous) — check
        # the next non-blank line for a company name.
        designation = first_text
        company = None
        if len(non_blank) > 1:
            second_text = cls._clean_marker_text(non_blank[1])
            if second_text and cls._looks_like_company(second_text):
                company = second_text
        return company, designation

    @staticmethod
    def _extract_date_range(markdown_text: str) -> Optional[str]:
        match = _DATE_RANGE_PATTERN.search(markdown_text)
        return match.group(0).strip() if match else None

    def match_jobs_to_spans(
        self, jobs: List[Dict[str, Any]], spans: List[JobSpan]
    ) -> Dict[int, JobSpan]:
        """Match each job (by index in `jobs`) to at most one of `spans`.

        Combines four signals into one confidence score in [0, 1] per
        (job, span) pair:

        1. Company similarity — fuzzy text similarity between the
           job's `company` and the span's `company`, OR whether the
           job's company appears verbatim inside the span's own
           markdown text (whichever is higher) — the latter matters
           because this locator's own company extraction is known to
           return None for many real, plain company names (see the
           module docstring), so a job's company can still be
           confirmed by searching the span's raw text directly.
        2. Designation similarity — the same two-part check, against
           the span's `designation` and its raw text.
        3. Date overlap — how much the job's start/end date range
           overlaps the span's own extracted date range, as a
           fraction in [0, 1].
        4. Occurrence order — how close the job's position (among
           `jobs`) is to the span's position (among `spans`), as a
           weak, tie-breaking signal only.

        Any signal that can't be computed (e.g. the job has no
        `company`) is simply excluded, and the remaining signals'
        weights are rescaled to sum to 1.0 — a job can still be
        matched confidently on designation and date alone, for
        instance. At least one of company/designation/date must be
        available; occurrence order alone is never enough (that would
        let two totally unidentifiable jobs "match" purely by
        position).

        A job is only matched if its single best-scoring span both (a)
        scores at least `_MIN_CONFIDENCE`, and (b) beats that job's
        second-best candidate span by at least `_AMBIGUITY_MARGIN` —
        i.e. the winning span must be *uniquely* this job's best
        candidate, not merely the highest of several similarly-weak
        options. If two different jobs both uniquely prefer the same
        span, it goes to whichever scores it higher; the other job is
        left unmatched rather than reassigned to a worse span.

        Args:
            jobs: The work_experience array exactly as extracted (a
                list of job dicts with `company`, `designation`,
                `start_date`, `end_date`, etc.).
            spans: The JobSpans `locate()` found for the same source
                markdown.

        Returns:
            A dict mapping a job's index (only for jobs that got a
            confident, unambiguous match) to a copy of its matched
            JobSpan, with `confidence` set to the winning score.
            Unmatched jobs are simply absent from the dict — never
            guessed at.
        """
        if not jobs or not spans:
            return {}

        num_jobs, num_spans = len(jobs), len(spans)
        matrix = [
            [
                _score_pair(job, span, job_index, span_index, num_jobs, num_spans)
                for span_index, span in enumerate(spans)
            ]
            for job_index, job in enumerate(jobs)
        ]

        candidate_by_job: Dict[int, int] = {}
        for job_index, row in enumerate(matrix):
            ranked = sorted(range(num_spans), key=lambda si: row[si], reverse=True)
            best_span_index = ranked[0]
            best_score = row[best_span_index]
            second_score = row[ranked[1]] if num_spans > 1 else -1.0
            if best_score >= _MIN_CONFIDENCE and (best_score - second_score) >= _AMBIGUITY_MARGIN:
                candidate_by_job[job_index] = best_span_index

        # Two jobs might still both uniquely prefer the same span —
        # award it to whichever scores it higher; the loser stays
        # unmatched rather than falling back to a second-choice span.
        span_claimed_by: Dict[int, Tuple[int, float]] = {}
        for job_index, span_index in candidate_by_job.items():
            score = matrix[job_index][span_index]
            current = span_claimed_by.get(span_index)
            if current is None or score > current[1]:
                span_claimed_by[span_index] = (job_index, score)

        return {
            job_index: replace(spans[span_index], confidence=score)
            for span_index, (job_index, score) in span_claimed_by.items()
        }


def _find_intermediate_heading_fallback_markers(
    lines: List[str],
    heading_positions: List[Tuple[int, int]],
    sole_marker: int,
    marker_level: int,
    body_end: int,
) -> Optional[List[int]]:
    """Milestone 23.30 (Candidate C3, per
    docs/MILESTONE_23_29_INTERMEDIATE_HEADING_FALLBACK_SAFETY_INVESTIGATION.md's
    own validated rule): when `_find_block_markers`'s primary heading
    tier selects exactly one marker at the first-found level, but real
    per-job headings sit one level deeper (an intermediate, non-job
    heading -- e.g. a subsection label like "IT Support Experience" --
    swallowing them, confirmed real on
    Dattatray S. Phad_Desktop Support Engineer_Intileo.md), recover
    those deeper headings as the real job markers instead -- but ONLY
    when every one of them is independently confirmed job-shaped by
    having its own date range within its own immediate block. See
    docs/MILESTONE_23_30_INTERMEDIATE_HEADING_FALLBACK_IMPLEMENTATION_RESULTS.md
    for this implementation's own dataset-wide verification, including
    the two real negative controls (`CA Ishika Sharma Resume`,
    `Krishankant Soni_GAD-UK_GT-Intileo`) this exact gate was built to
    correctly decline on.

    All of the following must hold, checked in this order, with an
    immediate `None` ("do not fall back -- keep the caller's existing,
    unmodified single-marker behavior") the instant any one fails:

    A. Exactly one marker at the selected level -- guaranteed by the
       caller (`_find_block_markers`) before this function is called
       at all; not re-checked here.
    B/C. At least two headings exist at a level deeper than
       `marker_level`, positioned after `sole_marker` -- a single
       deeper heading is ordinary sub-content (e.g. one
       "#### Responsibilities:" label), not repeated per-job
       structure, and is already handled correctly by the existing
       tier without this fallback.
    D. Every one of those deeper headings has a date-range match (the
       same `_DATE_RANGE_PATTERN` this module already uses everywhere
       else -- no new date-detection logic) somewhere within its own
       immediate block: from its own position up to (but not
       including) the next candidate's position, or `body_end` for
       the last one. A date belonging to a different (sibling or
       later) block never satisfies this condition for a heading that
       doesn't itself have one -- checked per-block, not pooled.
    E. Every one of those deeper headings shares the exact same
       level -- checked strictly: if the deeper headings span more
       than one distinct level, this returns `None` outright rather
       than silently filtering down to a same-level subset (which
       would be a broader, unvalidated rule than the one Milestone
       23.29 actually tested).
    F. Candidates are restricted to strictly after `sole_marker`
       (`index > sole_marker`), by construction.
    G. Candidates never extend past `body_end` (the caller's own
       section boundary), by construction -- the last candidate's
       "immediate block" is bounded by `body_end`, never beyond it.

    Args:
        lines: The full document's lines (0-indexed), exactly as
            `_find_block_markers` received them.
        heading_positions: Every heading found within the current
            section body, as (line index, level) pairs, in document
            order -- the same list `_find_block_markers` already
            computed; not recomputed here.
        sole_marker: The single marker position `_find_block_markers`'
            primary tier selected (its own `markers[0]`).
        marker_level: That marker's own heading level.
        body_end: The section body's own end boundary (exclusive),
            exactly as passed into `_find_block_markers`.

    Returns:
        The qualifying deeper-heading positions, in document order, if
        every condition holds; `None` otherwise.
    """
    deeper = [
        (index, level)
        for index, level in heading_positions
        if index > sole_marker and level > marker_level
    ]
    if len(deeper) < 2:
        return None

    levels = {level for _, level in deeper}
    if len(levels) != 1:
        return None

    candidates = [index for index, _ in deeper]

    for position, next_position in zip(candidates, candidates[1:] + [body_end]):
        block_text = "\n".join(lines[position:next_position])
        if not _DATE_RANGE_PATTERN.search(block_text):
            return None

    return candidates


def _normalize_text(text: str) -> str:
    """Lowercase and collapse any run of non-alphanumeric characters to
    a single space — shared normalization for every job-to-span
    similarity/presence check below."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _text_similarity(a: Optional[str], b: Optional[str]) -> Optional[float]:
    """Fuzzy character-level similarity in [0, 1] between two strings,
    after normalization, or None if either is missing/blank."""
    if not isinstance(a, str) or not a.strip() or not isinstance(b, str) or not b.strip():
        return None
    norm_a, norm_b = _normalize_text(a), _normalize_text(b)
    if not norm_a or not norm_b:
        return None
    return SequenceMatcher(None, norm_a, norm_b).ratio()


def _text_present_in_block(text: Optional[str], block_text: str) -> Optional[float]:
    """1.0 if `text`, normalized, appears as a whole-token substring
    inside `block_text` (also normalized); 0.0 if not; None if `text`
    is missing/blank. Whole-token (space-padded), not a raw substring
    check, so a short value like "AI" only matches its own token, not
    a run of letters inside an unrelated word."""
    if not isinstance(text, str) or not text.strip():
        return None
    norm_text = _normalize_text(text)
    if not norm_text:
        return None
    norm_block = _normalize_text(block_text)
    return 1.0 if f" {norm_text} " in f" {norm_block} " else 0.0


def _field_signal(
    value: Any, span_field_value: Optional[str], span_markdown_text: str
) -> Optional[float]:
    """The company/designation signal for one (job, span) pair. None if
    `value` itself is missing — nothing to score.

    When the span never captured this field at all (`span_field_value`
    is None — common for a format this locator only partially parses,
    e.g. one employer heading shared by several role sub-spans, where
    the company name is stated once and not repeated in each role's
    own block), the span hasn't actually asserted anything different
    from `value` — it simply has no data here. In that case, whether
    `value` appears verbatim anywhere in the span's raw text is the
    only thing there is to check: if it does, that's a real positive
    signal; if not, there's genuinely no information (not necessarily
    a mismatch), so this returns None (excluded from scoring) rather
    than a harsh 0.0 that would otherwise veto an otherwise-strong
    match on the *other* signals.

    When the span DOES have its own value for this field, this uses
    ONLY fuzzy similarity against that value — deliberately NOT a
    substring/presence check against the whole span text here. A
    presence check would wrongly give full credit to a short, genuinely
    different company whose name happens to be a prefix of a longer
    one in the same document (e.g. "Tech Solutions" is a verbatim
    substring of "Tech Solutions Group" — two distinct employers);
    fuzzy similarity correctly scores that as a partial, not perfect,
    match instead, preserving the ability to tell them apart."""
    if not isinstance(value, str) or not value.strip():
        return None
    if span_field_value is None:
        presence = _text_present_in_block(value, span_markdown_text)
        return presence if presence else None
    return _text_similarity(value, span_field_value)


def _parse_year_month(date_value: Any) -> Optional[Tuple[int, int]]:
    """Parse a date string into a (year, month) tuple, or None if it's
    missing, not a string, or unparseable. See the identical helper in
    work_experience_validator.py for why the fixed default matters;
    duplicated here rather than imported, to avoid a circular import
    (that module already imports this one)."""
    if not isinstance(date_value, str) or not date_value.strip():
        return None
    try:
        parsed = date_parser.parse(date_value, default=_DATE_PARSE_DEFAULT)
    except (ValueError, OverflowError, TypeError):
        return None
    return parsed.year, parsed.month


def _parse_date_range(
    date_range: str,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Parse a JobSpan's own extracted `date_range` string (e.g. "Jan
    2020 - Present") into a (start, end) pair of (year, month) tuples.
    `end` is None for an open-ended/ongoing range (e.g. "Present") —
    matching how job dicts represent an ongoing role. Anchored to the
    same date-token grammar `locate()` itself used to find this
    substring in the first place, so re-splitting it here is reliable
    rather than a naive hyphen split (which would mis-split a token
    that itself contains a hyphen)."""
    match = _DATE_RANGE_SPLIT_PATTERN.match(date_range.strip())
    if not match:
        return _parse_year_month(date_range), None
    left, right = match.group(1), match.group(2)
    start = _parse_year_month(left)
    end = None if _ONGOING_WORD_PATTERN.match(right.strip()) else _parse_year_month(right)
    return start, end


def _to_month_index(year_month: Tuple[int, int]) -> int:
    year, month = year_month
    return year * 12 + month


def _date_overlap_score(
    job_start: Tuple[int, int],
    job_end: Optional[Tuple[int, int]],
    span_start: Tuple[int, int],
    span_end: Optional[Tuple[int, int]],
) -> float:
    """Fraction (0-1) of overlap between the job's and the span's date
    ranges, as a Jaccard-style ratio of overlapping to total (union)
    months. An open-ended range (end=None, e.g. "currently working" or
    "Present") is resolved to extend at least as far as whichever is
    later of: its own start, the other range's start, or the other
    range's end — a deterministic, self-contained rule (never
    depending on today's real date) that lets two open-ended ranges
    starting around the same time score a high overlap, while an
    open-ended range paired against a much-earlier-ending finite range
    correctly scores low."""
    job_start_idx = _to_month_index(job_start)
    span_start_idx = _to_month_index(span_start)
    job_end_idx = _to_month_index(job_end) if job_end is not None else None
    span_end_idx = _to_month_index(span_end) if span_end is not None else None

    resolved_job_end = (
        job_end_idx
        if job_end_idx is not None
        else max(job_start_idx, span_start_idx, span_end_idx if span_end_idx is not None else span_start_idx)
    )
    resolved_span_end = (
        span_end_idx
        if span_end_idx is not None
        else max(span_start_idx, job_start_idx, job_end_idx if job_end_idx is not None else job_start_idx)
    )

    if job_start_idx > resolved_job_end or span_start_idx > resolved_span_end:
        return 0.0

    overlap = min(resolved_job_end, resolved_span_end) - max(job_start_idx, span_start_idx)
    union = max(resolved_job_end, resolved_span_end) - min(job_start_idx, span_start_idx)
    if overlap < 0:
        return 0.0
    if union == 0:
        return 1.0
    return overlap / union


def _date_signal(job: Dict[str, Any], span: JobSpan) -> Optional[float]:
    """The date-overlap signal for one (job, span) pair, or None if
    either side's date range can't be parsed at all."""
    job_start = _parse_year_month(job.get("start_date"))
    if job_start is None or not span.date_range:
        return None
    span_start, span_end = _parse_date_range(span.date_range)
    if span_start is None:
        return None
    job_end = _parse_year_month(job.get("end_date"))
    return _date_overlap_score(job_start, job_end, span_start, span_end)


def _order_signal(job_index: int, span_index: int, num_jobs: int, num_spans: int) -> float:
    """Weak positional signal: 1.0 when a job and a span sit at the
    same position among their own lists, decaying the further apart
    they are. Always available (never None) — this is why at least
    one *other* signal must also be available for a pair to be scored
    at all (see _score_pair): position alone should never be enough to
    call two otherwise-unidentifiable jobs a confident match."""
    denominator = max(num_jobs, num_spans, 1)
    return max(0.0, 1 - abs(job_index - span_index) / denominator)


def _score_pair(
    job: Dict[str, Any],
    span: JobSpan,
    job_index: int,
    span_index: int,
    num_jobs: int,
    num_spans: int,
) -> float:
    """Combined confidence score in [0, 1] for one (job, span) pair —
    see WorkExperienceLocator.match_jobs_to_spans for the full design.

    The primary score comes from whichever of company/designation/date
    are available, renormalized to sum to 1.0 (see
    _IDENTITY_SIGNAL_WEIGHTS). Occurrence order is then blended in at a
    small, fixed weight (_ORDER_TIEBREAK_WEIGHT) — never renormalized
    alongside the identity signals — so it can only break a near-tie
    between two candidates whose identity signals already point almost
    equally, and can never override a real difference between them
    (see the note on _ORDER_TIEBREAK_WEIGHT)."""
    company_signal = _field_signal(job.get("company"), span.company, span.markdown_text)
    designation_signal = _field_signal(
        job.get("designation"), span.designation, span.markdown_text
    )
    date_signal = _date_signal(job, span)

    identity_signals = {
        "company": company_signal,
        "designation": designation_signal,
        "date": date_signal,
    }
    available = {name: value for name, value in identity_signals.items() if value is not None}
    if not available:
        return 0.0

    total_weight = sum(_IDENTITY_SIGNAL_WEIGHTS[name] for name in available)
    weighted_sum = sum(_IDENTITY_SIGNAL_WEIGHTS[name] * value for name, value in available.items())
    primary_score = weighted_sum / total_weight

    order_signal = _order_signal(job_index, span_index, num_jobs, num_spans)
    return primary_score * (1 - _ORDER_TIEBREAK_WEIGHT) + order_signal * _ORDER_TIEBREAK_WEIGHT
