"""
services/education_locator.py

Education Locator (Milestone V2.6).

Additive, standalone deterministic evidence layer for Education, built
directly on measured corpus evidence (benchmark/baselines/
v2_6_education_heading_corpus_scan.csv and docs/
MILESTONE_V2_6_EDUCATION_LOCATOR_FOUNDATION.md's Step 2), following the
architectural shape of services/work_experience_locator.py where the
evidence supports it, and deliberately diverging from it where the
evidence does not.

Like WorkExperienceLocator, this module never reads any extracted JSON
and is not wired into EducationValidator or ParserPipeline -- it only
parses raw Stage-1 text (markdown or plain-text/native) and returns
structural evidence about what the document's Education section
actually contains.

CRITICAL SCOPE DIFFERENCE FROM WorkExperienceLocator: this module does
NOT attempt to extract institution/degree/date VALUES from a candidate
entry the way JobSpan extracts company/designation/date_range. Milestone
V2.6's own corpus scan found this would be unsafe to do reliably today:

- Entry structure varies far more than Work Experience's (at least four
  distinct real conventions measured across 22 resumes: markdown table
  rows, per-entry sub-headings, per-entry bold lines, and plain
  degree-keyword bullets -- see the milestone doc's Step 2).
- Which field a sub-heading names is itself ambiguous and inconsistent
  across real resumes in this dataset: Ritika Jain's sub-headings name
  the *degree* ("### Chartered Accountant (CA)"); Anushka_ (1)'s name
  the *institution* ("### Allenhouse Institute of Technology"). Nothing
  in the heading text itself distinguishes which convention a given
  resume uses.
- Native (`pdftotext -layout`) extraction of a real table can silently
  drop a column's value entirely for some rows and not others --
  confirmed on Krishankant Soni_GAD-UK_GT-Intileo, where the native
  text's "CA Final" row lost its Institution value (present in that
  same document's OCR-derived markdown table) while the very next row
  kept it.
- OCR (vision) transcription can mislabel section content outright --
  confirmed on AshishKumarSingh_RBM_Pune_Stonelam, where the real
  5-entry academic table appears under an unlabeled "Skills" heading
  (no education-family keyword at all), while the document's own
  "## Education" heading wraps one unrelated, single-line music-course
  entry. No structural rule can recover the correct section boundary
  here; the OCR transcription itself put the content in the wrong
  place.

Given that, this locator produces EVIDENCE about each candidate entry
(does it look degree-shaped? institution-shaped? dated?) rather than a
guessed field-value extraction that the above cases show would often be
wrong. Field-level value extraction, if ever attempted, is future work
requiring its own dataset-wide validation -- not assumed here.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# -- Section detection -------------------------------------------------
#
# Whole-heading text (after stripping leading #'s/trailing ':' and
# surrounding whitespace, lowercased) confirmed as a real Education
# section heading somewhere in this project's own 22-resume corpus
# (Milestone V2.6 Step 2 scan, both OCR- and native-derived text) --
# not guessed in advance. See
# benchmark/baselines/v2_6_education_heading_corpus_scan.csv for the
# exact source line each entry below was measured from.
_EDUCATION_SECTION_NAMES = frozenset(
    {
        "education",  # AMAN_CV_BCA, Ankush Mehta, Chirag Sharma, Kishan Yadav,
        # Monika Yathamshetti, Naukri_NileshShivramMagare, Nilesh-Shivram-Magare,
        # Ritika Jain, Anushka_ (1) (native)
        "academic qualifications",  # Anandu Chandran (OCR)
        "academic qualification",  # CA Ishika Sharma Resume
        "educational qualifications",  # Arjun Sharma (OCR)
        "educational qualification",  # PRAGNESH S. SHAH
        "professional and academic qualification",  # Bhavita Jain_GAD_GT
        "professional & educational qualifications",  # Krishankant Soni
        "academic credentials",  # AshishKumarSingh (native only -- see module
        # docstring: this document's own real education table is NOT reachable
        # via any heading match in its OCR text at all), Nikhil Khanna (native)
        "education & certifications",  # Rima_Karmakar_CV -- deliberately
        # included despite naming certifications too: the prompt's own rule 12
        # excludes certification-shaped content at the ENTRY level, which is
        # this locator's _looks_like_certification_only filter's job, not a
        # reason to exclude the section heading itself.
        "education and certifications",
    }
)

# Substrings (matched against a candidate END heading's own text) that
# mark the START of a different, non-Education section -- i.e. Education
# has ended. Deliberately keyword-based, mirroring
# WorkExperienceLocator's _NON_EXPERIENCE_SECTION_KEYWORDS design (real
# heading levels are inconsistent in this project's generated markdown,
# so heading-level comparison alone is not a reliable boundary). Every
# keyword below was confirmed as a REAL heading following a located
# Education section in this milestone's own corpus scan (AshishKumarSingh:
# "Technical Qualification", "Curricular Activities"; PRAGNESH: "Certification";
# Mukul Kanojia: "SKILLS"; Rima_Karmakar_CV: "Work Experience"; and the
# generic set already proven safe by WorkExperienceLocator's own
# equivalent list for the analogous problem).
_NON_EDUCATION_SECTION_KEYWORDS = (
    "experience",
    "skill",
    "project",
    "award",
    "achievement",
    "certificat",  # "Certification" (PRAGNESH) -- deliberately does NOT
    # collide with the "education & certifications" START heading above,
    # since the end-of-section scan only ever looks at headings AFTER
    # section_start, never re-matches the start line itself.
    "curricular activit",
    "language",
    "personal",
    "declaration",
    "hobbies",
    "interest",
    "reference",
    "publication",
    "summary",
    "objective",
    "contact",
    "strength",
)

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s*(.+?)\s*#*\s*$")
_MAX_SECTION_HEADING_WORDS = 6  # "Professional & Educational Qualifications"
# is 4 words; "Professional and Academic Qualification" is 4 words --
# 6 gives headroom without risking a long body sentence being mistaken
# for a heading. Kept close to (not copied blindly from)
# WorkExperienceLocator's own 5, since this module's real measured
# section names run slightly longer.

_BULLET_CHARS = ("•", "-", "*", "▪", "◦", "–", "—")
_LARGE_INTERNAL_GAP_PATTERN = re.compile(r" {3,}")


# -- Candidate entry detection -----------------------------------------

_TABLE_ROW_PATTERN = re.compile(r"^\s*\|.*\|")
_TABLE_SEPARATOR_ROW_PATTERN = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")
_STANDALONE_BOLD_PATTERN = re.compile(r"^\*\*(.+?)\*\*$")
_LEADING_LIST_MARKER_PATTERN = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
# A bullet-wrapped bold line, e.g. "- **Indra Gandhi National Open
# University**" (Mukul Kanojia) -- WorkExperienceLocator's own
# _STANDALONE_BOLD_PATTERN requires the WHOLE line to be "**...**" with
# no leading bullet; this project's real education bullet lists commonly
# wrap the bold text in a list marker, confirmed on Mukul Kanojia's
# document, so this is a distinct, additional pattern, not a copy.
_BULLETED_BOLD_PATTERN = re.compile(r"^(?:[-*+]|\d+[.)])\s+\*\*(.+?)\*\*\s*$")

# Degree/qualification-shaped keyword tokens -- taken directly from
# prompts/modules/education.md's own rule 10 worked examples plus every
# real degree token measured in this project's 22-resume corpus scan
# (10th/12th/SSC/HSC, B.Tech/B.Com/B.Sc/BCA/BBA/BA/B.E, M.Tech/M.Com/
# M.Sc/MBA/MCA/MA, PhD, Diploma, PGDM, CA Foundation/Intermediate/Final,
# Bachelor, Master, Graduation). Not an invented list -- every token is
# traceable to either the prompt's own contract or a real corpus match.
#
# Milestone V2.6B adds exactly the seven terms evidenced by V2.6A
# Experiment 1 (docs/MILESTONE_V2_6A_EDUCATION_FAILURE_DECOMPOSITION.md
# Step 5) and independently re-verified against the real source text of
# every resume that contains them
# (benchmark/baselines/v2_6b_term_evidence_and_impact.csv) before being
# added here -- each is additive only; no existing alternative above was
# changed or removed:
#   - "11th"                (Bhavita Jain_GAD_GT: GT[5] "11th")
#   - "plus two"             (Anandu Chandran...: GT[2] "Plus Two")
#   - "s.s.l.c"/"sslc"       (Anandu Chandran...: GT[3] "S.S.L.C")
#   - "senior secondary"     (Bhavita Jain_GAD_GT: GT[4] "Senior Secondary")
#   - "class" + roman numeral (CA Ishika Sharma Resume: GT[4]/GT[5]
#     "Class- XII"/"Class- X" -- added as a NEW alternative alongside the
#     existing "class\s*\d{1,2}" digit-only form, which does not match a
#     roman-numeral class level and is left completely unchanged)
#   - "ipcc"                 (Monika Yathamshetti...: an intermediate-
#     level CA exam stage not covered by the existing
#     "ca foundation/intermediate/final" alternation -- added as a new
#     alternative inside that SAME group, not a new top-level term)
#   - "chartered accountant" (Ritika Jain...: GT[0] "Chartered Accountant
#     (CA)" -- see the evidence CSV for this term's higher false-positive
#     risk profile relative to the other six, verified zero-impact on
#     this corpus but flagged for continued monitoring)
_DEGREE_KEYWORD_PATTERN = re.compile(
    r"\b("
    r"10th|11th|12th|ssc|hsc|s\.s\.c|h\.s\.c|s\.s\.l\.c|sslc|"
    r"plus\s*two|senior\s*secondary|"
    r"b\.?tech|b\.?e\.?|b\.?com|b\.?sc|bca|bba|b\.?a\.?|bachelor|"
    r"m\.?tech|m\.?com|m\.?sc|mba|mca|m\.?a\.?|master|"
    r"ph\.?d|doctorate|"
    r"diploma|pgdm|"
    r"ca[\s\-]?(?:foundation|intermediate|final|ipcc)|chartered\s*accountant|"
    r"graduation|"
    r"class\s*\d{1,2}|class[\s\-]*(?:xi{0,2}|i[xv]?)\b"
    r")\b",
    re.IGNORECASE,
)

# Institution-shaped keyword tokens -- mirrors
# WorkExperienceLocator._COMPANY_SUFFIX_KEYWORDS's "detect by suffix
# word" approach, using the institution-type words actually seen across
# this project's corpus (university, college, institute, school, board,
# academy).
_INSTITUTION_KEYWORD_PATTERN = re.compile(
    r"\b(university|college|institute|school|board|academy)\b", re.IGNORECASE
)

# Year/date evidence -- deliberately simpler than
# WorkExperienceLocator's full _DATE_RANGE_PATTERN grammar: education
# dates in this corpus are very often a bare year or a "YYYY-YYYY"
# range with no month at all (e.g. "2008-2010", "1998"), which a
# range-only pattern requiring two full date tokens would miss on the
# single-year case. A bare 4-digit year (1900-2099) is treated as
# sufficient date evidence.
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

# Certification-only keyword tokens -- used only to flag (never to
# silently drop) a candidate entry that looks like it names a
# certification body rather than a degree, per the education prompt's
# own rule 12 exclusion list. Kept narrow and evidence-based (Rima
# Karmakar_CV's real "ISTQB Foundation Level & ISTQB Agile Extension"
# line is the only real corpus example of this pattern measured in
# Milestone V2.6's own scan).
_CERTIFICATION_KEYWORD_PATTERN = re.compile(
    r"\b(istqb|pmi|aws certified|microsoft certified|six sigma|"
    r"cisco certified|comptia)\b",
    re.IGNORECASE,
)


@dataclass
class EducationCandidate:
    """One candidate education entry located in the source text.

    Deliberately does NOT include extracted institution/degree/date
    VALUES -- see the module docstring for why. Only evidence flags
    about the candidate's own raw text.

    Attributes:
        source_tier: Which detection tier found this candidate --
            "table_row", "subheading", "bold_line", or
            "degree_keyword_line".
        raw_text: The candidate's own raw text (for a table_row, the
            row's own line; for subheading/bold_line, the block from
            this marker up to the next marker or section end; for
            degree_keyword_line, the matching line only).
        start_line: 1-indexed start line within the original text.
        end_line: 1-indexed end line (inclusive).
        degree_evidence: True if `raw_text` contains a recognized
            degree/qualification keyword token.
        institution_evidence: True if `raw_text` contains a recognized
            institution-type keyword token.
        date_evidence: True if `raw_text` contains a 4-digit year.
        certification_flag: True if `raw_text` matches a recognized
            certification-body keyword -- a hint that this candidate
            may not be a real education entry per the prompt's own
            rule 12, NOT an automatic exclusion (this locator never
            drops candidates; that judgment belongs to a future
            consumer of this evidence).
    """

    source_tier: str
    raw_text: str
    start_line: int
    end_line: int
    degree_evidence: bool
    institution_evidence: bool
    date_evidence: bool
    certification_flag: bool


@dataclass
class EducationLocatorResult:
    """The full evidence this locator produced for one document.

    Attributes:
        section_found: Whether an Education-type section heading was
            located at all.
        section_start_line: 1-indexed line of the section heading, or
            None if section_found is False.
        section_end_line: 1-indexed line (exclusive) where the section
            ends (next non-education heading, or end of document), or
            None if section_found is False.
        candidates: Every EducationCandidate found within the section
            body, in document order. Empty if section_found is False,
            or if a section was found but no candidate-entry tier
            matched anything within it (this locator never falls back
            to treating the whole section body as one entry -- see
            module docstring: no evidence supports that being safe for
            Education the way it's evidenced safe for Work Experience).
        heading_text: The exact (lowercased, stripped) heading text
            that matched, or None.
    """

    section_found: bool
    section_start_line: Optional[int]
    section_end_line: Optional[int]
    candidates: List[EducationCandidate] = field(default_factory=list)
    heading_text: Optional[str] = None


class EducationLocator:
    """Locates the Education section and candidate entries within raw
    resume text (markdown or native/plain-text).

    Stateless: a single instance can be reused freely across resumes.
    """

    def locate(self, text: str) -> EducationLocatorResult:
        lines = text.splitlines()

        section_start, heading_text = self._find_section_start(lines)
        if section_start is None:
            return EducationLocatorResult(
                section_found=False, section_start_line=None, section_end_line=None
            )

        section_end = self._find_section_end(lines, section_start)
        body_start, body_end = section_start + 1, section_end

        candidates = self._find_candidates(lines, body_start, body_end)

        return EducationLocatorResult(
            section_found=True,
            section_start_line=section_start + 1,
            section_end_line=section_end + 1 if section_end < len(lines) else None,
            candidates=candidates,
            heading_text=heading_text,
        )

    # -- Section detection ----------------------------------------------

    @staticmethod
    def _plain_text_heading_text(line: str) -> Optional[str]:
        """Same convention as WorkExperienceLocator._plain_text_heading_text:
        the part of a line before its first ':' (native lines like
        "EDUCATIONAL QUALIFICATION: ---" attach trailing decoration),
        else the whole stripped line. None for a blank line."""
        stripped = line.strip()
        if not stripped:
            return None
        return stripped.split(":", 1)[0].strip().lower()

    @staticmethod
    def _looks_like_heading(stripped_line: str) -> bool:
        """Structural heading-likeness check, identical in shape to
        WorkExperienceLocator._looks_like_heading (Milestone V2.4E) --
        reused here for the same reason: a plain-text keyword match
        alone (e.g. "education" appearing inside a body sentence) is
        not sufficient evidence of a real heading."""
        if not stripped_line:
            return False
        if len(stripped_line.split()) > _MAX_SECTION_HEADING_WORDS:
            return False
        if stripped_line[0] in _BULLET_CHARS:
            return False
        if stripped_line.rstrip()[-1:] in (".", ",", ";"):
            return False
        if _LARGE_INTERNAL_GAP_PATTERN.search(stripped_line):
            return False
        return True

    @classmethod
    def _find_section_start(cls, lines: List[str]) -> Tuple[Optional[int], Optional[str]]:
        for index, line in enumerate(lines):
            match = _HEADING_PATTERN.match(line)
            if not match:
                continue
            text = match.group(2).strip().rstrip(":").strip().lower()
            if text in _EDUCATION_SECTION_NAMES:
                return index, text

        # Plain-text fallback (native/pdftotext-derived text has no `#`
        # syntax at all -- confirmed for every NATIVE representation in
        # this milestone's own corpus scan).
        for index, line in enumerate(lines):
            text = cls._plain_text_heading_text(line)
            if text is None:
                continue
            if len(text.split()) > _MAX_SECTION_HEADING_WORDS:
                continue
            if text in _EDUCATION_SECTION_NAMES and cls._looks_like_heading(line.strip()):
                return index, text
        return None, None

    @classmethod
    def _find_section_end(cls, lines: List[str], section_start: int) -> int:
        for index in range(section_start + 1, len(lines)):
            match = _HEADING_PATTERN.match(lines[index])
            if not match:
                continue
            text = match.group(2).strip().lower()
            if len(text.split()) > _MAX_SECTION_HEADING_WORDS:
                continue
            if any(keyword in text for keyword in _NON_EDUCATION_SECTION_KEYWORDS):
                return index

        for index in range(section_start + 1, len(lines)):
            stripped_line = lines[index].strip()
            text = cls._plain_text_heading_text(lines[index])
            if text is None:
                continue
            if not any(keyword in text for keyword in _NON_EDUCATION_SECTION_KEYWORDS):
                continue
            if cls._looks_like_heading(stripped_line):
                return index
        return len(lines)

    # -- Candidate entry detection ---------------------------------------

    @classmethod
    def _find_candidates(
        cls, lines: List[str], body_start: int, body_end: int
    ) -> List[EducationCandidate]:
        """Try each tier in order; the first tier that finds ANY
        candidate is used exclusively for this document (mirrors
        WorkExperienceLocator._find_block_markers' own tiered-fallback
        shape) -- tiers are not combined within one document, since
        measured evidence shows a single resume consistently uses one
        convention for all its own entries."""
        table_candidates = cls._find_table_row_candidates(lines, body_start, body_end)
        if table_candidates:
            return table_candidates

        subheading_candidates = cls._find_subheading_candidates(lines, body_start, body_end)
        if subheading_candidates:
            return subheading_candidates

        bold_candidates = cls._find_bold_line_candidates(lines, body_start, body_end)
        if bold_candidates:
            return bold_candidates

        return cls._find_degree_keyword_candidates(lines, body_start, body_end)

    @classmethod
    def _build_candidate(cls, tier: str, text: str, start: int, end: int) -> EducationCandidate:
        return EducationCandidate(
            source_tier=tier,
            raw_text=text,
            start_line=start + 1,
            end_line=end,
            degree_evidence=bool(_DEGREE_KEYWORD_PATTERN.search(text)),
            institution_evidence=bool(_INSTITUTION_KEYWORD_PATTERN.search(text)),
            date_evidence=bool(_YEAR_PATTERN.search(text)),
            certification_flag=bool(_CERTIFICATION_KEYWORD_PATTERN.search(text)),
        )

    @classmethod
    def _find_table_row_candidates(
        cls, lines: List[str], body_start: int, body_end: int
    ) -> List[EducationCandidate]:
        """Each real (non-separator) markdown table row within the
        section body is one candidate entry. Confirmed real pattern:
        Bhavita Jain_GAD_GT, Krishankant Soni_GAD-UK_GT-Intileo (OCR
        representations).

        Milestone V2.6C fixes the header-row-skip check below: it was
        gated on `index == body_start`, which in practice never held --
        a markdown heading is always followed by a blank line before
        its table, so the header row's own index is body_start + 2 (one
        blank line, one separator... actually one blank line then the
        header line itself), never body_start. Confirmed via direct
        corpus inspection this made the skip dead code on every real
        table in the corpus. The fix: check whether a row is the FIRST
        REAL (non-separator) table row encountered in the body,
        regardless of how many blank lines precede it.

        A second condition -- skip only when at least one OTHER real
        row in the same body shows year/degree evidence -- was added
        after this milestone's own regression testing found the naive
        first-real-row-only version incorrectly discarded a document's
        ONLY real data row when its degree name falls outside the
        recognized keyword list (Nilesh-Shivram-Magare-CV-Updated (1):
        'Instrumentation Engineering', with no other row in the body to
        compare against). Without this second condition, a table with a
        single, keyword-less data row could have that row wrongly
        treated as a header and discarded entirely -- exactly the kind
        of silent regression this milestone's regression gate is
        designed to catch."""
        rows_info: List[Tuple[int, str, bool]] = []
        for index in range(body_start, body_end):
            line = lines[index]
            if _TABLE_ROW_PATTERN.match(line) and not _TABLE_SEPARATOR_ROW_PATTERN.match(line):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                looks_like_header = not any(_YEAR_PATTERN.search(c) or _DEGREE_KEYWORD_PATTERN.search(c) for c in cells)
                rows_info.append((index, line.strip(), looks_like_header))

        if not rows_info:
            return []

        other_rows_have_evidence = any(not is_header for _, _, is_header in rows_info[1:])
        candidates = []
        for position, (index, text, is_header_like) in enumerate(rows_info):
            if position == 0 and is_header_like and other_rows_have_evidence:
                continue
            candidates.append(cls._build_candidate("table_row", text, index, index + 1))
        return candidates

    @classmethod
    def _find_subheading_candidates(
        cls, lines: List[str], body_start: int, body_end: int
    ) -> List[EducationCandidate]:
        """Level-N markdown sub-headings within the section body, each
        the start of one candidate entry -- only used when 2+ are
        found (a single sub-heading is ordinary sub-content, not a
        repeated per-entry marker, mirroring
        WorkExperienceLocator._find_block_markers' own heading tier).
        Confirmed real pattern: Ritika Jain (degree-named headings),
        Anushka_ (1) (institution-named headings) -- see module
        docstring for why this locator does not assume which."""
        heading_positions = []
        for index in range(body_start, body_end):
            match = _HEADING_PATTERN.match(lines[index])
            if match:
                heading_positions.append(index)
        if len(heading_positions) < 2:
            return []

        candidates = []
        for i, start in enumerate(heading_positions):
            end = heading_positions[i + 1] if i + 1 < len(heading_positions) else body_end
            text = "\n".join(lines[start:end]).strip()
            if text:
                candidates.append(cls._build_candidate("subheading", text, start, end))
        return candidates

    @classmethod
    def _find_bold_line_candidates(
        cls, lines: List[str], body_start: int, body_end: int
    ) -> List[EducationCandidate]:
        """A standalone bold line, or a bullet-wrapped bold line (e.g.
        "- **Indra Gandhi National Open University**"), each the start
        of one candidate entry -- only used when 2+ are found. Confirmed
        real pattern: Mukul Kanojia_Full Stack Developer_Intileo."""
        bold_positions = []
        for index in range(body_start, body_end):
            stripped = lines[index].strip()
            if _STANDALONE_BOLD_PATTERN.match(stripped) or _BULLETED_BOLD_PATTERN.match(stripped):
                bold_positions.append(index)
        if len(bold_positions) < 2:
            return []

        candidates = []
        for i, start in enumerate(bold_positions):
            end = bold_positions[i + 1] if i + 1 < len(bold_positions) else body_end
            text = "\n".join(lines[start:end]).strip()
            if text:
                candidates.append(cls._build_candidate("bold_line", text, start, end))
        return candidates

    @classmethod
    def _find_degree_keyword_candidates(
        cls, lines: List[str], body_start: int, body_end: int
    ) -> List[EducationCandidate]:
        """Last-resort tier: a plain bullet/line (no heading, no bold,
        no table) containing a recognized degree keyword, each its own
        candidate entry -- for resumes with no structural marker at
        all beyond the degree name itself. Confirmed real pattern:
        PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo (OCR
        representation: "- BCA – Yashwantrao Chavan Maharashtra Open
        University", no bold, no sub-heading, no date)."""
        candidates = []
        for index in range(body_start, body_end):
            line = lines[index].strip()
            if not line:
                continue
            if _DEGREE_KEYWORD_PATTERN.search(line):
                candidates.append(cls._build_candidate("degree_keyword_line", line, index, index + 1))
        return candidates
