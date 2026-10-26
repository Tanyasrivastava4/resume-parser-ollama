"""
validators/misrouting_detector.py

Misrouting Detector (Milestone T3/Phase 1, Part D).

Deterministic, GT-free, ZERO-LLM-CALL structural safety checks that
FLAG (never silently fix or remove) content that may have been
misrouted into the wrong module output, specifically the two real,
evidenced mechanisms this project has confirmed:

1. Certification content appearing in `skills` (V3.4/V3.5/V5.6/V6.5's
   real, repeated content-substitution finding for resume 005).
2. Project content appearing in `work_experience` (Milestone T2/Phase 0's
   real, traced finding for resume 001).

Design philosophy, matching the already-validated V6.4-V6.9 GT-free
fidelity-proxy research line: conservative, flag-only, never a hard
filter. A flag is a signal for review, not a correctness verdict --
false negatives (real misrouted content not caught) are preferable to
false positives (legitimate content wrongly flagged), per this
project's own established discipline throughout that research line.

`detect_certification_content_in_skills` was validated in this
milestone against real historical positive evidence (the actual
TRUE_UNSUPPORTED items from resume 005's real V5.6 baseline run3,
preserved in the committed benchmark/baselines/v6_5_flag_classification.csv)
and against all 22 resumes' real, already-committed Skills baseline
output (zero false positives found there). See
docs/MILESTONE_T3_PHASE1_SCHEMA_FOUNDATION.md Part D for the full,
disclosed recall/precision figures -- this detector's real recall on
that one historical case is 12/20 genuinely certification-shaped items
(60%), a known, disclosed limitation, not oversold as comprehensive.

`detect_project_content_in_work_experience` is a structural
CROSS-REFERENCE check: it does not try to classify a work_experience
entry as "really a project" from content patterns alone (evidenced,
in this milestone's own investigation, to be unreliable from a single
real example) -- instead, given a (possibly separately-produced)
projects result for the SAME resume, it flags any work_experience
entry whose `company` fuzzy-matches a project's `name`, i.e. the same
real-world thing appearing in both outputs. This only has something to
compare against once a Projects extraction actually exists for that
resume; with no projects result supplied, it returns no flags (it
never guesses).
"""
from __future__ import annotations

import re
from typing import Any


_CERT_KEYWORD_RE = re.compile(r"certif|credential", re.IGNORECASE)
_CRED_CODE_RE = re.compile(r"\([A-Z]{2,4}-?\d{2,4}\)")


def detect_certification_content_in_skills(skills: list[str]) -> list[dict[str, Any]]:
    """Flags Skills items that look like certification titles, using
    two conservative, real-evidence-derived signals: the word
    "certif"/"credential" appearing anywhere in the item (case-
    insensitive), or a parenthetical credential-code pattern like
    "(AZ-305)" or "(PL-300)". Returns a list of flag records, empty if
    none found. Never modifies `skills`.
    """
    flags = []
    for item in skills:
        if not isinstance(item, str):
            continue
        reasons = []
        if _CERT_KEYWORD_RE.search(item):
            reasons.append("contains 'certif'/'credential' keyword")
        m = _CRED_CODE_RE.search(item)
        if m:
            reasons.append(f"contains credential-code pattern {m.group(0)!r}")
        if reasons:
            flags.append({"item": item, "reasons": reasons, "check": "certification_in_skills"})
    return flags


def _normalize(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _fuzzy_contains(a: str, b: str) -> bool:
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= 0.6


def detect_project_content_in_work_experience(
    work_experience: list[dict[str, Any]],
    projects: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Structural cross-reference: flags a work_experience entry whose
    `company` fuzzy-matches a project's `name` from a SEPARATELY
    supplied projects result for the same resume. If `projects` is
    None or empty, returns no flags -- this check never infers a
    project exists from work_experience content alone (that was found,
    in this milestone's own investigation, to be unreliable from a
    single real example -- see the module docstring). Never modifies
    `work_experience`.
    """
    if not projects:
        return []
    flags = []
    for job in work_experience:
        if not isinstance(job, dict):
            continue
        company = job.get("company")
        if not company:
            continue
        for proj in projects:
            if not isinstance(proj, dict):
                continue
            name = proj.get("name")
            if name and _fuzzy_contains(company, name):
                flags.append({
                    "work_experience_company": company,
                    "matched_project_name": name,
                    "check": "project_in_work_experience",
                })
    return flags
