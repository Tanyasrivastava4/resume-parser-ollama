"""
scripts/_milestone_v6_4_fidelity_proxy.py

Milestone V6.4 -- GT-free Skills fidelity/grounding proxy (RESEARCH/
EVALUATION CODE ONLY, not production). Answers exactly one question:
"does this candidate Skills output appear to remain grounded in the
actual extracted Skills-section source text?" -- a FIDELITY/SAFETY
proxy, never a correctness or completeness evaluator (see the
milestone report's own explicit distinction, Section 3).

Core mechanism, empirically validated in this milestone against both
real historical data and adversarial synthetic cases before being
adopted (see the milestone report's own predeclared validation
protocol): whole-item, case/whitespace-normalized, WORD-BOUNDARY-
anchored substring containment against the real Skills-section source
body -- using Python's `(?<!\\w)...(?!\\w)` lookaround instead of `\\b`,
because raw `\\b...\\b` was found, in this milestone's own testing, to
fail for tokens that begin or end with punctuation (e.g. "C++", "C#"),
since `\\b` requires a word/non-word transition and two adjacent
non-word characters never produce one.

This is a DELIBERATE REFINEMENT of the two existing, unguarded/
partially-guarded containment checks already in this codebase
(`in_section_frac` in scripts/_milestone_v5_9_signal_corrections.py has
NO length guard at all; `fuzzy_eq` in
scripts/_milestone_v5_8_offline_evaluation.py has only a same-length
4-character floor, itself found in this milestone's own testing to
still admit a real, concrete false positive -- "Java" matching inside
"JavaScript" -- even with the length guard applied). The refinement is
narrow (word-boundary anchoring, not a new algorithm family) and
evidence-justified (10/10 adversarial+real test cases pass with it,
6/7 failed the naive fixed-length-guard approach) -- not a blind reuse
and not an unjustified reinvention.

ZERO LLM calls anywhere in this module.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _word_boundary_search(needle: str, haystack: str):
    if not needle:
        return None
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    return re.search(pattern, haystack)


def item_is_grounded(item: str, body_text: str) -> dict:
    """Whole-item, word-boundary-anchored containment check against the
    real Skills-section source body. Returns an EXPLAINABLE record, not
    a bare boolean -- a reviewer must be able to see why an item was
    judged grounded or not.

    Includes a SECONDARY, clearly-distinguished check found necessary
    during this milestone's own real-data validation (resume 018,
    baseline runs 2/5): the model sometimes repeats a category label on
    each split-out item from a single labeled source line (source:
    "MS Office Suite: Advanced Excel, Word, PowerPoint" -> candidate
    items "MS Office Suite: Word", "MS Office Suite: PowerPoint", ...).
    The compound "label: item" string never appears verbatim in the
    source (other comma-separated items sit between the label and any
    one split-out item), so the primary whole-item check correctly
    fails it -- but this is the same, already-established V5.6
    label-inclusion/segmentation-convention mechanism (resume 021), not
    hallucination. A candidate item containing a colon is given a
    SECOND, clearly-labeled chance: if the text AFTER the first colon
    (only) matches, it is classified "grounded_after_label_strip" --
    NEVER silently folded into the primary "grounded_direct" category,
    so a reviewer can always tell the two apart. Verified in this
    milestone (see tests) that this does not create a loophole for
    genuine hallucinations merely by adding a fake colon-prefixed
    label -- the remainder itself must still independently pass the
    same word-boundary check.
    """
    norm_item = _normalize(item)
    norm_body = _normalize(body_text)
    if not norm_item:
        return {"item": item, "grounded": False, "reason": "empty_or_non_string_item"}

    m = _word_boundary_search(norm_item, norm_body)
    if m:
        return {"item": item, "grounded": True, "reason": "grounded_direct", "matched_text": norm_body[m.start():m.end()]}

    if ":" in norm_item:
        remainder = norm_item.split(":", 1)[1].strip()
        m2 = _word_boundary_search(remainder, norm_body)
        if m2:
            return {
                "item": item, "grounded": True, "reason": "grounded_after_label_strip",
                "matched_text": norm_body[m2.start():m2.end()],
                "note": "whole item not found verbatim; matched only after stripping an apparent 'Label: ' prefix -- a known label-inclusion formatting pattern (V5.6), not hallucination",
            }

    return {"item": item, "grounded": False, "reason": "no_word_boundary_match_in_source"}


def compute_fidelity_proxy(candidate_items: Any, body_text: str, boundary_valid: bool) -> dict:
    """Compute the full, explainable fidelity proxy record for one
    candidate Skills extraction.

    IMPORTANT, explicit per Section 3 of the milestone report: the
    returned `grounding_fraction` is a FIDELITY/GROUNDING measure only.
    It is NOT a correctness score, NOT a completeness score, and must
    never be interpreted as either -- a fully-grounded output can still
    be badly incomplete (it says nothing about missing items at all),
    and this function has no access to ground truth to ever know that.
    """
    if not boundary_valid:
        return {
            "proxy_status": "boundary_invalid", "malformed": False,
            "n_items": None, "n_grounded": None, "n_ungrounded": None,
            "grounding_fraction": None, "ungrounded_items": [], "per_item_evidence": [],
            "note": "Skills-section boundary itself was not safely located (V5.9 boundary_content_implausible or not-found) -- no fidelity claim can be made at all.",
        }
    if not isinstance(candidate_items, list) or not all(isinstance(x, str) and x.strip() for x in candidate_items):
        return {
            "proxy_status": "malformed", "malformed": True,
            "n_items": None, "n_grounded": None, "n_ungrounded": None,
            "grounding_fraction": None, "ungrounded_items": [], "per_item_evidence": [],
            "note": "candidate output is not a well-formed list of non-empty strings.",
        }
    if not candidate_items:
        return {
            "proxy_status": "empty", "malformed": False,
            "n_items": 0, "n_grounded": 0, "n_ungrounded": 0,
            "grounding_fraction": None, "ungrounded_items": [], "per_item_evidence": [],
            "note": "candidate returned zero items -- fidelity is undefined, not zero.",
        }

    evidence = [item_is_grounded(item, body_text) for item in candidate_items]
    n_grounded = sum(1 for e in evidence if e["grounded"])
    n_grounded_direct = sum(1 for e in evidence if e.get("reason") == "grounded_direct")
    n_grounded_after_label_strip = sum(1 for e in evidence if e.get("reason") == "grounded_after_label_strip")
    n_total = len(evidence)
    ungrounded = [e["item"] for e in evidence if not e["grounded"]]
    frac = n_grounded / n_total

    return {
        "proxy_status": "evaluated", "malformed": False,
        "n_items": n_total, "n_grounded": n_grounded, "n_ungrounded": n_total - n_grounded,
        "n_grounded_direct": n_grounded_direct,
        "n_grounded_after_label_strip": n_grounded_after_label_strip,
        "grounding_fraction": round(frac, 4), "ungrounded_items": ungrounded,
        "per_item_evidence": evidence,
        "note": "grounding_fraction is a FIDELITY measure only -- not correctness, not completeness. "
                "n_grounded_after_label_strip counts items only matched after stripping an apparent "
                "category-label prefix -- kept visible, never silently folded into n_grounded_direct.",
    }
