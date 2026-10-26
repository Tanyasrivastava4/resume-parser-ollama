"""
scripts/_milestone_v6_9_fidelity_proxy_correction.py

Milestone V6.9 -- the ONE new secondary grounding check found, on
evidence, to be safely correctable among the remaining false-warning
categories V6.5/V6.8 left open: FALSE_WARNING_LABEL_OR_SEGMENTATION
(candidate item is a merged, comma-joined multi-technology string).

Composes on top of `item_is_grounded_v6_8` (V6.8, which itself composes
on the FROZEN V6.4 proxy) -- does not modify either. See
docs/MILESTONE_V6_9_SKILLS_FIDELITY_PROXY_REMAINING_LIMITATIONS.md
Section 1 for the frozen pre-analysis protocol this module is held to.

Design: if the whole candidate item fails the existing V6.8-composed
check, and the item contains at least one comma, split on commas and
re-check EVERY resulting part independently via `item_is_grounded_v6_8`
(so each part gets the full benefit of the frozen primary check, the
frozen label-strip check, AND V6.8's Mechanisms A/B). The item is
credited as grounded ONLY IF EVERY part is independently grounded --
never on a majority or "most parts" basis. This "require ALL parts"
rule is the deliberate safety design: real testing (this milestone,
Section 3) confirmed two real historical merged-string items contain a
MIX of real and unsupported/typo'd sub-parts, and the all-or-nothing
rule correctly leaves those flagged, while safely rescuing a third,
real merged-string item whose every sub-part is independently
verifiable.

The two other remaining categories (FALSE_WARNING_PARAPHRASE_OR_TYPO,
BOUNDARY_OR_SOURCE_AMBIGUITY) and the UNDETERMINED item are
DELIBERATELY NOT addressed here -- see the milestone report for why
each was found unsafe/inappropriate to correct.
"""
from __future__ import annotations

from typing import Any

from scripts._milestone_v6_4_fidelity_proxy import _normalize  # noqa: E402 -- FROZEN, unmodified
from scripts._milestone_v6_8_fidelity_proxy_correction import item_is_grounded_v6_8  # noqa: E402


def _grounded_after_segmentation_split(item: str, body_text: str) -> dict | None:
    """Mechanism: merged comma-joined multi-item string. Returns None
    if not applicable (no comma, or fewer than 2 non-empty parts) or if
    even one part fails; a match dict only if EVERY part independently
    grounds."""
    if "," not in item:
        return None
    parts = [p.strip() for p in item.split(",")]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None

    part_evidence = []
    for part in parts:
        r = item_is_grounded_v6_8(part, body_text)
        part_evidence.append(r)
        if not r.get("grounded"):
            return None  # even one ungrounded part blocks the whole item -- no partial credit

    return {
        "item": item,
        "grounded": True,
        "reason": "grounded_after_segmentation_split",
        "part_evidence": part_evidence,
    }


def item_is_grounded_v6_9(item: str, body_text: str) -> dict:
    """Runs the V6.8-composed check first (unmodified); only if it
    fails does the V6.9 segmentation-split secondary check run."""
    base = item_is_grounded_v6_8(item, body_text)
    if base.get("grounded"):
        return base

    seg = _grounded_after_segmentation_split(item, body_text)
    if seg is not None:
        return seg

    return base  # unchanged: ungrounded, V6.8/V6.4's own reason code


def compute_fidelity_proxy_v6_9(candidate_items: Any, body_text: str, boundary_valid: bool) -> dict:
    """Same contract/shape as the frozen `compute_fidelity_proxy` /
    V6.8's `compute_fidelity_proxy_v6_8`, using `item_is_grounded_v6_9`
    per item. A thin, separate wrapper -- does not modify either prior
    aggregate function."""
    if not boundary_valid:
        return {"proxy_status": "boundary_invalid", "grounding_fraction": None, "n_items": None,
                "n_grounded_direct": None, "n_grounded_after_label_strip": None,
                "n_grounded_after_v6_8_correction": None, "n_grounded_after_v6_9_correction": None,
                "ungrounded_items": None, "per_item_evidence": None}
    if not isinstance(candidate_items, list) or not all(isinstance(x, str) for x in candidate_items):
        return {"proxy_status": "malformed", "grounding_fraction": None, "n_items": None,
                "n_grounded_direct": None, "n_grounded_after_label_strip": None,
                "n_grounded_after_v6_8_correction": None, "n_grounded_after_v6_9_correction": None,
                "ungrounded_items": None, "per_item_evidence": None}
    if len(candidate_items) == 0:
        return {"proxy_status": "empty", "grounding_fraction": None, "n_items": 0,
                "n_grounded_direct": 0, "n_grounded_after_label_strip": 0,
                "n_grounded_after_v6_8_correction": 0, "n_grounded_after_v6_9_correction": 0,
                "ungrounded_items": [], "per_item_evidence": []}

    evidence = [item_is_grounded_v6_9(item, body_text) for item in candidate_items]
    n_items = len(evidence)
    n_grounded = sum(1 for e in evidence if e.get("grounded"))
    n_direct = sum(1 for e in evidence if e.get("reason") == "grounded_direct")
    n_label_strip = sum(1 for e in evidence if e.get("reason") == "grounded_after_label_strip")
    n_v6_8 = sum(1 for e in evidence if e.get("reason") in
                 ("grounded_after_punctuation_normalization", "grounded_after_elliptical_list_expansion"))
    n_v6_9 = sum(1 for e in evidence if e.get("reason") == "grounded_after_segmentation_split")
    ungrounded = [e["item"] for e in evidence if not e.get("grounded")]

    return {
        "proxy_status": "evaluated",
        "grounding_fraction": n_grounded / n_items,
        "n_items": n_items,
        "n_grounded_direct": n_direct,
        "n_grounded_after_label_strip": n_label_strip,
        "n_grounded_after_v6_8_correction": n_v6_8,
        "n_grounded_after_v6_9_correction": n_v6_9,
        "ungrounded_items": ungrounded,
        "per_item_evidence": evidence,
    }
