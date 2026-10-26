"""
scripts/_milestone_v6_8_fidelity_proxy_correction.py

Milestone V6.8 -- narrowly-scoped SECONDARY grounding checks for the two
false-warning mechanisms V6.7 left unresolved in the FROZEN V6.4 fidelity
proxy (scripts/_milestone_v6_4_fidelity_proxy.py, NOT modified by this
file -- only imported and composed).

Mechanism A: punctuation/spacing inconsistency (e.g. "DataFactory" vs.
"Data Factory").
Mechanism B: elliptical shared-prefix list expansion (e.g. source
"Windows Server 2012, 2016, 2019" implicitly grounds candidate item
"Windows Server 2016").

Both checks are SECONDARY -- they only run when the frozen proxy's own
`item_is_grounded` (direct + label-strip) already failed, exactly the
same layering pattern the frozen proxy itself uses for label-strip on
top of direct match. Both checks reuse the frozen proxy's own
`_word_boundary_search` lookaround-anchoring design rather than raw
substring containment, specifically to avoid the coincidental-match risk
V6.5's own (never-adopted) exploratory punctuation script was found to
have (see docstring there: "achivements" containing "hive" after naive
space-stripping). See docs/MILESTONE_V6_8_... section 1 for the frozen
pre-analysis protocol this module is held to.

Per that protocol, this module does NOT modify
scripts/_milestone_v6_4_fidelity_proxy.py; it composes on top of it.
"""
from __future__ import annotations

import re
from typing import Any

from scripts._milestone_v6_4_fidelity_proxy import (  # noqa: E402 -- FROZEN, unmodified
    _normalize,
    item_is_grounded,
)

# ---------------------------------------------------------------------------
# Mechanism A: punctuation/spacing-insensitive secondary check
# ---------------------------------------------------------------------------

_MIN_STRIPPED_LEN = 5  # reuses V6.5 exploratory script's own tested guard

# Matches a run of one-or-more word characters (letters/digits) --
# used to walk an item and insert flexibility ONLY between such runs,
# never around a period (periods stay rigid: "Node.js" vs "Node" must
# remain distinguishable, per the forbidden-relaxations list).
_WORDCHAR_RUN_RE = re.compile(r"\w+", re.UNICODE)


def _flexible_gap_pattern(item_norm: str) -> str | None:
    r"""Builds a regex pattern (as a string, to be wrapped in the same
    boundary lookaround the frozen proxy already uses) from a normalized
    item, allowing AT MOST one optional whitespace/hyphen at every
    character-to-character gap. This replaces V6.4/V6.5's earlier idea
    of stripping the ENTIRE body text of whitespace/hyphens before
    comparison -- that approach was found, in this milestone's own
    testing, to destroy TRUE inter-word boundaries in the surrounding
    body (e.g. "Internet and E-mail" -> "internetande-mail" -> stripped
    "internetandemail" loses the real boundary between "and" and
    "e-mail", causing a legitimate rescue candidate to be missed). By
    only inserting flexibility WITHIN the span the item itself occupies,
    and leaving the rest of the body's own punctuation/whitespace
    untouched, the surrounding body's real boundaries are preserved, and
    the final match is still anchored with the same `(?<!\w)`/`(?!\w)`
    lookaround the frozen proxy uses for its primary check -- so a
    coincidental prefix match (e.g. item "Excel" against body
    "Excellent") is still correctly rejected: inserting optional gaps
    between item's own characters does not change what character
    immediately follows the matched span in the body, which is what the
    closing boundary check inspects.
    """
    if len(item_norm) < _MIN_STRIPPED_LEN:
        return None
    parts: list[str] = []
    last_end = 0

    def _emit_gap(gap: str) -> None:
        if gap == "":
            return
        if re.fullmatch(r"[\s\-]+", gap):
            # the gap is PURELY whitespace/hyphen -- make it a single
            # optional separator, so extra/missing/differently-typed
            # whitespace-or-hyphen at this exact position matches either
            # way (this is the ONLY flexibility this function grants).
            parts.append(r"[\s\-]?")
        else:
            # the gap contains other punctuation (e.g. the "." in
            # "Node.js", the "+" in "C++", a "/", a "#") -- this must
            # stay a RIGID, exact, required match. Silently dropping or
            # loosening it would let "NodeJS" match "Node.js"'s pattern,
            # exactly the kind of identity-collapsing normalization the
            # governing brief forbids.
            parts.append(re.escape(gap))

    for m in _WORDCHAR_RUN_RE.finditer(item_norm):
        _emit_gap(item_norm[last_end:m.start()])
        run = m.group(0)
        parts.append(r"[\s\-]?".join(re.escape(ch) for ch in run))
        last_end = m.end()
    _emit_gap(item_norm[last_end:])
    return "".join(parts)


def _grounded_punctuation_insensitive(item: str, body_text: str) -> dict | None:
    """Mechanism A secondary check. Unlike V6.5's exploratory version
    (raw substring containment on a globally-stripped body + length
    guard only), and unlike this milestone's own first draft (also
    global-body stripping, found in testing to destroy real inter-word
    boundaries -- see `_flexible_gap_pattern` docstring), this builds a
    per-item flexible-gap pattern and searches the ORIGINAL
    (punctuation-INTACT) normalized body with the same boundary
    lookaround anchoring the frozen proxy already uses. Returns None if
    not applicable (below the length guard) or not found; a match dict
    otherwise.
    """
    item_norm = _normalize(item)
    pattern_body = _flexible_gap_pattern(item_norm)
    if pattern_body is None:
        return None
    norm_body = _normalize(body_text)
    pattern = r"(?<!\w)" + pattern_body + r"(?!\w)"
    m = re.search(pattern, norm_body)
    if not m:
        return None
    return {
        "item": item,
        "grounded": True,
        "reason": "grounded_after_punctuation_normalization",
        "matched_text": norm_body[m.start():m.end()],
    }


# ---------------------------------------------------------------------------
# Mechanism B: elliptical shared-prefix list expansion
# ---------------------------------------------------------------------------

# A trailing list-continuation token must contain at least one digit --
# this is the explicit structural signal that distinguishes "this is a
# version/model/year code continuing the enumeration" from "this is an
# unrelated coordinate item in the outer list" (e.g. it is what correctly
# excludes "Linux" from "Windows Server 2012, 2016, 2019 and Linux").
_LIST_TOKEN = r"[A-Za-z]*\d+[A-Za-z]*"
_PREFIX = r"(?:[A-Z][A-Za-z]*\s+){1,3}"

# "<Prefix><anchor-token>, <token>, <token> and <token>" / any subset of
# comma-and-and continuations, each individually digit-qualified.
_ELLIPTICAL_LIST_RE = re.compile(
    rf"(?P<prefix>{_PREFIX})(?P<anchor>{_LIST_TOKEN})"
    rf"(?P<continuations>(?:\s*,\s*{_LIST_TOKEN})*(?:\s+and\s+{_LIST_TOKEN})?)"
)

_CONT_TOKEN_RE = re.compile(rf"({_LIST_TOKEN})")


def _elliptical_list_expansions(body_text: str) -> set[str]:
    """Scans body_text LINE BY LINE (never across lines -- a structural
    scope limit) for explicit shared-prefix enumerations and returns the
    set of normalized "prefix + trailing token" expansions those
    enumerations explicitly justify. The anchor token itself is not
    included (it already appears verbatim in the source and is already
    handled by the frozen proxy's direct-match check)."""
    expansions: set[str] = set()
    for line in body_text.split("\n"):
        for match in _ELLIPTICAL_LIST_RE.finditer(line):
            prefix = match.group("prefix")
            continuations = match.group("continuations")
            if not continuations:
                continue
            for tok in _CONT_TOKEN_RE.findall(continuations):
                expansions.add(_normalize(f"{prefix}{tok}"))
    return expansions


def _grounded_elliptical_expansion(item: str, body_text: str) -> dict | None:
    """Mechanism B secondary check."""
    expansions = _elliptical_list_expansions(body_text)
    if _normalize(item) in expansions:
        return {
            "item": item,
            "grounded": True,
            "reason": "grounded_after_elliptical_list_expansion",
        }
    return None


# ---------------------------------------------------------------------------
# Combined V6.8 item-level check (frozen proxy result + both secondaries,
# each tagged with a DISTINCT reason code so results are never conflated)
# ---------------------------------------------------------------------------


def item_is_grounded_v6_8(item: str, body_text: str) -> dict:
    """Runs the frozen V6.4 check first (unmodified); only if it fails
    does either V6.8 secondary check run. Each path is separately
    reason-coded so callers/analysis can attribute every rescue to the
    exact mechanism responsible -- required so Mechanism A and Mechanism
    B are never pooled into a single, misleading combined number."""
    base = item_is_grounded(item, body_text)
    if base.get("grounded"):
        return base

    mech_a = _grounded_punctuation_insensitive(item, body_text)
    if mech_a is not None:
        return mech_a

    mech_b = _grounded_elliptical_expansion(item, body_text)
    if mech_b is not None:
        return mech_b

    return base  # unchanged: ungrounded, frozen proxy's own reason code


def compute_fidelity_proxy_v6_8(candidate_items: Any, body_text: str, boundary_valid: bool) -> dict:
    """Same contract/shape as the frozen `compute_fidelity_proxy`, but
    using `item_is_grounded_v6_8` per item. Kept as a thin, separate
    wrapper (not a modification of the frozen function) so V6.4's own
    aggregate function remains byte-for-byte untouched."""
    if not boundary_valid:
        return {"proxy_status": "boundary_invalid", "grounding_fraction": None, "n_items": None,
                "n_grounded_direct": None, "n_grounded_after_label_strip": None,
                "n_grounded_after_v6_8_correction": None, "ungrounded_items": None, "per_item_evidence": None}
    if not isinstance(candidate_items, list) or not all(isinstance(x, str) for x in candidate_items):
        return {"proxy_status": "malformed", "grounding_fraction": None, "n_items": None,
                "n_grounded_direct": None, "n_grounded_after_label_strip": None,
                "n_grounded_after_v6_8_correction": None, "ungrounded_items": None, "per_item_evidence": None}
    if len(candidate_items) == 0:
        return {"proxy_status": "empty", "grounding_fraction": None, "n_items": 0,
                "n_grounded_direct": 0, "n_grounded_after_label_strip": 0,
                "n_grounded_after_v6_8_correction": 0, "ungrounded_items": [], "per_item_evidence": []}

    evidence = [item_is_grounded_v6_8(item, body_text) for item in candidate_items]
    n_items = len(evidence)
    n_grounded = sum(1 for e in evidence if e.get("grounded"))
    n_direct = sum(1 for e in evidence if e.get("reason") == "grounded_direct")
    n_label_strip = sum(1 for e in evidence if e.get("reason") == "grounded_after_label_strip")
    n_v6_8 = sum(1 for e in evidence if e.get("reason") in
                 ("grounded_after_punctuation_normalization", "grounded_after_elliptical_list_expansion"))
    ungrounded = [e["item"] for e in evidence if not e.get("grounded")]

    return {
        "proxy_status": "evaluated",
        "grounding_fraction": n_grounded / n_items,
        "n_items": n_items,
        "n_grounded_direct": n_direct,
        "n_grounded_after_label_strip": n_label_strip,
        "n_grounded_after_v6_8_correction": n_v6_8,
        "ungrounded_items": ungrounded,
        "per_item_evidence": evidence,
    }
