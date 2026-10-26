"""
scripts/_milestone_v6_5_exploratory_punctuation_insensitive.py

Milestone V6.5 -- EXPLORATORY ONLY. Not part of the frozen V6.4 proxy
validation (see benchmark/baselines/v6_5_full_corpus_item_results.csv
and v6_5_flag_classification.csv for the frozen, unmodified results).
This module is NOT imported by, and does NOT modify, the frozen V6.4
proxy (scripts/_milestone_v6_4_fidelity_proxy.py).

The full-corpus classification (§ milestone report) found the single
largest false-warning category (17 of 78 flagged items, ~22%) was
FALSE_WARNING_FORMATTING -- specifically, spacing/hyphenation
inconsistency between the model's own output and the source text
(e.g. candidate "DataFactory" vs. source "Data Factory"; candidate
"Azure Devops" vs. source "AzureDevops"; candidate "Email" vs. source
"E-mail"). This is a narrow, well-defined, recurring pattern -- a
plausible candidate for a secondary check, investigated here ONLY
after the frozen validation was completed and preserved.

Design: strip spaces AND hyphens (not periods -- "Node.js" vs "Node"
must remain distinguishable) from both item and body before the same
word-boundary-style containment check, but ONLY as a SECONDARY check
(never replacing the primary word-boundary match), and ONLY for items
whose punctuation-stripped form is >=5 characters -- a length guard
found NECESSARY by direct testing below (a naive, unguarded version
was found to coincidentally match "Hive" inside "achivements" once
spaces are stripped from a large body -- exactly the class of risk
V6.4's original design was built to avoid for the primary check).

This has NOT been validated to the same standard as the frozen V6.4
proxy (single corpus pass, no adversarial stress-testing beyond the
tests in this file, no independent full-corpus re-classification) and
must NOT be described as validated production readiness.
"""
from __future__ import annotations

import re


def _strip_punct(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\s\-]+", "", text)
    return text


def is_grounded_punctuation_insensitive(item: str, body_text: str, min_len: int = 5) -> dict:
    """EXPLORATORY secondary check. Returns a record distinct from the
    frozen proxy's own `item_is_grounded` -- callers must not conflate
    the two."""
    stripped_item = _strip_punct(item)
    stripped_body = _strip_punct(body_text)
    if len(stripped_item) < min_len:
        return {"item": item, "grounded_exploratory": False, "reason": "below_min_length_guard"}
    if stripped_item in stripped_body:
        return {"item": item, "grounded_exploratory": True, "reason": "grounded_after_punctuation_strip"}
    return {"item": item, "grounded_exploratory": False, "reason": "no_match_even_after_punctuation_strip"}
