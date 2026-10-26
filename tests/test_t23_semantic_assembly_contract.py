"""
tests/test_t23_semantic_assembly_contract.py

Milestone T23/Phase 2K -- tests for the ambiguity detector, the LLM
decision contract validator, and the hybrid assembler's fallback/
non-LLM-path behavior.

Zero LLM calls in this file -- pure unit tests against real corpus
fixtures and hand-constructed LLM responses.
"""
from __future__ import annotations

import pytest

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._milestone_t18_blocks import blockify  # noqa: E402
from scripts._milestone_t20_candidates import generate_candidates_t20  # noqa: E402
from scripts._milestone_t22_assemble import load_t21b_decisions, is_accepted  # noqa: E402
from scripts._milestone_t23_ambiguity_detection import (  # noqa: E402
    classify_junction,
    detect_ambiguities_for_resume,
    detect_all,
)
from scripts._milestone_t23_llm_decision import (  # noqa: E402
    validate_response,
    _extract_json_object,
    ALLOWED_DECISIONS,
)
from scripts._milestone_t23_assemble import assemble_entities_hybrid  # noqa: E402

STEMS = {
    "U1": "178480991686_Vachan_Premugh_Dicruz_Founder_Office_BTTR.MEN",
    "U2": "8th_April_Akasa_Kapoor_Resume",
    "U3": "AARTI BAGHOTIYA_Economic Urban Planner_Intileo",
    "U4": "AASHISH PARMAR - Email Marketing Executive - Intileo",
    "U5": "_Adarsh_Bhagat Resume",
    "001": "AMAN_CV_BCA",
    "006": "AshishKumarSingh_RBM_Pune_Stonelam",
    "013": "Kishan Yadav_Data Scientist_Intileo",
    "021": "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo",
}


@pytest.mark.local_data('corpus_markdown', 'unseen_markdown')
class TestAmbiguityDetectorReusability:
    """The detector must be reusable, not hardcoded to U2 -- verified
    by running it blind across all 9 resumes and checking the result
    matches the pre-registered, evidence-derived expectation exactly."""

    def test_exactly_two_ambiguous_junctions_across_the_whole_corpus(self):
        records = detect_all(STEMS)
        ambiguous = [r for r in records if r.classification == "SEMANTICALLY_AMBIGUOUS"]
        assert len(ambiguous) == 2
        assert {(r.resume_id, r.candidate_a_id, r.candidate_b_id) for r in ambiguous} == {
            ("U2", "C10", "C11"), ("U2", "C13", "C14"),
        }

    def test_total_junction_count_matches_pre_registration(self):
        records = detect_all(STEMS)
        assert len(records) == 23

    def test_021_has_zero_ambiguous_junctions(self):
        decisions = load_t21b_decisions()
        records = detect_ambiguities_for_resume("021", STEMS["021"], decisions)
        assert all(r.classification != "SEMANTICALLY_AMBIGUOUS" for r in records)

    def test_classification_partition_is_exhaustive_and_exclusive(self):
        records = detect_all(STEMS)
        for r in records:
            assert r.classification in ("CLEAR_MERGE", "CLEAR_SEPARATE", "SEMANTICALLY_AMBIGUOUS")


class TestClassifyJunctionRule:
    def test_first_has_date_is_clear_separate(self):
        text = "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n\n### Widget Inc\n**Manager**\n*2018 - 2020*\n"
        blocks = blockify(text)
        candidates = generate_candidates_t20(blocks)
        rec = classify_junction(blocks, candidates[0], candidates[1])
        assert rec.classification == "CLEAR_SEPARATE"

    def test_first_lacks_date_second_has_date_is_clear_merge(self):
        text = "## Acme Corp\n- **Designation:** Engineer\n*Jan 2020 - Dec 2022*\n- Did a thing\n"
        blocks = blockify(text)
        candidates = generate_candidates_t20(blocks)
        if len(candidates) >= 2:
            rec = classify_junction(blocks, candidates[0], candidates[1])
            assert rec.classification == "CLEAR_MERGE"

    def test_neither_has_date_is_semantically_ambiguous(self):
        text = "# Acting Manager\n- Did a thing\n- Did another\n\n# Analyst\n- Did something else\n"
        blocks = blockify(text)
        candidates = generate_candidates_t20(blocks)
        rec = classify_junction(blocks, candidates[0], candidates[1])
        assert rec.classification == "SEMANTICALLY_AMBIGUOUS"


class TestLLMContractValidation:
    VALID_IDS = {"C10", "C11"}

    def test_valid_merge_response_accepted(self):
        parsed = {"ambiguity_id": "U2_AMB_001", "decision": "MERGE_SPECIFIED", "merge_groups": [["C10", "C11"]]}
        assert validate_response(parsed, "U2_AMB_001", self.VALID_IDS) == []

    def test_valid_separate_response_accepted(self):
        parsed = {"ambiguity_id": "U2_AMB_001", "decision": "KEEP_SEPARATE", "merge_groups": []}
        assert validate_response(parsed, "U2_AMB_001", self.VALID_IDS) == []

    def test_invented_candidate_id_rejected(self):
        parsed = {"ambiguity_id": "U2_AMB_001", "decision": "MERGE_SPECIFIED", "merge_groups": [["C10", "C99"]]}
        errors = validate_response(parsed, "U2_AMB_001", self.VALID_IDS)
        assert any("INVENTED" in e for e in errors)

    def test_out_of_group_candidate_id_rejected(self):
        # C12 is a real candidate ID elsewhere in the resume, but not
        # part of THIS ambiguity's own pair -- must be rejected.
        parsed = {"ambiguity_id": "U2_AMB_001", "decision": "MERGE_SPECIFIED", "merge_groups": [["C10", "C12"]]}
        errors = validate_response(parsed, "U2_AMB_001", self.VALID_IDS)
        assert any("INVENTED" in e or "does not exactly match" in e for e in errors)

    def test_mismatched_ambiguity_id_rejected(self):
        parsed = {"ambiguity_id": "WRONG_ID", "decision": "KEEP_SEPARATE", "merge_groups": []}
        errors = validate_response(parsed, "U2_AMB_001", self.VALID_IDS)
        assert any("ambiguity_id" in e for e in errors)

    def test_unsupported_decision_rejected(self):
        parsed = {"ambiguity_id": "U2_AMB_001", "decision": "MAYBE", "merge_groups": []}
        errors = validate_response(parsed, "U2_AMB_001", self.VALID_IDS)
        assert any("unsupported decision" in e for e in errors)

    def test_merge_specified_requires_nonempty_groups(self):
        parsed = {"ambiguity_id": "U2_AMB_001", "decision": "MERGE_SPECIFIED", "merge_groups": []}
        errors = validate_response(parsed, "U2_AMB_001", self.VALID_IDS)
        assert any("MERGE_SPECIFIED must include" in e for e in errors)

    def test_keep_separate_requires_empty_groups(self):
        parsed = {"ambiguity_id": "U2_AMB_001", "decision": "KEEP_SEPARATE", "merge_groups": [["C10", "C11"]]}
        errors = validate_response(parsed, "U2_AMB_001", self.VALID_IDS)
        assert any("KEEP_SEPARATE must have empty" in e for e in errors)

    def test_allowed_decisions_are_exactly_two(self):
        assert ALLOWED_DECISIONS == {"KEEP_SEPARATE", "MERGE_SPECIFIED"}

    def test_non_dict_response_rejected(self):
        assert validate_response("not a dict", "U2_AMB_001", self.VALID_IDS) == ["response is not a JSON object"]


class TestJsonExtraction:
    def test_extracts_plain_json(self):
        assert _extract_json_object('{"a": 1}') == '{"a": 1}'

    def test_extracts_after_think_block(self):
        assert _extract_json_object("<think>...</think>\n{\"a\": 1}") == '{"a": 1}'

    def test_returns_none_when_absent(self):
        assert _extract_json_object("no json") is None


@pytest.mark.local_data('corpus_markdown', 'unseen_markdown')
class TestHybridAssemblerNonLLMPath:
    """For resumes with zero ambiguous junctions, the hybrid assembler
    must produce output identical to T22's own deterministic-only
    result, with ZERO LLM calls (client=None must not raise)."""

    def test_021_requires_no_llm_client(self):
        decisions = load_t21b_decisions()
        blocks = blockify(open(f"output/markdown/{STEMS['021']}.md").read())
        candidates = generate_candidates_t20(blocks)
        dmap = decisions.get("021", {})
        accepted = sorted([c for c in candidates if is_accepted(dmap.get(f"C{c.candidate_id}"))],
                           key=lambda c: c.extent_start)
        entities, llm_calls = assemble_entities_hybrid("021", blocks, accepted, client=None)
        assert llm_calls == []
        assert len(entities) == 8  # matches T22's own frozen result exactly

    def test_missing_client_raises_only_when_ambiguity_actually_present(self):
        decisions = load_t21b_decisions()
        blocks = blockify(open(f"output/markdown/{STEMS['U2']}.md").read())
        candidates = generate_candidates_t20(blocks)
        dmap = decisions.get("U2", {})
        accepted = sorted([c for c in candidates if is_accepted(dmap.get(f"C{c.candidate_id}"))],
                           key=lambda c: c.extent_start)
        import pytest
        with pytest.raises(RuntimeError):
            assemble_entities_hybrid("U2", blocks, accepted, client=None)
