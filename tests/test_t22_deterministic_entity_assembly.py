"""
tests/test_t22_deterministic_entity_assembly.py

Milestone T22/Phase 2J -- tests for the one frozen deterministic
assembly rule in scripts/_milestone_t22_assemble.py.

Zero LLM calls. Pure unit tests against hand-constructed and real
corpus fixtures.
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
from scripts._milestone_t22_assemble import (  # noqa: E402
    assemble_entities,
    is_accepted,
    load_t21b_decisions,
    run_resume,
    ACCEPTED_DECISIONS,
    ACCEPTED_CLASSIFICATION,
)


class TestAcceptedCandidateRule:
    def test_keep_work_experience_is_accepted(self):
        assert is_accepted({"decision": "KEEP", "classification": "WORK_EXPERIENCE"})

    def test_classify_work_experience_is_accepted(self):
        assert is_accepted({"decision": "CLASSIFY", "classification": "WORK_EXPERIENCE"})

    def test_discard_is_not_accepted(self):
        assert not is_accepted({"decision": "DISCARD", "classification": "OTHER_NON_EXPERIENCE"})

    def test_classify_project_is_not_accepted(self):
        assert not is_accepted({"decision": "CLASSIFY", "classification": "PROJECT"})

    def test_missing_decision_is_not_accepted(self):
        assert not is_accepted(None)

    def test_rule_matches_t21b_vocabulary(self):
        assert ACCEPTED_DECISIONS == {"KEEP", "CLASSIFY"}
        assert ACCEPTED_CLASSIFICATION == "WORK_EXPERIENCE"


class TestAssemblyRule:
    def test_single_candidate_becomes_one_entity(self):
        text = "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n- Did a thing\n"
        blocks = blockify(text)
        candidates = generate_candidates_t20(blocks)
        entities = assemble_entities(blocks, candidates)
        assert len(entities) == 1
        assert entities[0].candidate_ids == [f"C{candidates[0].candidate_id}"]

    def test_dateless_header_merges_forward_into_dated_body(self):
        # Simulates the 021-style split: a bare company heading with no
        # date, immediately (0-gap) followed by a date-bearing block.
        text = "## Acme Corp\n- **Designation:** Engineer\n*Jan 2020 - Dec 2022*\n- Did a thing\n"
        blocks = blockify(text)
        candidates = generate_candidates_t20(blocks)
        entities = assemble_entities(blocks, candidates)
        # Both candidates (if generated as two) end up in one entity.
        assert len(entities) == 1

    def test_two_dated_adjacent_candidates_remain_separate_entities(self):
        # Each already has its own date -> must NOT merge, even though
        # 0-gap adjacent. This is the U1 Cadell-Group/Dali-Ceramicco
        # regression case.
        text = (
            "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n- Did a thing\n"
            "### Widget Inc\n**Manager**\n*2018 - 2020*\n- Did another thing\n"
        )
        blocks = blockify(text)
        candidates = generate_candidates_t20(blocks)
        entities = assemble_entities(blocks, candidates)
        assert len(entities) == 2
        assert entities[0].candidate_ids != entities[1].candidate_ids

    def test_same_company_different_dated_roles_remain_two_entities(self):
        # "Same employer does NOT automatically mean same entity" --
        # both roles have their own dates, so no merge, regardless of
        # the company string being identical.
        text = (
            "### Acme Corp - Senior Engineer\n*Jan 2022 - Present*\n- Led things\n"
            "### Acme Corp - Junior Engineer\n*2020 - 2022*\n- Did things\n"
        )
        blocks = blockify(text)
        candidates = generate_candidates_t20(blocks)
        entities = assemble_entities(blocks, candidates)
        assert len(entities) == 2

    def test_intervening_unaccepted_candidate_prevents_merge_even_if_first_lacks_date(self):
        # C0 ("## Acme Corp") has no date of its own, but a genuine
        # intervening candidate (a Project heading, not accepted) sits
        # structurally between it and the next real job -- this must
        # break 0-gap adjacency and prevent the merge, even though C0
        # lacks its own date.
        text = (
            "## Acme Corp\n- **Designation:** Engineer\n\n"
            "### Side Project\n- Built a thing\n\n"
            "### Widget Inc\n**Manager**\n*2018 - 2020*\n"
        )
        blocks = blockify(text)
        candidates = generate_candidates_t20(blocks)
        # Simulate T21B: the Project candidate (index 1) is NOT accepted.
        accepted = [candidates[0], candidates[2]]
        entities = assemble_entities(blocks, accepted)
        assert len(entities) == 2
        assert entities[0].candidate_ids == [f"C{candidates[0].candidate_id}"]
        assert entities[1].candidate_ids == [f"C{candidates[2].candidate_id}"]

    def test_empty_accepted_list_produces_no_entities(self):
        assert assemble_entities([], []) == []


class TestTraceability:
    def test_every_candidate_id_appears_in_exactly_one_entity(self):
        text = (
            "## Acme Corp\n- **Designation:** Engineer\n*Jan 2020 - Dec 2022*\n- Did a thing\n"
            "### Widget Inc\n**Manager**\n*2018 - 2020*\n- Did another thing\n"
        )
        blocks = blockify(text)
        candidates = generate_candidates_t20(blocks)
        entities = assemble_entities(blocks, candidates)
        all_ids_in_entities = [cid for e in entities for cid in e.candidate_ids]
        assert len(all_ids_in_entities) == len(set(all_ids_in_entities))  # no duplication across entities
        expected_ids = {f"C{c.candidate_id}" for c in candidates}
        assert set(all_ids_in_entities) == expected_ids  # every candidate assigned

    def test_no_entity_references_a_nonexistent_candidate_id(self):
        text = "### Acme Corp\n**Engineer**\n*Jan 2020 - Dec 2022*\n"
        blocks = blockify(text)
        candidates = generate_candidates_t20(blocks)
        entities = assemble_entities(blocks, candidates)
        valid_ids = {f"C{c.candidate_id}" for c in candidates}
        for e in entities:
            for cid in e.candidate_ids:
                assert cid in valid_ids

    def test_candidate_order_is_preserved_within_an_entity(self):
        text = "## Acme Corp\n- **Designation:** Engineer\n*Jan 2020 - Dec 2022*\n- Did a thing\n"
        blocks = blockify(text)
        candidates = generate_candidates_t20(blocks)
        entities = assemble_entities(blocks, candidates)
        for e in entities:
            ids_as_ints = [int(cid[1:]) for cid in e.candidate_ids]
            assert ids_as_ints == sorted(ids_as_ints)


@pytest.mark.local_data('corpus_markdown')
class TestRealCorpusScenarios:
    """Uses the real, frozen T21B decisions and real corpus markdown --
    no GT is read here, only structural behavior is asserted."""

    def test_001_discarded_candidates_never_enter_an_entity(self):
        decisions = load_t21b_decisions()
        blocks, candidates, accepted, entities = run_resume("001", "AMAN_CV_BCA", decisions)
        accepted_ids = {f"C{c.candidate_id}" for c in accepted}
        for e in entities:
            for cid in e.candidate_ids:
                assert cid in accepted_ids  # never a discarded/project candidate

    def test_001_project_candidates_excluded_from_entities(self):
        decisions = load_t21b_decisions()
        blocks, candidates, accepted, entities = run_resume("001", "AMAN_CV_BCA", decisions)
        project_ids = {
            f"C{c.candidate_id}" for c in candidates
            if decisions["001"].get(f"C{c.candidate_id}", {}).get("classification") == "PROJECT"
        }
        entity_ids = {cid for e in entities for cid in e.candidate_ids}
        assert not (project_ids & entity_ids)

    def test_021_tail_candidates_all_traceable_to_some_entity(self):
        decisions = load_t21b_decisions()
        blocks, candidates, accepted, entities = run_resume(
            "021", "PRAGNESH S. SHAH_Sr. Associate-IT Ops_Intileo", decisions
        )
        accepted_ids = {f"C{c.candidate_id}" for c in accepted}
        entity_ids = {cid for e in entities for cid in e.candidate_ids}
        assert accepted_ids == entity_ids  # every accepted candidate assigned, none dropped, none duplicated

    def test_no_llm_call_capability_is_imported_anywhere_in_the_assembler(self):
        import scripts._milestone_t22_assemble as mod
        src = Path(mod.__file__).read_text()
        for banned in ("ollama", "Ollama", "generate_text", "generate_vision"):
            assert banned not in src
