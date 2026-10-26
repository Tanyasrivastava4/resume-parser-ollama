"""
scripts/_milestone_v6_5_classify_flagged.py

Milestone V6.5 -- manual, evidence-based classification of every item
the FROZEN V6.4 proxy flagged as ungrounded across the full-corpus run
(benchmark/baselines/v6_5_full_corpus_item_results.csv). Every
classification below is grounded in direct inspection performed in
this milestone (real source text, real full-document text, and for
005 specifically a real, newly-discovered boundary-detector defect --
see the milestone report) -- not automated guessing.

Categories (defined before this classification was finalized, per the
milestone's own predeclared protocol):
  TRUE_UNSUPPORTED               -- item genuinely lacks source support
  FALSE_WARNING_FORMATTING       -- source support exists; a formatting/
                                     spacing/hyphenation/typo/elliptical-
                                     list-expansion difference prevented
                                     literal recognition
  FALSE_WARNING_LABEL_OR_SEGMENTATION -- support exists but the model
                                     merged multiple real items (each
                                     individually verifiable) into one
                                     non-atomic string
  FALSE_WARNING_PARAPHRASE_OR_TYPO -- likely legitimate, but lexical
                                     support is insufficient (a
                                     correction of a source typo, or a
                                     genuine paraphrase)
  BOUNDARY_OR_SOURCE_AMBIGUITY    -- cannot confidently determine
                                     support because of a source
                                     extraction/boundary issue
  UNDETERMINED                    -- evidence insufficient to classify

No LLM calls. Writes:
  benchmark/baselines/v6_5_flag_classification.csv
"""
from __future__ import annotations

import csv

# Each entry: (resume_id, data_source, item_text_prefix_or_exact, category, evidence_note)
# item matched by exact string; long items matched by a stable prefix.
CLASSIFICATIONS = [
    # --- 001: Git/GitHub/VS Code/Postman/Vercel -- all found in a
    # separate, same-level "## Tools" heading immediately after
    # "## Technical Skills" in the real source document (confirmed by
    # direct line-by-line inspection). The V5.9 boundary detector
    # correctly stops the "Technical Skills" body at the next same-
    # level heading -- this is CORRECT boundary behavior for its own
    # purpose, but means a real, topically-adjacent "Tools" section is
    # (correctly, by the detector's own design) excluded from the body
    # this proxy checks against, even though a human would consider
    # "Tools" part of this candidate's skill set.
    ("001", "milestone_23_67_single_baseline", "Git", "BOUNDARY_OR_SOURCE_AMBIGUITY",
     "Found verbatim in a separate, same-level '## Tools' heading immediately following '## Technical Skills' -- correctly excluded by the boundary detector's own (correct) same-level-heading rule, not a hallucination."),
    ("001", "milestone_23_67_single_baseline", "GitHub", "BOUNDARY_OR_SOURCE_AMBIGUITY",
     "Same as 'Git' -- found in the adjacent '## Tools' section."),
    ("001", "milestone_23_67_single_baseline", "VS Code", "BOUNDARY_OR_SOURCE_AMBIGUITY",
     "Same as 'Git' -- found in the adjacent '## Tools' section."),
    ("001", "milestone_23_67_single_baseline", "Postman", "BOUNDARY_OR_SOURCE_AMBIGUITY",
     "Same as 'Git' -- found in the adjacent '## Tools' section."),
    ("001", "milestone_23_67_single_baseline", "Vercel", "BOUNDARY_OR_SOURCE_AMBIGUITY",
     "Same as 'Git' -- found in the adjacent '## Tools' section."),

    # --- 002: "Email" vs source "E-mail" (hyphenated) ---
    ("002", "milestone_23_67_single_baseline", "Email", "FALSE_WARNING_FORMATTING",
     "Source literally reads 'Internet and E-mail' (hyphenated). Proxy normalization lowercases/collapses whitespace but does not strip hyphens, so 'email' (no hyphen) is not a substring of 'e-mail'. Confirmed via direct hyphen-stripped re-check: matches once hyphens are removed."),

    # --- 019: all 6 items found in Summary narrative / Work-Experience
    # subheadings, not the Skills section itself ---
    ("019", "milestone_23_67_single_baseline", "Scenario Modeling", "TRUE_UNSUPPORTED",
     "Not found verbatim anywhere; source has 'Scenario Analysis' (a different, real Skills-section item) -- a plausible but distinct paraphrase, not a literal match. Classified TRUE_UNSUPPORTED relative to the Skills section specifically (the proxy's own defined scope), while noting the item is thematically close to real content."),
    ("019", "milestone_23_67_single_baseline", "Databricks", "TRUE_UNSUPPORTED",
     "Confirmed absent from the Skills section; found instead in the Professional Summary narrative prose ('Proficient in ... Databricks, and Snowflake'). Not part of the Skills-section source this proxy is scoped to check."),
    ("019", "milestone_23_67_single_baseline", "Snowflake", "TRUE_UNSUPPORTED",
     "Same as 'Databricks' -- Summary narrative, not the Skills section."),
    ("019", "milestone_23_67_single_baseline", "Quality Assurance", "TRUE_UNSUPPORTED",
     "Confirmed absent from Skills section; found as a Work-Experience job-description subheading ('#### Data Governance & Quality Assurance:'), not Skills content."),
    ("019", "milestone_23_67_single_baseline", "Executive Reporting", "TRUE_UNSUPPORTED",
     "Confirmed absent from Skills section; found as a Work-Experience subheading ('#### Executive Reporting & Data Storytelling:')."),
    ("019", "milestone_23_67_single_baseline", "Agile Project Management", "TRUE_UNSUPPORTED",
     "Not found verbatim; source Skills section has 'Agile' and 'Project Management' as two SEPARATE items, never merged into this compound phrase -- the model synthesized a new combined phrase."),

    # --- 021: elliptical shared-prefix list expansion (Windows Server
    # 2012, 2016, 2019) and conjunction list (Fortigate 310B and 80E) ---
    ("021", "milestone_23_67_single_baseline", "Windows Server 2016", "FALSE_WARNING_FORMATTING",
     "Source reads 'Windows Server 2012, 2016, 2019' -- an elliptical list where the shared prefix 'Windows Server' applies to all three trailing years. '2016' alone is a literal substring but the full compound 'Windows Server 2016' never appears contiguously. A real, disclosed list-writing convention this proxy's literal matching cannot expand."),
    ("021", "milestone_23_67_single_baseline", "Windows Server 2019", "FALSE_WARNING_FORMATTING",
     "Same elliptical-list mechanism as 'Windows Server 2016'."),
    ("021", "milestone_23_67_single_baseline", "Fortigate 80E", "FALSE_WARNING_FORMATTING",
     "Source reads 'Fortigate 310B and 80E' -- a conjunction-joined list where 'Fortigate' is shared. 'Fortigate' and '80E' both appear, but not contiguously as 'Fortigate 80E'."),
    ("021", "v5_6_checkpoint", "Windows Server 2016", "FALSE_WARNING_FORMATTING", "Same mechanism, V5.6 checkpoint (own correct source lineage, identical real source pattern)."),
    ("021", "v5_6_checkpoint", "Windows Server 2019", "FALSE_WARNING_FORMATTING", "Same mechanism, V5.6 checkpoint."),
    ("021", "v5_6_checkpoint", "Firewall & Security: Fortigate 310B and 80E, Sophos XGS 2100 SonicWall, Seqrite Endpoint Security, Bitdefender Gravity Zone",
     "FALSE_WARNING_LABEL_OR_SEGMENTATION",
     "A label-prefixed, fully-merged copy of the real 'Firewall & Security' source line -- but the model DROPPED one real item ('Symantec Endpoint Antivirus') from the middle while merging, breaking literal containment of the whole remainder. Every individual technology name in this string is independently, directly verifiable in the source (confirmed: splitting on commas, every resulting part matches via word-boundary search) -- a genuine merge-and-partial-omission segmentation artifact, not hallucination."),

    # --- 022: Work-Experience responsibility bullets pulled into Skills ---
    ("022", "milestone_23_67_single_baseline", "Define end-to-end test strategy", "TRUE_UNSUPPORTED",
     "Verbatim opening clause of a real Work-Experience bullet ('Define end-to-end test strategy covering web, API, and database layers.') -- Skills-section content-substitution from Work Experience, structurally similar in kind (though smaller in degree) to the confirmed V3.4/V3.5/V5.6 mechanism for resume 005."),
    ("022", "milestone_23_67_single_baseline", "Drive automation framework evolution", "TRUE_UNSUPPORTED", "Same mechanism -- verbatim opening clause of a Work-Experience responsibility bullet."),
    ("022", "milestone_23_67_single_baseline", "Support shift-left testing practices", "TRUE_UNSUPPORTED", "Same mechanism -- verbatim opening clause of a Work-Experience responsibility bullet."),
    ("022", "milestone_23_67_single_baseline", "Own test planning, execution governance, defect management, and reporting", "TRUE_UNSUPPORTED", "Same mechanism -- verbatim Work-Experience responsibility bullet."),
    ("022", "milestone_23_67_single_baseline", "Collaborate with product owners, developers, and stakeholders", "TRUE_UNSUPPORTED", "Same mechanism -- verbatim Work-Experience responsibility bullet."),
    ("022", "milestone_23_67_single_baseline", "Conduct RQA and implement preventive quality controls", "TRUE_UNSUPPORTED", "Same mechanism -- verbatim Work-Experience responsibility bullet."),
    ("022", "milestone_23_67_single_baseline", "Hands-on in exploratory testing and automation scripting", "TRUE_UNSUPPORTED", "Same mechanism -- verbatim Work-Experience responsibility bullet."),

    # --- 018: already characterized directly in V6.4, reproduced here ---
    ("018", "v6_1_checkpoint", "Power BI", "TRUE_UNSUPPORTED",
     "Confirmed absent from both real GT and the real Skills-section source text (V6.4's own direct finding, reproduced here) -- a genuine, real hallucination."),
    ("018", "v6_1_checkpoint", "Advance Excel", "FALSE_WARNING_PARAPHRASE_OR_TYPO",
     "Source reads 'Advanced Excel'; candidate wrote 'Advance Excel' -- a one-character model-side typo/variant of real, present content (V6.4's own direct finding, reproduced here)."),

    # --- 005: BOUNDARY-OVERRUN CAVEAT (see milestone report) applies to
    # every item below -- the Skills-section boundary for this specific
    # resume/source overruns past the real Skills content into Awards/
    # Exam-Qualified/Reference/Personal-Details sections (none of which
    # use '#'-prefixed markdown headings, so V5.9's boundary detector
    # never finds a terminating heading). This does not change the
    # TRUE_UNSUPPORTED classification for the certification items below
    # (independently, strongly confirmed since V3.4/V3.5 by direct
    # human inspection, not solely by this proxy) but means confidence
    # in this resume's specific unsupported/supported item counts is
    # LOWER than for resumes with a cleanly-terminated boundary.
    ("005", "v5_6_checkpoint", "IAC-Terraform", "FALSE_WARNING_PARAPHRASE_OR_TYPO",
     "Source itself contains a real spelling error, 'IAC-Terrafrom' (confirmed, V5.7's own prior finding). The model wrote the CORRECTED spelling 'IAC-Terraform' -- semantically the same real skill, but not a literal match against the source's own typo. [Boundary-overrun caveat applies -- see above.]"),
    ("005", "v5_6_checkpoint", "Microsoft Certified: Azure Solution Architect Expert", "TRUE_UNSUPPORTED",
     "Certification content, not Skills -- confirmed content-substitution mechanism, independently established since V3.4/V3.5 by direct human inspection, reconfirmed V5.6/V6.4. [Boundary-overrun caveat applies -- see above; does not weaken this independently-established classification.]"),
    ("005", "v5_6_checkpoint", "Oracle Certified: Oracle Cloud Infrastructure", "TRUE_UNSUPPORTED",
     "Certification content -- same confirmed content-substitution mechanism as the other 'X Certified:' items. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Microsoft Certified: Microsoft DevOps Engineer Expert", "TRUE_UNSUPPORTED",
     "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Microsoft Certified: Azure Data Engineer Associate", "TRUE_UNSUPPORTED",
     "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Microsoft Certified: Azure AI Engineer Associate", "TRUE_UNSUPPORTED",
     "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Microsoft Certified: Azure Cosmos DB Developer Specialty", "TRUE_UNSUPPORTED",
     "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Microsoft Certified: Azure Administrator Associate", "TRUE_UNSUPPORTED",
     "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Microsoft Certified: Designing Microsoft Azure Infrastructure", "TRUE_UNSUPPORTED",
     "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Microsoft Certified: PowerBi Data Analyst Associate", "TRUE_UNSUPPORTED",
     "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Azure Data Engg., Data Bricks, Snowflake & sol. architecture expertise", "TRUE_UNSUPPORTED",
     "A synthesized descriptive summary phrase, not a literal quotation of any source list -- does not match the source's own comma-separated technology-list structure. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Python, VSC, SQL, Elastic search, Grafana, Snowflake, MuleSoft, Logstash, Apex AiOps, AWS, Databricks, DataFactory, Linux, Azure Devops, Cribl, Ansible, GitHub, Qracle DB, PySpark, BI, Postman, Kubernetes, Terraform",
     "FALSE_WARNING_LABEL_OR_SEGMENTATION",
     "A merged, non-atomic string combining many real source technology names (most independently verifiable in the source's own 'Software Dev.'/'SharePoint Infra.' lines) into one giant comma-joined item -- checked as a single whole string, so it does not match verbatim even though most of its constituent parts are real. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "R, PowerBI, Tableau, MySQL, Python, SQL, Azure-Data Bricks, ADF, ASA, Hive, ELK, Snowflake, PySpark, Access, Moogsoft, Grafana, Kubernetes",
     "FALSE_WARNING_LABEL_OR_SEGMENTATION",
     "Same mechanism as the previous row -- a merged multi-item string, most constituent technology names independently real. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "AISSE (CBSE) 2014, 10th (CBSE) 2012, AISSCE (CBSE) 2013", "TRUE_UNSUPPORTED",
     "Education/exam-board content (10th/12th-grade board exam names), not Skills content at all -- genuine content-substitution. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Azure Solution Architect Expert", "TRUE_UNSUPPORTED", "Certification content (run4, without the 'Microsoft Certified:' prefix this time) -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Oracle Cloud Infrastructure 2025 Certified DevOps Professional", "TRUE_UNSUPPORTED", "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Microsoft DevOps Engineer Expert", "TRUE_UNSUPPORTED", "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Azure Data Engineer Associate", "TRUE_UNSUPPORTED", "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Azure AI Engineer Associate", "TRUE_UNSUPPORTED", "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Azure Cosmos DB Developer Specialty", "TRUE_UNSUPPORTED", "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Azure Administrator Associate", "TRUE_UNSUPPORTED", "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "DataBricks Certified: Accredited Generative AI Fundamentals", "TRUE_UNSUPPORTED", "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Designing Microsoft Azure Infrastructure Solutions", "TRUE_UNSUPPORTED", "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "PowerBi Data Analyst Associate", "TRUE_UNSUPPORTED", "Certification content -- same confirmed mechanism. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "VSC", "UNDETERMINED",
     "Not found verbatim or as a clear formatting variant of any single identified source token; insufficient evidence to confidently classify as a genuine abbreviation-of-something-real vs. a fabricated token. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Logstash", "TRUE_UNSUPPORTED", "Not found in the Skills-section source body under any spacing/case variant checked; classified unsupported. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "DataFactory", "FALSE_WARNING_FORMATTING", "Source has 'Data Factory' (with space); model wrote 'DataFactory' (no space) -- confirmed via space-stripped re-check. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Linux", "TRUE_UNSUPPORTED", "Not found in the Skills-section source body; classified unsupported. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Azure Devops", "FALSE_WARNING_FORMATTING", "Source has 'AzureDevops' (no space); model wrote 'Azure Devops' (with space) -- confirmed via space-stripped re-check. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Qracle DB", "FALSE_WARNING_PARAPHRASE_OR_TYPO", "Almost certainly a model-side typo of 'Oracle DB' (source has 'MySQL, Oracle, Elastic Search...') -- 'Qracle' for 'Oracle' is a one-character substitution error. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Postman", "TRUE_UNSUPPORTED", "Not found in the Skills-section source body; classified unsupported. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Terraform", "FALSE_WARNING_PARAPHRASE_OR_TYPO", "Same mechanism as 'IAC-Terraform' above -- source has 'IAC-Terrafrom' (typo); model's corrected spelling doesn't literally match. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "PowerBI", "FALSE_WARNING_FORMATTING", "Source has 'Power BI' (with space, in the Visualization Tools line); model wrote 'PowerBI' (no space). [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Hive", "TRUE_UNSUPPORTED", "Not found in the Skills-section source body under any variant checked. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "Azure-Data Bricks", "FALSE_WARNING_FORMATTING", "Source has 'AZURE-DataBricks' -- a hyphenation/spacing variant of the same real token. [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "ADF", "TRUE_UNSUPPORTED", "Not found in the Skills-section source body; classified unsupported (no strong evidence of a specific formatting variant). [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "ASA", "TRUE_UNSUPPORTED", "Not found in the Skills-section source body; classified unsupported (no strong evidence of a specific formatting variant). [Boundary-overrun caveat applies.]"),
    ("005", "v5_6_checkpoint", "PowerApps", "FALSE_WARNING_FORMATTING", "Source has 'Power Apps' (with space, in the Software Dev. line); model wrote 'PowerApps' (no space). [Boundary-overrun caveat applies.]"),
]


def main() -> None:
    with open("benchmark/baselines/v6_5_full_corpus_item_results.csv", newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    flagged = [r for r in all_rows if r["grounded"] == "False"]

    out_rows = []
    unmatched = []
    for r in flagged:
        match = None
        for rid, ds, item_key, category, note in CLASSIFICATIONS:
            if r["resume_id"] == rid and r["data_source"] == ds and r["item"].startswith(item_key):
                match = (category, note)
                break
        if match is None:
            unmatched.append(r)
            category, note = "UNDETERMINED", "No classification rule matched -- flagged for manual follow-up."
        else:
            category, note = match
        out_rows.append({
            "resume_id": r["resume_id"], "data_source": r["data_source"], "condition": r["condition"],
            "run_index": r["run_index"], "item": r["item"], "category": category, "evidence_note": note,
        })

    with open("benchmark/baselines/v6_5_flag_classification.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote v6_5_flag_classification.csv ({len(out_rows)} rows)")

    if unmatched:
        print(f"\nWARNING: {len(unmatched)} flagged items had no classification rule:")
        for r in unmatched:
            print(f"  {r['resume_id']} | {r['data_source']} | {r['item'][:60]!r}")

    from collections import Counter
    print("\n=== Category distribution ===")
    for cat, count in Counter(r["category"] for r in out_rows).most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
