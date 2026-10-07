"""
Evaluator Benchmark Dataset Generator and Verifier.

Generates and validates the LongTracer v0.3.0 benchmark dataset covering 14 categories,
stratified into calibration (~40%) and held-out (~60%) splits.

Ensures:
- Strict schema adherence matching longtracer.contracts types.
- Leakage detection: verifies no near-duplicate pairs (token Jaccard >= 0.8) across splits.
- Deterministic IDs and metadata pinning.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

DATASET_VERSION = "0.3.0"
DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"


def _jaccard_similarity(s1: str, s2: str) -> float:
    """Calculate token-level Jaccard similarity between two texts."""
    tokens1 = set(re.findall(r"\w+", s1.lower()))
    tokens2 = set(re.findall(r"\w+", s2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _source_text(case: Dict[str, Any]) -> str:
    return " ".join(s["text"] for s in case.get("sources", []))


def check_dataset_leakage(cases: List[Dict[str, Any]], threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """Check for near-duplicate leakage between calibration and held-out splits.

    A pair leaks if the normalised responses are identical, or the token Jaccard
    similarity of either the responses or the concatenated sources is >= threshold.
    Empty responses are compared on sources only (every empty answer is "identical").
    """
    calib = [c for c in cases if c.get("split") == "calibration"]
    heldout = [c for c in cases if c.get("split") == "heldout"]
    leaks = []
    for c1 in calib:
        for c2 in heldout:
            r1, r2 = _normalise(c1["response"]), _normalise(c2["response"])
            sims = [_jaccard_similarity(_source_text(c1), _source_text(c2))]
            if r1 and r2:
                sims.append(1.0 if r1 == r2 else _jaccard_similarity(r1, r2))
            sim = max(sims)
            if sim >= threshold:
                leaks.append((c1["case_id"], c2["case_id"], sim))
    return leaks


def is_reviewed(case: Dict[str, Any]) -> bool:
    """A case counts as human-reviewed only with status 'reviewed' and two named reviewers."""
    review = case.get("review", {})
    return review.get("status") == "reviewed" and len(review.get("reviewers", [])) >= 2


def validate_cases(cases: List[Dict[str, Any]]) -> List[str]:
    """Validate every case against the longtracer.contracts types. Returns error strings."""
    from longtracer.contracts.evidence import SourceEvidence
    from longtracer.contracts.result import AssessmentAvailability, ClaimAssessment, ExecutionStatus, QualityGate

    errors: List[str] = []
    seen = set()
    for c in cases:
        cid = c.get("case_id", "?")
        if cid in seen:
            errors.append(f"{cid}: duplicate case_id")
        seen.add(cid)
        if c.get("split") not in ("calibration", "heldout"):
            errors.append(f"{cid}: bad split {c.get('split')!r}")
        try:
            for s in c.get("sources", []):
                SourceEvidence(**s)
            exp = c["expected"]
            ExecutionStatus(exp["execution"])
            AssessmentAvailability(exp["availability"])
            QualityGate(exp["quality_gate"])
            for claim in exp.get("claims", []):
                ClaimAssessment(claim["assessment"])
        except Exception as e:  # report every problem, don't stop at the first
            errors.append(f"{cid}: {type(e).__name__}: {e}")
    return errors


def build_raw_cases() -> List[Dict[str, Any]]:
    """Construct the curated benchmark test cases across all 14 categories."""
    cases = []

    # Helper to construct case
    def add_case(
        cat: str,
        idx: int,
        split: str,
        response: str,
        sources: List[str],
        gate: str,
        claim_assessments: List[Tuple[str, str]],
        notes: str,
        availability: str = "ASSESSED",
    ):
        case_id = f"case_{cat}_{idx:03d}"
        source_objs = [
            {
                "source_id": f"src_{i + 1}",
                "text": s,
                "metadata": {"title": f"Document {i + 1}"},
            }
            for i, s in enumerate(sources)
        ]
        expected_claims = [{"claim_text": c_text, "assessment": c_ass} for c_text, c_ass in claim_assessments]
        cases.append(
            {
                "case_id": case_id,
                "split": split,
                "category": cat,
                "response": response,
                "sources": source_objs,
                "expected": {
                    "execution": "SUCCESS",
                    "availability": availability,
                    "quality_gate": gate,
                    "claims": expected_claims,
                },
                "review": {
                    # Drafted, not yet human-reviewed. A case only counts as reviewed
                    # once two named human reviewers have signed off (see README.md).
                    "reviewers": [],
                    "status": "unreviewed",
                    "notes": notes,
                },
            }
        )

    # 1. Support
    add_case(
        "support",
        1,
        "calibration",
        "The Eiffel Tower is located in Paris and was completed in 1889.",
        ["The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, completed in 1889."],
        "PASS",
        [("The Eiffel Tower is located in Paris and was completed in 1889.", "SUPPORTED")],
        "Ground truth historical fact supported directly.",
    )
    add_case(
        "support",
        2,
        "heldout",
        "Water freezes at 0 degrees Celsius under standard atmospheric pressure.",
        ["Under standard atmospheric pressure, the freezing point of pure water is 0 degrees Celsius."],
        "PASS",
        [("Water freezes at 0 degrees Celsius under standard atmospheric pressure.", "SUPPORTED")],
        "Physical science fact directly grounded.",
    )
    add_case(
        "support",
        3,
        "calibration",
        "Python was designed by Guido van Rossum and released in 1991.",
        ["Guido van Rossum began implementing Python in December 1989 and released version 0.9.0 in February 1991."],
        "PASS",
        [("Python was designed by Guido van Rossum and released in 1991.", "SUPPORTED")],
        "Software history fact supported.",
    )
    add_case(
        "support",
        4,
        "heldout",
        "Mount Everest is the highest mountain above sea level on Earth.",
        [
            "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas."
        ],
        "PASS",
        [("Mount Everest is the highest mountain above sea level on Earth.", "SUPPORTED")],
        "Geographical fact supported.",
    )
    add_case(
        "support",
        5,
        "heldout",
        "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
        [
            "The Pacific Ocean is the largest and deepest of the world ocean basins, covering over 60 million square miles."
        ],
        "PASS",
        [("The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.", "SUPPORTED")],
        "Earth science fact supported.",
    )
    add_case(
        "support",
        6,
        "heldout",
        "DNA contains the genetic instructions used in the development and functioning of all known living organisms.",
        [
            "Deoxyribonucleic acid is a polymer composed of two polynucleotide chains that coil around each other to carry genetic instructions."
        ],
        "PASS",
        [
            (
                "DNA contains the genetic instructions used in the development and functioning of all known living organisms.",
                "SUPPORTED",
            )
        ],
        "Biology fact supported.",
    )
    add_case(
        "support",
        7,
        "calibration",
        "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
        [
            "The speed of light in vacuum, commonly denoted c, is universally defined as exactly 299,792,458 metres per second."
        ],
        "PASS",
        [("The speed of light in a vacuum is approximately 299,792 kilometers per second.", "SUPPORTED")],
        "Physics constant grounded.",
    )
    add_case(
        "support",
        8,
        "heldout",
        "The Amazon River discharges more water than any other river system in the world.",
        [
            "The Amazon River by far has the greatest total river discharge in the world, representing roughly a fifth of global river flow."
        ],
        "PASS",
        [("The Amazon River discharges more water than any other river system in the world.", "SUPPORTED")],
        "Geography fact supported.",
    )
    add_case(
        "support",
        9,
        "heldout",
        "Mercury is the smallest planet in the Solar System and closest to the Sun.",
        ["Mercury is the smallest planet in the Solar System and the one closest to the Sun."],
        "PASS",
        [("Mercury is the smallest planet in the Solar System and closest to the Sun.", "SUPPORTED")],
        "Astronomy fact supported.",
    )
    add_case(
        "support",
        10,
        "heldout",
        "Photosynthesis enables plants to convert sunlight, carbon dioxide, and water into glucose and oxygen.",
        [
            "Through photosynthesis, green plants synthesize glucose from carbon dioxide and water, releasing oxygen as a byproduct."
        ],
        "PASS",
        [
            (
                "Photosynthesis enables plants to convert sunlight, carbon dioxide, and water into glucose and oxygen.",
                "SUPPORTED",
            )
        ],
        "Biochemistry fact supported.",
    )

    # 2. Contradiction
    add_case(
        "contradiction",
        1,
        "calibration",
        "The Eiffel Tower is located in downtown Berlin, Germany.",
        ["The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France."],
        "FAIL",
        [("The Eiffel Tower is located in downtown Berlin, Germany.", "CONTRADICTED")],
        "Explicit geographical contradiction.",
    )
    add_case(
        "contradiction",
        2,
        "heldout",
        "Water boils at 20 degrees Celsius under standard atmospheric pressure.",
        ["Under standard atmospheric pressure at sea level, pure water boils at 100 degrees Celsius."],
        "FAIL",
        [("Water boils at 20 degrees Celsius under standard atmospheric pressure.", "CONTRADICTED")],
        "Physical property contradiction.",
    )
    add_case(
        "contradiction",
        3,
        "calibration",
        "Guido van Rossum created the Java programming language at Sun Microsystems.",
        ["Guido van Rossum created Python. James Gosling created Java at Sun Microsystems."],
        "FAIL",
        [("Guido van Rossum created the Java programming language at Sun Microsystems.", "CONTRADICTED")],
        "Creator contradiction.",
    )
    add_case(
        "contradiction",
        4,
        "heldout",
        "Mount Everest is located entirely within the borders of South America.",
        ["Mount Everest is situated in the Himalayas on the border between Nepal and China in Asia."],
        "FAIL",
        [("Mount Everest is located entirely within the borders of South America.", "CONTRADICTED")],
        "Continent contradiction.",
    )
    add_case(
        "contradiction",
        5,
        "heldout",
        "The Pacific Ocean is the smallest and shallowest of Earth's oceans.",
        ["The Pacific Ocean is the largest and deepest ocean basin on Earth."],
        "FAIL",
        [("The Pacific Ocean is the smallest and shallowest of Earth's oceans.", "CONTRADICTED")],
        "Direct opposite property contradiction.",
    )
    add_case(
        "contradiction",
        6,
        "heldout",
        "Mercury is the planet located furthest from the Sun in the solar system.",
        ["Mercury is the innermost planet, orbiting closest to the Sun."],
        "FAIL",
        [("Mercury is the planet located furthest from the Sun in the solar system.", "CONTRADICTED")],
        "Orbital position contradiction.",
    )
    add_case(
        "contradiction",
        7,
        "calibration",
        "Humans have three sets of natural lungs located in the thoracic cavity.",
        ["Human anatomy features two lungs situated in the chest on either side of the heart."],
        "FAIL",
        [("Humans have three sets of natural lungs located in the thoracic cavity.", "CONTRADICTED")],
        "Anatomical quantity contradiction.",
    )
    add_case(
        "contradiction",
        8,
        "heldout",
        "The Apollo 11 mission was an unmanned satellite launch that never carried astronauts.",
        [
            "Apollo 11 was the American spaceflight that first landed humans, Neil Armstrong and Buzz Aldrin, on the Moon."
        ],
        "FAIL",
        [("The Apollo 11 mission was an unmanned satellite launch that never carried astronauts.", "CONTRADICTED")],
        "Historical mission contradiction.",
    )
    add_case(
        "contradiction",
        9,
        "heldout",
        "The Sahara Desert is located primarily in North America.",
        ["The Sahara is a vast desert located on the African continent."],
        "FAIL",
        [("The Sahara Desert is located primarily in North America.", "CONTRADICTED")],
        "Location contradiction.",
    )
    add_case(
        "contradiction",
        10,
        "heldout",
        "Sound travels faster in a complete vacuum than through solid steel.",
        ["Sound requires a material medium to propagate and cannot travel through a vacuum."],
        "FAIL",
        [("Sound travels faster in a complete vacuum than through solid steel.", "CONTRADICTED")],
        "Acoustic physics contradiction.",
    )

    # 3. Insufficient Evidence
    add_case(
        "insufficient_evidence",
        1,
        "calibration",
        "Gustave Eiffel personally preferred strawberry ice cream over chocolate.",
        ["The Eiffel Tower was designed by Gustave Eiffel's engineering company in Paris."],
        "FAIL",
        [("Gustave Eiffel personally preferred strawberry ice cream over chocolate.", "INSUFFICIENT_EVIDENCE")],
        "Unverifiable biographical detail absent from context.",
    )
    add_case(
        "insufficient_evidence",
        2,
        "heldout",
        "The secret recipe for the elixir of immortality was discovered in 1842.",
        ["Standard drinking water contains trace minerals like calcium and magnesium."],
        "FAIL",
        [("The secret recipe for the elixir of immortality was discovered in 1842.", "INSUFFICIENT_EVIDENCE")],
        "Completely ungrounded statement with unrelated source.",
    )
    add_case(
        "insufficient_evidence",
        3,
        "calibration",
        "The global market capitalization of widgets will exceed ten trillion dollars next year.",
        ["Manufacturing widgets requires precision tooling and specialized steel alloys."],
        "FAIL",
        [
            (
                "The global market capitalization of widgets will exceed ten trillion dollars next year.",
                "INSUFFICIENT_EVIDENCE",
            )
        ],
        "Forward-looking financial projection not in text.",
    )
    add_case(
        "insufficient_evidence",
        4,
        "heldout",
        "The ancient kingdom possessed advanced quantum computing devices in 500 BC.",
        ["Ancient architectural ruins in the valley date back to the fifth century BC."],
        "FAIL",
        [("The ancient kingdom possessed advanced quantum computing devices in 500 BC.", "INSUFFICIENT_EVIDENCE")],
        "Anachronistic claim absent from archaeological context.",
    )
    add_case(
        "insufficient_evidence",
        5,
        "heldout",
        "Certain species of deep sea cephalopods communicate exclusively through psychic telepathy.",
        ["Deep sea squids use bioluminescent photophores and chromatophores to display visual patterns."],
        "FAIL",
        [
            (
                "Certain species of deep sea cephalopods communicate exclusively through psychic telepathy.",
                "INSUFFICIENT_EVIDENCE",
            )
        ],
        "Ungrounded pseudoscientific assertion.",
    )
    add_case(
        "insufficient_evidence",
        6,
        "heldout",
        "The conference keynote speaker was wearing custom purple suede sneakers on stage.",
        ["The technology summit commenced with opening remarks on modern artificial intelligence safety."],
        "FAIL",
        [
            (
                "The conference keynote speaker was wearing custom purple suede sneakers on stage.",
                "INSUFFICIENT_EVIDENCE",
            )
        ],
        "Irrelevant sartorial detail absent from summary.",
    )
    add_case(
        "insufficient_evidence",
        7,
        "calibration",
        "The company founder celebrated their thirtieth birthday in Honolulu, Hawaii.",
        ["The corporation was founded in 2010 to develop open source observability software."],
        "FAIL",
        [("The company founder celebrated their thirtieth birthday in Honolulu, Hawaii.", "INSUFFICIENT_EVIDENCE")],
        "Personal biographical detail missing from business text.",
    )
    add_case(
        "insufficient_evidence",
        8,
        "heldout",
        "Domestic feline sleep cycles are directly influenced by Jupiter's gravitational field.",
        ["Domestic cats typically spend twelve to sixteen hours per day sleeping or resting."],
        "FAIL",
        [
            (
                "Domestic feline sleep cycles are directly influenced by Jupiter's gravitational field.",
                "INSUFFICIENT_EVIDENCE",
            )
        ],
        "Ungrounded astrological claim.",
    )
    add_case(
        "insufficient_evidence",
        9,
        "heldout",
        "The original blueprint was drafted using green ink imported from Belgium.",
        ["The architectural blueprints were completed in March and submitted for building permits."],
        "FAIL",
        [("The original blueprint was drafted using green ink imported from Belgium.", "INSUFFICIENT_EVIDENCE")],
        "Unverifiable drafting detail.",
    )
    add_case(
        "insufficient_evidence",
        10,
        "heldout",
        "Ancient inhabitants of the island preferred listening to acoustic guitar melodies.",
        ["Archaeologists discovered stone pottery and bone fishhooks at the excavation site."],
        "FAIL",
        [
            (
                "Ancient inhabitants of the island preferred listening to acoustic guitar melodies.",
                "INSUFFICIENT_EVIDENCE",
            )
        ],
        "Cultural anachronism unsupported by evidence.",
    )

    # 4. Conflicting Sources
    add_case(
        "conflicting_sources",
        1,
        "calibration",
        "The historic fortress was constructed in the year 1889.",
        [
            "Archival records confirm the fortress was built and dedicated in 1889.",
            "Recent municipal surveys prove the fortress was built in 1925, not 1889.",
        ],
        "FAIL",
        [("The historic fortress was constructed in the year 1889.", "CONFLICTING_SOURCES")],
        "Document 1 entails while Document 2 explicitly refutes the construction year.",
    )
    add_case(
        "conflicting_sources",
        2,
        "heldout",
        "The company reported ten million dollars in annual operating profit.",
        [
            "The press release states the firm earned ten million dollars in net profit this fiscal year.",
            "The audited regulatory filing clarifies the firm suffered a five million dollar loss, contradicting claims of ten million in profit.",
        ],
        "FAIL",
        [("The company reported ten million dollars in annual operating profit.", "CONFLICTING_SOURCES")],
        "Source financial disagreement.",
    )
    add_case(
        "conflicting_sources",
        3,
        "calibration",
        "The expedition team successfully reached the northern summit on Tuesday.",
        [
            "Base camp logs confirm the mountaineers reached the northern summit on Tuesday morning.",
            "Radio transmissions verify the team had to abort before the summit due to severe blizzard conditions.",
        ],
        "FAIL",
        [("The expedition team successfully reached the northern summit on Tuesday.", "CONFLICTING_SOURCES")],
        "Mission outcome conflicting sources.",
    )
    add_case(
        "conflicting_sources",
        4,
        "heldout",
        "The newly opened transit bridge was designed by Chief Engineer Martinez.",
        [
            "The city municipal gazette names Martinez as the principal design architect of the bridge.",
            "The state infrastructure commission records document that Engineer Jenkins designed the bridge, not Martinez.",
        ],
        "FAIL",
        [("The newly opened transit bridge was designed by Chief Engineer Martinez.", "CONFLICTING_SOURCES")],
        "Attribution conflict across independent archives.",
    )
    add_case(
        "conflicting_sources",
        5,
        "heldout",
        "Clinical trial results showed the drug reduced symptom duration by four days.",
        [
            "Phase III trial summaries report a significant four-day reduction in illness duration.",
            "The independent peer review panel concluded the drug showed zero statistical difference in duration.",
        ],
        "FAIL",
        [("Clinical trial results showed the drug reduced symptom duration by four days.", "CONFLICTING_SOURCES")],
        "Medical study conflicting conclusions.",
    )
    add_case(
        "conflicting_sources",
        6,
        "heldout",
        "The ancient manuscript contains exactly forty-two illustrated botanical plates.",
        [
            "The library catalog notes that the manuscript contains forty-two botanical plates.",
            "The restoration team's physical audit discovered only thirty-eight plates exist in the codex.",
        ],
        "FAIL",
        [("The ancient manuscript contains exactly forty-two illustrated botanical plates.", "CONFLICTING_SOURCES")],
        "Disputed archival enumeration.",
    )
    add_case(
        "conflicting_sources",
        7,
        "calibration",
        "The island population grew by fifteen percent over the last decade.",
        [
            "Census bureau estimates show an approximate fifteen percent increase in resident population.",
            "Demographic research from the local university demonstrated the population was completely flat with zero net growth.",
        ],
        "FAIL",
        [("The island population grew by fifteen percent over the last decade.", "CONFLICTING_SOURCES")],
        "Demographic study disagreement.",
    )
    add_case(
        "conflicting_sources",
        8,
        "heldout",
        "The spacecraft achieved an orbital speed exceeding twenty-eight thousand kilometers per hour.",
        [
            "Telemetry streams showed the vehicle maintained an orbital speed of twenty-eight thousand kilometers per hour.",
            "Subsequent flight analysis confirmed the vehicle failed to reach orbit, maxing out at sixteen thousand kilometers per hour.",
        ],
        "FAIL",
        [
            (
                "The spacecraft achieved an orbital speed exceeding twenty-eight thousand kilometers per hour.",
                "CONFLICTING_SOURCES",
            )
        ],
        "Telemetry vs post-flight investigation conflict.",
    )
    add_case(
        "conflicting_sources",
        9,
        "heldout",
        "The treaty was officially ratified in Geneva on October twelfth.",
        [
            "Diplomatic cables state the delegates ratified the treaty in Geneva on October twelfth.",
            "National legislative records show the ratification occurred in Bern two months later, not in Geneva.",
        ],
        "FAIL",
        [("The treaty was officially ratified in Geneva on October twelfth.", "CONFLICTING_SOURCES")],
        "Treaty ratification location conflict.",
    )
    add_case(
        "conflicting_sources",
        10,
        "heldout",
        "The solar array provides eighty percent of the facility's electric power.",
        [
            "The environmental impact statement cites solar power providing eighty percent of baseline energy.",
            "Utility billing statements show municipal grid power accounts for sixty percent, with solar providing under forty percent.",
        ],
        "FAIL",
        [("The solar array provides eighty percent of the facility's electric power.", "CONFLICTING_SOURCES")],
        "Renewable energy generation claim conflict.",
    )

    # 5. Numeric Change
    add_case(
        "numeric_change",
        1,
        "calibration",
        "The software company achieved ninety-five percent customer satisfaction.",
        ["In the latest annual customer survey, seventy-two percent of respondents reported being satisfied."],
        "FAIL",
        [("The software company achieved ninety-five percent customer satisfaction.", "CONTRADICTED")],
        "Numeric percentage altered from 72% to 95%.",
    )
    add_case(
        "numeric_change",
        2,
        "heldout",
        "The project completed forty-eight hours ahead of schedule.",
        ["Due to logistics bottlenecks, the construction project was delivered twelve hours behind schedule."],
        "FAIL",
        [("The project completed forty-eight hours ahead of schedule.", "CONTRADICTED")],
        "Numeric timeframe hallucination.",
    )
    add_case(
        "numeric_change",
        3,
        "calibration",
        "The server rack contains fifty-four individual blade computing nodes.",
        ["Each standard datacenter cabinet houses forty-two modular blade servers."],
        "FAIL",
        [("The server rack contains fifty-four individual blade computing nodes.", "CONTRADICTED")],
        "Count altered from 42 to 54.",
    )
    add_case(
        "numeric_change",
        4,
        "heldout",
        "The aircraft has a maximum certified passenger capacity of eight hundred people.",
        ["The twin-engine commercial airliner is configured to seat up to three hundred and eighty passengers."],
        "FAIL",
        [("The aircraft has a maximum certified passenger capacity of eight hundred people.", "CONTRADICTED")],
        "Passenger capacity numeric exaggeration.",
    )
    add_case(
        "numeric_change",
        5,
        "heldout",
        "The database cluster maintains three hundred replicas across global availability zones.",
        ["The primary production database cluster is configured with three read replicas."],
        "FAIL",
        [("The database cluster maintains three hundred replicas across global availability zones.", "CONTRADICTED")],
        "Replica count inflated by two orders of magnitude.",
    )
    add_case(
        "numeric_change",
        6,
        "heldout",
        "The company employs approximately twelve thousand full-time software engineers.",
        ["The startup has grown its engineering division to forty-five full-time developers."],
        "FAIL",
        [("The company employs approximately twelve thousand full-time software engineers.", "CONTRADICTED")],
        "Staff headcount exaggeration.",
    )
    add_case(
        "numeric_change",
        7,
        "calibration",
        "The battery pack provides fifteen hours of continuous high-definition video playback.",
        ["Testing revealed the integrated lithium battery lasts for six hours of media streaming."],
        "FAIL",
        [("The battery pack provides fifteen hours of continuous high-definition video playback.", "CONTRADICTED")],
        "Battery runtime numeric modification.",
    )
    add_case(
        "numeric_change",
        8,
        "heldout",
        "The mountain pass reaches a peak elevation of six thousand meters above sea level.",
        ["The alpine highway summit attains an elevation of two thousand four hundred meters."],
        "FAIL",
        [("The mountain pass reaches a peak elevation of six thousand meters above sea level.", "CONTRADICTED")],
        "Elevation numeric change.",
    )
    add_case(
        "numeric_change",
        9,
        "heldout",
        "The pipeline throughput is rated at five hundred megabytes per second.",
        ["Benchmarking established peak streaming throughput of fifty megabytes per second."],
        "FAIL",
        [("The pipeline throughput is rated at five hundred megabytes per second.", "CONTRADICTED")],
        "Throughput numeric scale error.",
    )
    add_case(
        "numeric_change",
        10,
        "heldout",
        "The city subway network spans four hundred operational passenger stations.",
        ["The metropolitan transit authority operates eighty-eight underground stations."],
        "FAIL",
        [("The city subway network spans four hundred operational passenger stations.", "CONTRADICTED")],
        "Station count numeric alteration.",
    )

    # 6. Unit Change
    add_case(
        "unit_change",
        1,
        "calibration",
        "The transmission distance between relay towers is eighty miles.",
        ["The maximum line-of-sight distance between relay towers was measured at eighty kilometers."],
        "FAIL",
        [("The transmission distance between relay towers is eighty miles.", "CONTRADICTED")],
        "Unit swapped from kilometers to miles.",
    )
    add_case(
        "unit_change",
        2,
        "heldout",
        "The dry cargo container weighs approximately two thousand pounds when empty.",
        ["The empty tare weight of the standard shipping container is two thousand kilograms."],
        "FAIL",
        [("The dry cargo container weighs approximately two thousand pounds when empty.", "CONTRADICTED")],
        "Mass unit swapped from kilograms to pounds.",
    )
    add_case(
        "unit_change",
        3,
        "calibration",
        "The reactor coolant temperature rose by fifty degrees Fahrenheit during operation.",
        ["Telemetry recordings showed the primary coolant temperature increased by fifty degrees Celsius."],
        "FAIL",
        [("The reactor coolant temperature rose by fifty degrees Fahrenheit during operation.", "CONTRADICTED")],
        "Temperature unit Celsius changed to Fahrenheit.",
    )
    add_case(
        "unit_change",
        4,
        "heldout",
        "The chemical storage tank holds twenty thousand gallons of industrial solvent.",
        ["The cylindrical containment vessel is certified to hold twenty thousand liters."],
        "FAIL",
        [("The chemical storage tank holds twenty thousand gallons of industrial solvent.", "CONTRADICTED")],
        "Volume unit liters changed to gallons.",
    )
    add_case(
        "unit_change",
        5,
        "heldout",
        "The microchip fabrication gate length measures five micrometers in thickness.",
        ["The semiconductor manufacturing process utilizes a five nanometer lithography standard."],
        "FAIL",
        [("The microchip fabrication gate length measures five micrometers in thickness.", "CONTRADICTED")],
        "Length scale unit changed from nanometers to micrometers.",
    )
    add_case(
        "unit_change",
        6,
        "heldout",
        "The engine generates four hundred kilowatts of rotational power at peak throttle.",
        ["Dynamometer measurements recorded an engine output of four hundred horsepower."],
        "FAIL",
        [("The engine generates four hundred kilowatts of rotational power at peak throttle.", "CONTRADICTED")],
        "Power unit horsepower swapped to kilowatts.",
    )
    add_case(
        "unit_change",
        7,
        "calibration",
        "The satellite completes one full terrestrial orbit every ninety seconds.",
        ["The orbital mechanics determine a low Earth orbit period of ninety minutes."],
        "FAIL",
        [("The satellite completes one full terrestrial orbit every ninety seconds.", "CONTRADICTED")],
        "Time unit minutes swapped to seconds.",
    )
    add_case(
        "unit_change",
        8,
        "heldout",
        "The fiber optic connection transmits data at twenty gigabits per day.",
        ["The dedicated interconnect operates at a sustained rate of twenty gigabits per second."],
        "FAIL",
        [("The fiber optic connection transmits data at twenty gigabits per day.", "CONTRADICTED")],
        "Time rate unit seconds changed to days.",
    )
    add_case(
        "unit_change",
        9,
        "heldout",
        "The parcel delivery package has a declared weight of five ounces.",
        ["The postal receipt lists the package weight as five kilograms."],
        "FAIL",
        [("The parcel delivery package has a declared weight of five ounces.", "CONTRADICTED")],
        "Weight unit kilograms changed to ounces.",
    )
    add_case(
        "unit_change",
        10,
        "heldout",
        "The subterranean pressure valve operates at forty Pascals of hydraulic pressure.",
        ["The high pressure valve is rated for forty atmospheres of hydraulic load."],
        "FAIL",
        [("The subterranean pressure valve operates at forty Pascals of hydraulic pressure.", "CONTRADICTED")],
        "Pressure unit atmospheres changed to Pascals.",
    )

    # 7. Date Change
    add_case(
        "date_change",
        1,
        "calibration",
        "The treaty of peace was signed in the palace in 1995.",
        ["The historic armistice was officially signed in the palace in 1918."],
        "FAIL",
        [("The treaty of peace was signed in the palace in 1995.", "CONTRADICTED")],
        "Historical year altered from 1918 to 1995.",
    )
    add_case(
        "date_change",
        2,
        "heldout",
        "The university established its campus in December 1980.",
        ["The university charter was ratified and the campus founded in September 1850."],
        "FAIL",
        [("The university established its campus in December 1980.", "CONTRADICTED")],
        "Founding date altered by over a century.",
    )
    add_case(
        "date_change",
        3,
        "calibration",
        "The product launch occurred on the first of July.",
        ["Marketing teams organized the product unveiling for the first of November."],
        "FAIL",
        [("The product launch occurred on the first of July.", "CONTRADICTED")],
        "Month changed from November to July.",
    )
    add_case(
        "date_change",
        4,
        "heldout",
        "The constitutional convention concluded in the summer of 1887.",
        ["Delegates gathered in Philadelphia where the constitutional convention concluded in 1787."],
        "FAIL",
        [("The constitutional convention concluded in the summer of 1887.", "CONTRADICTED")],
        "Centennial date error.",
    )
    add_case(
        "date_change",
        5,
        "heldout",
        "The patent application was granted in February 2024.",
        ["Records at the patent office show the invention patent was issued in August 2012."],
        "FAIL",
        [("The patent application was granted in February 2024.", "CONTRADICTED")],
        "Patent grant year altered.",
    )
    add_case(
        "date_change",
        6,
        "heldout",
        "The railway line commenced passenger service in 1960.",
        ["Construction completed and passenger rail service commenced in 1904."],
        "FAIL",
        [("The railway line commenced passenger service in 1960.", "CONTRADICTED")],
        "Commencement year altered.",
    )
    add_case(
        "date_change",
        7,
        "calibration",
        "The lunar landing occurred in July 1989.",
        ["Apollo 11 astronaut Neil Armstrong stepped onto the lunar surface in July 1969."],
        "FAIL",
        [("The lunar landing occurred in July 1989.", "CONTRADICTED")],
        "Apollo landing year altered from 1969 to 1989.",
    )
    add_case(
        "date_change",
        8,
        "heldout",
        "The company released its flagship operating system in 2001.",
        ["The computer company launched its flagship desktop operating system in 1984."],
        "FAIL",
        [("The company released its flagship operating system in 2001.", "CONTRADICTED")],
        "Software launch date shifted forward.",
    )
    add_case(
        "date_change",
        9,
        "heldout",
        "The global epidemic subsided completely by April 1910.",
        ["Medical historians record that the influenza pandemic subsided by the spring of 1920."],
        "FAIL",
        [("The global epidemic subsided completely by April 1910.", "CONTRADICTED")],
        "Pandemic timeline shifted backward.",
    )
    add_case(
        "date_change",
        10,
        "heldout",
        "The international space station was first occupied in November 2015.",
        ["Expedition 1 arrived and permanent human residency aboard the station began in November 2000."],
        "FAIL",
        [("The international space station was first occupied in November 2015.", "CONTRADICTED")],
        "Space station occupancy date altered.",
    )

    # 8. Negation
    add_case(
        "negation",
        1,
        "calibration",
        "The audit committee did not find any financial compliance irregularities.",
        ["The internal audit uncovered severe financial compliance irregularities in the procurement division."],
        "FAIL",
        [("The audit committee did not find any financial compliance irregularities.", "CONTRADICTED")],
        "Inserted 'did not find' reversing source findings.",
    )
    add_case(
        "negation",
        2,
        "heldout",
        "The chemical reaction produces toxic chlorine gas during synthesis.",
        ["The patented synthesis reaction is non-toxic and does not produce chlorine gas."],
        "FAIL",
        [("The chemical reaction produces toxic chlorine gas during synthesis.", "CONTRADICTED")],
        "Removed negation to falsely assert toxicity.",
    )
    add_case(
        "negation",
        3,
        "calibration",
        "The software application requires administrative privileges to execute.",
        ["The utility is self-contained and does not require administrative privileges."],
        "FAIL",
        [("The software application requires administrative privileges to execute.", "CONTRADICTED")],
        "Privilege requirement inverted.",
    )
    add_case(
        "negation",
        4,
        "heldout",
        "The proposed municipal zoning law does not allow commercial retail stores.",
        ["The updated zoning amendment explicitly permits commercial retail development in the district."],
        "FAIL",
        [("The proposed municipal zoning law does not allow commercial retail stores.", "CONTRADICTED")],
        "Zoning permission negated.",
    )
    add_case(
        "negation",
        5,
        "heldout",
        "The patient was not allergic to common penicillin antibiotics.",
        ["Medical history intake forms note the patient suffers from an acute penicillin allergy."],
        "FAIL",
        [("The patient was not allergic to common penicillin antibiotics.", "CONTRADICTED")],
        "Allergy status negated.",
    )
    add_case(
        "negation",
        6,
        "heldout",
        "The bridge was structurally undamaged following the seismic event.",
        ["Engineers detected severe structural damage to the suspension cables following the earthquake."],
        "FAIL",
        [("The bridge was structurally undamaged following the seismic event.", "CONTRADICTED")],
        "Structural damage denied.",
    )
    add_case(
        "negation",
        7,
        "calibration",
        "The algorithm is not susceptible to adversarial perturbation attacks.",
        ["Empirical research proved the machine learning algorithm is vulnerable to subtle adversarial perturbations."],
        "FAIL",
        [("The algorithm is not susceptible to adversarial perturbation attacks.", "CONTRADICTED")],
        "Security vulnerability negated.",
    )
    add_case(
        "negation",
        8,
        "heldout",
        "The agreement failed to win the necessary parliamentary majority.",
        ["The multilateral accord successfully secured a commanding majority during parliamentary voting."],
        "FAIL",
        [("The agreement failed to win the necessary parliamentary majority.", "CONTRADICTED")],
        "Legislative success inverted to failure.",
    )
    add_case(
        "negation",
        9,
        "heldout",
        "The water filtration system does not eliminate heavy metal contaminants.",
        ["Laboratory tests demonstrate the carbon filtration matrix effectively eliminates heavy metals."],
        "FAIL",
        [("The water filtration system does not eliminate heavy metal contaminants.", "CONTRADICTED")],
        "Filtration efficacy negated.",
    )
    add_case(
        "negation",
        10,
        "heldout",
        "The mobile device battery is removable by end users.",
        ["The chassis is hermetically sealed and the internal battery is non-removable."],
        "FAIL",
        [("The mobile device battery is removable by end users.", "CONTRADICTED")],
        "Non-removable hardware inverted to removable.",
    )

    # 9. Mixed Supported / Unsupported
    add_case(
        "mixed_supported_unsupported",
        1,
        "calibration",
        "The Eiffel Tower is located in Paris. Gustave Eiffel was a champion chess grandmaster.",
        ["The Eiffel Tower is located on the Champ de Mars in Paris, France."],
        "FAIL",
        [
            ("The Eiffel Tower is located in Paris.", "SUPPORTED"),
            ("Gustave Eiffel was a champion chess grandmaster.", "INSUFFICIENT_EVIDENCE"),
        ],
        "First claim supported, second claim ungrounded hallucination.",
    )
    add_case(
        "mixed_supported_unsupported",
        2,
        "heldout",
        "Water freezes at 0 degrees Celsius. Pure water tastes like peppermint candy.",
        ["Water freezes at 0 degrees Celsius under standard atmospheric pressure."],
        "FAIL",
        [
            ("Water freezes at 0 degrees Celsius.", "SUPPORTED"),
            ("Pure water tastes like peppermint candy.", "INSUFFICIENT_EVIDENCE"),
        ],
        "First claim supported, second claim absurd ungrounded assertion.",
    )
    add_case(
        "mixed_supported_unsupported",
        3,
        "calibration",
        "Python was released in 1991. Python syntax is based on ancient Egyptian hieroglyphics.",
        ["Guido van Rossum released Python in 1991 as an accessible scripting language."],
        "FAIL",
        [
            ("Python was released in 1991.", "SUPPORTED"),
            ("Python syntax is based on ancient Egyptian hieroglyphics.", "INSUFFICIENT_EVIDENCE"),
        ],
        "Factual origin supported, mythological syntax ungrounded.",
    )
    add_case(
        "mixed_supported_unsupported",
        4,
        "heldout",
        "Mount Everest is Earth's highest mountain. Space aliens built a radio telescope on its summit.",
        ["Mount Everest is the highest mountain above sea level on Earth."],
        "FAIL",
        [
            ("Mount Everest is Earth's highest mountain.", "SUPPORTED"),
            ("Space aliens built a radio telescope on its summit.", "INSUFFICIENT_EVIDENCE"),
        ],
        "Geography supported, sci-fi telescope ungrounded.",
    )
    add_case(
        "mixed_supported_unsupported",
        5,
        "heldout",
        "Mercury is the planet closest to the Sun. Mercury is completely covered in liquid chocolate.",
        ["Mercury is the innermost planet orbiting nearest to the Sun in the solar system."],
        "FAIL",
        [
            ("Mercury is the planet closest to the Sun.", "SUPPORTED"),
            ("Mercury is completely covered in liquid chocolate.", "INSUFFICIENT_EVIDENCE"),
        ],
        "Astronomy supported, chocolate ocean ungrounded.",
    )
    add_case(
        "mixed_supported_unsupported",
        6,
        "heldout",
        "The Pacific Ocean is the largest ocean. Deep sea mermaids manage fish trade routes there.",
        ["The Pacific Ocean is the largest and deepest body of water on Earth."],
        "FAIL",
        [
            ("The Pacific Ocean is the largest ocean.", "SUPPORTED"),
            ("Deep sea mermaids manage fish trade routes there.", "INSUFFICIENT_EVIDENCE"),
        ],
        "Hydrography supported, fantasy creature ungrounded.",
    )
    add_case(
        "mixed_supported_unsupported",
        7,
        "calibration",
        "Light travels at approximately 300,000 kilometers per second. Sound waves travel even faster.",
        ["Light moves through a vacuum at roughly 300,000 kilometers per second."],
        "FAIL",
        [
            ("Light travels at approximately 300,000 kilometers per second.", "SUPPORTED"),
            ("Sound waves travel even faster.", "CONTRADICTED"),
        ],
        "Speed of light supported, sound speed contradicted.",
    )
    add_case(
        "mixed_supported_unsupported",
        8,
        "heldout",
        "DNA carries genetic instructions. DNA was first synthesized by medieval alchemists in Venice.",
        ["DNA molecules store genetic instructions that direct biological cellular growth."],
        "FAIL",
        [
            ("DNA carries genetic instructions.", "SUPPORTED"),
            ("DNA was first synthesized by medieval alchemists in Venice.", "INSUFFICIENT_EVIDENCE"),
        ],
        "Biochemistry supported, alchemical history ungrounded.",
    )
    add_case(
        "mixed_supported_unsupported",
        9,
        "heldout",
        "The Amazon River has the largest river discharge. The Amazon River flows backward into Canada.",
        ["The Amazon River discharges the highest volume of freshwater of any river on Earth."],
        "FAIL",
        [
            ("The Amazon River has the largest river discharge.", "SUPPORTED"),
            ("The Amazon River flows backward into Canada.", "CONTRADICTED"),
        ],
        "River discharge supported, geographic flow contradicted.",
    )
    add_case(
        "mixed_supported_unsupported",
        10,
        "heldout",
        "Photosynthesis produces glucose and oxygen. Plants communicate by sending text messages.",
        ["In photosynthesis, green plants synthesize carbohydrates and release oxygen."],
        "FAIL",
        [
            ("Photosynthesis produces glucose and oxygen.", "SUPPORTED"),
            ("Plants communicate by sending text messages.", "INSUFFICIENT_EVIDENCE"),
        ],
        "Botany supported, telecommunications ungrounded.",
    )

    # 10. Multi-Passage Support
    add_case(
        "multi_passage_support",
        1,
        "calibration",
        "The spacecraft was assembled in California and launched from Florida.",
        [
            "Manufacturing engineers assembled the robotic spacecraft at the facility in Pasadena, California.",
            "The launch vehicle carrying the satellite lifted off from Cape Canaveral, Florida.",
        ],
        "PASS",
        [("The spacecraft was assembled in California and launched from Florida.", "SUPPORTED")],
        "Sentence synthesizes facts from two separate source passages.",
    )
    add_case(
        "multi_passage_support",
        2,
        "heldout",
        "The researcher earned her bachelor's degree in Chicago and completed her doctorate in Boston.",
        [
            "She graduated with a Bachelor of Science from the University of Chicago.",
            "Following undergraduate studies, she defended her doctoral thesis at Harvard University in Boston.",
        ],
        "PASS",
        [
            (
                "The researcher earned her bachelor's degree in Chicago and completed her doctorate in Boston.",
                "SUPPORTED",
            )
        ],
        "Biographical synthesis across academic records.",
    )
    add_case(
        "multi_passage_support",
        3,
        "calibration",
        "The engine combines a turbocharger made in Germany with electronics produced in Japan.",
        [
            "Precision turbochargers are fabricated at the engineering plant in Stuttgart, Germany.",
            "Electronic control units are manufactured in Tokyo, Japan before final integration.",
        ],
        "PASS",
        [("The engine combines a turbocharger made in Germany with electronics produced in Japan.", "SUPPORTED")],
        "Supply chain multi-source integration.",
    )
    add_case(
        "multi_passage_support",
        4,
        "heldout",
        "The novel was written in Paris during the winter and published in London the following autumn.",
        [
            "During a cold winter in Paris, the author completed the original manuscript draft.",
            "The prestigious publishing house released the hardcover edition in London the following autumn.",
        ],
        "PASS",
        [
            (
                "The novel was written in Paris during the winter and published in London the following autumn.",
                "SUPPORTED",
            )
        ],
        "Literary history multi-passage grounding.",
    )
    add_case(
        "multi_passage_support",
        5,
        "heldout",
        "The festival began with an acoustic concert on Friday and ended with fireworks on Sunday.",
        [
            "Opening festivities kicked off Friday evening with an acoustic music performance in the plaza.",
            "A grand fireworks spectacular illuminated the night sky on Sunday to conclude the celebrations.",
        ],
        "PASS",
        [("The festival began with an acoustic concert on Friday and ended with fireworks on Sunday.", "SUPPORTED")],
        "Event schedule synthesized from opening and closing documents.",
    )
    add_case(
        "multi_passage_support",
        6,
        "heldout",
        "The architectural plan combines granite quarried in Vermont with timber harvested in Oregon.",
        [
            "High durability grey granite slabs were sourced from quarry deposits in Barre, Vermont.",
            "Heavy structural Douglas fir timber beams were harvested from managed forests in Oregon.",
        ],
        "PASS",
        [("The architectural plan combines granite quarried in Vermont with timber harvested in Oregon.", "SUPPORTED")],
        "Materials source synthesis.",
    )
    add_case(
        "multi_passage_support",
        7,
        "calibration",
        "The team designed the user interface in London while the database was optimized in Singapore.",
        [
            "Design specialists in the London studio crafted the mobile application user interface.",
            "Senior database administrators in the Singapore office re-indexed the distributed cluster.",
        ],
        "PASS",
        [
            (
                "The team designed the user interface in London while the database was optimized in Singapore.",
                "SUPPORTED",
            )
        ],
        "Global operations synthesis.",
    )
    add_case(
        "multi_passage_support",
        8,
        "heldout",
        "The clinical trial recruited patients in Denmark and conducted molecular sequencing in Switzerland.",
        [
            "Clinical researchers enrolled seventy-five patient participants across hospitals in Denmark.",
            "Biopsy tissue samples were dispatched to the central genomics laboratory in Zurich, Switzerland for sequencing.",
        ],
        "PASS",
        [
            (
                "The clinical trial recruited patients in Denmark and conducted molecular sequencing in Switzerland.",
                "SUPPORTED",
            )
        ],
        "Clinical trial multi-site synthesis.",
    )
    add_case(
        "multi_passage_support",
        9,
        "heldout",
        "The documentary was filmed across rural Iceland and edited at a studio in Toronto.",
        [
            "Principal cinematography took place amidst the volcanic landscapes of southern Iceland.",
            "Post-production sound mixing and color grading were completed at the production studio in Toronto.",
        ],
        "PASS",
        [("The documentary was filmed across rural Iceland and edited at a studio in Toronto.", "SUPPORTED")],
        "Film production multi-location support.",
    )
    add_case(
        "multi_passage_support",
        10,
        "heldout",
        "The vehicle battery cells were manufactured in Seoul and assembled into modular packs in Michigan.",
        [
            "Advanced cylindrical lithium battery cells are produced at the industrial plant in Seoul.",
            "Final automotive battery pack enclosures are assembled and tested at the plant in Michigan.",
        ],
        "PASS",
        [
            (
                "The vehicle battery cells were manufactured in Seoul and assembled into modular packs in Michigan.",
                "SUPPORTED",
            )
        ],
        "Automotive manufacturing cross-border synthesis.",
    )

    # 11. Table-Derived Text
    add_case(
        "table_derived_text",
        1,
        "calibration",
        "Region North achieved three million dollars in revenue while Region South achieved two million.",
        ["| Region | Revenue | Costs |\n| North | $3M | $1M |\n| South | $2M | $0.8M |"],
        "PASS",
        [
            (
                "Region North achieved three million dollars in revenue while Region South achieved two million.",
                "SUPPORTED",
            )
        ],
        "Text correctly summarizes markdown table figures.",
    )
    add_case(
        "table_derived_text",
        2,
        "heldout",
        "Model Beta has an inference latency of thirty milliseconds with ninety percent accuracy.",
        ["| Model | Latency | Accuracy |\n| Alpha | 15ms | 82% |\n| Beta | 30ms | 90% |\n| Gamma | 65ms | 94% |"],
        "PASS",
        [("Model Beta has an inference latency of thirty milliseconds with ninety percent accuracy.", "SUPPORTED")],
        "Evaluation metrics table correctly extracted.",
    )
    add_case(
        "table_derived_text",
        3,
        "calibration",
        "Item A costs ten dollars, Item B costs twenty dollars, and Item C costs thirty dollars.",
        ["Product Catalog Table:\nItem A: $10\nItem B: $20\nItem C: $30"],
        "PASS",
        [("Item A costs ten dollars, Item B costs twenty dollars, and Item C costs thirty dollars.", "SUPPORTED")],
        "Tabular price list accurately transcribed.",
    )
    add_case(
        "table_derived_text",
        4,
        "heldout",
        "The server CPU utilization averaged forty percent while memory usage reached seventy percent.",
        ["Metrics Table:\nTimestamp | CPU | RAM | Disk\n12:00:00 | 40% | 70% | 12GB"],
        "PASS",
        [
            (
                "The server CPU utilization averaged forty percent while memory usage reached seventy percent.",
                "SUPPORTED",
            )
        ],
        "System monitor table grounded.",
    )
    add_case(
        "table_derived_text",
        5,
        "heldout",
        "Quarter three sales reached five hundred units, exceeding quarter one sales of three hundred units.",
        ["Quarterly Volume Summary Table:\nQ1: 300 units\nQ2: 420 units\nQ3: 500 units\nQ4: 480 units"],
        "PASS",
        [
            (
                "Quarter three sales reached five hundred units, exceeding quarter one sales of three hundred units.",
                "SUPPORTED",
            )
        ],
        "Comparative table summary grounded.",
    )
    add_case(
        "table_derived_text",
        6,
        "heldout",
        "The patient group received a fifty milligram daily dosage while the control group received a placebo.",
        ["| Group | Treatment | Daily Dose |\n| Active | Compound X | 50mg |\n| Control | Placebo | 0mg |"],
        "PASS",
        [
            (
                "The patient group received a fifty milligram daily dosage while the control group received a placebo.",
                "SUPPORTED",
            )
        ],
        "Clinical dosing table grounded.",
    )
    add_case(
        "table_derived_text",
        7,
        "calibration",
        "Flight 202 departs at nine o'clock and arrives at eleven o'clock.",
        ["Flight Schedule:\nFlight | Origin | Dest | Depart | Arrive\nFL202 | JFK | ORD | 09:00 | 11:00"],
        "PASS",
        [("Flight 202 departs at nine o'clock and arrives at eleven o'clock.", "SUPPORTED")],
        "Flight schedule table grounded.",
    )
    add_case(
        "table_derived_text",
        8,
        "heldout",
        "Tier Silver allows ten users while Tier Gold supports up to fifty users.",
        ["Subscription Matrix:\nTier | Max Users | Storage\nBronze | 3 | 5GB\nSilver | 10 | 25GB\nGold | 50 | 100GB"],
        "PASS",
        [("Tier Silver allows ten users while Tier Gold supports up to fifty users.", "SUPPORTED")],
        "SaaS subscription matrix grounded.",
    )
    add_case(
        "table_derived_text",
        9,
        "heldout",
        "The blue widget has a diameter of fifteen centimeters and weighs two kilograms.",
        ["Inventory Table:\nSKU | Color | Diameter | Weight\nWD-01 | Blue | 15cm | 2kg\nWD-02 | Red | 20cm | 3kg"],
        "PASS",
        [("The blue widget has a diameter of fifteen centimeters and weighs two kilograms.", "SUPPORTED")],
        "Physical product specification table grounded.",
    )
    add_case(
        "table_derived_text",
        10,
        "heldout",
        "District West recorded eighty votes while District East recorded one hundred and twenty votes.",
        ["Election Ballot Tally:\nDistrict | Votes Cast\nWest | 80\nEast | 120"],
        "PASS",
        [
            (
                "District West recorded eighty votes while District East recorded one hundred and twenty votes.",
                "SUPPORTED",
            )
        ],
        "Voting tabulation table grounded.",
    )

    # 12. Empty Answers
    add_case(
        "empty_answers",
        1,
        "calibration",
        "",
        ["The Eiffel Tower is located in Paris, France."],
        "FAIL",
        [],
        "Zero-length empty string response must fail.",
        availability="NO_ASSESSABLE_CLAIMS",
    )
    add_case(
        "empty_answers",
        2,
        "heldout",
        "   \n  \t  \n  ",
        ["Under standard pressure, water boils at 100 degrees Celsius."],
        "FAIL",
        [],
        "Whitespace-only response must fail.",
        availability="NO_ASSESSABLE_CLAIMS",
    )
    add_case(
        "empty_answers",
        3,
        "calibration",
        "No.",
        ["Python was created by Guido van Rossum in 1991."],
        "FAIL",
        [],
        "Three-character response below claim splitter length threshold.",
        availability="NO_ASSESSABLE_CLAIMS",
    )
    add_case(
        "empty_answers",
        4,
        "heldout",
        "Hi there!",
        ["Mount Everest is the highest mountain on Earth."],
        "FAIL",
        [],
        "Short conversational greeting containing zero factual assertions.",
        availability="NO_ASSESSABLE_CLAIMS",
    )
    add_case(
        "empty_answers",
        5,
        "heldout",
        "...",
        ["The Pacific Ocean is the largest oceanic basin on the planet."],
        "FAIL",
        [],
        "Punctuation-only response.",
        availability="NO_ASSESSABLE_CLAIMS",
    )
    add_case(
        "empty_answers",
        6,
        "heldout",
        "Okay.",
        ["Mercury is the smallest planet in the solar system."],
        "FAIL",
        [],
        "One-word acknowledgement response.",
        availability="NO_ASSESSABLE_CLAIMS",
    )
    add_case(
        "empty_answers",
        7,
        "calibration",
        "Thanks!",
        ["Photosynthesis produces glucose and oxygen from sunlight."],
        "FAIL",
        [],
        "Exclamation greeting.",
        availability="NO_ASSESSABLE_CLAIMS",
    )
    add_case(
        "empty_answers",
        8,
        "heldout",
        "\n\n\n\n",
        ["DNA stores biological genetic information."],
        "FAIL",
        [],
        "Newline characters only.",
        availability="NO_ASSESSABLE_CLAIMS",
    )
    add_case(
        "empty_answers",
        9,
        "heldout",
        "Sure.",
        ["The speed of light in vacuum is 299,792 kilometers per second."],
        "FAIL",
        [],
        "Monosyllabic conversational filler.",
        availability="NO_ASSESSABLE_CLAIMS",
    )
    add_case(
        "empty_answers",
        10,
        "heldout",
        "N/A",
        ["The Amazon River discharges the highest water volume."],
        "FAIL",
        [],
        "Abbreviation placeholder answer.",
        availability="NO_ASSESSABLE_CLAIMS",
    )

    # 13. Refusals & Honest Uncertainty
    add_case(
        "refusals",
        1,
        "calibration",
        "The provided documents do not contain information regarding this topic.",
        ["The company announced quarterly earnings and operating revenues for fiscal year 2023."],
        "PASS",
        [("The provided documents do not contain information regarding this topic.", "INSUFFICIENT_EVIDENCE")],
        "Classic honest refusal statement correctly identified as honest uncertainty.",
    )
    add_case(
        "refusals",
        2,
        "heldout",
        "I cannot find information about the requested topic in the provided text.",
        ["Standard safety guidelines require safety goggles in the chemical laboratory."],
        "PASS",
        [("I cannot find information about the requested topic in the provided text.", "INSUFFICIENT_EVIDENCE")],
        "Direct honest uncertainty response.",
    )
    add_case(
        "refusals",
        3,
        "calibration",
        "There is no information about the executive founder's salary in the context.",
        ["The enterprise produces enterprise database monitoring tools and distributed tracing agents."],
        "PASS",
        [("There is no information about the executive founder's salary in the context.", "INSUFFICIENT_EVIDENCE")],
        "Contextual absence refusal.",
    )
    add_case(
        "refusals",
        4,
        "heldout",
        "The context does not provide details concerning the specified contract deadline.",
        ["Terms of service dictate acceptable usage parameters for cloud compute credits."],
        "PASS",
        [("The context does not provide details concerning the specified contract deadline.", "INSUFFICIENT_EVIDENCE")],
        "Context omission statement.",
    )
    add_case(
        "refusals",
        5,
        "heldout",
        "I don't have information in the provided sources to answer this question accurately.",
        ["The transit schedule outlines weekday bus routes between downtown and the airport."],
        "PASS",
        [
            (
                "I don't have information in the provided sources to answer this question accurately.",
                "INSUFFICIENT_EVIDENCE",
            )
        ],
        "Polite assistant refusal.",
    )
    add_case(
        "refusals",
        6,
        "heldout",
        "No specific information regarding that historical figure is mentioned in the text.",
        ["Geological formations in the valley indicate ancient limestone deposit sedimentation."],
        "PASS",
        [
            (
                "No specific information regarding that historical figure is mentioned in the text.",
                "INSUFFICIENT_EVIDENCE",
            )
        ],
        "Subject matter absence acknowledgement.",
    )
    add_case(
        "refusals",
        7,
        "calibration",
        "The documents do not contain any references to the requested chemical formula.",
        ["The logistics manual instructs handlers to keep freight pallets shrink-wrapped."],
        "PASS",
        [("The documents do not contain any references to the requested chemical formula.", "INSUFFICIENT_EVIDENCE")],
        "Document limitation declaration.",
    )
    add_case(
        "refusals",
        8,
        "heldout",
        "I could not find relevant facts about the software license in the attached files.",
        ["Network configuration files specify local gateway router IP addresses."],
        "PASS",
        [
            (
                "I could not find relevant facts about the software license in the attached files.",
                "INSUFFICIENT_EVIDENCE",
            )
        ],
        "Search failure honest disclosure.",
    )
    add_case(
        "refusals",
        9,
        "heldout",
        "The provided context does not provide any answers for the query submitted.",
        ["Maintenance personnel inspect the ventilation filters on the first day of each month."],
        "PASS",
        [("The provided context does not provide any answers for the query submitted.", "INSUFFICIENT_EVIDENCE")],
        "Standard uncertainty formula.",
    )
    add_case(
        "refusals",
        10,
        "heldout",
        "There is no information about international shipping rates in these excerpts.",
        ["The customer support portal operates Monday through Friday from nine until five."],
        "PASS",
        [("There is no information about international shipping rates in these excerpts.", "INSUFFICIENT_EVIDENCE")],
        "Catalog boundary refusal.",
    )

    # 14. Citation Mistakes & Attribution Errors
    add_case(
        "citation_mistakes",
        1,
        "calibration",
        "According to Document One, the population of the village reached ten thousand residents.",
        [
            "Document One: The village economy relies on artisan pottery.\nDocument Two: The census recorded ten thousand village residents."
        ],
        "FAIL",
        [
            (
                "According to Document One, the population of the village reached ten thousand residents.",
                "INSUFFICIENT_EVIDENCE",
            )
        ],
        "Claim attributes population statistic to Document One when it resides in Document Two.",
    )
    add_case(
        "citation_mistakes",
        2,
        "heldout",
        "As stated in the safety appendix, wearing leather gloves is strictly prohibited.",
        ["Safety Appendix: Nitrile gloves are mandatory. Leather gloves are permitted for heavy lifting."],
        "FAIL",
        [("As stated in the safety appendix, wearing leather gloves is strictly prohibited.", "CONTRADICTED")],
        "Citation references section that contradicts the prohibition.",
    )
    add_case(
        "citation_mistakes",
        3,
        "calibration",
        "Source Document Alpha proves that the reactor was designed by Engineer Davis.",
        [
            "Document Alpha: The electrical substation was designed by Engineer Davis.\nDocument Beta: The nuclear reactor was designed by Chief Engineer Kowalski."
        ],
        "FAIL",
        [("Source Document Alpha proves that the reactor was designed by Engineer Davis.", "CONTRADICTED")],
        "Cross-document entity attribution mistake.",
    )
    add_case(
        "citation_mistakes",
        4,
        "heldout",
        "The financial report in Section Four reports zero long-term corporate debt.",
        ["Section Four: The company carries twenty million dollars in long-term commercial bonds."],
        "FAIL",
        [("The financial report in Section Four reports zero long-term corporate debt.", "CONTRADICTED")],
        "Section citation refutes debt claim.",
    )
    add_case(
        "citation_mistakes",
        5,
        "heldout",
        "According to Table Three, the product warranty period is five full years.",
        ["Table Two: Warranty coverage is valid for one year.\nTable Three: Shipping options include express ground."],
        "FAIL",
        [("According to Table Three, the product warranty period is five full years.", "INSUFFICIENT_EVIDENCE")],
        "Wrong table cited for warranty term.",
    )
    add_case(
        "citation_mistakes",
        6,
        "heldout",
        "As explained in the user guide, pressing the reset switch formats the storage drive.",
        ["User Guide: Pressing the reset switch power-cycles the device without erasing user storage."],
        "FAIL",
        [("As explained in the user guide, pressing the reset switch formats the storage drive.", "CONTRADICTED")],
        "Operational manual citation contradicts data loss.",
    )
    add_case(
        "citation_mistakes",
        7,
        "calibration",
        "Exhibit B documents that the company was founded in the year 2005.",
        [
            "Exhibit A: Incorporating paperwork was filed in 2005.\nExhibit B: Intellectual property patents were acquired in 2018."
        ],
        "FAIL",
        [("Exhibit B documents that the company was founded in the year 2005.", "INSUFFICIENT_EVIDENCE")],
        "Exhibit misattribution.",
    )
    add_case(
        "citation_mistakes",
        8,
        "heldout",
        "Per the clinical study conclusion, the therapeutic intervention showed zero efficacy.",
        ["Clinical Study Conclusion: The therapeutic intervention demonstrated robust positive clinical efficacy."],
        "FAIL",
        [("Per the clinical study conclusion, the therapeutic intervention showed zero efficacy.", "CONTRADICTED")],
        "Attributed conclusion directly opposite of evidence.",
    )
    add_case(
        "citation_mistakes",
        9,
        "heldout",
        "According to the weather bulletin, snowfall will commence at midnight.",
        ["Weather Bulletin: Precipitation will remain rain, with zero possibility of freezing snowfall."],
        "FAIL",
        [("According to the weather bulletin, snowfall will commence at midnight.", "CONTRADICTED")],
        "Forecast citation contradicts precipitation type.",
    )
    add_case(
        "citation_mistakes",
        10,
        "heldout",
        "Source Document Gamma proves that the team completed twenty sprints in total.",
        ["Document Gamma: Sprint velocity averaged thirty story points per iteration."],
        "FAIL",
        [("Source Document Gamma proves that the team completed twenty sprints in total.", "INSUFFICIENT_EVIDENCE")],
        "Attributed document discusses velocity, not sprint total count.",
    )

    return cases


def generate_and_save_dataset() -> None:
    """Generate the full benchmark dataset and save to disk."""
    cases = build_raw_cases()

    # Verify leakage
    leaks = check_dataset_leakage(cases, threshold=0.8)
    if leaks:
        print(f"Warning: Found {len(leaks)} potential leakage pairs between calibration and heldout!")
        for c1, c2, sim in leaks:
            print(f"  Leakage: {c1} <-> {c2} (similarity: {sim:.2f})")
    else:
        print("✓ Zero leakage detected between calibration and held-out splits (Jaccard threshold 0.8).")

    from collections import Counter

    errors = validate_cases(cases)
    if errors:
        raise SystemExit("Dataset validation failed:\n" + "\n".join(errors))

    dataset_dict = {
        "dataset_version": DATASET_VERSION,
        "metadata": {
            "categories": len({c["category"] for c in cases}),
            "total_cases": len(cases),
            "calibration_cases": sum(1 for c in cases if c["split"] == "calibration"),
            "heldout_cases": sum(1 for c in cases if c["split"] == "heldout"),
            "review_status": dict(Counter(c["review"]["status"] for c in cases)),
            "human_reviewed_cases": sum(1 for c in cases if is_reviewed(c)),
            "language": "en",
            "content": "Synthetic, general-knowledge English sentences. No customer data, no secrets.",
            "models_evaluated": {
                "sts": "sentence-transformers/all-MiniLM-L6-v2",
                "sts_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                "nli": "cross-encoder/nli-deberta-v3-xsmall",
                "nli_revision": "a150876415327c80daeff35ca6f68f5ed8cf5c24",
            },
            "thresholds": {
                "support_threshold": 0.40,
                "nli_gate_avg_similarity": 0.25,
                "contradiction": 0.5,
                "entailment": 0.5,
            },
            "preprocessing": {
                "claim_splitter": "longtracer.guard.claim_splitter.split_into_claims "
                "(whitespace-normalised; responses <= 10 chars and sentences <= 15 chars dropped)",
                "source_sentences": "HybridVerificationModel.split_into_sentences (sentences <= 10 chars dropped)",
            },
            "leakage_check": {
                "method": "Across calibration x held-out pairs: exact match after lower-casing and "
                "whitespace normalisation, plus token Jaccard >= 0.8 on the response and on the "
                "concatenated source text.",
                "pairs_flagged": len(leaks),
            },
        },
        "cases": cases,
    }

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=2)
    print(f"✓ Saved {len(cases)} benchmark cases to {DATASET_PATH}")


if __name__ == "__main__":
    generate_and_save_dataset()
