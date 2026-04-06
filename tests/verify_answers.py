"""
ANSWER / RESPONSE VERIFICATION SCRIPT
Validates the NLI model's actual decision quality across many scenarios.
"""

import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from longtracer import check

PASS = "✅"
FAIL = "❌"
results = []

def test(name, response, sources, expect_verdict, expect_hallucinations=None):
    """Run a verification and check the result matches expectations."""
    r = check(response=response, sources=sources)
    verdict_ok = r.verdict == expect_verdict
    hall_ok = True
    if expect_hallucinations is not None:
        hall_ok = r.hallucination_count == expect_hallucinations

    status = PASS if (verdict_ok and hall_ok) else FAIL
    detail = f"verdict={r.verdict}(expect={expect_verdict}) score={r.trust_score:.2f} hall={r.hallucination_count}"
    if expect_hallucinations is not None:
        detail += f"(expect={expect_hallucinations})"
    results.append({"status": status, "test": name, "detail": detail})
    print(f"  {status} {name} — {detail}")
    return r


print("\n" + "="*70)
print("  ANSWER VALIDATION: SUPPORTED CLAIMS (expect PASS)")
print("="*70)

test("Exact match",
     response="Water boils at 100 degrees Celsius.",
     sources=["Water boils at 100 degrees Celsius at standard atmospheric pressure."],
     expect_verdict="PASS", expect_hallucinations=0)

test("Paraphrased claim",
     response="Python was made by Guido van Rossum.",
     sources=["Python is a programming language created by Guido van Rossum in 1991."],
     expect_verdict="PASS", expect_hallucinations=0)

test("Multi-source support",
     response="The Earth orbits the Sun. Gravity keeps it in orbit.",
     sources=[
         "The Earth orbits the Sun once every 365.25 days.",
         "Gravity is the force that keeps planets in their orbits around the Sun."
     ],
     expect_verdict="PASS", expect_hallucinations=0)

test("Technical claim",
     response="JavaScript runs in the browser. Node.js allows JavaScript on the server.",
     sources=[
         "JavaScript is a programming language used in web browsers.",
         "Node.js is a runtime that allows JavaScript to run on the server side."
     ],
     expect_verdict="PASS", expect_hallucinations=0)


print("\n" + "="*70)
print("  ANSWER VALIDATION: HALLUCINATED CLAIMS (expect FAIL)")
print("="*70)

test("Wrong location",
     response="The Eiffel Tower is located in Berlin, Germany.",
     sources=["The Eiffel Tower is a wrought-iron lattice tower in Paris, France."],
     expect_verdict="FAIL")

test("Wrong date",
     response="World War 2 ended in 1960.",
     sources=["World War II ended in 1945 with the surrender of Germany and Japan."],
     expect_verdict="FAIL")

test("Fabricated fact",
     response="The Great Wall of China is made of plastic and was built last year.",
     sources=["The Great Wall of China is made of stone, brick, tamped earth and other materials. Construction began in the 7th century BC."],
     expect_verdict="FAIL")

test("Contradicted attribution",
     response="Albert Einstein invented the telephone.",
     sources=["Alexander Graham Bell is credited with inventing the telephone in 1876. Einstein was a theoretical physicist."],
     expect_verdict="FAIL")


print("\n" + "="*70)
print("  ANSWER VALIDATION: EDGE CASES")
print("="*70)

test("Empty response",
     response="",
     sources=["Some source text here."],
     expect_verdict="PASS", expect_hallucinations=0)

test("No sources at all",
     response="This claim cannot be verified without any sources.",
     sources=[],
     expect_verdict="FAIL")

test("Very short claim",
     response="Yes.",
     sources=["The answer is yes, confirmed by the data."],
     expect_verdict="PASS")

test("Claim with numbers",
     response="The speed of light is approximately 300,000 km/s.",
     sources=["Light travels at approximately 299,792 kilometers per second in a vacuum."],
     expect_verdict="PASS", expect_hallucinations=0)


print("\n" + "="*70)
print("  ANSWER VALIDATION: MIXED CLAIMS (some true, some false)")
print("="*70)

test("Mixed: 1 true + 1 false",
     response="Paris is the capital of France. Tokyo is the capital of China.",
     sources=[
         "Paris is the capital and most populous city of France.",
         "Tokyo is the capital of Japan. Beijing is the capital of China."
     ],
     expect_verdict="FAIL")

test("Mixed: 2 true + 1 false",
     response="Python is open source. It was created by Guido van Rossum. Python was released in 2020.",
     sources=["Python is an open-source programming language created by Guido van Rossum. It was first released in 1991."],
     expect_verdict="FAIL")


print("\n" + "="*70)
print("  CLI INTEGRATION: JSON PURITY")
print("="*70)

import subprocess
try:
    proc = subprocess.run(
        ["longtracer", "check", "--json", "The sky is blue.", "The sky appears blue due to Rayleigh scattering."],
        capture_output=True, text=True, timeout=120
    )
    stdout = proc.stdout.strip()
    parsed = json.loads(stdout)
    assert "verdict" in parsed
    assert "trust_score" in parsed
    assert "claims" in parsed
    print(f"  {PASS} CLI --json output is pure JSON (no pollution)")
    results.append({"status": PASS, "test": "CLI JSON purity", "detail": "stdout parses as clean JSON"})
except json.JSONDecodeError as e:
    print(f"  {FAIL} CLI --json output is NOT pure JSON — {e}")
    print(f"       Raw stdout: {stdout[:200]}")
    results.append({"status": FAIL, "test": "CLI JSON purity", "detail": str(e)})
except Exception as e:
    print(f"  {FAIL} CLI test error — {e}")
    results.append({"status": FAIL, "test": "CLI JSON purity", "detail": str(e)})

# CLI hallucination detection
try:
    proc = subprocess.run(
        ["longtracer", "check", "--json",
         "The Eiffel Tower is in Berlin.",
         "The Eiffel Tower is in Paris, France."],
        capture_output=True, text=True, timeout=120
    )
    parsed = json.loads(proc.stdout.strip())
    assert parsed["verdict"] == "FAIL"
    assert parsed["hallucination_count"] >= 1
    print(f"  {PASS} CLI detects hallucination (verdict=FAIL, hall={parsed['hallucination_count']})")
    results.append({"status": PASS, "test": "CLI hallucination detection", "detail": f"verdict={parsed['verdict']}"})
except Exception as e:
    print(f"  {FAIL} CLI hallucination detection — {e}")
    results.append({"status": FAIL, "test": "CLI hallucination detection", "detail": str(e)})

# CLI human-readable
try:
    proc = subprocess.run(
        ["longtracer", "check",
         "Python is a programming language.",
         "Python is a high-level programming language."],
        capture_output=True, text=True, timeout=120
    )
    assert "✓" in proc.stdout or "✗" in proc.stdout
    print(f"  {PASS} CLI human-readable output contains verdict icons")
    results.append({"status": PASS, "test": "CLI human-readable", "detail": "contains ✓/✗"})
except Exception as e:
    print(f"  {FAIL} CLI human-readable — {e}")
    results.append({"status": FAIL, "test": "CLI human-readable", "detail": str(e)})

# CLI trace listing
try:
    proc = subprocess.run(
        ["longtracer", "view", "--limit", "3"],
        capture_output=True, text=True, timeout=30
    )
    assert "RECENT TRACES" in proc.stdout or "No traces" in proc.stdout
    print(f"  {PASS} CLI 'view' lists traces without crash")
    results.append({"status": PASS, "test": "CLI view", "detail": "lists or empty"})
except Exception as e:
    print(f"  {FAIL} CLI view — {e}")
    results.append({"status": FAIL, "test": "CLI view", "detail": str(e)})


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  ANSWER VALIDATION SUMMARY")
print("="*70)
total = len(results)
passed = sum(1 for r in results if PASS in r["status"])
failed = sum(1 for r in results if FAIL in r["status"])
print(f"\n  Total: {total}  |  Passed: {passed}  |  Failed: {failed}")
print(f"  Pass Rate: {passed/total*100:.1f}%\n")

if failed > 0:
    print("  FAILED TESTS:")
    for r in results:
        if FAIL in r["status"]:
            print(f"    ❌ {r['test']}: {r['detail']}")
    print()

print("="*70)
sys.exit(1 if failed > 0 else 0)
