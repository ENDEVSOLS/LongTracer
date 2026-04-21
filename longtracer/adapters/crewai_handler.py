"""
CrewAI adapter for LongTracer.

Wraps a CrewAI ``Crew`` to automatically verify task outputs after
crew execution (kickoff). Each task's output is verified against its
context sources.

Usage:
    from longtracer import instrument_crewai

    crew = Crew(agents=[...], tasks=[...])
    instrument_crewai(crew)

    result = crew.kickoff()
    # Verification results attached to result:
    for r in result._longtracer_results:
        print(r.verdict, r.trust_score)

    # Or use standalone:
    from longtracer.adapters.crewai_handler import verify_crew_output
    results = verify_crew_output(crew_output, sources)
"""

import logging
import time
from typing import Any, Dict, List, Optional

try:
    import crewai as _crewai_mod
    _CREWAI_AVAILABLE = True
except ImportError:
    _CREWAI_AVAILABLE = False

from longtracer.core import LongTracer
from longtracer.logging_config import log_span, log_trace_id

logger = logging.getLogger("longtracer")


def _check_crewai() -> None:
    """Raise ImportError with install instructions if crewai is missing."""
    if not _CREWAI_AVAILABLE:
        raise ImportError(
            "The crewai package is required for the CrewAI adapter. "
            "Install with: pip install 'longtracer[crewai]'"
        )


def _extract_task_output(task: Any) -> str:
    """Extract the output text from a CrewAI task.

    Handles multiple output attribute formats across CrewAI versions.

    Returns:
        The task output text, or empty string.
    """
    output = getattr(task, "output", None)
    if output is None:
        return ""

    # CrewAI TaskOutput object
    if hasattr(output, "raw"):
        return str(output.raw or "")

    # String output
    if isinstance(output, str):
        return output

    # Dict output
    if isinstance(output, dict):
        return str(output.get("raw", output.get("result", str(output))))

    return str(output)


def _extract_context_sources(task: Any, all_tasks: List[Any]) -> List[str]:
    """Extract source texts from a task's context tasks.

    In CrewAI, a task's ``context`` is a list of other tasks whose
    outputs feed into this task. We use those outputs as the
    verification sources.

    Args:
        task: The task to extract context for.
        all_tasks: All tasks in the crew (for fallback).

    Returns:
        List of source text strings from context tasks.
    """
    sources: List[str] = []

    # Primary: explicit context tasks
    context_tasks = getattr(task, "context", None)
    if context_tasks and isinstance(context_tasks, (list, tuple)):
        for ctx_task in context_tasks:
            text = _extract_task_output(ctx_task)
            if text:
                sources.append(text[:1000])

    # If no explicit context, try the task's description + expected_output
    # as a lightweight check (the task knows what it should produce)
    if not sources:
        description = getattr(task, "description", "")
        expected = getattr(task, "expected_output", "")
        if expected:
            sources.append(str(expected)[:1000])
        if description:
            sources.append(str(description)[:1000])

    return sources


def _verify_task_output(
    task: Any,
    all_tasks: List[Any],
    threshold: float = 0.5,
) -> Any:
    """Run verification on a single task's output.

    Returns:
        VerificationResult or None.
    """
    response = _extract_task_output(task)
    if not response or not response.strip():
        return None

    sources = _extract_context_sources(task, all_tasks)
    if not sources:
        return None

    try:
        from longtracer.guard.verifier import CitationVerifier

        verifier = CitationVerifier(threshold=threshold)
        result = verifier.verify_parallel(response, sources)
        return result
    except Exception as exc:
        logger.warning("LongTracer: CrewAI task verification failed: %s", exc)
        return None


def _log_verification_to_tracer(
    task: Any,
    result: Any,
    task_index: int,
    verify_ms: float,
) -> None:
    """Log task verification to LongTracer tracer if enabled."""
    tracer = LongTracer.get_tracer() if LongTracer.is_enabled() else None
    if not tracer or result is None:
        return

    task_name = getattr(task, "description", f"task_{task_index}")[:80]

    claims_data = []
    for i, claim in enumerate(result.claims):
        claims_data.append({
            "claim_id": f"task{task_index}_claim_{i}",
            "text": claim["claim"][:200],
            "status": "supported" if claim["supported"] else "unsupported",
            "score": claim["score"],
            "is_hallucination": claim.get("is_hallucination", False),
        })

    with tracer.span(f"crewai_task_{task_index}", run_type="chain") as span:
        span.set_output({
            "task": task_name,
            "claims": claims_data,
            "total_claims": len(claims_data),
            "trust_score": result.trust_score,
            "verdict": result.verdict,
            "verify_ms": round(verify_ms, 1),
        })

    if LongTracer.is_verbose():
        log_span(
            f"crewai_task_{task_index}",
            task=task_name[:40],
            verdict=result.verdict,
            score=f"{result.trust_score:.2f}",
        )


def instrument_crewai(
    crew: Any,
    threshold: float = 0.5,
    verbose: Optional[bool] = None,
) -> Any:
    """Instrument a CrewAI Crew to verify task outputs after kickoff.

    Wraps ``crew.kickoff()`` to run hallucination verification on each
    task's output after the crew completes execution.

    Args:
        crew: A ``crewai.Crew`` instance.
        threshold: Verification threshold (default 0.5).
        verbose: Override verbose setting.

    Returns:
        The crew (same instance, modified in-place).

    Usage::

        from crewai import Crew, Agent, Task
        from longtracer import instrument_crewai

        crew = Crew(agents=[...], tasks=[...])
        instrument_crewai(crew)

        result = crew.kickoff()
        # Access verification results:
        for vr in getattr(result, '_longtracer_results', []):
            print(vr.verdict, vr.trust_score)
    """
    _check_crewai()

    if not LongTracer.is_enabled():
        LongTracer.init(verbose=verbose)

    # Guard against double-patching
    if getattr(crew, "_longtracer_patched", False):
        logger.debug("LongTracer: Crew already instrumented, skipping")
        return crew

    original_kickoff = crew.kickoff

    def patched_kickoff(*args: Any, **kwargs: Any) -> Any:
        """Wrapped kickoff that verifies task outputs after execution."""
        # Initialize tracer span for the full crew run
        tracer = LongTracer.get_tracer() if LongTracer.is_enabled() else None
        if tracer:
            tracer.start_root(inputs={"adapter": "crewai"})

        crew_start = time.time()
        crew_output = original_kickoff(*args, **kwargs)
        crew_ms = (time.time() - crew_start) * 1000

        # Verify each task's output
        verification_results = []
        tasks = getattr(crew, "tasks", [])
        for i, task in enumerate(tasks):
            verify_start = time.time()
            result = _verify_task_output(task, tasks, threshold=threshold)
            verify_ms = (time.time() - verify_start) * 1000

            if result is not None:
                verification_results.append(result)
                _log_verification_to_tracer(task, result, i, verify_ms)

        # Attach results to crew output
        try:
            crew_output._longtracer_results = verification_results
        except (AttributeError, TypeError):
            pass

        # Close tracer root
        if tracer and tracer.root_run:
            total_tasks = len(tasks)
            verified_tasks = len(verification_results)
            passed = sum(1 for r in verification_results if r.verdict == "PASS")
            tracer.end_root(outputs={
                "adapter": "crewai",
                "total_tasks": total_tasks,
                "verified_tasks": verified_tasks,
                "passed": passed,
                "crew_ms": round(crew_ms, 1),
            })
            if LongTracer.is_verbose():
                log_span(
                    "crewai_complete",
                    tasks=total_tasks,
                    verified=verified_tasks,
                    passed=passed,
                )
                trace_id = tracer.root_run.get("trace_id", "") if tracer.root_run else ""
                if trace_id:
                    log_trace_id(trace_id)

        return crew_output

    crew.kickoff = patched_kickoff
    crew._longtracer_patched = True
    logger.info("LongTracer: CrewAI crew instrumented")
    return crew


def verify_crew_output(
    crew_output: Any,
    sources: List[str],
    threshold: float = 0.5,
) -> Any:
    """Manually verify a CrewAI crew output against sources.

    Standalone function — no monkey-patching needed.

    Args:
        crew_output: A ``CrewOutput`` object or string.
        sources: Source texts to verify against.
        threshold: Verification threshold.

    Returns:
        VerificationResult or None.

    Usage::

        result = crew.kickoff()
        vr = verify_crew_output(result, ["source text..."])
        print(vr.verdict, vr.trust_score)
    """
    _check_crewai()

    # Extract text from CrewOutput
    if hasattr(crew_output, "raw"):
        response = str(crew_output.raw or "")
    elif isinstance(crew_output, str):
        response = crew_output
    else:
        response = str(crew_output)

    if not response or not sources:
        return None

    try:
        from longtracer.guard.verifier import CitationVerifier

        verifier = CitationVerifier(threshold=threshold)
        return verifier.verify_parallel(response, sources)
    except Exception as exc:
        logger.error("LongTracer: CrewAI output verification failed: %s", exc)
        return None
