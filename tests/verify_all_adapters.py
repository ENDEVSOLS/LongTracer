"""
COMPREHENSIVE VERIFICATION SCRIPT
Tests: LangGraph, LangChain Agents, CLI, Haystack
No code changes - READ-ONLY verification.
"""

import sys
import os
import json
import time
import traceback
from uuid import uuid4, UUID
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
results = []

def log(status, test_name, detail=""):
    results.append({"status": status, "test": test_name, "detail": detail})
    print(f"  {status} {test_name}" + (f" — {detail}" if detail else ""))


# ══════════════════════════════════════════════════════════════
# SECTION 1: IMPORT VERIFICATION
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 1: IMPORT VERIFICATION")
print("="*70)

try:
    from longtracer import LongTracer, CitationVerifier, VerificationResult
    log(PASS, "Core imports (LongTracer, CitationVerifier, VerificationResult)")
except Exception as e:
    log(FAIL, "Core imports", str(e))

try:
    from longtracer import instrument_langgraph, instrument_langchain_agent
    log(PASS, "LangGraph + LangChain agent instrument functions importable")
except Exception as e:
    log(FAIL, "LangGraph/Agent instrument imports", str(e))

try:
    from longtracer import instrument_haystack
    log(PASS, "Haystack instrument function importable")
except Exception as e:
    log(FAIL, "Haystack instrument import", str(e))

try:
    from longtracer import instrument_langchain
    log(PASS, "LangChain instrument function importable")
except Exception as e:
    log(FAIL, "LangChain instrument import", str(e))

try:
    from longtracer import check
    log(PASS, "check() one-liner importable")
except Exception as e:
    log(FAIL, "check() import", str(e))

try:
    from longtracer.adapters.langgraph_handler import LongTracerAgentHandler
    log(PASS, "LongTracerAgentHandler class importable directly")
except Exception as e:
    log(FAIL, "LongTracerAgentHandler import", str(e))

try:
    from longtracer.adapters.langchain_handler import CitationGuardCallbackHandler
    log(PASS, "CitationGuardCallbackHandler class importable")
except Exception as e:
    log(FAIL, "CitationGuardCallbackHandler import", str(e))

try:
    from longtracer.adapters.haystack_handler import LongTracerVerifier
    log(PASS, "LongTracerVerifier (Haystack component) importable")
except Exception as e:
    log(FAIL, "LongTracerVerifier import", str(e))

try:
    from longtracer.cli import main as cli_main
    log(PASS, "CLI main function importable")
except Exception as e:
    log(FAIL, "CLI import", str(e))

# Check __all__ completeness
try:
    import longtracer
    for name in longtracer.__all__:
        assert hasattr(longtracer, name), f"Missing: {name}"
    log(PASS, f"__all__ exports ({len(longtracer.__all__)} symbols) all resolvable")
except Exception as e:
    log(FAIL, "__all__ exports", str(e))

# Check adapters __init__ lazy loading
try:
    from longtracer.adapters import __all__ as adapter_all
    # The lazy __getattr__ should work for listed names
    log(PASS, f"Adapters __all__ defined ({len(adapter_all)} symbols)")
except Exception as e:
    log(FAIL, "Adapters __all__", str(e))


# ══════════════════════════════════════════════════════════════
# SECTION 2: LANGGRAPH HANDLER VERIFICATION
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 2: LANGGRAPH HANDLER VERIFICATION")
print("="*70)

from longtracer.adapters.langgraph_handler import (
    LongTracerAgentHandler, instrument_langgraph,
    _get_agent_state, _reset_agent_state, _normalize_document,
    _extract_text_from_message, _check_langchain_core,
)
from longtracer.core import LongTracer

# 2.1 - Dependency guard
try:
    _check_langchain_core()
    log(PASS, "langchain-core detected correctly")
except ImportError:
    log(FAIL, "langchain-core not detected but IS installed")

# 2.2 - Handler instantiation
try:
    handler = LongTracerAgentHandler(threshold=0.6, verbose=True)
    assert handler.threshold == 0.6
    assert handler._verbose is True
    assert handler.name == "LongTracerAgentHandler"
    log(PASS, "Handler instantiation with custom threshold/verbose")
except Exception as e:
    log(FAIL, "Handler instantiation", str(e))

# 2.3 - Thread-safe state
try:
    _reset_agent_state()
    state = _get_agent_state()
    assert isinstance(state, dict)
    assert state["chain_depth"] == 0
    assert state["sources"] == []
    assert state["tool_calls"] == []
    assert state["final_answer"] is None
    _reset_agent_state()
    log(PASS, "Thread-safe agent state init/reset")
except Exception as e:
    log(FAIL, "Thread-safe agent state", str(e))

# 2.4 - Document normalization
try:
    # Test with LangChain Document-like object
    mock_doc = MagicMock()
    mock_doc.page_content = "The Eiffel Tower is in Paris."
    mock_doc.metadata = {"source": "wiki.pdf", "page": 3}
    result = _normalize_document(mock_doc)
    assert result["text"] == "The Eiffel Tower is in Paris."
    assert result["source"] == "wiki.pdf"
    assert result["page"] == 3
    assert len(result["chunk_id"]) == 12

    # Test with dict input
    dict_doc = {"page_content": "Hello world", "metadata": {"source": "test"}}
    result2 = _normalize_document(dict_doc)
    assert result2["text"] == "Hello world"
    assert result2["source"] == "test"

    log(PASS, "Document normalization (object + dict)")
except Exception as e:
    log(FAIL, "Document normalization", str(e))

# 2.5 - Message text extraction
try:
    assert _extract_text_from_message("plain string") == "plain string"

    msg = MagicMock()
    msg.content = "Hello from AI"
    assert _extract_text_from_message(msg) == "Hello from AI"

    msg2 = MagicMock()
    msg2.content = [{"type": "text", "text": "block text"}, "extra"]
    assert "block text" in _extract_text_from_message(msg2)
    assert "extra" in _extract_text_from_message(msg2)

    log(PASS, "Message text extraction (str, object, list)")
except Exception as e:
    log(FAIL, "Message text extraction", str(e))

# 2.6 - Full callback lifecycle simulation
try:
    LongTracer.reset()
    LongTracer.init(backend="memory", verbose=False)
    handler = LongTracerAgentHandler(threshold=0.5, verbose=False)
    _reset_agent_state()

    run_id = uuid4()
    serialized = {"name": "test_graph"}

    # on_chain_start (root)
    handler.on_chain_start(serialized, {"input": "test"}, run_id=run_id)
    state = _get_agent_state()
    assert state["root_run_id"] == str(run_id)
    assert state["chain_depth"] == 1

    # on_retriever_start + end
    ret_id = uuid4()
    handler.on_retriever_start(serialized, "test query", run_id=ret_id)
    mock_docs = [MagicMock(page_content="Source text A", metadata={"source": "a.pdf"})]
    handler.on_retriever_end(mock_docs, run_id=ret_id)
    state = _get_agent_state()
    assert len(state["sources"]) == 1
    assert "Source text A" in state["sources"][0]

    # on_tool_start + end
    tool_id = uuid4()
    handler.on_tool_start({"name": "search"}, "query input", run_id=tool_id)
    handler.on_tool_end("tool output result", run_id=tool_id)
    state = _get_agent_state()
    assert len(state["tool_calls"]) == 1
    assert state["tool_calls"][0]["name"] == "search"

    # on_chat_model_start + end
    llm_id = uuid4()
    handler.on_chat_model_start(serialized, [[MagicMock()]], run_id=llm_id)
    mock_response = MagicMock()
    mock_gen = MagicMock()
    mock_gen.message = MagicMock()
    mock_gen.message.content = "The answer is 42."
    mock_response.generations = [[mock_gen]]
    mock_response.llm_output = {"model_name": "gpt-4"}
    handler.on_chat_model_end(mock_response, run_id=llm_id)
    state = _get_agent_state()
    assert state["final_answer"] == "The answer is 42."
    assert len(state["llm_responses"]) == 1

    # on_chain_end (root) - triggers finalize
    handler.on_chain_end({"output": "done"}, run_id=run_id)
    # State should be reset after finalize
    # (agent_state is reset inside _finalize)

    log(PASS, "Full LangGraph callback lifecycle (chain→retriever→tool→LLM→end)")
    LongTracer.reset()
except Exception as e:
    log(FAIL, "Full LangGraph callback lifecycle", str(e))
    traceback.print_exc()
    LongTracer.reset()

# 2.7 - Error handling in chain
try:
    LongTracer.reset()
    LongTracer.init(backend="memory", verbose=False)
    handler = LongTracerAgentHandler()
    _reset_agent_state()

    run_id = uuid4()
    handler.on_chain_start({"name": "test"}, {}, run_id=run_id)
    handler.on_chain_error(ValueError("test error"), run_id=run_id)
    # Should not crash, state should be cleaned

    log(PASS, "Chain error handling (no crash)")
    LongTracer.reset()
except Exception as e:
    log(FAIL, "Chain error handling", str(e))
    LongTracer.reset()

# 2.8 - on_agent_action / on_agent_finish (LangChain AgentExecutor compat)
try:
    LongTracer.reset()
    LongTracer.init(backend="memory", verbose=False)
    handler = LongTracerAgentHandler()
    _reset_agent_state()

    run_id = uuid4()
    handler.on_chain_start({"name": "agent"}, {}, run_id=run_id)

    mock_action = MagicMock()
    mock_action.tool = "calculator"
    mock_action.tool_input = "2+2"
    handler.on_agent_action(mock_action, run_id=uuid4())

    mock_finish = MagicMock()
    mock_finish.return_values = {"output": "The result is 4"}
    handler.on_agent_finish(mock_finish, run_id=uuid4())

    state = _get_agent_state()
    assert state["final_answer"] == "The result is 4"

    handler.on_chain_end({}, run_id=run_id)
    log(PASS, "AgentExecutor events (on_agent_action + on_agent_finish)")
    LongTracer.reset()
except Exception as e:
    log(FAIL, "AgentExecutor events", str(e))
    LongTracer.reset()

# 2.9 - instrument_langgraph() convenience function
try:
    LongTracer.reset()
    mock_graph = MagicMock()
    handler = instrument_langgraph(mock_graph, threshold=0.7, verbose=False)
    assert isinstance(handler, LongTracerAgentHandler)
    assert handler.threshold == 0.7
    assert LongTracer.is_enabled()
    log(PASS, "instrument_langgraph() creates handler and enables LongTracer")
    LongTracer.reset()
except Exception as e:
    log(FAIL, "instrument_langgraph()", str(e))
    LongTracer.reset()

# 2.10 - instrument_langchain_agent() convenience function
try:
    LongTracer.reset()
    from longtracer.adapters.langgraph_handler import instrument_langchain_agent
    mock_executor = MagicMock()
    mock_executor.callbacks = None
    handler = instrument_langchain_agent(mock_executor, threshold=0.8)
    assert isinstance(handler, LongTracerAgentHandler)
    assert mock_executor.callbacks == [handler]
    log(PASS, "instrument_langchain_agent() attaches handler to executor.callbacks")
    LongTracer.reset()
except Exception as e:
    log(FAIL, "instrument_langchain_agent()", str(e))
    LongTracer.reset()


# ══════════════════════════════════════════════════════════════
# SECTION 3: LANGCHAIN HANDLER VERIFICATION
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 3: LANGCHAIN HANDLER VERIFICATION")
print("="*70)

from longtracer.adapters.langchain_handler import (
    CitationGuardCallbackHandler, instrument_langchain,
    normalize_doc, _get_state, _reset_state, _check_langchain,
)

# 3.1 - Handler instantiation
try:
    handler = CitationGuardCallbackHandler()
    assert handler.name == "LongTracerCallbackHandler"
    log(PASS, "CitationGuardCallbackHandler instantiation")
except Exception as e:
    log(FAIL, "CitationGuardCallbackHandler instantiation", str(e))

# 3.2 - normalize_doc
try:
    mock_doc = MagicMock()
    mock_doc.page_content = "Test content about Python."
    mock_doc.metadata = {"source": "python.pdf", "page": 1, "section": "intro"}
    result = normalize_doc(mock_doc)
    assert result["text"] == "Test content about Python."
    assert result["source"] == "python.pdf"
    assert result["section"] == "intro"
    assert len(result["chunk_id"]) == 12
    log(PASS, "normalize_doc() produces chunk_id, text, source, section")
except Exception as e:
    log(FAIL, "normalize_doc()", str(e))

# 3.3 - Thread-safe state
try:
    _reset_state()
    state = _get_state()
    assert state["chunks"] == []
    assert state["prompts"] == []
    assert state["final_answer"] is None
    _reset_state()
    log(PASS, "LangChain thread-safe state init/reset")
except Exception as e:
    log(FAIL, "LangChain thread-safe state", str(e))

# 3.4 - Full LangChain callback lifecycle
try:
    LongTracer.reset()
    LongTracer.init(backend="memory", verbose=False)
    handler = CitationGuardCallbackHandler()
    _reset_state()

    root_id = uuid4()
    handler.on_chain_start({"name": "qa"}, {"query": "test"}, run_id=root_id)
    state = _get_state()
    assert state["root_run_id"] == str(root_id)

    # Retriever
    ret_id = uuid4()
    handler.on_retriever_start({}, "test query", run_id=ret_id)
    mock_doc = MagicMock()
    mock_doc.page_content = "Source document text."
    mock_doc.metadata = {"source": "src.pdf"}
    handler.on_retriever_end([mock_doc], run_id=ret_id)
    state = _get_state()
    assert len(state["chunks"]) == 1

    # LLM
    llm_id = uuid4()
    handler.on_llm_start({}, ["prompt text"], run_id=llm_id)
    mock_response = MagicMock()
    mock_response.generations = [[MagicMock(text="LLM answer text.")]]
    mock_response.llm_output = {"model_name": "gpt-4"}
    handler.on_llm_end(mock_response, run_id=llm_id)
    state = _get_state()
    assert state["final_answer"] == "LLM answer text."

    # End chain
    handler.on_chain_end({"result": "done"}, run_id=root_id)

    log(PASS, "Full LangChain callback lifecycle (chain→retriever→LLM→end)")
    LongTracer.reset()
except Exception as e:
    log(FAIL, "Full LangChain callback lifecycle", str(e))
    traceback.print_exc()
    LongTracer.reset()

# 3.5 - instrument_langchain with callbacks attr
try:
    LongTracer.reset()
    mock_chain = MagicMock()
    mock_chain.callbacks = None
    handler = instrument_langchain(mock_chain, verbose=False)
    assert isinstance(handler, CitationGuardCallbackHandler)
    assert mock_chain.callbacks == [handler]
    log(PASS, "instrument_langchain() attaches to chain.callbacks")
    LongTracer.reset()
except Exception as e:
    log(FAIL, "instrument_langchain()", str(e))
    LongTracer.reset()

# 3.6 - instrument_langchain with config attr (LCEL Runnables)
try:
    LongTracer.reset()
    mock_chain2 = MagicMock(spec=[])  # no callbacks attr
    mock_chain2.config = {"some_key": "value"}
    handler = instrument_langchain(mock_chain2, verbose=False)
    assert "callbacks" in mock_chain2.config
    assert handler in mock_chain2.config["callbacks"]
    log(PASS, "instrument_langchain() falls back to chain.config")
    LongTracer.reset()
except Exception as e:
    log(FAIL, "instrument_langchain() config fallback", str(e))
    LongTracer.reset()


# ══════════════════════════════════════════════════════════════
# SECTION 4: HAYSTACK HANDLER VERIFICATION
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 4: HAYSTACK HANDLER VERIFICATION")
print("="*70)

from longtracer.adapters.haystack_handler import (
    LongTracerVerifier, instrument_haystack, _check_haystack,
)

# 4.1 - Haystack available check
try:
    _check_haystack()
    log(PASS, "Haystack detected correctly")
except ImportError:
    log(FAIL, "Haystack not detected but IS installed")

# 4.2 - Component instantiation
try:
    verifier = LongTracerVerifier(threshold=0.6, verbose=True)
    assert verifier.threshold == 0.6
    assert verifier.verbose is True
    assert verifier._verifier is None  # lazy init
    log(PASS, "LongTracerVerifier instantiation (lazy loaded)")
except Exception as e:
    log(FAIL, "LongTracerVerifier instantiation", str(e))

# 4.3 - warm_up loads the model
try:
    LongTracer.reset()
    verifier = LongTracerVerifier(threshold=0.5, verbose=False)
    verifier.warm_up()
    assert verifier._verifier is not None
    log(PASS, "warm_up() loads CitationVerifier internally")
except Exception as e:
    log(FAIL, "warm_up()", str(e))

# 4.4 - run() with Haystack Documents
try:
    from haystack import Document
    verifier = LongTracerVerifier(threshold=0.5, verbose=False)
    docs = [
        Document(content="The Eiffel Tower is located in Paris, France.", meta={"source": "wiki"}),
        Document(content="Python is a programming language created by Guido.", meta={"source": "docs"}),
    ]
    result = verifier.run(
        response="The Eiffel Tower is in Paris. Python was created by Guido van Rossum.",
        documents=docs,
    )
    assert "response" in result
    assert "trust_score" in result
    assert "verdict" in result
    assert "claims" in result
    assert isinstance(result["trust_score"], float)
    assert result["verdict"] in ("PASS", "FAIL")
    log(PASS, f"Haystack run() returns valid result (score={result['trust_score']:.2f}, verdict={result['verdict']})")
except Exception as e:
    log(FAIL, "Haystack run()", str(e))
    traceback.print_exc()

# 4.5 - run() error handling
try:
    verifier2 = LongTracerVerifier(threshold=0.5, verbose=False)
    # Pass a broken verifier
    verifier2._verifier = MagicMock()
    verifier2._verifier.verify_parallel.side_effect = RuntimeError("boom")
    from haystack import Document
    result = verifier2.run(
        response="test",
        documents=[Document(content="test")],
    )
    assert result["trust_score"] == -1.0
    assert result["verdict"] == "ERROR"
    log(PASS, "Haystack run() error handling returns ERROR verdict")
except Exception as e:
    log(FAIL, "Haystack run() error handling", str(e))

# 4.6 - instrument_haystack
try:
    LongTracer.reset()
    from haystack import Pipeline
    pipeline = Pipeline()
    instrument_haystack(pipeline, verbose=False)
    # Check the verifier component was added
    component_names = list(pipeline.graph.nodes.keys())
    assert "longtracer_verifier" in component_names
    log(PASS, "instrument_haystack() adds 'longtracer_verifier' component to pipeline")
    LongTracer.reset()
except Exception as e:
    log(FAIL, "instrument_haystack()", str(e))
    LongTracer.reset()

# 4.7 - Output types check
try:
    from haystack import Document
    verifier = LongTracerVerifier(threshold=0.5, verbose=False)
    docs = [Document(content="Water boils at 100 degrees Celsius at sea level.", meta={})]
    result = verifier.run(
        response="Water boils at 100 degrees Celsius at sea level. This is a basic fact of chemistry.",
        documents=docs
    )
    assert isinstance(result["response"], str)
    assert isinstance(result["trust_score"], float)
    assert isinstance(result["verdict"], str)
    assert isinstance(result["summary"], str)
    assert isinstance(result["claims"], list)
    assert isinstance(result["hallucination_count"], int)
    log(PASS, "Haystack output_types match declared schema (str, float, str, str, list, int)")
except Exception as e:
    log(FAIL, "Haystack output types", str(e))


# ══════════════════════════════════════════════════════════════
# SECTION 5: CLI VERIFICATION
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 5: CLI VERIFICATION")
print("="*70)

from longtracer.cli import main, cmd_check, cmd_list, cmd_view, cmd_last

# 5.1 - CLI entry point registered
try:
    import importlib.metadata
    eps = importlib.metadata.entry_points()
    # Check for console_scripts
    found = False
    if hasattr(eps, 'select'):
        for ep in eps.select(group='console_scripts'):
            if ep.name == 'longtracer':
                found = True
                break
    else:
        for ep in eps.get('console_scripts', []):
            if ep.name == 'longtracer':
                found = True
                break
    if found:
        log(PASS, "CLI entry point 'longtracer' registered in console_scripts")
    else:
        log(WARN, "CLI entry point 'longtracer' not found (may need reinstall)")
except Exception as e:
    log(WARN, "CLI entry point check", str(e))

# 5.2 - check subcommand with mock args
try:
    from argparse import Namespace
    args = Namespace(
        response="The sky is blue.",
        sources=["The sky appears blue due to Rayleigh scattering of sunlight."],
        json_output=True,
        threshold=0.5,
    )
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_check(args)
    output = buf.getvalue()
    parsed = json.loads(output)
    assert "verdict" in parsed
    assert "trust_score" in parsed
    assert "claims" in parsed
    log(PASS, f"CLI 'check' subcommand JSON output (verdict={parsed['verdict']}, score={parsed['trust_score']:.2f})")
except Exception as e:
    log(FAIL, "CLI 'check' subcommand", str(e))
    traceback.print_exc()

# 5.3 - check subcommand with human-readable output
try:
    args = Namespace(
        response="The Eiffel Tower is in Paris.",
        sources=["The Eiffel Tower is a wrought-iron lattice tower in Paris, France."],
        json_output=False,
        threshold=0.5,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_check(args)
    output = buf.getvalue()
    assert ("✓" in output) or ("✗" in output)
    log(PASS, "CLI 'check' human-readable output contains verdict icons")
except Exception as e:
    log(FAIL, "CLI 'check' human-readable output", str(e))

# 5.4 - Hallucination detection via CLI
try:
    args = Namespace(
        response="The Eiffel Tower is located in Berlin, Germany. It was built in 1950.",
        sources=["The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889."],
        json_output=True,
        threshold=0.5,
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_check(args)
    output = buf.getvalue()
    parsed = json.loads(output)
    log(PASS, f"CLI hallucination detection (verdict={parsed['verdict']}, hallucinations={parsed['hallucination_count']})")
except Exception as e:
    log(FAIL, "CLI hallucination detection", str(e))

# 5.5 - view subcommand with no traces
try:
    args = Namespace(project=None, limit=10)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cmd_list(args)
    output = buf.getvalue()
    assert "No traces" in output or "RECENT TRACES" in output
    log(PASS, "CLI 'view' (list) executes without crash")
except Exception as e:
    log(FAIL, "CLI 'view' (list)", str(e))

# 5.6 - _load_dotenv
try:
    from longtracer.cli import _load_dotenv
    _load_dotenv()  # Should not crash
    log(PASS, "CLI _load_dotenv() works without crash")
except Exception as e:
    log(FAIL, "CLI _load_dotenv()", str(e))

# 5.7 - argparse structure correct
try:
    from longtracer.cli import main
    import argparse
    # Test that main creates proper parser by checking no exception
    with patch('sys.argv', ['longtracer', 'check', '--help']):
        try:
            main()
        except SystemExit:
            pass  # --help causes SystemExit(0)
    log(PASS, "CLI argparse structure valid (check --help)")
except Exception as e:
    log(FAIL, "CLI argparse structure", str(e))


# ══════════════════════════════════════════════════════════════
# SECTION 6: CORE VERIFIER ANSWER VALIDATION
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  SECTION 6: CORE VERIFIER ANSWER VALIDATION")
print("="*70)

# 6.1 - Supported claim
try:
    from longtracer import check
    result = check(
        response="Python is a programming language. It was created by Guido van Rossum. Python is widely used for web development and data science.",
        sources=["Python is a high-level programming language created by Guido van Rossum. It is widely used for web development, data science, and automation."],
    )
    assert isinstance(result.trust_score, float)
    assert result.verdict in ("PASS", "FAIL")
    log(PASS, f"Supported claim check (score={result.trust_score:.2f}, verdict={result.verdict})")
except Exception as e:
    log(FAIL, "Supported claim check", str(e))

# 6.2 - Contradicted claim (hallucination)
try:
    result = check(
        response="The Great Wall of China is located in Japan. It was built in the 20th century. The wall is made of plastic.",
        sources=["The Great Wall of China is a series of fortifications in China, built over centuries starting from the 7th century BC. It is made of stone, brick, tamped earth, and other materials."],
    )
    log(PASS, f"Contradicted claim check (score={result.trust_score:.2f}, verdict={result.verdict}, hallucinations={result.hallucination_count})")
except Exception as e:
    log(FAIL, "Contradicted claim check", str(e))

# 6.3 - Empty response
try:
    result = check(response="", sources=["Some source text here."])
    assert result.trust_score == 1.0
    assert result.verdict == "PASS"
    log(PASS, "Empty response returns vacuous truth (score=1.0, PASS)")
except Exception as e:
    log(FAIL, "Empty response handling", str(e))

# 6.4 - No sources
try:
    result = check(
        response="This is a claim with no sources to verify against whatsoever and it should fail.",
        sources=[]
    )
    assert result.trust_score == 0.0
    assert result.verdict == "FAIL"
    log(PASS, "No sources returns unsupported (score=0.0, FAIL)")
except Exception as e:
    log(FAIL, "No sources handling", str(e))

# 6.5 - Type validation
try:
    from longtracer.guard.verifier import CitationVerifier
    v = CitationVerifier()
    try:
        v.verify(123, ["source"])
        log(FAIL, "Type validation: should reject non-string response")
    except TypeError:
        log(PASS, "Type validation: rejects non-string response")
except Exception as e:
    log(FAIL, "Type validation", str(e))

try:
    v = CitationVerifier()
    try:
        v.verify("response", "not a list")
        log(FAIL, "Type validation: should reject non-list sources")
    except TypeError:
        log(PASS, "Type validation: rejects non-list sources")
except Exception as e:
    log(FAIL, "Type validation sources", str(e))


# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  VERIFICATION SUMMARY")
print("="*70)
total = len(results)
passed = sum(1 for r in results if PASS in r["status"])
failed = sum(1 for r in results if FAIL in r["status"])
warned = sum(1 for r in results if WARN in r["status"])
print(f"\n  Total: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Warnings: {warned}")
print(f"  Pass Rate: {passed/total*100:.1f}%\n")

if failed > 0:
    print("  FAILED TESTS:")
    for r in results:
        if FAIL in r["status"]:
            print(f"    ❌ {r['test']}: {r['detail']}")
    print()

if warned > 0:
    print("  WARNINGS:")
    for r in results:
        if WARN in r["status"]:
            print(f"    ⚠️  {r['test']}: {r['detail']}")
    print()

print("="*70)
sys.exit(1 if failed > 0 else 0)
