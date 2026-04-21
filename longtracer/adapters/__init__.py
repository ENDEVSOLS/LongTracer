"""
LongTracer Adapters.

Framework-specific instrumentation functions.
Imports are lazy — importing this module does NOT require any framework.
"""


def __getattr__(name: str):
    """Lazy attribute access for adapter classes."""
    if name in ("CitationGuardCallbackHandler", "instrument_langchain"):
        from longtracer.adapters.langchain_handler import (
            CitationGuardCallbackHandler,
            instrument_langchain,
        )
        return CitationGuardCallbackHandler if name == "CitationGuardCallbackHandler" else instrument_langchain

    if name in ("CitationGuardLlamaIndexHandler", "instrument_llamaindex"):
        from longtracer.adapters.llamaindex_handler import (
            CitationGuardLlamaIndexHandler,
            instrument_llamaindex,
        )
        return CitationGuardLlamaIndexHandler if name == "CitationGuardLlamaIndexHandler" else instrument_llamaindex

    if name in ("LongTracerVerifier", "instrument_haystack"):
        from longtracer.adapters.haystack_handler import (
            LongTracerVerifier,
            instrument_haystack,
        )
        return LongTracerVerifier if name == "LongTracerVerifier" else instrument_haystack

    if name in ("LongTracerAgentHandler", "instrument_langgraph", "instrument_langchain_agent"):
        from longtracer.adapters.langgraph_handler import (
            LongTracerAgentHandler,
            instrument_langgraph,
            instrument_langchain_agent,
        )
        mapping = {
            "LongTracerAgentHandler": LongTracerAgentHandler,
            "instrument_langgraph": instrument_langgraph,
            "instrument_langchain_agent": instrument_langchain_agent,
        }
        return mapping[name]

    if name in ("instrument_openai_assistant", "verify_assistant_run"):
        from longtracer.adapters.openai_handler import (
            instrument_openai_assistant,
            verify_assistant_run,
        )
        return instrument_openai_assistant if name == "instrument_openai_assistant" else verify_assistant_run

    if name in ("instrument_crewai", "verify_crew_output"):
        from longtracer.adapters.crewai_handler import (
            instrument_crewai,
            verify_crew_output,
        )
        return instrument_crewai if name == "instrument_crewai" else verify_crew_output

    if name in ("instrument_autogen", "verify_autogen_result"):
        from longtracer.adapters.autogen_handler import (
            instrument_autogen,
            verify_autogen_result,
        )
        return instrument_autogen if name == "instrument_autogen" else verify_autogen_result

    raise AttributeError(f"module 'longtracer.adapters' has no attribute {name!r}")


__all__ = [
    # LangChain
    "CitationGuardCallbackHandler",
    "instrument_langchain",
    # LlamaIndex
    "CitationGuardLlamaIndexHandler",
    "instrument_llamaindex",
    # Haystack
    "LongTracerVerifier",
    "instrument_haystack",
    # LangGraph / LangChain Agent
    "LongTracerAgentHandler",
    "instrument_langgraph",
    "instrument_langchain_agent",
    # OpenAI Assistants
    "instrument_openai_assistant",
    "verify_assistant_run",
    # CrewAI
    "instrument_crewai",
    "verify_crew_output",
    # AutoGen
    "instrument_autogen",
    "verify_autogen_result",
]
