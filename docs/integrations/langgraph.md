# LangGraph & LangChain Agent Integration

LongTracer provides a single callback handler that traces and verifies agent executions across the entire LangChain/LangGraph ecosystem.

## Install

```bash
pip install "longtracer[langgraph]"
```

## Supported Agent Patterns

| Pattern | Function |
|---------|----------|
| LangGraph `create_react_agent` | `instrument_langgraph(graph)` |
| LangGraph custom `StateGraph` | `instrument_langgraph(graph)` |
| LangGraph Functional API | Pass handler in `config` |
| LangChain `AgentExecutor` | `instrument_langchain_agent(executor)` |
| LangChain `create_react_agent` | `instrument_langchain_agent(executor)` |
| LangChain `create_tool_calling_agent` | `instrument_langchain_agent(executor)` |

---

## LangGraph `create_react_agent`

The most common LangGraph pattern:

```python
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from longtracer import LongTracer, instrument_langgraph

LongTracer.init(verbose=True)

llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, tools=[retriever_tool, search_tool])

# Get the handler
handler = instrument_langgraph(agent)

# Pass handler in config
result = agent.invoke(
    {"messages": [("user", "What is the capital of France?")]},
    config={"callbacks": [handler]}
)
```

## LangGraph Custom StateGraph

```python
from langgraph.graph import StateGraph, START, END
from longtracer import instrument_langgraph

handler = instrument_langgraph(graph)

app = graph.compile()
result = app.invoke(
    {"messages": [("user", "Summarize the document")]},
    config={"callbacks": [handler]}
)
```

## LangGraph Streaming

```python
handler = instrument_langgraph(agent)

for chunk in agent.stream(
    {"messages": [("user", "What is X?")]},
    config={"callbacks": [handler]}
):
    for node, update in chunk.items():
        print(f"Node: {node}")
# Verification runs automatically at stream end
```

## LangChain AgentExecutor

```python
from langchain.agents import AgentExecutor, create_react_agent
from longtracer import instrument_langchain_agent

handler = instrument_langchain_agent(agent_executor)
result = agent_executor.invoke({"input": "What is X?"})
```

## LangChain Tool-Calling Agent

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from longtracer import instrument_langchain_agent

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

handler = instrument_langchain_agent(executor)
result = executor.invoke({"input": "Search for recent news"})
```

## Direct Handler Usage

For maximum control, use the handler class directly:

```python
from longtracer.adapters.langgraph_handler import LongTracerAgentHandler

handler = LongTracerAgentHandler(threshold=0.7)

# Works with any LangChain/LangGraph Runnable
result = any_runnable.invoke(
    input,
    config={"callbacks": [handler]}
)
```

---

## What Gets Captured

| Span | What it records |
|------|----------------|
| `retrieval` | Retrieved documents, count, latency |
| `tool_call` | Tool name, input, output, latency |
| `llm_call` | LLM response, model name, latency |
| `agent_action` | Agent's decision to call a tool |
| `eval_claims` | Per-claim verification results |
| `grounding` | Trust score, verdict, hallucination count |

## How It Works

1. The handler accumulates all retrieved documents across agent steps
2. It captures every tool call and LLM response as spans
3. When the agent finishes, it runs STS + NLI verification on the final answer against all accumulated sources
4. Results are logged as `eval_claims` and `grounding` spans

This means verification adds zero overhead during agent execution — it only runs once at the end.

## Viewing Results

```bash
longtracer view --last
longtracer view --html <trace_id>
```

## Notes

- Sources are accumulated across multiple retriever/tool calls — multi-step agents are fully supported
- Verification runs once at agent finish, not after every intermediate LLM call
- Thread-safe: concurrent agent invocations don't interfere
- A failing verification never crashes your agent — all errors are logged as warnings
- Works with both sync and streaming execution modes
