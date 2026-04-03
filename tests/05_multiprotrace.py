import os
import json
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from typing import List

from longtracer import LongTracer, instrument_langchain

# Load environment variables
load_dotenv()

# Initialize LongTracer for two different applications/projects.
LongTracer.init(project_name="search-api", backend="sqlite", verbose=False)
LongTracer.init(project_name="chatbot-prod", backend="sqlite", verbose=False)

# Get project-specific tracers exactly as requested
chatbot = LongTracer.get_tracer("chatbot-prod")
search  = LongTracer.get_tracer("search-api")

class DummySearchRetriever(BaseRetriever):
    """
    Demonstrates manual tracing for a secondary project ('search-api').
    """
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # Start a root trace for the search sub-system
        search.start_root(inputs={"query": query})
        docs = []
        try:
            with search.span("fetch_documents", run_type="retriever") as span:
                docs = [
                    Document(
                        page_content="LangChain and LongTracer can be integrated seamlessly for multi-project monitoring.",
                        metadata={"source": "docs.txt"}
                    )
                ]
                span.set_output({"doc_count": len(docs)})
                return docs
        finally:
            search.end_root(outputs={"status": "success", "docs_returned": len(docs)})

# 1. Create the Chatbot with LangChain
retriever = DummySearchRetriever()
prompt = PromptTemplate.from_template(
    "You are a helpful chatbot. Based on the context below, answer the question.\n\nContext: {context}\n\nQuestion: {question}"
)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chatbot_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 2. Instrument the LangChain Chatbot
# We ensure 'chatbot-prod' is the default so the handler attaches to it
LongTracer._default_project = "chatbot-prod"
handler = instrument_langchain(chatbot_chain)

if __name__ == "__main__":
    print("Starting multi-project tracing demo...\n")
    query = "How do LangChain and LongTracer work together for multiple projects?"
    
    # Explicitly start the root trace for the chatbot project
    chatbot.start_root(inputs={"query": query})
    try:
        result = chatbot_chain.invoke(query, config={'callbacks': [handler]})
    finally:
        chatbot.end_root(outputs={"answer": result.content})
    
    print("==== Chatbot Output ====")
    print(result.content)
    print("=========================\n")
    
    print("Fetching recorded traces from the SQLite backend...\n")
    
    # Fetch recent traces for both projects to show they are recorded
    chatbot_traces = chatbot.list_recent_traces(limit=2, project_name="chatbot-prod")
    search_traces = search.list_recent_traces(limit=2, project_name="search-api")
    
    print("--- CHATBOT TRACES (chatbot.list_recent_traces) ---")
    for t in chatbot_traces:
        print(f"Project: {t.get('project_name')} | Trace ID: {t.get('trace_id')}")
        print(f"Inputs: {t.get('inputs')}")
        print(f"Outputs: {t.get('outputs')}")
        print("-" * 20)
        
    print("\n--- SEARCH API TRACES (search.list_recent_traces) ---")
    for t in search_traces:
        print(f"Project: {t.get('project_name')} | Trace ID: {t.get('trace_id')}")
        print(f"Inputs: {t.get('inputs')}")
        print(f"Outputs: {t.get('outputs')}")
        print("-" * 20)