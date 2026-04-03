from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings, Document, VectorStoreIndex

from longtracer import LongTracer, instrument_llamaindex

import os
from dotenv import load_dotenv
load_dotenv()
if "GOOGLE_API_KEY" in os.environ and "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]

# 1. Initialize LongTracer
LongTracer.init(verbose=True)

# 2. Configure LlamaIndex to use Gemini
print("Initializing Gemini models with google_genai...")
llm = GoogleGenAI(model="gemini-2.5-flash")
embed_model = GoogleGenAIEmbedding(model_name="gemini-embedding-001")

Settings.llm = llm
Settings.embed_model = embed_model

# 3. Create a basic VectorStoreIndex
docs = [
    Document(text="LongTracer is a robust platform designed to analyze, trace, and instrument Large Language Model pipelines."),
    Document(text="LlamaIndex is an advanced data framework designed to connect custom data sources to large language models for applications like RAG.")
]

print("Building vector index...")
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()

# 4. Instrument real query engine with LongTracer
print("Instrumenting LlamaIndex with LongTracer...")
instrument_llamaindex(query_engine)

# 5. Run a query to test the tracing
print("\nQuerying...")
response = query_engine.query("What is LongTracer and how does it compare to LlamaIndex?")

print("\n==== Response ====")
print(response)
print("==================")