from longtracer.guard.verifier import CitationVerifier
import json

print("Initializing CitationVerifier...")
verifier = CitationVerifier()

print("\n" + "="*50)
print("1. Working with a Custom RAG Pipeline")
print("="*50)

# Imagine this is your custom python RAG code
def my_custom_rag_pipeline(query: str):
    # Does retrieval...
    retrieved_docs = [
        {"content": "Paris is the capital of France. It is known for the Eiffel Tower.", "file": "geography.txt"},
        {"content": "The Eiffel Tower was constructed in 1889.", "file": "history.txt"}
    ]
    # Does generation...
    llm_response = "The capital of France is Paris, and it features the Eiffel Tower which was built in 1889."
    
    return llm_response, retrieved_docs

response_text, docs = my_custom_rag_pipeline("Tell me about Paris")

# Just pass the strings directly to LongTracer
result_custom = verifier.verify_parallel(
    response=response_text,
    sources=[doc["content"] for doc in docs],
    source_metadata=[{"source": doc["file"]} for doc in docs]
)
print(f"Response: {response_text}")
print("Verification Result:")
# result_custom is a Pydantic model (VerificationResult) or Dict, so we can convert it or print it.
# Usually, models print nicely or can be dumped to dict
if hasattr(result_custom, "model_dump"):
    print(json.dumps(result_custom.model_dump(), indent=2))
else:
    print(result_custom)


print("\n" + "="*50)
print("2. Working with simulated Haystack output")
print("="*50)

# In Haystack v2, a RAG pipeline might return a result like this:
class HaystackDocument:
    def __init__(self, content, meta):
        self.content = content
        self.meta = meta

haystack_output = {
    "llm": {
        "replies": ["The speed of light is approximately 299,792 kilometers per second in a vacuum."]
    },
    "retriever": {
        "documents": [
            HaystackDocument("In a vacuum, light travels at 299,792 km/s.", {"url": "physics.com"})
        ]
    }
}

# Extract strings and metadata
haystack_reply = haystack_output["llm"]["replies"][0]
haystack_docs = haystack_output["retriever"]["documents"]

result_haystack = verifier.verify_parallel(
    response=haystack_reply,
    sources=[doc.content for doc in haystack_docs],
    source_metadata=[doc.meta for doc in haystack_docs]
)
print(f"Response: {haystack_reply}")
print("Verification Result:")
if hasattr(result_haystack, "model_dump"):
    print(json.dumps(result_haystack.model_dump(), indent=2))
else:
    print(result_haystack)