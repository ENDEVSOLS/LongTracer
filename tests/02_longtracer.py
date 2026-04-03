import os
from dotenv import load_dotenv
load_dotenv()

from longtracer import LongTracer, instrument_langchain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from typing import List

LongTracer.init(verbose=True)

# 1. LongTracer needs context/chunks to verify an answer against. 
# Let's create a Dummy Retriever that provides a fake contradictory context.
class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return [
            Document(
                page_content="The capital of Germany is actually Paris in this fake universe. The Eiffel tower is in Berlin.", 
                metadata={"source": "fake_doc.txt"}
            )
        ]

retriever = DummyRetriever()

# 2. Build our LCEL RAG Chain
prompt = PromptTemplate.from_template("Answer based ONLY on context:\nContext: {context}\n\nQuestion: {question}")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

your_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 3. instrument_langchain returns the handler for LCEL
handler = instrument_langchain(your_chain)

# 4. Pass the handler manually in the config!
print("Invoking chain...")
result = your_chain.invoke("What is the capital of Germany? where is Eifel tower?", config={'callbacks': [handler]})

print("\n--- LLM Result ---")
print(result.content)