from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

script_current_dir = Path(__file__).parent
# Define the path and collection name
chroma_vectordb_dir = script_current_dir.parent / 'chroma_db'
COLLECTION_NAME = "nike-10k-2023"

# Define the *same* embedding function used during creation
embeddings = CohereEmbeddings(model="embed-english-v3.0") # Or another embedding model (e.g., HuggingFaceEmbeddings)

# Load the vector store from the persistent directory
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=chroma_vectordb_dir
)

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# from typing import Literal

# def retrieve_context(query: str, section: Literal["beginning", "middle", "end"]):

from langchain.agents import create_agent


tools = [retrieve_context]
# If desired, specify custom instructions
custom_system_prompt = (
    "You have access to a tool that retrieves context from a pdf file. "
    "Use the tool to help answer user queries."
)

llm_model = ChatGroq(model="openai/gpt-oss-20b")
agent = create_agent(llm_model, tools, system_prompt=custom_system_prompt)

UserMessage = "What was Nike's revenue in 2023?"

result = agent.invoke({
    "messages": UserMessage
})

output_string = result["messages"][-1].content
print(f"\nUser Question: {UserMessage}")
print(f"\nAI Answer based on RAG Context: {output_string}\n")