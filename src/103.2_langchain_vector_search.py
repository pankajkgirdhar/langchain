from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
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


results = vector_store.similarity_search_with_relevance_scores(query="What was Nike's revenue in 2023?", k=3)
print(results)

# results = vector_store.similarity_search(query="What was Nike's revenue in 2023?", k=2)
# print(results)
