import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_cohere import CohereEmbeddings
from datetime import datetime
from langchain.agents import create_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass

load_dotenv()
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma

script_current_dir = Path(__file__).parent
file_path = script_current_dir.parent / 'data' / 'nike-10k-2023.pdf'
print(file_path)


loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))

embeddings = CohereEmbeddings(model="embed-english-v3.0")
#embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

#########################   commented Section is to test Embeddings generation  - START #########################
# vector_1 = embeddings.embed_query(all_splits[0].page_content)
# vector_2 = embeddings.embed_query(all_splits[1].page_content)

# assert len(vector_1) == len(vector_2)
# print(f"Generated vectors of length {len(vector_1)}\n")
# print(vector_1[:10])
#########################   commented Section is to test Embeddings generation  - END #########################


chroma_vectordb_dir = script_current_dir.parent / 'chroma_db'

vector_store = Chroma(
    collection_name="nike-10k-2023",
    embedding_function=embeddings,
    persist_directory=chroma_vectordb_dir,  # Where to save data locally, remove if not necessary
)

ids = vector_store.add_documents(documents=all_splits)

