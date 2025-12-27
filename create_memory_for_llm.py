import os
import time
from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm  # You need to install this: uv pip install tqdm

# 1. Load Environment Variables
load_dotenv()

# Verify API Key
if not os.getenv("PINECONE_API_KEY"):
    raise ValueError("PINECONE_API_KEY is missing. Check your .env file.")

# Configuration
DATA_PATH = "Data/"
INDEX_NAME = "education-expert"

# --- Step 1: Load Raw PDFs ---
def load_pdf_files(data_path):
    print(f"Loading PDFs from {data_path}...")
    loader = DirectoryLoader(data_path,
                             glob='*.pdf',
                             loader_cls=PyMuPDFLoader)
    documents = loader.load()
    return documents

# --- Step 2: Create Chunks ---
def create_chunks(extracted_data):
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# --- Step 3: Get Embedding Model ---
def get_embedding_model():
    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# --- Main Execution Flow ---
if __name__ == "__main__":
    # A. Load Documents
    documents = load_pdf_files(DATA_PATH)
    print(f"Length of PDF pages: {len(documents)}")

    # B. Chunk Documents
    text_chunks = create_chunks(documents)
    total_chunks = len(text_chunks)
    print(f"Length of text chunks: {total_chunks}")

    # C. Initialize Embeddings
    embedding_model = get_embedding_model()

    # D. Initialize Pinecone Vector Store (Empty first)
    print(f"Connecting to Pinecone index '{INDEX_NAME}'...")
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embedding_model
    )

    # E. Batch Upload (The Fix!)
    BATCH_SIZE = 100
    print(f"Uploading {total_chunks} chunks in batches of {BATCH_SIZE}...")

    # Using tqdm for a progress bar
    for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Uploading Batches"):
        batch = text_chunks[i : i + BATCH_SIZE]
        vector_store.add_documents(batch)
        # Optional: Sleep for 0.5s to be nice to the API limits
        time.sleep(0.5)
    
    print("✅ Success! All data uploaded to Pinecone cloud.")