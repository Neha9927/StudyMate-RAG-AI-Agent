import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Setup
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore.from_existing_index(
    index_name="education-expert",
    embedding=embeddings
)

# Test Search
print("Testing connection...")
results = vector_store.similarity_search("Newton's second law", k=3)

if len(results) == 0:
    print("❌ Connection success, but NO documents found matching the query.")
else:
    print(f"✅ Success! Found {len(results)} documents.")
    print(f"Content preview: {results[0].page_content[:200]}")