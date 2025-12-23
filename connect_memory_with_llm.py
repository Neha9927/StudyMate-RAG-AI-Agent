import streamlit as st
import os
from dotenv import load_dotenv

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration
st.set_page_config(
    page_title="Education Expert",
    page_icon="📚",
    layout="centered"
)

st.title("🎓 NCERT Education Expert")

# 2. Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY is missing! Please check your .env file.")
    st.stop()

# 3. Cache the Heavy Resource (Vector Database)
# This prevents reloading the database on every user interaction
@st.cache_resource
def load_vectorstore():
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        return None

# Load the DB
vector_store = load_vectorstore()

if vector_store is None:
    st.error("Failed to load vector store. Please ensure the 'vectorstore/db_faiss' folder exists.")
    st.stop()

# 4. Initialize LLM and Prompts
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=512,
    api_key=GROQ_API_KEY
)

system_prompt = (
    "You are a precise technical assistant. Answer the user's question based ONLY on the following context:\n\n"
    "{context}\n\n"
    "### INSTRUCTIONS:\n"
    "1. **Direct Answer:** Start with a clear summary sentence.\n"
    "2. **Formatting:** Use valid Markdown. Use Bullet points for lists.\n"
    "3. **No Fluff:** Do not say 'According to the context' or 'The document says'. Just state the facts.\n"
    "4. **Missing Data:** If the answer is not in the context, say exactly: 'Data not available in the provided documents.'"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Create the RAG Chain
combine_doc_chains = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
rag_chain = create_retrieval_chain(retriever, combine_doc_chains)

# 6. Chat Interface & History
# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. Handle User Input
if user_query := st.chat_input("Ask a question from 9th and 10th PCM..."):
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                response = rag_chain.invoke({'input': user_query})
                answer = response["answer"]
                
                # Display text answer
                st.markdown(answer)
                
                # Optional: Expandable Source viewer
                with st.expander("🔍 View Source Documents"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                        st.caption(f"{doc.page_content[:300]}...")
                        st.divider()

                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            
            except Exception as e:
                st.error(f"An error occurred: {e}")