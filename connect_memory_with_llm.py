import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 1. Page Configuration
st.set_page_config(
    page_title="Education Expert",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 NCERT Education Expert")

# 2. Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY is missing! Please check your .env file.")
    st.stop()

# 3. Sidebar: Mode Selection
with st.sidebar:
    st.header("⚙️ Mode Selection")
    mode = st.radio(
        "Choose your assistant:",
        ["📚 Concept Search (RAG)", "📝 Step-by-Step Math Tutor"],
        captions=["Best for theory & searching notes", "Best for solving problems with full steps"]
    )
    st.divider()
    st.info("Tip: The 'Math Tutor' uses Llama 3.3 to generate textbook-style solutions.")

# 4. Cache Vector Store (Only needed for RAG)
@st.cache_resource
def load_vectorstore():
    DB_FAISS_PATH = "vectorstore/db_faiss"
    if not os.path.exists(DB_FAISS_PATH):
        return None
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        return None

# 5. Define the Math Tutor Function (The Fix)
def get_math_solution(question):
    """
    Uses Llama 3.3 to generate a structured, step-by-step mathematical explanation.
    """
    # Initialize the Smart Model
    llm_math = ChatGroq(
        model="llama-3.3-70b-versatile", # powerful reasoning model
        temperature=0.1, # Low temperature for precision
        api_key=GROQ_API_KEY
    )

    # The "Teacher" System Prompt
    system_prompt = """
    You are an expert Math Tutor for Grade 9-10 students. 
    Your goal is to solve the problem analytically and explain every step clearly.

    ### FORMATTING RULES:
    1. **Structure:** Use 'Step 1', 'Step 2', etc., headings.
    2. **Math:** Use LaTeX formatting for all equations. Enclose them in double dollar signs ($$).
       - Example: $$x^2 + y^2 = r^2$$
    3. **Explanation:** Briefly explain the theorem or logic used (e.g., "Pythagoras theorem").
    4. **Final Answer:** State the final result clearly at the end and final result can be more than one give answer based the question asked.

    ### RESTRICTIONS:
    - Do NOT write Python code.
    - Do NOT give a direct answer without steps.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    chain = prompt | llm_math
    
    try:
        response = chain.invoke({"input": question})
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# 6. Initialize RAG Chain (Only if in RAG mode)
if mode == "📚 Concept Search (RAG)":
    vector_store = load_vectorstore()
    if vector_store:
        llm_rag = ChatGroq(
            model="llama-3.1-8b-instant", # Fast model for reading text
            temperature=0.3,
            api_key=GROQ_API_KEY
        )
        
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer based ONLY on context:\n{context}"),
            ("human", "{input}"),
        ])
        
        combine_docs = create_stuff_documents_chain(llm_rag, rag_prompt)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        rag_chain = create_retrieval_chain(retriever, combine_docs)

# 7. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 8. Handle User Input
if user_query := st.chat_input("Ask a question..."):
    
    # Display User Message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate Response
    with st.chat_message("assistant"):
        
        # --- MODE 1: MATH TUTOR (Structured Output) ---
        if mode == "📝 Step-by-Step Math Tutor":
            with st.spinner("📐 Solving step-by-step..."):
                answer = get_math_solution(user_query)
                st.markdown(answer) # Markdown renders the LaTeX $$ automatically
                st.session_state.messages.append({"role": "assistant", "content": answer})

        # --- MODE 2: RAG SEARCH ---
        else:
            if not load_vectorstore():
                st.error("Vector DB not found. Please ingest documents.")
            else:
                with st.spinner("📖 Searching documents..."):
                    try:
                        response = rag_chain.invoke({'input': user_query})
                        st.markdown(response["answer"])
                        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    except Exception as e:
                        st.error(f"Error: {e}")