import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

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

# 5. Define the Math Tutor Function (UPDATED FOR MEMORY)
def get_math_solution(question, chat_history):
    """
    Uses Llama 3.3 to generate a structured, step-by-step mathematical explanation
    while remembering previous context.
    """
    # Initialize the Smart Model
    llm_math = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
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
    4. **Final Answer:** State the final result clearly at the end.
    
    ### MEMORY:
    - If the user asks a follow-up question (e.g., "How did you get step 2?"), refer to the previous conversation history provided.
    """

    # Build the conversation list
    messages = [("system", system_prompt)]
    
    # Add history
    for msg in chat_history:
        role = "ai" if msg["role"] == "assistant" else "human"
        messages.append((role, msg["content"]))
    
    # Add new question
    messages.append(("human", question))

    # Create Prompt & Chain
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | llm_math
    
    try:
        response = chain.invoke({})
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# 6. Initialize RAG Chain (UPDATED FOR MEMORY)
if mode == "📚 Concept Search (RAG)":
    vector_store = load_vectorstore()
    if vector_store:
        llm_rag = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )

        # A. History-Aware Retriever
        # (This rewrites the user's question to include context from history)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm_rag, retriever, contextualize_q_prompt
        )

        # B. Answer Generation Chain
        system_prompt_rag = (
            "You are a precise technical assistant. Answer the user's question based ONLY on the following context:\n\n"
            "{context}\n\n"
            "If the answer is not in the context, say: 'Data not available in the provided documents.'"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_rag),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm_rag, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

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
                # Pass the full chat history!
                answer = get_math_solution(user_query, st.session_state.messages)
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

        # --- MODE 2: RAG SEARCH ---
        else:
            if not load_vectorstore():
                st.error("Vector DB not found. Please ingest documents.")
            else:
                with st.spinner("📖 Searching documents..."):
                    try:
                        # Convert session state to LangChain format for RAG
                        chat_history_lc = []
                        for msg in st.session_state.messages[:-1]: # Skip the latest msg (it's in 'input')
                            if msg["role"] == "user":
                                chat_history_lc.append(HumanMessage(content=msg["content"]))
                            else:
                                chat_history_lc.append(AIMessage(content=msg["content"]))
                        
                        response = rag_chain.invoke({
                            "input": user_query,
                            "chat_history": chat_history_lc
                        })
                        
                        st.markdown(response["answer"])
                        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    except Exception as e:
                        st.error(f"Error: {e}")