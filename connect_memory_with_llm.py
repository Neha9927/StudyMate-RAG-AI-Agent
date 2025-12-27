import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore  # <--- CHANGED THIS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 1. Page Configuration
st.set_page_config(
    page_title="Education Expert",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Education Expert")

# 2. Load Environment Variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY") # <--- ADDED THIS

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY is missing! Please check your .env file.")
    st.stop()

if not PINECONE_API_KEY:
    st.error("⚠️ PINECONE_API_KEY is missing! Please check your .env file.")
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

# 4. Connect to Pinecone Vector Store (UPDATED FOR CLOUD)
@st.cache_resource
def load_vectorstore():
    # We no longer look for a local path like "vectorstore/db_faiss"
    try:
        # 1. Define the Embedding Model (Must match what you uploaded!)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 2. Connect to the Pinecone Cloud Index
        vector_store = PineconeVectorStore.from_existing_index(
            index_name="education-expert",
            embedding=embedding_model
        )
        return vector_store
    except Exception as e:
        st.error(f"Failed to connect to Pinecone: {e}")
        return None

# 5. Define the Math Tutor Function
def get_math_solution(question, chat_history):
    """
    Uses Llama 3.3 to generate a structured, step-by-step mathematical explanation
    Safely handles LaTeX formatting using MessagesPlaceholder.
    """
    # Initialize the Smart Model
    llm_math = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=GROQ_API_KEY
    )

    # 1. The "Teacher" System Prompt
    system_prompt_text = """
    You are an expert Math Tutor for Grade 9-10 students. 
    Your goal is to solve the problem analytically and explain every step clearly.
    
    ### FORMATTING RULES:
    1. **Structure:** Use 'Step 1', 'Step 2', etc., headings.
    2. **Math:** Use LaTeX formatting for all equations. Enclose them in double dollar signs ($$).
       - Example: $$x^2 + y^2 = r^2$$
    3. **Explanation:** Briefly explain the theorem or logic used.
    4. **Final Answer:** State the final result clearly at the end.
    
    ### MEMORY:
    - If the user asks a follow-up question, refer to the conversation history.
    """

    # 2. Create Prompt Template using Placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        MessagesPlaceholder(variable_name="chat_history"), 
        ("human", "{input}"),
    ])

    chain = prompt | llm_math
    
    # 3. Convert Session State History to LangChain Message Objects
    history_buffer = []
    for msg in chat_history:
        if msg["role"] == "user":
            history_buffer.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_buffer.append(AIMessage(content=msg["content"]))
    
    try:
        response = chain.invoke({
            "chat_history": history_buffer,
            "input": question
        })
        return response.content
    except Exception as e:
        return f"Error: {str(e)}"

# 6. Initialize RAG Chain
if mode == "📚 Concept Search (RAG)":
    vector_store = load_vectorstore()
    if vector_store:
        llm_rag = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=GROQ_API_KEY
        )

        # A. History-Aware Retriever
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
if user_query := st.chat_input("Ask a question from 9th or 10th Grade Math/Science..."):
    
    # Display User Message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate Response
    with st.chat_message("assistant"):
        
        # --- MODE 1: MATH TUTOR (Structured Output) ---
        if mode == "📝 Step-by-Step Math Tutor":
            with st.spinner("📐 Solving step-by-step..."):
                answer = get_math_solution(user_query, st.session_state.messages)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

        # --- MODE 2: RAG SEARCH ---
        else:
            if not load_vectorstore():
                st.error("Vector DB connection failed.")
            else:
                with st.spinner("📖 Searching documents..."):
                    try:
                        chat_history_lc = []
                        for msg in st.session_state.messages[:-1]:
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