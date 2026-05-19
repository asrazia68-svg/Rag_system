import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# --- 1. CHANGES: Title, Icon aur Layout ko WIDE kiya taake sidebar fit ho sakay ---
st.set_page_config(page_title="DocuQuery AI", page_icon="📝", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b0c16 !important;
    color: #e8e8f0 !important;
}
.stApp { background: #0b0c16; }
#MainMenu, footer {visibility: hidden;}

/* Sidebar Background Aura Styling */
[data-testid="stSidebar"] {
    background-color: #0f1123 !important;
    border-right: 1px solid #1f2347;
}

[data-testid="stChatMessage"] {
    background: #1c1c27 !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 16px !important;
    padding: 14px 18px !important;
    margin-bottom: 12px !important;
    color: #e8e8f0 !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div {
    color: #e8e8f0 !important;
}
[data-testid="stChatInputTextArea"] {
    background: #1c1c27 !important;
    color: #e8e8f0 !important;
}
.stChatInputContainer {
    background: #1c1c27 !important;
    border: 1px solid #2a2a3d !important;
    border-radius: 14px !important;
}
.stChatInputContainer:focus-within {
    border-color: #7c6aff !important;
    box-shadow: 0 0 0 3px rgba(124,106,255,0.12) !important;
}

/* Premium Gradient Button Style */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: bold !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5) !important;
}

/* Tech Badges Styling */
code {
    background-color: #1e1b4b !important;
    color: #a5b4fc !important;
    border: 1px solid #312e81 !important;
    padding: 4px 8px !important;
    border-radius: 6px !important;
}

.online-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.3);
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.75rem; color: #4ade80; font-weight: 500;
}
.dot {
    width: 7px; height: 7px; background: #4ade80;
    border-radius: 50%; box-shadow: 0 0 6px #4ade80;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
.logo-area { display:flex; align-items:center; gap:12px; padding:10px 0 24px 0; }
.logo-icon {
    width:42px; height:42px;
    background:linear-gradient(135deg,#4f46e5,#7c3aed);
    border-radius:12px; display:flex; align-items:center;
    justify-content:center; font-size:20px;
}
.logo-title {
    font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800;
    background:linear-gradient(135deg,#ffffff,#a5b4fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;
}
.logo-sub { font-size:0.75rem; color:#6b6b8a; margin:0; }
.divider { border:none; border-top:1px solid #2a2a3d; margin:8px 0 20px 0; }
.hint { text-align:center; font-size:0.72rem; color:#6b6b8a; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

# --- 2. CHANGES: Left Sidebar bana kar us mein File Uploader aur Tech Stack shift kiya ---
with st.sidebar:
    st.markdown("""
    <div class="logo-area">
        <div class="logo-icon">📝</div>
        <div>
            <p class="logo-title">DocuQuery AI</p>
            <p class="logo-sub">Intelligent Document Search</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    st.markdown("<b style='color: #9ca3af; font-size:0.85rem;'>UPLOAD DOCUMENTS</b>", unsafe_allow_html=True)
    # File uploader yahan add kiya jo left panel mein show hoga
    uploaded_file = st.file_uploader("Supports PDF & TXT", type=["pdf", "txt"], label_visibility="collapsed")
    
    if st.button("⚡ Process & Index Documents"):
        if uploaded_file is not None:
            st.success("Document added successfully! (Simulated)")
        else:
            st.warning("Please upload a document first.")
            
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<b style='color: #6b7280; font-size: 0.8rem;'>TECH STACK</b>", unsafe_allow_html=True)
    st.markdown("`Pinecone` `Groq LLM` `FastAPI` `LLaMA` `HuggingFace` `Python`")


# --- 3. Main Content Screen (Center Panel) ---
col1, col2 = st.columns([3, 1])
with col1:
    # Top main area text ko up to date kiya
    st.markdown("<h2 style='font-family:\"Syne\", sans-serif; font-weight:800;'>Chat Workspace</h2>", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="padding-top:10px; text-align:right;">
        <span class="online-badge"><span class="dot"></span>Online</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

@st.cache_resource
def load_rag():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    model = ChatGroq(model="llama-3.3-70b-versatile")
    return db, model

db, model = load_rag()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 4. CHANGES: Welcome screen text ko premium aur clean design diya ---
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;">
        <div style="font-size:3.5rem; margin-bottom:16px;">✨</div>
        <h1 style="font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800;
            background:linear-gradient(135deg, #ffffff, #a5b4fc);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:15px;">
            Welcome to DocuQuery AI
        </h1>
        <p style="color:#9ca3af; font-size:1.05rem; line-height:1.6; max-width:600px; margin:0 auto;">
            Upload your PDF or text documents on the left panel, then ask any question in natural language. 
            I will retrieve the most relevant context and give you precise answers.
        </p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask me anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retriever = db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(prompt)

            combined_input = f"""Based on the following documents, answer: {prompt}
Documents:
{chr(10).join([f"- {doc.page_content}" for doc in docs])}
If you cant find the answer say: I dont have enough information."""

            messages = [
                SystemMessage(content="You are a helpful assistant that answers questions based on provided documents."),
            ] + st.session_state.chat_history + [
                HumanMessage(content=combined_input)
            ]

            result = model.invoke(messages)
            answer = result.content

            st.session_state.chat_history.append(HumanMessage(content=prompt))
            st.session_state.chat_history.append(AIMessage(content=answer))
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.write(answer)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Clear History button layout settings
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

st.markdown('<p class="hint">Press Enter to send • Powered by DocuQuery AI & Groq</p>', unsafe_allow_html=True)