import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Chat", page_icon="🧠", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f !important;
    color: #e8e8f0 !important;
}
.stApp { background: #0a0a0f; }
#MainMenu, footer {visibility: hidden;}

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
.stButton > button {
    background: linear-gradient(135deg, #7c6aff, #ff6a9b) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
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
    background:linear-gradient(135deg,#7c6aff,#ff6a9b);
    border-radius:12px; display:flex; align-items:center;
    justify-content:center; font-size:20px;
}
.logo-title {
    font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:800;
    background:linear-gradient(135deg,#7c6aff,#ff6a9b);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;
}
.logo-sub { font-size:0.72rem; color:#6b6b8a; margin:0; }
.divider { border:none; border-top:1px solid #2a2a3d; margin:8px 0 20px 0; }
.hint { text-align:center; font-size:0.72rem; color:#6b6b8a; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div class="logo-area">
        <div class="logo-icon">🧠</div>
        <div>
            <p class="logo-title">RAG Chat</p>
            <p class="logo-sub">Document Intelligence</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="padding-top:18px; text-align:right;">
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

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:40px 20px;">
        <div style="font-size:3rem; margin-bottom:16px;">✨</div>
        <h2 style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700;
            background:linear-gradient(135deg,#7c6aff,#ff6a9b);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            Ask Your Documents!
        </h2>
        <p style="color:#6b6b8a; font-size:0.9rem; line-height:1.6;">
            I have read all your documents.<br>Ask me anything — I will answer!
        </p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
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

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

st.markdown('<p class="hint">Press Enter to send • RAG powered by Groq</p>', unsafe_allow_html=True)