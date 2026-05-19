import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader  # NEW IMPORT FOR PDF
from dotenv import load_dotenv

load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="DocuQuery AI", page_icon="📝", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght=700;800&family=DM+Sans:wght=300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0b0c16 !important; color: #e8e8f0 !important; }
.stApp { background: #0b0c16; }
#MainMenu, footer {visibility: hidden;}
[data-testid="stSidebar"] { background-color: #0f1123 !important; border-right: 1px solid #1f2347; }
[data-testid="stChatMessage"] { background: #1c1c27 !important; border: 1px solid #2a2a3d !important; border-radius: 16px !important; padding: 14px 18px !important; margin-bottom: 12px !important; color: #e8e8f0 !important; }
[data-testid="stChatInputTextArea"] { background: #1c1c27 !important; color: #e8e8f0 !important; }
.stChatInputContainer { background: #1c1c27 !important; border: 1px solid #2a2a3d !important; border-radius: 14px !important; }
.stButton > button { background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 10px 24px !important; font-weight: bold !important; width: 100% !important; box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important; }
code { background-color: #1e1b4b !important; color: #a5b4fc !important; border: 1px solid #312e81 !important; padding: 4px 8px !important; border-radius: 6px !important; }
.online-badge { display: inline-flex; align-items: center; gap: 6px; background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.3); border-radius: 20px; padding: 4px 12px; font-size: 0.75rem; color: #4ade80; font-weight: 500; }
.dot { width: 7px; height: 7px; background: #4ade80; border-radius: 50%; box-shadow: 0 0 6px #4ade80; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
.logo-area { display:flex; align-items:center; gap:12px; padding:10px 0 24px 0; }
.logo-icon { width:42px; height:42px; background:linear-gradient(135deg,#4f46e5,#7c3aed); border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:20px; }
.logo-title { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; background:linear-gradient(135deg,#ffffff,#a5b4fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0; }
.logo-sub { font-size:0.75rem; color:#6b6b8a; margin:0; }
.divider { border:none; border-top:1px solid #2a2a3d; margin:8px 0 20px 0; }
.hint { text-align:center; font-size:0.72rem; color:#6b6b8a; margin-top:8px; }
</style>
""", unsafe_allow_html=True)

# --- Embedding & LLM Setup ---
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = Chroma(
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.3-70b-versatile")

model = load_llm()

# --- Sidebar UI Panel ---
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
    
    # MODIFIED: Ab yeh PDF aur TXT dono formats accept karega
    uploaded_file = st.file_uploader("Supports PDF & TXT", type=["pdf", "txt"], label_visibility="collapsed")
    
    if st.button("⚡ Process & Index Documents"):
        if uploaded_file is not None:
            with st.spinner("Processing document and generating vector embeddings..."):
                try:
                    raw_text = ""
                    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
                    
                    # 1. Agar file PDF hai toh aise read karein
                    if file_ext == ".pdf":
                        pdf_reader = PdfReader(uploaded_file)
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                raw_text += page_text + "\n"
                    
                    # 2. Agar file TXT hai toh aise read karein
                    elif file_ext == ".txt":
                        raw_text = uploaded_file.read().decode("utf-8")
                    
                    if not raw_text.strip():
                        st.error("The document seems to be empty or non-readable.")
                        st.stop()
                    
                    # 3. Text Chunks (Splitting)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
                    chunks = text_splitter.split_text(raw_text)
                    
                    # 4. Save to vector store
                    st.session_state.vector_store = Chroma.from_texts(
                        texts=chunks,
                        embedding=embedding_model,
                        collection_metadata={"hnsw:space": "cosine"}
                    )
                    
                    st.success(f"🎉 '{uploaded_file.name}' successfully vectorized and stored!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error processing file: {e}")
        else:
            st.warning("Please upload a document first.")
            
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<b style='color: #6b7280; font-size: 0.8rem;'>TECH STACK</b>", unsafe_allow_html=True)
    st.markdown("`Chroma DB` `Groq LLM` `FastAPI` `LLaMA 3.3` `HuggingFace` `Python`")

# --- Main Chat Screen Workspace ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<h2 style='font-family:\"Syne\", sans-serif; font-weight:800;'>Chat Workspace</h2>", unsafe_allow_html=True)
with col2:
    st.markdown("""<div style="padding-top:10px; text-align:right;"><span class="online-badge"><span class="dot"></span>Online</span></div>""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "messages" not in st.session_state: st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:60px 20px;">
        <div style="font-size:3.5rem; margin-bottom:16px;">✨</div>
        <h1 style="font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800; background:linear-gradient(135deg, #ffffff, #a5b4fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:15px;">Welcome to DocuQuery AI</h1>
        <p style="color:#9ca3af; font-size:1.05rem; line-height:1.6; max-width:600px; margin:0 auto;">Upload your PDF or TXT documents on the left panel, click Process, then ask any question. I will retrieve the most relevant context and give you precise answers.</p>
    </div>
    """, unsafe_allow_html=True)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Ask me anything about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            db_instance = st.session_state.vector_store
            retriever = db_instance.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(prompt)
            
            context = "\n".join([doc.page_content for doc in docs])

            combined_input = f"""Based on the following documents content, answer the user question accurately.
            
Documents Content:
{context}

Question: {prompt}

If you cannot find the answer in the provided documents content, strictly say: I dont have enough information."""

            messages = [
                SystemMessage(content="You are a helpful assistant that answers questions based on provided documents."),
            ] + st.session_state.chat_history + [HumanMessage(content=combined_input)]

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
        st.session_state.vector_store = Chroma(embedding_function=embedding_model, collection_metadata={"hnsw:space": "cosine"})
        st.rerun()

st.markdown('<p class="hint">Press Enter to send • Powered by DocuQuery AI & Groq</p>', unsafe_allow_html=True)