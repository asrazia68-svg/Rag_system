from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = Chroma(
    persist_directory="db/chroma_db",
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)
model = ChatGroq(model="llama-3.3-70b-versatile")
chat_history = []

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/ask")
def ask(data: Question):
    global chat_history
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(data.question)
    
    combined_input = f"""Based on the following documents, answer this question: {data.question}
    Documents:
    {chr(10).join([f"- {doc.page_content}" for doc in docs])}
    If you cant find the answer say: I dont have enough information.
    """
    
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    chat_history.append(HumanMessage(content=data.question))
    chat_history.append(AIMessage(content=answer))
    
    return {"answer": answer}

@app.post("/clear")
def clear_history():
    global chat_history
    chat_history = []
    return {"message": "History cleared!"}

app.mount("/static", StaticFiles(directory="static"), name="static")