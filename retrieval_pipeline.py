from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os

load_dotenv()
load_dotenv(dotenv_path=r"C:\Users\DELL\Desktop\Rag system\Rag system\.env")

print("API KEY:", os.getenv("GROQ_API_KEY"))
persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

# Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # ✅ Naya model
    api_key=os.getenv("GROQ_API_KEY")
)

# Define the messages for the model
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

# Invoke the model
result = llm.invoke(messages)  # ✅ llm use kiya

print("\n--- Generated Response ---")
print("Content only:")
print(result.content)