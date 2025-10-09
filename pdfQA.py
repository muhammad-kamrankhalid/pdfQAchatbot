import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()
store = {}

retriever = None
conversational_rag_chain = None


# ============================
# 📄 Helper: Create Vectorstore from PDF
# ============================
def create_vectorstore_from_pdf(pdf: UploadFile):
    """Loads a PDF, splits into chunks, embeds, and stores in FAISS."""
    temp_pdf_path = f"temp_{pdf.filename}"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf.file.read())

    # Load and split PDF
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()

    # Create embeddings (you can switch model if needed)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Cleanup
    os.remove(temp_pdf_path)

    return vectorstore


# ============================
# 🧠 Session Chat History
# ============================
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create a chat message history for a given session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ============================
# 🏠 Root Route
# ============================
@app.get("/")
async def root():
    return {"message": "✅ FastAPI PDF Q&A backend is running successfully!"}


# ============================
# 📤 Upload PDF
# ============================
@app.post("/upload_pdf/")
async def upload_pdf(pdf: UploadFile = File(...), session_id: str = Form("default_session")):
    """
    Upload a PDF and create a vectorstore retriever.
    The Groq API key is loaded automatically from the .env file.
    """
    global retriever, conversational_rag_chain

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return JSONResponse({"error": "❌ GROQ_API_KEY not found in .env file."}, status_code=400)

    # Initialize LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

    # Create vectorstore retriever from uploaded PDF
    vectorstore = create_vectorstore_from_pdf(pdf)
    retriever = vectorstore.as_retriever()

    # Build contextual retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question "
                   "which might reference context in the chat history, "
                   "formulate a standalone question. Do NOT answer the question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    # Create history-aware retriever
    from langchain.chains import create_history_aware_retriever
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Create QA chain
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant for question-answering tasks. "
                   "Use retrieved context to answer concisely. "
                   "If you don't know, say you don't know.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Bind with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return {"message": "✅ PDF uploaded and processed successfully.", "session_id": session_id}


# ============================
# 💬 Ask Questions about PDF
# ============================
@app.post("/ask/")
async def ask_question(question: str = Form(...), session_id: str = Form("default_session")):
    """
    Ask a question related to the uploaded PDF.
    Requires that the PDF has been uploaded first.
    """
    global conversational_rag_chain

    if conversational_rag_chain is None:
        return JSONResponse({"error": "❌ Please upload a PDF first using /upload_pdf/."}, status_code=400)

    # Run question through RAG pipeline
    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}}
    )

    return {"question": question, "answer": response["answer"]}
