import streamlit as st
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
import pickle

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def load_documents():
    loader = CSVLoader(file_path="Fintech.csv", source_column="content")
    return loader.load()

def get_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store_path = "faiss_store.pkl"
    
    if os.path.exists(vector_store_path):
        with open(vector_store_path, "rb") as f:
            return pickle.load(f)
    
    vector_store = FAISS.from_documents(docs, embeddings)
    with open(vector_store_path, "wb") as f:
        pickle.dump(vector_store, f)
    return vector_store

def setup_rag(vector_store):
    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_api_key)
    retriever = vector_store.as_retriever()
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

st.title("Fintech RAG Chatbot")
st.write("Ask any fintech-related question: ")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

docs = load_documents()
vector_store = get_vector_store(docs)
qa_chain = setup_rag(vector_store)

user_query = st.text_input("Enter your question below:")
if user_query:
    response = qa_chain.invoke({
        "question": user_query, 
        "chat_history": st.session_state.chat_history
    })
    
    st.session_state.chat_history.append((user_query, response["answer"]))
    st.write("**Answer:**", response["answer"])
