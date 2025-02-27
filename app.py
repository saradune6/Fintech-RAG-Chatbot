import streamlit as st
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Load documents
def load_documents():
    loader = CSVLoader("Fintech.csv", source_column="content")
    return loader.load()

# Create vector store
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Setup LLM and RAG chain
def setup_rag(vector_store):
    llm = ChatGroq(model_name="llama3-8b-8192", groq_api_key=groq_api_key)
    retriever = vector_store.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    return qa_chain

# Streamlit UI
st.title("Fintech RAG Chatbot")
user_query = st.text_input("Ask a fintech-related question:")

docs = load_documents()
vector_store = create_vector_store(docs)
qa_chain = setup_rag(vector_store)

if user_query:
    response = qa_chain.invoke({"question": user_query, "chat_history": []})
    st.write(response["answer"])