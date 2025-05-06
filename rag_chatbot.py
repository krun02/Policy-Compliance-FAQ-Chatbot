import os
from pathlib import Path
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Load documents and embed once (caching with Streamlit)
@st.cache_resource
def load_bot():
    
    # Load docs
    doc_path = Path("docs")
    all_docs = []
    for file in doc_path.glob("*.txt"):
        loader = TextLoader(str(file))
        all_docs.extend(loader.load())

    # Split chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    # Embedding and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # LLM via Ollama
    llm = OllamaLLM(model="mistral")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

#  UI Layout
st.set_page_config(page_title="GRC RAG Chatbot", layout="centered")
st.title(" Policy & Compliance FAQ Chatbot")
st.markdown("Ask questions about SOC 2, HIPAA, or NIST policies.")

qa_chain = load_bot()

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat form
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="e.g. What does HIPAA say about PHI?")
    submitted = st.form_submit_button("Ask")

if submitted and user_input:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({"query": user_input})
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

# Display chat history
for speaker, text in st.session_state.chat_history:
    st.chat_message(speaker).write(text)
