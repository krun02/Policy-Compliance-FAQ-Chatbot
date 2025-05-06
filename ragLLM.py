from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

#  Load your documents
doc_path = Path("docs")
all_docs = []
for file in doc_path.glob("*.txt"):
    loader = TextLoader(str(file))
    all_docs.extend(loader.load())

#  Chunk the text
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

#  Vectorize using local embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever()

#  Use Ollama LLM (local)
llm = OllamaLLM(model="mistral")
  # You can also try "llama3" or others

#  Create the Retrieval-Augmented QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

#  Chat loop
print("\n Policy & Compliance RAG Chatbot (Powered by Ollama)")
while True:
    query = input("\nAsk your policy question (or type 'exit'): ")
    if query.lower().strip() == "exit":
        print("👋 Goodbye!")
        break
    answer = qa_chain.invoke({"query": query})

    print(" Answer:", answer)
