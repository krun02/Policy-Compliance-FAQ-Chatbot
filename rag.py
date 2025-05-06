import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

#  NO OpenAI key needed anymore
# os.environ["OPENAI_API_KEY"] = "your-key" ← REMOVE this line!

#  Load .txt files from docs/ folder
doc_path = Path("docs")
text_files = list(doc_path.glob("*.txt"))

if not doc_path.exists() or not text_files:
    print("[ ERROR] No .txt files found in the 'docs/' folder.")
    exit()

all_docs = []
for file in text_files:
    loader = TextLoader(str(file))
    all_docs.extend(loader.load())

print(f"[ INFO] Loaded {len(all_docs)} document(s).")

#  Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)
print(f"[ INFO] Split into {len(chunks)} chunks.")

if not chunks:
    print("[ ERROR] No chunks created. Check .txt content.")
    exit()

#  Use FREE local embeddings (no API!)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
retriever = vectorstore.as_retriever()

#  Use dummy LLM (prints context for now since OpenAI is off)
def dummy_answer(context, question):
    print(f"\n Retrieved Context:\n{context}")
    return "This is a placeholder answer. Plug in a real LLM if needed."

#  Simple loop interface
print("\n Policy & Compliance FAQ Bot (Offline Version) ")
while True:
    query = input("\nAsk a policy question (or type 'exit'): ")
    if query.lower().strip() == "exit":
        print(" Goodbye!")
        break
    docs = retriever.get_relevant_documents(query)
    context = "\n---\n".join([doc.page_content for doc in docs])
    print(" Answer:", dummy_answer(context, query))
