# Policy-Compliance-FAQ-Chatbot
Offline RAG chatbot for querying SOC 2, HIPAA, and NIST policies. Built with LangChain, FAISS, HuggingFace embeddings, and Mistral (Ollama). Streamlit UI for a private, secure, and resume-ready compliance assistant.


## 💡 What It Can Do
- Ask questions like:  
  “What is HIPAA's policy on data retention?”  
  “What does SOC 2 say about access control?”

- Gives answers based on your documents
- Runs completely offline using **Ollama** (local AI model)
- Simple chat interface built with **Streamlit**

## 🧠 How It Works

1. Loads `.txt` files from the `docs/` folder (these are your policies)
2. Splits them into smaller chunks
3. Converts the chunks into vectors (smart searchable format)
4. When you ask a question:
   - It searches the most relevant chunks
   - Sends them to the local AI model (like **Mistral*)
   - The model reads the info and gives an answer
