# 📄 RAG-based PDF Question Answering with Watsonx & Gradio

This project implements a **Retrieval-Augmented Generation (RAG)** system that allows users to upload a PDF document and ask natural language questions.  
The system retrieves relevant document chunks and generates answers using **IBM watsonx foundation models**.

---

## 🚀 Features

- Upload PDF documents
- Ask questions about the document content
- Retrieval-Augmented Generation (RAG)
- Vector database using **Chroma**
- Embeddings and LLM powered by **IBM watsonx**
- Interactive UI using **Gradio**

---

## 🧠 Architecture Overview

1. **PDF Loader** – Loads uploaded PDF
2. **Text Splitter** – Splits text into overlapping chunks
3. **Embedding Model** – Converts chunks to embeddings (watsonx)
4. **Vector Store** – Stores embeddings using Chroma
5. **Retriever** – Retrieves relevant chunks
6. **LLM** – Generates answers using retrieved context
7. **Gradio UI** – User interaction

---

## 🛠 Tech Stack

- Python
- IBM watsonx.ai
- LangChain
- ChromaDB
- Gradio
- Hugging Face Hub

---

## 📦 Installation

```bash
git clone https://github.com/vannyminanda888/rag-watsonx-pdf-chatbot.git
cd rag-watsonx-pdf-chatbot
pip install -r requirements.txt
